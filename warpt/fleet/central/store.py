"""Fleet store — multi-writer store behind the central ingest/dashboard API.

SQLAlchemy Core against any supported URL: ``postgresql://`` (with
TimescaleDB for scale) in production, ``sqlite:///`` for dev and small
fleets. Do **not** point this at the node's DuckDB — that stays node-local.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
    text,
)

from warpt.utils.logger import Logger

metadata = MetaData()

nodes = Table(
    "nodes",
    metadata,
    Column("node_id", String(64), primary_key=True),
    Column("hostname", String(255)),
    Column("first_seen", DateTime, default=datetime.utcnow),
    Column("last_seen", DateTime),
    Column("last_heartbeat", DateTime),
    # Lifetime energy odometer totals (from heartbeat payloads).
    Column("energy_kwh", Float),
    Column("co2_grams", Float),
    Column("cost_usd", Float),
    Column("energy_updated", DateTime),
)

fleet_vitals = Table(
    "fleet_vitals",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("node_id", String(64), nullable=False),
    Column("ts", DateTime, nullable=False),
    Column("payload", JSON),
    Index("ix_fleet_vitals_node_ts", "node_id", "ts"),
)

fleet_events = Table(
    "fleet_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("node_id", String(64), nullable=False),
    Column("node_event_id", BigInteger, nullable=False),
    Column("ts", DateTime),
    Column("kind", String(64)),
    Column("severity", String(16)),
    Column("gpu_guid", String(128)),
    Column("summary", Text),
    Column("payload", JSON),
    UniqueConstraint("node_id", "node_event_id", name="uq_events_node"),
    Index("ix_fleet_events_node_ts", "node_id", "ts"),
)

fleet_cases = Table(
    "fleet_cases",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("node_id", String(64), nullable=False),
    Column("node_case_id", BigInteger, nullable=False),
    Column("title", Text),
    Column("status", String(32)),
    Column("severity", String(16)),
    Column("hypothesis", Text),
    Column("confidence_pct", Float),
    Column("opened_at", DateTime),
    Column("updated_at", DateTime),
    Column("payload", JSON),
    UniqueConstraint("node_id", "node_case_id", name="uq_cases_node"),
    Index("ix_fleet_cases_status", "status"),
)

fleet_activity = Table(
    "fleet_activity",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("node_id", String(64), nullable=False),
    Column("ts", DateTime),
    Column("agent", String(64)),
    Column("activity", String(64)),
    Column("case_ref", BigInteger),
    Column("detail", JSON),
    Index("ix_fleet_activity_id", "id"),
)


def _parse_ts(value: Any) -> datetime | None:
    """Best-effort ISO timestamp parse; None on failure."""
    if value is None or isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


class FleetStore:
    """Multi-writer fleet store over SQLAlchemy Core.

    Parameters
    ----------
    url
        Database URL (``postgresql://...`` or ``sqlite:///path``).
    """

    def __init__(self, url: str) -> None:
        connect_args = {}
        engine_kwargs: dict[str, Any] = {}
        if url.startswith("sqlite"):
            # The API serves from multiple threads; sqlite needs this.
            connect_args["check_same_thread"] = False
            if url in ("sqlite://", "sqlite:///:memory:"):
                # Pure in-memory DBs are per-connection; share one connection
                # across threads (dev/test only).
                from sqlalchemy.pool import StaticPool

                engine_kwargs["poolclass"] = StaticPool
        self._engine = create_engine(
            url, future=True, connect_args=connect_args, **engine_kwargs
        )
        metadata.create_all(self._engine)
        self._ensure_energy_columns()
        self._log = Logger.get("fleet.store")
        self._log.info("Fleet store ready: %s", url.split("@")[-1])

    def _ensure_energy_columns(self) -> None:
        """Add odometer columns to pre-existing ``nodes`` tables.

        ``create_all`` only creates missing tables, never missing columns, so
        stores created before the energy odometer need a lightweight ALTER.
        Each statement is independent and failure-tolerant (column exists).
        """
        for column, sql_type in (
            ("energy_kwh", "FLOAT"),
            ("co2_grams", "FLOAT"),
            ("cost_usd", "FLOAT"),
            ("energy_updated", "TIMESTAMP"),
        ):
            try:
                with self._engine.begin() as conn:
                    conn.execute(
                        text(f"ALTER TABLE nodes ADD COLUMN {column} {sql_type}")
                    )
            except Exception:
                pass  # Column already exists.

    # -------------------------------------------------------------- ingest

    def ingest(self, messages: list[dict[str, Any]]) -> int:
        """Ingest a batch of node messages; returns the accepted count."""
        accepted = 0
        with self._engine.begin() as conn:
            for msg in messages:
                kind = msg.get("kind")
                node_id = msg.get("node_id")
                if not node_id or not kind:
                    continue
                ts = _parse_ts(msg.get("ts")) or datetime.utcnow()
                payload = msg.get("payload") or {}
                self._touch_node(
                    conn, node_id, msg.get("hostname", ""), ts, kind, payload
                )
                if kind == "vitals":
                    conn.execute(
                        fleet_vitals.insert().values(
                            node_id=node_id, ts=ts, payload=payload
                        )
                    )
                elif kind == "event":
                    self._ingest_event(conn, node_id, ts, payload)
                elif kind == "case":
                    self._ingest_case(conn, node_id, payload)
                elif kind == "activity":
                    conn.execute(
                        fleet_activity.insert().values(
                            node_id=node_id,
                            ts=ts,
                            agent=payload.get("agent", ""),
                            activity=payload.get("activity", ""),
                            case_ref=payload.get("case_id"),
                            detail=payload,
                        )
                    )
                # heartbeat: node touch above is all it needs
                accepted += 1
        return accepted

    @staticmethod
    def _touch_node(
        conn: Any,
        node_id: str,
        hostname: str,
        ts: datetime,
        kind: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {"last_seen": ts}
        if hostname:
            values["hostname"] = hostname
        if kind == "heartbeat":
            values["last_heartbeat"] = ts
            energy = (payload or {}).get("energy")
            if isinstance(energy, dict) and energy.get("energy_kwh") is not None:
                values["energy_kwh"] = float(energy["energy_kwh"])
                values["co2_grams"] = float(energy.get("co2_grams") or 0.0)
                values["cost_usd"] = float(energy.get("cost_usd") or 0.0)
                values["energy_updated"] = ts
        result = conn.execute(
            nodes.update().where(nodes.c.node_id == node_id).values(**values)
        )
        if result.rowcount == 0:
            conn.execute(
                nodes.insert().values(node_id=node_id, first_seen=ts, **values)
            )

    @staticmethod
    def _ingest_event(
        conn: Any, node_id: str, ts: datetime, payload: dict[str, Any]
    ) -> None:
        node_event_id = payload.get("event_id")
        if node_event_id is None:
            return
        exists = conn.execute(
            select(fleet_events.c.id).where(
                fleet_events.c.node_id == node_id,
                fleet_events.c.node_event_id == int(node_event_id),
            )
        ).first()
        if exists:
            return
        conn.execute(
            fleet_events.insert().values(
                node_id=node_id,
                node_event_id=int(node_event_id),
                ts=ts,
                kind=payload.get("kind", ""),
                severity=payload.get("severity", ""),
                gpu_guid=payload.get("gpu_guid"),
                summary=payload.get("summary", ""),
                payload=payload,
            )
        )

    @staticmethod
    def _ingest_case(conn: Any, node_id: str, payload: dict[str, Any]) -> None:
        node_case_id = payload.get("case_id")
        if node_case_id is None:
            return
        values = {
            "title": payload.get("title", ""),
            "status": payload.get("status", ""),
            "severity": payload.get("severity"),
            "hypothesis": payload.get("hypothesis"),
            "confidence_pct": payload.get("confidence_pct"),
            "opened_at": _parse_ts(payload.get("opened_at")),
            "updated_at": _parse_ts(payload.get("updated_at")),
            "payload": payload,
        }
        result = conn.execute(
            fleet_cases.update()
            .where(
                fleet_cases.c.node_id == node_id,
                fleet_cases.c.node_case_id == int(node_case_id),
            )
            .values(**values)
        )
        if result.rowcount == 0:
            conn.execute(
                fleet_cases.insert().values(
                    node_id=node_id,
                    node_case_id=int(node_case_id),
                    **values,
                )
            )

    # ------------------------------------------------------------- queries

    def list_nodes(self) -> list[dict[str, Any]]:
        """All nodes with open-case counts and health rollup."""
        with self._engine.connect() as conn:
            rows = conn.execute(select(nodes).order_by(nodes.c.hostname)).mappings()
            result = []
            for row in rows:
                open_cases = conn.execute(
                    select(func.count())
                    .select_from(fleet_cases)
                    .where(
                        fleet_cases.c.node_id == row["node_id"],
                        fleet_cases.c.status == "open",
                    )
                ).scalar_one()
                critical = conn.execute(
                    select(func.count())
                    .select_from(fleet_cases)
                    .where(
                        fleet_cases.c.node_id == row["node_id"],
                        fleet_cases.c.status == "open",
                        fleet_cases.c.severity == "critical",
                    )
                ).scalar_one()
                result.append(
                    {
                        "node_id": row["node_id"],
                        "hostname": row["hostname"],
                        "first_seen": _iso(row["first_seen"]),
                        "last_seen": _iso(row["last_seen"]),
                        "last_heartbeat": _iso(row["last_heartbeat"]),
                        "open_cases": open_cases,
                        "critical_cases": critical,
                        "energy_kwh": row["energy_kwh"],
                        "co2_grams": row["co2_grams"],
                        "cost_usd": row["cost_usd"],
                        "energy_updated": _iso(row["energy_updated"]),
                    }
                )
            return result

    def get_cases(
        self,
        status: str | None = None,
        severity: str | None = None,
        node_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Fleet-wide case query — e.g. every open critical case."""
        stmt = select(fleet_cases)
        if status:
            stmt = stmt.where(fleet_cases.c.status == status)
        if severity:
            stmt = stmt.where(fleet_cases.c.severity == severity)
        if node_id:
            stmt = stmt.where(fleet_cases.c.node_id == node_id)
        stmt = stmt.order_by(fleet_cases.c.updated_at.desc()).limit(limit)
        with self._engine.connect() as conn:
            return [
                {
                    "node_id": r["node_id"],
                    "case_id": r["node_case_id"],
                    "title": r["title"],
                    "status": r["status"],
                    "severity": r["severity"],
                    "hypothesis": r["hypothesis"],
                    "confidence_pct": r["confidence_pct"],
                    "opened_at": _iso(r["opened_at"]),
                    "updated_at": _iso(r["updated_at"]),
                    "detail": r["payload"],
                }
                for r in conn.execute(stmt).mappings()
            ]

    def get_vitals(
        self, node_id: str, hours: float = 1.0, limit: int = 2000
    ) -> list[dict[str, Any]]:
        """Recent vitals payloads for one node."""
        since = datetime.utcnow() - timedelta(hours=hours)
        stmt = (
            select(fleet_vitals)
            .where(
                fleet_vitals.c.node_id == node_id,
                fleet_vitals.c.ts > since,
            )
            .order_by(fleet_vitals.c.ts)
            .limit(limit)
        )
        with self._engine.connect() as conn:
            return [
                {"ts": _iso(r["ts"]), **(r["payload"] or {})}
                for r in conn.execute(stmt).mappings()
            ]

    def get_activity(self, since_id: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        """Agent-activity entries after *since_id* (the live stream feed)."""
        stmt = (
            select(fleet_activity)
            .where(fleet_activity.c.id > since_id)
            .order_by(fleet_activity.c.id)
            .limit(limit)
        )
        with self._engine.connect() as conn:
            return [
                {
                    "id": r["id"],
                    "node_id": r["node_id"],
                    "ts": _iso(r["ts"]),
                    "agent": r["agent"],
                    "activity": r["activity"],
                    "case_id": r["case_ref"],
                    "detail": r["detail"],
                }
                for r in conn.execute(stmt).mappings()
            ]

    def max_activity_id(self) -> int:
        """Return the high-water mark of the activity stream."""
        with self._engine.connect() as conn:
            value = conn.execute(select(func.max(fleet_activity.c.id))).scalar_one()
            return int(value or 0)

    def get_events(
        self, node_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Recent events, fleet-wide or for one node."""
        stmt = select(fleet_events)
        if node_id:
            stmt = stmt.where(fleet_events.c.node_id == node_id)
        stmt = stmt.order_by(fleet_events.c.ts.desc()).limit(limit)
        with self._engine.connect() as conn:
            return [
                {
                    "node_id": r["node_id"],
                    "event_id": r["node_event_id"],
                    "ts": _iso(r["ts"]),
                    "kind": r["kind"],
                    "severity": r["severity"],
                    "gpu_guid": r["gpu_guid"],
                    "summary": r["summary"],
                }
                for r in conn.execute(stmt).mappings()
            ]

    def healthy(self) -> bool:
        """Return True when the database answers a trivial query."""
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _json_safe(value: Any) -> Any:  # pragma: no cover - trivial
    """Round-trip through JSON to coerce non-serializable values."""
    return json.loads(json.dumps(value, default=str))
