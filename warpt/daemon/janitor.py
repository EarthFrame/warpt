"""Janitor — bounded data lifecycle for the node-local DuckDB.

Keeps the Case File from growing without bound on long-lived nodes:

- ``vitals`` rows older than ``retention.vitals_days`` are pruned (cases,
  events, tool calls, and actions are the durable record; raw vitals are
  high-volume telemetry).
- Closed cases older than ``retention.closed_cases_days`` are pruned along
  with their events/tool_calls/actions.
- A ``CHECKPOINT`` after pruning lets DuckDB reclaim space.

Open cases are never touched.
"""

from __future__ import annotations

import threading
from typing import Any

from warpt.daemon.casefile import CaseFile
from warpt.utils.logger import Logger

DEFAULT_VITALS_DAYS = 14
DEFAULT_CLOSED_CASES_DAYS = 90
DEFAULT_INTERVAL_H = 6.0


class Janitor:
    """Periodic retention pruning on the node's Case File.

    Parameters
    ----------
    casefile
        CaseFile to prune (shared daemon connection).
    config
        Full daemon config; reads the ``retention`` block.
    """

    def __init__(self, casefile: CaseFile, config: dict[str, Any]) -> None:
        retention = config.get("retention", {}) or {}
        self._casefile = casefile
        self._vitals_days = float(retention.get("vitals_days", DEFAULT_VITALS_DAYS))
        self._closed_cases_days = float(
            retention.get("closed_cases_days", DEFAULT_CLOSED_CASES_DAYS)
        )
        self._interval_s = (
            float(retention.get("interval_h", DEFAULT_INTERVAL_H)) * 3600.0
        )
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._log = Logger.get("daemon.janitor")

    def start(self) -> None:
        """Start the pruning loop (first pass runs immediately)."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="daemon-janitor", daemon=True
        )
        self._thread.start()
        self._log.info(
            "Janitor started (vitals>%gd pruned, closed cases>%gd pruned, "
            "every %.1fh)",
            self._vitals_days,
            self._closed_cases_days,
            self._interval_s / 3600.0,
        )

    def stop(self) -> None:
        """Stop the pruning loop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.prune_once()
            except Exception:
                self._log.exception("Prune pass failed (will retry)")
            self._stop_event.wait(self._interval_s)

    def prune_once(self) -> dict[str, int]:
        """Run one prune pass; returns per-table deleted counts."""
        deleted: dict[str, int] = {}

        before = self._count("vitals")
        self._casefile.execute(
            f"""
            DELETE FROM vitals
            WHERE ts < current_timestamp - INTERVAL '{self._vitals_days} DAYS'
            """
        )
        deleted["vitals"] = before - self._count("vitals")

        # Expired closed cases and their satellite rows.
        rows = self._casefile.query(
            f"""
            SELECT case_id FROM cases
            WHERE status != 'open'
              AND closed_at IS NOT NULL
              AND closed_at <
                  current_timestamp - INTERVAL '{self._closed_cases_days} DAYS'
            """
        )
        case_ids = [r[0] for r in rows]
        for table, column in (
            ("events", "case_id"),
            ("tool_calls", "case_id"),
            ("actions", "case_id"),
            ("cases", "case_id"),
        ):
            count = 0
            for case_id in case_ids:
                before = self._count(table)
                self._casefile.execute(
                    f"DELETE FROM {table} WHERE {column} = ?", [case_id]
                )
                count += before - self._count(table)
            deleted[table] = count

        try:
            self._casefile.execute("CHECKPOINT")
        except Exception:
            # In-memory DBs and some states don't support CHECKPOINT.
            self._log.debug("CHECKPOINT skipped")

        if any(deleted.values()):
            self._log.info(
                "Pruned: %s",
                ", ".join(f"{k}={v}" for k, v in deleted.items() if v),
            )
        return deleted

    def _count(self, table: str) -> int:
        return self._casefile.query(f"SELECT count(*) FROM {table}")[0][0]
