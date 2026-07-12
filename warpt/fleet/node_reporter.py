"""NodeReporter — buffered push of node data to the central control plane.

Design rules (see ER_PRODUCTION_PLAN.md Phase 3):

- **Node autonomy is sacred.** Every failure here is logged and swallowed;
  the daemon's observe → diagnose → report loop never depends on central.
- **Buffered + retried on disk.** New rows are appended to a JSONL outbox
  before any network attempt; when central is unreachable the outbox grows
  (bounded by ``fleet.max_buffer_mb``, oldest dropped first) and backfills
  on reconnect.
- **Cursor-tracked.** A small state file records how far each table has been
  exported, so restarts neither duplicate nor skip data.

The agent-activity stream is derived from the Phase-2 audit tables
(``tool_calls``, ``actions``) plus events — what each agent did and why,
without new plumbing on the hot path.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import uuid
from pathlib import Path
from typing import Any

import requests

from warpt.daemon.casefile import CaseFile
from warpt.utils.logger import Logger

DEFAULT_PUSH_INTERVAL_S = 30.0
DEFAULT_MAX_BUFFER_MB = 64
_BATCH_LIMIT = 500  # messages per POST
_ROW_LIMIT = 500  # max rows pulled per table per cycle

_VITALS_COLUMNS = (
    "ts",
    "cpu_utilization_pct",
    "cpu_power_w",
    "mem_total_bytes",
    "mem_available_bytes",
    "mem_wired_bytes",
    "mem_utilization_pct",
    "gpus",
    "total_power_w",
    "collection_type",
)

_EVENT_COLUMNS = (
    "event_id",
    "ts",
    "kind",
    "severity",
    "gpu_guid",
    "summary",
    "metadata",
    "case_id",
    "triggered_by",
)

_CASE_COLUMNS = (
    "case_id",
    "title",
    "status",
    "opened_at",
    "closed_at",
    "updated_at",
    "hypothesis",
    "severity",
    "confidence_pct",
    "recommended_action",
    "reasoning_chain",
    "evidence",
    "tools_used",
    "baseline_deviation_pct",
    "report_content",
    "diagnostician_model",
    "prompt_snapshot",
)


def get_node_id(warpt_dir: str) -> str:
    """Return the stable node id, generating and persisting one if absent.

    Parameters
    ----------
    warpt_dir
        The warpt data directory (e.g. ``~/.warpt``).
    """
    path = Path(warpt_dir) / "node_id"
    if path.exists():
        node_id = path.read_text().strip()
        if node_id:
            return node_id
    node_id = uuid.uuid4().hex
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(node_id)
    return node_id


class NodeReporter:
    """Pushes vitals, events, cases, and agent activity to central.

    Parameters
    ----------
    casefile
        CaseFile for reading node-local data (shared connection; queries
        are serialized by DuckDB like the other daemon threads).
    config
        Full daemon config dict; reads the ``fleet`` block.
    warpt_dir
        The warpt data directory for outbox/state/node-id files.
    """

    def __init__(
        self,
        casefile: CaseFile,
        config: dict[str, Any],
        warpt_dir: str,
    ) -> None:
        fleet = config.get("fleet", {}) or {}
        self._casefile = casefile
        self._central_url = str(fleet.get("central_url", "")).rstrip("/")
        self._push_interval = float(
            fleet.get("push_interval_s", DEFAULT_PUSH_INTERVAL_S)
        )
        self._max_buffer_bytes = (
            int(fleet.get("max_buffer_mb", DEFAULT_MAX_BUFFER_MB)) * 1024 * 1024
        )
        token_env = fleet.get("token_env", "WARPT_FLEET_TOKEN")
        self._token = os.environ.get(token_env, "").strip() or None

        self._warpt_dir = Path(warpt_dir)
        self._outbox_dir = self._warpt_dir / "fleet_outbox"
        self._state_path = self._warpt_dir / "fleet_state.json"
        self._node_id = get_node_id(warpt_dir)
        self._hostname = socket.gethostname()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._outbox_counter = 0
        self._log = Logger.get("fleet.node_reporter")

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Start the reporter thread."""
        self._outbox_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="fleet-reporter", daemon=True
        )
        self._thread.start()
        self._log.info(
            "NodeReporter started (node=%s, central=%s, interval=%.0fs)",
            self._node_id[:12],
            self._central_url or "(unset)",
            self._push_interval,
        )

    def stop(self) -> None:
        """Stop the reporter thread (final cycle finishes or times out)."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._thread = None
        self._log.info("NodeReporter stopped.")

    def _run_loop(self) -> None:
        """Collect → buffer → flush on a fixed interval until stopped."""
        while not self._stop_event.is_set():
            try:
                self.run_cycle()
            except Exception:
                # Autonomy rule: reporting must never hurt the node.
                self._log.exception("Reporter cycle failed (node unaffected)")
            self._stop_event.wait(self._push_interval)

    # ----------------------------------------------------------- one cycle

    def run_cycle(self) -> dict[str, int]:
        """Run one collect/buffer/flush cycle.

        Returns
        -------
            ``{"collected": n, "flushed": m}`` counters (for status/tests).
        """
        state = self._load_state()
        messages = self._collect(state)
        if messages:
            self._append_outbox(messages)
            self._save_state(state)
        self._enforce_buffer_cap()
        flushed = self._flush_outbox()
        return {"collected": len(messages), "flushed": flushed}

    # ------------------------------------------------------------- collect

    def _collect(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Pull new rows since the cursors in *state*; advance cursors."""
        messages: list[dict[str, Any]] = [
            self._message("heartbeat", self._now_iso(), {"status": "ok"})
        ]
        messages.extend(self._collect_vitals(state))
        messages.extend(self._collect_events(state))
        messages.extend(self._collect_cases(state))
        messages.extend(self._collect_activity(state))
        return messages

    def _collect_vitals(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        last_ts = state.get("last_vitals_ts")
        clause = "WHERE ts > ?::TIMESTAMP" if last_ts else ""
        params = [last_ts] if last_ts else None
        rows = self._casefile.query(
            f"""
            SELECT {", ".join(_VITALS_COLUMNS)} FROM vitals {clause}
            ORDER BY ts LIMIT {_ROW_LIMIT}
            """,
            params,
        )
        messages = []
        for row in rows:
            payload = dict(zip(_VITALS_COLUMNS, row, strict=True))
            ts = str(payload["ts"])
            payload["ts"] = ts
            messages.append(self._message("vitals", ts, payload))
            state["last_vitals_ts"] = ts
        return messages

    def _collect_events(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        last_id = int(state.get("last_event_id", 0))
        rows = self._casefile.query(
            f"""
            SELECT {", ".join(_EVENT_COLUMNS)} FROM events
            WHERE event_id > ? ORDER BY event_id LIMIT {_ROW_LIMIT}
            """,
            [last_id],
        )
        messages = []
        for row in rows:
            payload = dict(zip(_EVENT_COLUMNS, row, strict=True))
            payload["ts"] = str(payload["ts"])
            messages.append(self._message("event", payload["ts"], payload))
            state["last_event_id"] = int(payload["event_id"])
        return messages

    def _collect_cases(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        last_updated = state.get("last_case_updated_at")
        clause = "WHERE updated_at > ?::TIMESTAMP" if last_updated else ""
        params = [last_updated] if last_updated else None
        rows = self._casefile.query(
            f"""
            SELECT {", ".join(_CASE_COLUMNS)} FROM cases {clause}
            ORDER BY updated_at LIMIT {_ROW_LIMIT}
            """,
            params,
        )
        messages = []
        for row in rows:
            payload = {
                col: (str(val) if col.endswith("_at") and val is not None else val)
                for col, val in zip(_CASE_COLUMNS, row, strict=True)
            }
            messages.append(self._message("case", payload["updated_at"], payload))
            state["last_case_updated_at"] = payload["updated_at"]
        return messages

    def _collect_activity(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Derive the agent-activity stream from the Phase-2 audit tables."""
        messages: list[dict[str, Any]] = []

        last_tc = int(state.get("last_tool_call_id", 0))
        rows = self._casefile.query(
            f"""
            SELECT tool_call_id, case_id, ts, tool_name, arguments, status,
                   latency_ms
            FROM tool_calls WHERE tool_call_id > ?
            ORDER BY tool_call_id LIMIT {_ROW_LIMIT}
            """,
            [last_tc],
        )
        for tc_id, case_id, ts, tool_name, arguments, status, latency in rows:
            messages.append(
                self._message(
                    "activity",
                    str(ts),
                    {
                        "agent": "attending",
                        "activity": "tool_call",
                        "case_id": case_id,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "status": status,
                        "latency_ms": latency,
                    },
                )
            )
            state["last_tool_call_id"] = int(tc_id)

        last_action = int(state.get("last_action_id", 0))
        rows = self._casefile.query(
            f"""
            SELECT action_id, case_id, ts, action_type, target, reason,
                   policy_verdict, status
            FROM actions WHERE action_id > ?
            ORDER BY action_id LIMIT {_ROW_LIMIT}
            """,
            [last_action],
        )
        for a_id, case_id, ts, action_type, target, reason, verdict, status in rows:
            messages.append(
                self._message(
                    "activity",
                    str(ts),
                    {
                        "agent": "attending",
                        "activity": "proposed_action",
                        "case_id": case_id,
                        "action_type": action_type,
                        "target": target,
                        "reason": reason,
                        "policy_verdict": verdict,
                        "status": status,
                    },
                )
            )
            state["last_action_id"] = int(a_id)

        return messages

    def _message(self, kind: str, ts: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "node_id": self._node_id,
            "hostname": self._hostname,
            "kind": kind,
            "ts": ts,
            "payload": payload,
        }

    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime

        return datetime.now().isoformat()

    # -------------------------------------------------------------- outbox

    def _append_outbox(self, messages: list[dict[str, Any]]) -> None:
        """Append messages to a new JSONL outbox file (crash-safe buffer)."""
        self._outbox_counter += 1
        import time as _time

        stamp = int(_time.time() * 1000)
        name = f"outbox-{stamp:015d}-{self._outbox_counter:06d}.jsonl"
        self._outbox_dir.mkdir(parents=True, exist_ok=True)
        path = self._outbox_dir / name
        with open(path, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg, default=str) + "\n")

    def _outbox_files(self) -> list[Path]:
        if not self._outbox_dir.exists():
            return []
        return sorted(self._outbox_dir.glob("outbox-*.jsonl"))

    def _enforce_buffer_cap(self) -> None:
        """Drop oldest outbox files when the buffer exceeds its cap."""
        files = self._outbox_files()
        total = sum(f.stat().st_size for f in files)
        while total > self._max_buffer_bytes and files:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink(missing_ok=True)
            self._log.critical(
                "Fleet outbox over %d MB — dropped oldest buffer file %s "
                "(central unreachable too long?)",
                self._max_buffer_bytes // (1024 * 1024),
                oldest.name,
            )

    # --------------------------------------------------------------- flush

    def _flush_outbox(self) -> int:
        """POST buffered files to central, oldest first; stop on failure.

        Returns
        -------
            Number of messages successfully flushed this cycle.
        """
        if not self._central_url:
            return 0
        flushed = 0
        for path in self._outbox_files():
            try:
                lines = [
                    json.loads(line)
                    for line in path.read_text().splitlines()
                    if line.strip()
                ]
            except (OSError, json.JSONDecodeError):
                self._log.warning("Unreadable outbox file %s; dropping", path.name)
                path.unlink(missing_ok=True)
                continue
            if not self._post_messages(lines):
                # Central unreachable — keep the file, backfill next cycle.
                return flushed
            flushed += len(lines)
            path.unlink(missing_ok=True)
        return flushed

    def _post_messages(self, messages: list[dict[str, Any]]) -> bool:
        """POST messages in batches; True only if every batch succeeded."""
        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        for i in range(0, len(messages), _BATCH_LIMIT):
            batch = messages[i : i + _BATCH_LIMIT]
            try:
                resp = requests.post(
                    f"{self._central_url}/api/v1/ingest",
                    json={"messages": batch},
                    headers=headers,
                    timeout=10,
                )
                resp.raise_for_status()
            except requests.RequestException as e:
                self._log.warning(
                    "Central unreachable (%s); %d message(s) buffered",
                    e.__class__.__name__,
                    len(messages) - i,
                )
                return False
        return True

    # --------------------------------------------------------------- state

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {}
        try:
            return json.loads(self._state_path.read_text())
        except (OSError, json.JSONDecodeError):
            self._log.warning("Corrupt fleet state file; starting cursors over")
            return {}

    def _save_state(self, state: dict[str, Any]) -> None:
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(self._state_path)
