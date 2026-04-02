"""ChargeNurse — event recording and case creation from threshold breaches."""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable
from typing import Any

from warpt.daemon.casefile import CaseFile
from warpt.utils.logger import Logger

_SENTINEL = None  # Placed on queue to signal worker to exit


class ChargeNurse:
    """Receives threshold breach events and records them as events + cases.

    Parameters
    ----------
    casefile
        CaseFile instance for database writes.
    pipeline_fn
        Optional callable ``(case_id, event) -> None`` dispatched
        asynchronously via a single worker thread. When ``None``,
        only synchronous DB work is performed (Phase 1 behavior).
    """

    def __init__(
        self,
        casefile: CaseFile,
        pipeline_fn: Callable[[int, dict], None] | None = None,
    ) -> None:
        self._casefile = casefile
        self._log = Logger.get("daemon.charge_nurse")
        self._pipeline_fn = pipeline_fn
        self._queue: queue.Queue | None = None
        self._worker: threading.Thread | None = None

        if pipeline_fn is not None:
            self._queue = queue.Queue()
            self._worker = threading.Thread(
                target=self._worker_loop, name="charge-nurse-pipeline", daemon=True
            )
            self._worker.start()

    def handle_breach(self, event: dict[str, Any]) -> None:
        """Process a threshold breach event from VitalsNurse.

        Creates an event row and opens or reuses a case.

        Parameters
        ----------
        event
            Dict with keys: metric, value, threshold, gpu_guid,
            sustained_seconds.
        """
        metric = event["metric"]
        value = event["value"]
        threshold = event["threshold"]
        gpu_guid = event.get("gpu_guid")
        sustained_seconds = event.get("sustained_seconds", 0.0)

        severity = self._compute_severity(value, threshold)
        summary = self._build_summary(metric, value, gpu_guid, sustained_seconds)
        metadata = {
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "sustained_seconds": sustained_seconds,
        }

        case_id = self._find_or_create_case(metric, gpu_guid, summary)

        self._casefile.execute(
            """
            INSERT INTO events (ts, kind, severity, gpu_guid, summary,
                                metadata, case_id, triggered_by)
            VALUES (current_timestamp, 'threshold_breach', ?, ?, ?,
                    ?::JSON, ?, 'vitals_nurse')
            """,
            [severity, gpu_guid, summary, json.dumps(metadata), case_id],
        )
        self._log.info(
            "Event created: %s [severity=%s, case_id=%s]", summary, severity, case_id
        )

        if self._queue is not None:
            self._queue.put((case_id, event))
            self._log.info("Pipeline enqueued for case #%s", case_id)

    def shutdown(self, timeout: float = 30.0) -> None:
        """Drain the pipeline queue and join the worker thread.

        Parameters
        ----------
        timeout
            Maximum seconds to wait for the worker to finish.
        """
        if self._queue is None or self._worker is None:
            return
        self._queue.put(_SENTINEL)
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            self._log.warning("Pipeline worker did not finish within %.1fs", timeout)

    def _worker_loop(self) -> None:
        """Consume pipeline work items from the queue until sentinel."""
        while True:
            item = self._queue.get()  # type: ignore[union-attr]
            if item is _SENTINEL:
                break
            case_id, event = item
            try:
                self._pipeline_fn(case_id, event)  # type: ignore[misc]
            except Exception:
                self._log.exception("Pipeline failed for case #%s", case_id)

    def _find_or_create_case(
        self, metric: str, gpu_guid: str | None, summary: str
    ) -> int:
        """Find an existing open case for this GPU+metric, or create one."""
        if gpu_guid:
            rows = self._casefile.query(
                """
                SELECT c.case_id FROM cases c
                JOIN events e ON e.case_id = c.case_id
                WHERE c.status = 'open'
                  AND e.gpu_guid = ?
                  AND json_extract_string(e.metadata, '$.metric') = ?
                ORDER BY c.opened_at DESC LIMIT 1
                """,
                [gpu_guid, metric],
            )
            if rows:
                self._log.info("Reusing case #%s", rows[0][0])
                return rows[0][0]

        self._casefile.execute(
            """
            INSERT INTO cases (title, status)
            VALUES (?, 'open')
            """,
            [summary],
        )
        rows = self._casefile.query("SELECT max(case_id) FROM cases")
        new_id = rows[0][0]
        self._log.info("Opened new case #%s: %s", new_id, summary)
        return new_id

    @staticmethod
    def _compute_severity(value: float, threshold: float) -> str:
        """Derive severity from how far the value exceeds the threshold."""
        if threshold == 0:
            return "critical"
        overshoot_pct = ((value - threshold) / threshold) * 100
        if overshoot_pct >= 25:
            return "critical"
        if overshoot_pct >= 10:
            return "warning"
        return "info"

    @staticmethod
    def _build_summary(
        metric: str,
        value: float,
        gpu_guid: str | None,
        sustained_seconds: float,
    ) -> str:
        """Build a human-readable one-liner for the event."""
        gpu_prefix = f"{gpu_guid}: " if gpu_guid else ""
        duration = _format_duration(sustained_seconds)
        return f"{gpu_prefix}{metric} at {value:.1f}% for {duration}"


def _format_duration(seconds: float) -> str:
    """Format seconds into the most relevant human-readable duration."""
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s:
        parts.append(f"{s}s")
    return " ".join(parts)
