"""ChargeNurse — event recording and case creation from threshold breaches."""

from __future__ import annotations

import json
from typing import Any

from warpt.daemon.casefile import CaseFile


class ChargeNurse:
    """Receives threshold breach events and records them as events + cases.

    Parameters
    ----------
    casefile
        CaseFile instance for database writes.
    """

    def __init__(self, casefile: CaseFile) -> None:
        self._casefile = casefile

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
                return rows[0][0]

        self._casefile.execute(
            """
            INSERT INTO cases (title, status)
            VALUES (?, 'open')
            """,
            [summary],
        )
        rows = self._casefile.query("SELECT max(case_id) FROM cases")
        return rows[0][0]

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
        duration_min = int(sustained_seconds // 60)
        gpu_prefix = f"{gpu_guid}: " if gpu_guid else ""
        return f"{gpu_prefix}{metric} at {value:.1f}% for {duration_min}m"
