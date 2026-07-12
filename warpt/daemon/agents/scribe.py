"""Scribe — formats diagnosed cases into human-readable reports."""

from __future__ import annotations

import json

from warpt.daemon.casefile import CaseFile
from warpt.utils.logger import Logger


class Scribe:
    """Formats a diagnosed case into a human-readable report.

    Parameters
    ----------
    casefile
        CaseFile instance for database queries and writes.
    """

    def __init__(self, casefile: CaseFile) -> None:
        self._casefile = casefile
        self._log = Logger.get("daemon.agents.scribe")

    def report(self, case_id: int) -> str:
        """Query case, format to human-readable text, log it, write to case row.

        Parameters
        ----------
        case_id
            The case to report on.

        Returns
        -------
            Formatted report text.
        """
        rows = self._casefile.query(
            """
            SELECT title, status, hypothesis, confidence_pct,
                   recommended_action, reasoning_chain,
                   baseline_deviation_pct, diagnostician_model,
                   opened_at, severity, evidence, tools_used
            FROM cases WHERE case_id = ?
            """,
            [case_id],
        )
        if not rows:
            msg = f"Case #{case_id}: not found."
            self._log.warning(msg)
            return msg

        (
            title,
            status,
            hypothesis,
            confidence_pct,
            recommended_action,
            reasoning_chain,
            baseline_deviation_pct,
            diagnostician_model,
            opened_at,
            severity,
            evidence,
            tools_used,
        ) = rows[0]

        if hypothesis is None:
            report = (
                f"Case #{case_id}: {title}\n"
                f"Status: {status}\n"
                f"Opened: {opened_at}\n"
                f"Diagnosis: pending — no diagnosis available yet."
            )
        else:
            lines = [
                f"Case #{case_id}: {title}",
                f"Status: {status}",
                f"Opened: {opened_at}",
                f"Hypothesis: {hypothesis}",
            ]
            if severity:
                lines.append(f"Severity: {severity}")
            # Only show a valid 0-100% confidence; NULL/invalid is suppressed
            # (degraded diagnoses store no confidence).
            if confidence_pct is not None and 0.0 <= confidence_pct <= 100.0:
                lines.append(f"Confidence: {confidence_pct}%")
            if recommended_action:
                lines.append(_format_action(recommended_action))
            if reasoning_chain:
                lines.append(f"Reasoning: {reasoning_chain}")
            evidence_items = _parse_json_list(evidence)
            if evidence_items:
                lines.append("Evidence:")
                lines.extend(f"  - {item}" for item in evidence_items[:5])
            tools = _parse_json_list(tools_used)
            if tools:
                lines.append(f"Tools Used: {', '.join(str(t) for t in tools)}")
            if baseline_deviation_pct is not None:
                lines.append(f"Baseline Deviation: {baseline_deviation_pct}%")
            if diagnostician_model:
                lines.append(f"Model: {diagnostician_model}")
            report = "\n".join(lines)

        self._casefile.execute(
            "UPDATE cases SET report_content = ?,"
            " updated_at = current_timestamp WHERE case_id = ?",
            [report, case_id],
        )
        self._log.info("Report generated for case #%d", case_id)
        return report


def _format_action(recommended_action: str) -> str:
    """Render a recommended action — structured JSON or plain string.

    Diagnoses store a structured object (``{"action_type", "target",
    "description"}``); degraded pipeline rungs store a plain string.
    """
    try:
        parsed = json.loads(recommended_action)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        description = parsed.get("description", "")
        action_type = parsed.get("action_type", "")
        suffix = f" [{action_type}]" if action_type else ""
        return f"Recommended Action: {description}{suffix}"
    return f"Recommended Action: {recommended_action}"


def _parse_json_list(value: str | None) -> list:
    """Parse a JSON array column value; anything else yields []."""
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []
