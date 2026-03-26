"""Scribe — formats diagnosed cases into human-readable reports."""

from __future__ import annotations

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
                   opened_at
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
            if confidence_pct is not None:
                lines.append(f"Confidence: {confidence_pct}%")
            if recommended_action:
                lines.append(f"Recommended Action: {recommended_action}")
            if reasoning_chain:
                lines.append(f"Reasoning: {reasoning_chain}")
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
