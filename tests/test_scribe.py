"""Tests for the Scribe — case formatting and report generation."""

from __future__ import annotations

from warpt.daemon.agents.scribe import Scribe
from warpt.daemon.casefile import CaseFile


def _insert_diagnosed_case(cf: CaseFile) -> int:
    """Insert a case with diagnosis fields populated, return case_id."""
    cf.execute(
        """
        INSERT INTO cases (title, status, hypothesis, confidence_pct,
                           recommended_action, reasoning_chain,
                           baseline_deviation_pct, diagnostician_model)
        VALUES ('GPU-aaa: utilization_pct at 95.0% for 5m', 'open',
                'Sustained compute load from ML training',
                -1.23,
                'Monitor for thermal throttling',
                'High utilization consistent with training workload',
                18.5,
                'llama3:70b')
        """,
    )
    rows = cf.query("SELECT max(case_id) FROM cases")
    return rows[0][0]


def _insert_undiagnosed_case(cf: CaseFile) -> int:
    """Insert a case with no diagnosis fields (NULLs)."""
    cf.execute(
        """
        INSERT INTO cases (title, status)
        VALUES ('GPU-bbb: temperature_c at 82.0% for 3m', 'open')
        """,
    )
    rows = cf.query("SELECT max(case_id) FROM cases")
    return rows[0][0]


def test_scribe_formats_diagnosed_case() -> None:
    """Scribe.report() returns text containing hypothesis and recommended_action."""
    cf = CaseFile(":memory:")
    case_id = _insert_diagnosed_case(cf)
    scribe = Scribe(casefile=cf)

    report = scribe.report(case_id)

    assert "Sustained compute load from ML training" in report
    assert "Monitor for thermal throttling" in report
    cf.close()


def test_scribe_writes_report_content_to_case() -> None:
    """After report(), the report_content column is populated."""
    cf = CaseFile(":memory:")
    case_id = _insert_diagnosed_case(cf)
    scribe = Scribe(casefile=cf)

    scribe.report(case_id)

    rows = cf.query("SELECT report_content FROM cases WHERE case_id = ?", [case_id])
    assert rows[0][0] is not None
    assert "Sustained compute load from ML training" in rows[0][0]
    cf.close()


def test_scribe_handles_undiagnosed_case() -> None:
    """A case with NULL hypothesis returns a sensible message."""
    cf = CaseFile(":memory:")
    case_id = _insert_undiagnosed_case(cf)
    scribe = Scribe(casefile=cf)

    report = scribe.report(case_id)

    assert "no diagnosis" in report.lower() or "pending" in report.lower()
    assert str(case_id) in report
    cf.close()
