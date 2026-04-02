"""Tests for the warpt inspect command."""

from __future__ import annotations

import json

from warpt.commands.inspect_cmd import list_cases, show_case, show_latest
from warpt.daemon.casefile import CaseFile


def _seed_case(cf: CaseFile, **overrides) -> int:
    """Insert a case and return its case_id."""
    defaults = {
        "title": "GPU-3365cb89: utilization_percent breach",
        "status": "open",
        "hypothesis": "Thermal management issue likely due to high GPU utilization.",
        "confidence_pct": -1.23,
        "recommended_action": "Monitor system temperature and adjust thermal settings.",
        "observation": json.dumps({
            "baseline": {
                "avg_1h": 67.3,
                "avg_24h": 60.8,
                "avg_7d": 60.8,
                "deviation_pct": 48.6,
                "current_value": 100.0,
            },
            "interpretation": (
                "The current GPU utilization is 100%, "
                "which is significantly higher than baseline."
            ),
        }),
        "reasoning_chain": repr([
            {
                "category": "Thermal / Power",
                "finding": "The GPU utilization is at 100%.",
                "implication": "Thermal management might be compromised.",
            },
            {
                "category": "Memory",
                "finding": "Memory utilization is within normal range.",
                "implication": "No memory pressure detected.",
            },
        ]),
        "diagnostician_model": "llama3:8b",
    }
    defaults.update(overrides)

    cols = ", ".join(defaults.keys())
    placeholders = ", ".join(["?"] * len(defaults))
    cf.execute(
        f"INSERT INTO cases ({cols}) VALUES ({placeholders})",
        list(defaults.values()),
    )
    rows = cf.query("SELECT MAX(case_id) FROM cases")
    return rows[0][0]


def _seed_event(cf: CaseFile, case_id: int, **overrides) -> None:
    """Insert an event linked to a case."""
    defaults = {
        "ts": "2026-04-02 10:58:21",
        "kind": "threshold_breach",
        "severity": "critical",
        "gpu_guid": "GPU-3365cb89",
        "summary": "utilization_percent at 100.0%",
        "case_id": case_id,
    }
    defaults.update(overrides)
    cols = ", ".join(defaults.keys())
    placeholders = ", ".join(["?"] * len(defaults))
    cf.execute(
        f"INSERT INTO events ({cols}) VALUES ({placeholders})",
        list(defaults.values()),
    )


# -- show_case tests ----------------------------------------------------------

def test_show_case_contains_sections(capsys) -> None:
    """show_case prints hypothesis, baseline, events, and reasoning."""
    cf = CaseFile(":memory:")
    cid = _seed_case(cf)
    _seed_event(cf, cid)

    show_case(cf, cid)
    out = capsys.readouterr().out

    assert f"WARPT CASE #{cid}" in out
    assert "HYPOTHESIS" in out
    assert "Thermal management issue" in out
    assert "CONFIDENCE" in out
    assert "-1.23%" in out
    assert "RECOMMENDED ACTION" in out
    assert "BASELINE" in out
    assert "67.3%" in out
    assert "LLM INTERPRETATION (llama3:8b)" in out
    assert "TRIAGE REASONING" in out
    assert "Thermal / Power" in out
    assert "EVENTS" in out
    assert "[CRITICAL]" in out
    assert "utilization_percent at 100.0%" in out

    cf.close()


def test_show_case_not_found(capsys) -> None:
    """show_case prints a message when case ID does not exist."""
    cf = CaseFile(":memory:")

    show_case(cf, 999)
    out = capsys.readouterr().out

    assert "No case found with ID 999" in out

    cf.close()


# -- show_latest tests --------------------------------------------------------

def test_show_latest_picks_most_recent(capsys) -> None:
    """show_latest selects the case with the most recent opened_at."""
    cf = CaseFile(":memory:")
    _seed_case(cf, title="GPU-old: metric_a breach", opened_at="2026-04-01 08:00:00")
    cid2 = _seed_case(
        cf, title="GPU-new: metric_b breach",
        opened_at="2026-04-02 12:00:00",
    )

    show_latest(cf)
    out = capsys.readouterr().out

    assert f"WARPT CASE #{cid2}" in out
    assert "GPU-new" in out

    cf.close()


def test_show_latest_empty_db(capsys) -> None:
    """show_latest prints a message when there are no cases."""
    cf = CaseFile(":memory:")

    show_latest(cf)
    out = capsys.readouterr().out

    assert "No cases found" in out

    cf.close()


# -- list_cases tests ---------------------------------------------------------

def test_list_cases_contains_id_and_title(capsys) -> None:
    """list_cases shows a table with case IDs and titles."""
    cf = CaseFile(":memory:")
    _seed_case(cf, title="GPU-aaaa: temperature_c breach", status="open")
    _seed_case(cf, title="GPU-bbbb: power_w breach", status="closed")

    list_cases(cf)
    out = capsys.readouterr().out

    assert "ID" in out
    assert "Status" in out
    assert "GPU-aaaa" in out
    assert "GPU-bbbb" in out
    assert "open" in out
    assert "closed" in out

    cf.close()


def test_list_cases_empty_db(capsys) -> None:
    """list_cases prints a message when there are no cases."""
    cf = CaseFile(":memory:")

    list_cases(cf)
    out = capsys.readouterr().out

    assert "No cases found" in out

    cf.close()


# -- pending diagnosis --------------------------------------------------------

def test_pending_diagnosis_shows_message(capsys) -> None:
    """A case with no hypothesis shows a pending diagnosis message."""
    cf = CaseFile(":memory:")
    cid = _seed_case(
        cf,
        hypothesis=None,
        confidence_pct=None,
        recommended_action=None,
        observation=None,
        reasoning_chain=None,
        diagnostician_model=None,
    )
    _seed_event(cf, cid)

    show_case(cf, cid)
    out = capsys.readouterr().out

    assert "DIAGNOSIS PENDING" in out
    assert "HYPOTHESIS" not in out
    assert "CONFIDENCE" not in out
    assert "EVENTS" in out

    cf.close()


# -- read_only CaseFile -------------------------------------------------------

def test_casefile_read_only(tmp_path) -> None:
    """CaseFile opens in read-only mode without running migrations."""
    db_path = str(tmp_path / "warpt.db")

    # Create the DB with schema first
    cf_rw = CaseFile(db_path)
    cf_rw.execute(
        "INSERT INTO cases (title, status) VALUES (?, ?)",
        ["GPU-test: metric breach", "open"],
    )
    cf_rw.close()

    # Open read-only
    cf_ro = CaseFile(db_path, read_only=True)
    rows = cf_ro.query("SELECT title FROM cases")
    assert len(rows) == 1
    assert rows[0][0] == "GPU-test: metric breach"
    cf_ro.close()
