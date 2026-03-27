"""Tests for warpt.daemon.agents.pipeline — degradation ladder."""

from unittest.mock import MagicMock, patch

from warpt.daemon.agents.ollama_client import OllamaPermanentError
from warpt.daemon.agents.pipeline import run_intelligence_pipeline

_EVENT = {
    "gpu_guid": "GPU-test-1234",
    "metric": "utilization_pct",
    "value": 95.0,
    "threshold": 85.0,
    "sustained_seconds": 120.0,
}

_CHART_RESULT = {
    "gpu_guid": "GPU-test-1234",
    "metric": "utilization_pct",
    "current_value": 95.0,
    "baseline": {"1h_avg": 62.3, "24h_avg": 58.1, "7d_avg": 55.0},
    "deviation_pct": 52.6,
    "interpretation": "Above baseline",
    "model_used": "llama3:8b",
}

_DIAGNOSIS = {
    "hypothesis": "Training job",
    "confidence_pct": -1.23,
    "recommended_action": "Monitor",
    "reasoning": "High utilization",
}


def test_pipeline_full_success():
    """All three stages succeed — chart_nurse, attending, scribe all called."""
    chart_nurse = MagicMock()
    chart_nurse.analyze.return_value = _CHART_RESULT

    attending = MagicMock()
    attending.diagnose.return_value = _DIAGNOSIS

    scribe = MagicMock()
    casefile = MagicMock()
    log = MagicMock()

    run_intelligence_pipeline(
        case_id=1,
        event=_EVENT,
        chart_nurse=chart_nurse,
        attending=attending,
        scribe=scribe,
        casefile=casefile,
        log=log,
    )

    chart_nurse.analyze.assert_called_once_with(
        "GPU-test-1234", "utilization_pct", 95.0
    )
    attending.diagnose.assert_called_once_with(_CHART_RESULT, 1)
    scribe.report.assert_called_once_with(1)


_RAW_ANALYTICS = {
    "gpu_guid": "GPU-test-1234",
    "metric": "utilization_pct",
    "current_value": 95.0,
    "baseline": {"1h_avg": 62.3, "24h_avg": 58.1, "7d_avg": 55.0},
    "deviation_pct": 52.6,
    "interpretation": None,
    "model_used": None,
}


def test_pipeline_degrades_on_chart_nurse_failure():
    """When chart_nurse.analyze() fails, falls back to analyze_without_llm()."""
    chart_nurse = MagicMock()
    chart_nurse.analyze.side_effect = RuntimeError("Ollama down")
    chart_nurse.analyze_without_llm.return_value = _RAW_ANALYTICS

    attending = MagicMock()
    scribe = MagicMock()
    casefile = MagicMock()
    log = MagicMock()

    with patch("warpt.daemon.agents.pipeline.time.sleep"):
        run_intelligence_pipeline(
            case_id=1,
            event=_EVENT,
            chart_nurse=chart_nurse,
            attending=attending,
            scribe=scribe,
            casefile=casefile,
            log=log,
            retries=2,
        )

    # analyze was retried
    assert chart_nurse.analyze.call_count == 2
    # Fell back to analyze_without_llm
    chart_nurse.analyze_without_llm.assert_called_once()
    # Attending NOT called (Chart Nurse LLM failed)
    attending.diagnose.assert_not_called()
    # Degraded observation written to case
    casefile.execute.assert_called_once()
    sql_arg = casefile.execute.call_args[0][0]
    assert "UPDATE cases" in sql_arg
    # Scribe still runs
    scribe.report.assert_called_once_with(1)


def test_pipeline_permanent_error_skips_retry():
    """OllamaPermanentError in Chart Nurse skips retry, degrades immediately."""
    chart_nurse = MagicMock()
    chart_nurse.analyze.side_effect = OllamaPermanentError("model not found")
    chart_nurse.analyze_without_llm.return_value = _RAW_ANALYTICS

    attending = MagicMock()
    scribe = MagicMock()
    casefile = MagicMock()
    log = MagicMock()

    run_intelligence_pipeline(
        case_id=1,
        event=_EVENT,
        chart_nurse=chart_nurse,
        attending=attending,
        scribe=scribe,
        casefile=casefile,
        log=log,
        retries=3,
    )

    # Only called once — no retries on permanent error
    assert chart_nurse.analyze.call_count == 1
    chart_nurse.analyze_without_llm.assert_called_once()
    attending.diagnose.assert_not_called()
    scribe.report.assert_called_once_with(1)


def test_pipeline_degrades_on_attending_failure():
    """When attending.diagnose() fails, case gets Chart Nurse analysis + note."""
    chart_nurse = MagicMock()
    chart_nurse.analyze.return_value = _CHART_RESULT

    attending = MagicMock()
    attending.diagnose.side_effect = RuntimeError("Ollama down")

    scribe = MagicMock()
    casefile = MagicMock()
    log = MagicMock()

    with patch("warpt.daemon.agents.pipeline.time.sleep"):
        run_intelligence_pipeline(
            case_id=1,
            event=_EVENT,
            chart_nurse=chart_nurse,
            attending=attending,
            scribe=scribe,
            casefile=casefile,
            log=log,
            retries=2,
        )

    # Chart Nurse succeeded on first try
    chart_nurse.analyze.assert_called_once()
    # Attending retried and exhausted
    assert attending.diagnose.call_count == 2
    # Degraded observation written with "Attending unavailable"
    casefile.execute.assert_called_once()
    params = casefile.execute.call_args[0][1]
    assert any("Attending unavailable" in str(p) for p in params)
    # Scribe still runs
    scribe.report.assert_called_once_with(1)
