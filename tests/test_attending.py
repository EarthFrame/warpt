"""Tests for warpt.daemon.agents.attending."""

import json
from unittest.mock import MagicMock

from warpt.daemon.agents.attending import CONFIDENCE_SENTINEL, Attending
from warpt.daemon.agents.ollama_client import OllamaClient
from warpt.daemon.casefile import CaseFile
from warpt.daemon.vitals_nurse import VitalsNurse

GPU_GUID = "GPU-test-1234"

# Minimal valid snapshot that VitalsNurse.feed_snapshot() accepts
_SNAPSHOT = {
    "timestamp": "2025-01-01T12:00:00",
    "cpu_utilization_percent": 45.0,
    "cpu_power_watts": 80.0,
    "total_memory_bytes": 16_000_000_000,
    "available_memory_bytes": 8_000_000_000,
    "wired_memory_bytes": 2_000_000_000,
    "memory_utilization_percent": 50.0,
    "gpu_usage": [
        {
            "guid": GPU_GUID,
            "index": 0,
            "model": "RTX 4090",
            "utilization_percent": 85.0,
            "memory_utilization_percent": 60.0,
            "power_watts": 300.0,
        }
    ],
}

# Chart Nurse output that the Attending consumes
_CHART_NURSE_RESULT = {
    "gpu_guid": GPU_GUID,
    "metric": "utilization_pct",
    "current_value": 95.0,
    "baseline": {"1h_avg": 62.3, "24h_avg": 58.1, "7d_avg": 55.0},
    "current_hour_profile": {"hour": 14, "mean": 60.0, "stddev": 7.2},
    "deviation_pct": 52.6,
    "prior_cases": [],
    "event_count_7d": 3,
    "interpretation": "GPU utilization is significantly above baseline.",
    "model_used": "llama3:8b",
}

_VALID_LLM_RESPONSE = json.dumps(
    {
        "hypothesis": "Sustained compute load from training job",
        "confidence": 75,
        "recommended_action": "Monitor for thermal throttling",
        "reasoning": "Utilization 52% above 1h baseline with no memory pressure",
    }
)

_CONFIG = {
    "models": {"attending": "llama3:70b"},
    "triage_order": ["thermal_power", "memory", "compute", "storage_io"],
}


def _make_attending(casefile, llm_response):
    """Build an Attending with a mocked OllamaClient."""
    client = OllamaClient.__new__(OllamaClient)
    client.model = "llama3:70b"
    client.generate = MagicMock(return_value=llm_response)

    vitals = VitalsNurse(casefile=casefile, heartbeat_interval=9999)
    vitals.feed_snapshot(_SNAPSHOT)

    return (
        Attending(
            casefile=casefile,
            ollama_client=client,
            vitals_nurse=vitals,
            config=_CONFIG,
        ),
        client,
    )


def test_valid_diagnosis_writes_to_case():
    """diagnose() with valid LLM JSON updates the case row in DuckDB."""
    cf = CaseFile(":memory:")
    cf.execute("INSERT INTO cases (title, status) VALUES ('High util', 'open')")
    case_id = cf.query("SELECT max(case_id) FROM cases")[0][0]

    attending, _client = _make_attending(cf, _VALID_LLM_RESPONSE)
    result = attending.diagnose(_CHART_NURSE_RESULT, case_id)

    # Return value has the expected keys
    assert result["hypothesis"] == "Sustained compute load from training job"
    assert result["recommended_action"] == "Monitor for thermal throttling"
    assert result["reasoning"] is not None

    # Case row updated in DuckDB
    sql = (
        "SELECT hypothesis, recommended_action, diagnostician_model"
        " FROM cases WHERE case_id = ?"
    )
    row = cf.query(sql, [case_id])[0]
    assert row[0] == "Sustained compute load from training job"
    assert row[1] == "Monitor for thermal throttling"
    assert row[2] == "llama3:70b"


def test_triage_order_in_llm_prompt():
    """diagnose() sends the configured triage order to the LLM system prompt."""
    cf = CaseFile(":memory:")
    cf.execute("INSERT INTO cases (title, status) VALUES ('Test', 'open')")
    case_id = cf.query("SELECT max(case_id) FROM cases")[0][0]

    custom_config = {
        "models": {"attending": "llama3:70b"},
        "triage_order": ["memory", "compute", "thermal_power", "storage_io"],
    }

    client = OllamaClient.__new__(OllamaClient)
    client.model = "llama3:70b"
    client.generate = MagicMock(return_value=_VALID_LLM_RESPONSE)

    vitals = VitalsNurse(casefile=cf, heartbeat_interval=9999)
    vitals.feed_snapshot(_SNAPSHOT)

    attending = Attending(
        casefile=cf, ollama_client=client, vitals_nurse=vitals, config=custom_config
    )
    attending.diagnose(_CHART_NURSE_RESULT, case_id)

    # The system prompt (second arg to generate) should list Memory before Compute
    system_prompt = client.generate.call_args[0][1]
    mem_pos = system_prompt.index("Memory")
    compute_pos = system_prompt.index("Compute")
    thermal_pos = system_prompt.index("Thermal")
    assert mem_pos < compute_pos < thermal_pos


def test_confidence_is_sentinel_value():
    """confidence_pct is always the sentinel, regardless of LLM output."""
    cf = CaseFile(":memory:")
    cf.execute("INSERT INTO cases (title, status) VALUES ('Test', 'open')")
    case_id = cf.query("SELECT max(case_id) FROM cases")[0][0]

    # LLM returns confidence=75, but result should use sentinel
    attending, _client = _make_attending(cf, _VALID_LLM_RESPONSE)
    result = attending.diagnose(_CHART_NURSE_RESULT, case_id)

    assert result["confidence_pct"] == CONFIDENCE_SENTINEL

    # Also verify the DB column
    row = cf.query("SELECT confidence_pct FROM cases WHERE case_id = ?", [case_id])
    assert row[0][0] == CONFIDENCE_SENTINEL


def test_malformed_llm_response_uses_fallback():
    """diagnose() degrades gracefully when LLM returns unparseable output."""
    cf = CaseFile(":memory:")
    cf.execute("INSERT INTO cases (title, status) VALUES ('Test', 'open')")
    case_id = cf.query("SELECT max(case_id) FROM cases")[0][0]

    garbage = "I'm not JSON, just some random LLM chatter about GPUs"
    attending, _client = _make_attending(cf, garbage)
    result = attending.diagnose(_CHART_NURSE_RESULT, case_id)

    # Should not raise — returns fallback values
    assert "Unable to parse" in result["hypothesis"]
    assert result["recommended_action"] is not None
    assert result["confidence_pct"] == CONFIDENCE_SENTINEL

    # Case still gets updated (with fallback)
    row = cf.query("SELECT hypothesis FROM cases WHERE case_id = ?", [case_id])
    assert "Unable to parse" in row[0][0]


def test_prompt_includes_chart_nurse_and_vitals():
    """diagnose() sends both Chart Nurse analysis and vitals snapshot to the LLM."""
    cf = CaseFile(":memory:")
    cf.execute("INSERT INTO cases (title, status) VALUES ('Test', 'open')")
    case_id = cf.query("SELECT max(case_id) FROM cases")[0][0]

    attending, client = _make_attending(cf, _VALID_LLM_RESPONSE)
    attending.diagnose(_CHART_NURSE_RESULT, case_id)

    # The user prompt (first arg to generate) should contain both data sources
    user_prompt = client.generate.call_args[0][0]
    prompt_data = json.loads(user_prompt)

    assert "chart_nurse_analysis" in prompt_data
    assert prompt_data["chart_nurse_analysis"]["gpu_guid"] == GPU_GUID
    assert prompt_data["chart_nurse_analysis"]["deviation_pct"] == 52.6

    assert "current_vitals_snapshot" in prompt_data
    assert prompt_data["current_vitals_snapshot"]["cpu_utilization_percent"] == 45.0


def test_attending_retries_malformed_then_succeeds():
    """diagnose() retries on malformed LLM output, uses valid response."""
    cf = CaseFile(":memory:")
    cf.execute("INSERT INTO cases (title, status) VALUES ('Test', 'open')")
    case_id = cf.query("SELECT max(case_id) FROM cases")[0][0]

    garbage = "not json at all"
    client = OllamaClient.__new__(OllamaClient)
    client.model = "llama3:70b"
    # First call returns garbage, second returns valid JSON
    client.generate = MagicMock(side_effect=[garbage, _VALID_LLM_RESPONSE])

    vitals = VitalsNurse(casefile=cf, heartbeat_interval=9999)
    vitals.feed_snapshot(_SNAPSHOT)

    attending = Attending(
        casefile=cf, ollama_client=client, vitals_nurse=vitals, config=_CONFIG
    )
    result = attending.diagnose(_CHART_NURSE_RESULT, case_id)

    # Should have the real hypothesis from the second attempt
    assert result["hypothesis"] == "Sustained compute load from training job"
    assert client.generate.call_count == 2


def test_attending_persistent_malformed_uses_fallback():
    """diagnose() uses fallback after exhausting malformed retries."""
    cf = CaseFile(":memory:")
    cf.execute("INSERT INTO cases (title, status) VALUES ('Test', 'open')")
    case_id = cf.query("SELECT max(case_id) FROM cases")[0][0]

    garbage = "still not json"
    attending, client = _make_attending(cf, garbage)
    result = attending.diagnose(_CHART_NURSE_RESULT, case_id)

    # Fallback used
    assert "Unable to parse" in result["hypothesis"]
    # Called 3 times (default retries)
    assert client.generate.call_count == 3
