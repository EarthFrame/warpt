"""Tests for warpt.daemon.agents.chart_nurse."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from warpt.daemon.agents.chart_nurse import ChartNurse
from warpt.daemon.agents.ollama_client import OllamaClient
from warpt.daemon.casefile import CaseFile

GPU_GUID = "GPU-test-1234"


def _seed_vitals(casefile, gpu_guid, values, hours_ago_start=0, interval_minutes=10):
    """Insert vitals rows with known GPU metric values.

    Parameters
    ----------
    values
        List of utilization_pct floats, inserted newest-first going backwards.
    hours_ago_start
        How many hours ago the most recent value is.
    interval_minutes
        Minutes between each value.
    """
    now = datetime.now()
    for i, val in enumerate(values):
        ts = now - timedelta(hours=hours_ago_start, minutes=i * interval_minutes)
        casefile.execute(
            """
            INSERT INTO vitals (
                ts, cpu_utilization_pct, cpu_power_w,
                mem_total_bytes, mem_available_bytes,
                mem_utilization_pct, gpus, collection_type
            ) VALUES (?, 50.0, 100.0, 8000000000, 4000000000, 50.0, ?, 'heartbeat')
            """,
            [
                ts.isoformat(),
                [
                    {
                        "gpu_guid": gpu_guid,
                        "gpu_index": 0,
                        "utilization_pct": val,
                        "mem_utilization_pct": 40.0,
                        "power_w": 200.0,
                        "temperature_c": 65.0,
                        "mem_used_bytes": None,
                        "mem_total_bytes": None,
                        "throttle_reasons": None,
                    }
                ],
            ],
        )


def test_rolling_averages_from_seeded_vitals():
    """analyze() computes 1h rolling average from vitals data."""
    cf = CaseFile(":memory:")
    client = OllamaClient.__new__(OllamaClient)
    client.model = "llama3:8b"
    client.generate = MagicMock(return_value="test interpretation")

    # Seed 6 values over the last hour (every 10 min), all utilization_pct
    _seed_vitals(cf, GPU_GUID, [60.0, 70.0, 80.0, 50.0, 60.0, 70.0])

    nurse = ChartNurse(casefile=cf, ollama_client=client)
    result = nurse.analyze(GPU_GUID, "utilization_pct", 95.0)

    assert result["baseline"]["1h_avg"] is not None
    # Average of [60, 70, 80, 50, 60, 70] = 65.0
    assert abs(result["baseline"]["1h_avg"] - 65.0) < 1.0


def test_hourly_profile_for_current_hour():
    """analyze() computes mean and stddev for the current hour-of-day."""
    cf = CaseFile(":memory:")
    client = OllamaClient.__new__(OllamaClient)
    client.model = "llama3:8b"
    client.generate = MagicMock(return_value="test interpretation")

    # Seed values all within the current hour
    _seed_vitals(cf, GPU_GUID, [60.0, 70.0, 80.0], interval_minutes=5)

    nurse = ChartNurse(casefile=cf, ollama_client=client)
    result = nurse.analyze(GPU_GUID, "utilization_pct", 95.0)

    profile = result["current_hour_profile"]
    assert profile is not None
    assert profile["hour"] == datetime.now().hour
    # Mean of [60, 70, 80] = 70.0
    assert abs(profile["mean"] - 70.0) < 1.0
    assert profile["stddev"] is not None
    assert profile["stddev"] > 0


def test_empty_db_returns_none_baselines():
    """analyze() on empty DB returns None baselines and no profile."""
    cf = CaseFile(":memory:")
    client = OllamaClient.__new__(OllamaClient)
    client.model = "llama3:8b"
    client.generate = MagicMock(return_value="no data available")

    nurse = ChartNurse(casefile=cf, ollama_client=client)
    result = nurse.analyze(GPU_GUID, "utilization_pct", 50.0)

    assert result["baseline"]["1h_avg"] is None
    assert result["baseline"]["24h_avg"] is None
    assert result["baseline"]["7d_avg"] is None
    assert result["current_hour_profile"] is None
    assert result["deviation_pct"] is None
    assert result["prior_cases"] == []
    assert result["event_count_7d"] == 0


def test_prior_cases_listed():
    """analyze() lists prior cases associated with the GPU."""
    cf = CaseFile(":memory:")
    client = OllamaClient.__new__(OllamaClient)
    client.model = "llama3:8b"
    client.generate = MagicMock(return_value="interpretation")

    # Create a case + event for this GPU
    cf.execute("INSERT INTO cases (title, status) VALUES ('High utilization', 'open')")
    case_id = cf.query("SELECT max(case_id) FROM cases")[0][0]
    cf.execute(
        """
        INSERT INTO events (ts, kind, severity, gpu_guid, summary, case_id)
        VALUES (current_timestamp, 'threshold_breach', 'warning', ?, 'test', ?)
        """,
        [GPU_GUID, case_id],
    )

    # Seed at least one vitals row so baselines don't blow up
    _seed_vitals(cf, GPU_GUID, [50.0])

    nurse = ChartNurse(casefile=cf, ollama_client=client)
    result = nurse.analyze(GPU_GUID, "utilization_pct", 95.0)

    assert len(result["prior_cases"]) == 1
    assert result["prior_cases"][0]["case_id"] == case_id
    assert result["prior_cases"][0]["title"] == "High utilization"


def test_event_count_7d():
    """analyze() counts recent events for the GPU."""
    cf = CaseFile(":memory:")
    client = OllamaClient.__new__(OllamaClient)
    client.model = "llama3:8b"
    client.generate = MagicMock(return_value="interpretation")

    # Insert 3 events for this GPU
    for _ in range(3):
        cf.execute(
            """
            INSERT INTO events (ts, kind, severity, gpu_guid, summary)
            VALUES (current_timestamp, 'threshold_breach', 'info', ?, 'test')
            """,
            [GPU_GUID],
        )

    _seed_vitals(cf, GPU_GUID, [50.0])

    nurse = ChartNurse(casefile=cf, ollama_client=client)
    result = nurse.analyze(GPU_GUID, "utilization_pct", 95.0)

    assert result["event_count_7d"] == 3


def test_deviation_percentage_calculated():
    """analyze() computes deviation_pct from 1h average."""
    cf = CaseFile(":memory:")
    client = OllamaClient.__new__(OllamaClient)
    client.model = "llama3:8b"
    client.generate = MagicMock(return_value="interpretation")

    # Seed uniform values so 1h avg = 50.0
    _seed_vitals(cf, GPU_GUID, [50.0, 50.0, 50.0])

    nurse = ChartNurse(casefile=cf, ollama_client=client)
    # Current value 75.0 with baseline 50.0 → deviation = 50%
    result = nurse.analyze(GPU_GUID, "utilization_pct", 75.0)

    assert result["deviation_pct"] == 50.0


def test_full_output_contract_with_interpretation():
    """analyze() returns complete output with LLM interpretation."""
    cf = CaseFile(":memory:")
    client = OllamaClient.__new__(OllamaClient)
    client.model = "llama3:8b"
    client.generate = MagicMock(
        return_value="GPU utilization is significantly above baseline."
    )

    _seed_vitals(cf, GPU_GUID, [60.0, 70.0, 80.0])

    nurse = ChartNurse(casefile=cf, ollama_client=client)
    result = nurse.analyze(GPU_GUID, "utilization_pct", 95.0)

    # All top-level keys present
    assert result["gpu_guid"] == GPU_GUID
    assert result["metric"] == "utilization_pct"
    assert result["current_value"] == 95.0
    assert "baseline" in result
    assert "current_hour_profile" in result
    assert "deviation_pct" in result
    assert "prior_cases" in result
    assert "event_count_7d" in result
    expected = "GPU utilization is significantly above baseline."
    assert result["interpretation"] == expected
    assert result["model_used"] == "llama3:8b"

    # LLM was called exactly once
    client.generate.assert_called_once()


def test_analyze_without_llm_returns_analytics():
    """analyze_without_llm() returns baselines with interpretation=None."""
    cf = CaseFile(":memory:")
    client = OllamaClient.__new__(OllamaClient)
    client.model = "llama3:8b"
    client.generate = MagicMock(return_value="should not be called")

    _seed_vitals(cf, GPU_GUID, [50.0, 50.0, 50.0])

    nurse = ChartNurse(casefile=cf, ollama_client=client)
    result = nurse.analyze_without_llm(GPU_GUID, "utilization_pct", 75.0)

    # Baselines should be present
    assert result["baseline"]["1h_avg"] is not None
    assert abs(result["baseline"]["1h_avg"] - 50.0) < 1.0

    # Deviation computed
    assert result["deviation_pct"] == 50.0

    # No LLM call
    assert result["interpretation"] is None
    assert result["model_used"] is None
    client.generate.assert_not_called()
