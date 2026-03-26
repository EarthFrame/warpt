"""Tests for the ChargeNurse — event recording and case creation."""

from __future__ import annotations

import threading
import time

from warpt.daemon.casefile import CaseFile
from warpt.daemon.charge_nurse import ChargeNurse


def _breach_event(
    metric: str = "utilization_percent",
    value: float = 95.0,
    threshold: float = 80.0,
    gpu_guid: str = "GPU-aaa-bbb-ccc",
    sustained_seconds: float = 310.0,
) -> dict:
    """Build a threshold breach event matching VitalsNurse callback shape."""
    return {
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "gpu_guid": gpu_guid,
        "sustained_seconds": sustained_seconds,
    }


def test_breach_event_creates_event_and_case() -> None:
    """A threshold breach creates one event row and one case."""
    cf = CaseFile(":memory:")
    nurse = ChargeNurse(casefile=cf)

    nurse.handle_breach(_breach_event())

    events = cf.query("SELECT kind, severity, gpu_guid, triggered_by FROM events")
    assert len(events) == 1
    assert events[0][0] == "threshold_breach"
    assert events[0][2] == "GPU-aaa-bbb-ccc"
    assert events[0][3] == "vitals_nurse"

    cases = cf.query("SELECT title, status FROM cases")
    assert len(cases) == 1
    assert cases[0][1] == "open"

    cf.close()


def test_event_links_to_case() -> None:
    """The event's case_id references the newly created case."""
    cf = CaseFile(":memory:")
    nurse = ChargeNurse(casefile=cf)

    nurse.handle_breach(_breach_event())

    rows = cf.query(
        "SELECT e.case_id, c.case_id, c.title "
        "FROM events e JOIN cases c ON e.case_id = c.case_id"
    )
    assert len(rows) == 1
    assert rows[0][0] == rows[0][1]  # event.case_id == case.case_id
    assert "GPU-aaa-bbb-ccc" in rows[0][2]  # title contains GPU GUID

    cf.close()


def test_duplicate_breach_reuses_open_case() -> None:
    """Second breach for same GPU+metric attaches to existing open case."""
    cf = CaseFile(":memory:")
    nurse = ChargeNurse(casefile=cf)

    nurse.handle_breach(_breach_event(value=85.0))
    nurse.handle_breach(_breach_event(value=92.0))

    cases = cf.query("SELECT case_id FROM cases")
    assert len(cases) == 1  # Only one case created

    events = cf.query("SELECT case_id FROM events ORDER BY event_id")
    assert len(events) == 2
    assert events[0][0] == events[1][0]  # Both linked to same case

    cf.close()


def test_severity_scales_with_overshoot() -> None:
    """Severity escalates based on how far above the threshold."""
    cf = CaseFile(":memory:")
    nurse = ChargeNurse(casefile=cf)

    # 85 vs 80 threshold = 6.25% overshoot → info
    nurse.handle_breach(_breach_event(value=85.0, threshold=80.0, gpu_guid="GPU-info"))
    # 90 vs 80 threshold = 12.5% overshoot → warning
    nurse.handle_breach(_breach_event(value=90.0, threshold=80.0, gpu_guid="GPU-warn"))
    # 100 vs 80 threshold = 25% overshoot → critical
    nurse.handle_breach(_breach_event(value=100.0, threshold=80.0, gpu_guid="GPU-crit"))

    rows = cf.query("SELECT gpu_guid, severity FROM events ORDER BY event_id")
    assert rows[0] == ("GPU-info", "info")
    assert rows[1] == ("GPU-warn", "warning")
    assert rows[2] == ("GPU-crit", "critical")

    cf.close()


def test_different_gpu_creates_separate_case() -> None:
    """Breaches on different GPUs create separate cases."""
    cf = CaseFile(":memory:")
    nurse = ChargeNurse(casefile=cf)

    nurse.handle_breach(_breach_event(gpu_guid="GPU-aaa"))
    nurse.handle_breach(_breach_event(gpu_guid="GPU-bbb"))

    cases = cf.query("SELECT case_id FROM cases ORDER BY case_id")
    assert len(cases) == 2  # Two distinct cases

    cf.close()


# --- Slice 2: pipeline_fn=None preserves Phase 1 behavior ---


def test_breach_without_pipeline_works_as_before() -> None:
    """ChargeNurse with pipeline_fn=None still creates events+cases."""
    cf = CaseFile(":memory:")
    nurse = ChargeNurse(casefile=cf, pipeline_fn=None)

    nurse.handle_breach(_breach_event())

    events = cf.query("SELECT kind FROM events")
    assert len(events) == 1
    cases = cf.query("SELECT status FROM cases")
    assert len(cases) == 1
    assert cases[0][0] == "open"

    cf.close()


# --- Slice 3: pipeline dispatch ---


def test_breach_with_pipeline_dispatches_async() -> None:
    """When pipeline_fn is set, handle_breach dispatches it asynchronously."""
    cf = CaseFile(":memory:")
    calls: list[tuple] = []
    call_event = threading.Event()

    def mock_pipeline(case_id: int, event: dict) -> None:
        calls.append((case_id, event))
        call_event.set()

    nurse = ChargeNurse(casefile=cf, pipeline_fn=mock_pipeline)
    evt = _breach_event()
    nurse.handle_breach(evt)

    # Wait for async dispatch
    assert call_event.wait(timeout=2.0), "pipeline_fn was not called within timeout"
    assert len(calls) == 1
    assert calls[0][0] >= 1  # valid case_id
    assert calls[0][1]["metric"] == "utilization_percent"

    nurse.shutdown()
    cf.close()


# --- Slice 4: single-flight queuing ---


def test_single_flight_queues_concurrent_breaches() -> None:
    """Two rapid breaches are processed sequentially by the single worker."""
    cf = CaseFile(":memory:")
    order: list[int] = []
    lock = threading.Lock()

    def slow_pipeline(case_id: int, _event: dict) -> None:
        time.sleep(0.1)
        with lock:
            order.append(case_id)

    nurse = ChargeNurse(casefile=cf, pipeline_fn=slow_pipeline)
    nurse.handle_breach(_breach_event(gpu_guid="GPU-first"))
    nurse.handle_breach(_breach_event(gpu_guid="GPU-second"))

    nurse.shutdown(timeout=5.0)

    assert len(order) == 2
    # Both completed (sequential order guaranteed by single worker)
    cf.close()


# --- Slice 5: shutdown ---


def test_shutdown_waits_for_active_pipeline() -> None:
    """Shutdown waits for an in-flight pipeline to complete."""
    cf = CaseFile(":memory:")
    completed = threading.Event()

    def slow_pipeline(_case_id: int, _event: dict) -> None:
        time.sleep(0.2)
        completed.set()

    nurse = ChargeNurse(casefile=cf, pipeline_fn=slow_pipeline)
    nurse.handle_breach(_breach_event())
    time.sleep(0.05)  # Let worker pick it up

    nurse.shutdown(timeout=5.0)

    assert completed.is_set(), "Pipeline was interrupted before completion"
    cf.close()


def test_shutdown_without_pipeline_is_noop() -> None:
    """Shutdown with no pipeline configured doesn't crash."""
    cf = CaseFile(":memory:")
    nurse = ChargeNurse(casefile=cf)
    nurse.shutdown()  # Should not raise
    cf.close()
