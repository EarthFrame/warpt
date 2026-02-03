"""Tests for the stress test results collection."""

import json
from io import StringIO

from warpt.stress.results import OutputFormat, TestResults


def test_add_result_and_finalize():
    """Test adding results and finalization."""
    results = TestResults()
    results.add_result("Test1", {"metric": 10})
    results.add_error("Test2", "Failed")

    assert "Test1" in results.results
    assert results.results["Test1"]["metric"] == 10
    assert "Test2" in results.errors

    results.finalize()
    assert results.to_dict()["metadata"]["timestamp_end"] is not None


def test_generate_summary():
    """Test results summary generation."""
    results = TestResults()
    results.add_result("P1", {"val": 1})
    results.add_result("P2", {"val": 2})
    results.add_error("F1", "Error")

    summary = results.to_dict()["summary"]
    assert summary["total_tests"] == 3
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    assert summary["success_rate"] == 2 / 3


def test_emit_json():
    """Test JSON emission."""
    results = TestResults()
    results.add_result("Test", {"val": 1})

    output = StringIO()
    results.emit(output, format=OutputFormat.JSON)

    data = json.loads(output.getvalue())
    assert "results" in data
    assert data["results"]["Test"]["val"] == 1
