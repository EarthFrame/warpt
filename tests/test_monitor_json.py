"""Tests for the --json output flag on warpt monitor."""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import patch

from warpt.monitoring import GPUUsage, ResourceSnapshot


def _make_snapshot(timestamp: float = 1711200000.0) -> ResourceSnapshot:
    """Build a sample snapshot for testing."""
    return ResourceSnapshot(
        timestamp=timestamp,
        cpu_utilization_percent=45.2,
        cpu_power_watts=125.0,
        total_memory_bytes=8_589_934_592,
        available_memory_bytes=2_684_354_560,
        wired_memory_bytes=1_073_741_824,
        memory_utilization_percent=68.7,
        gpu_usage=[
            GPUUsage(
                index=0,
                model="NVIDIA RTX 4090",
                utilization_percent=82.3,
                memory_utilization_percent=45.1,
                power_watts=280.5,
                guid="GPU-12345678-abcd-1234-abcd-123456789abc",
            ),
        ],
    )


def test_json_output_is_valid_json() -> None:
    """run_monitor with output_json=True emits valid JSON lines to stdout."""
    snapshot = _make_snapshot()
    captured = StringIO()

    with (
        patch(
            "warpt.commands.monitor_cmd.SystemMonitorDaemon"
        ) as mock_daemon_cls,
        patch("sys.stdout", captured),
    ):
        daemon_instance = mock_daemon_cls.return_value
        daemon_instance.get_latest_snapshot.return_value = snapshot

        from warpt.commands.monitor_cmd import run_monitor

        run_monitor(
            interval_seconds=0.1,
            duration_seconds=0.15,
            output_json=True,
        )

    output = captured.getvalue().strip()
    assert output, "Expected JSON output but got nothing"

    line = output.split("\n")[0]
    data = json.loads(line)

    assert "timestamp" in data
    assert "cpu_utilization_percent" in data
    assert "gpu_usage" in data
    assert len(data["gpu_usage"]) == 1
    assert data["gpu_usage"][0]["guid"] == "GPU-12345678-abcd-1234-abcd-123456789abc"


def test_json_output_contains_all_snapshot_fields() -> None:
    """JSON output includes every field from ResourceSnapshot.to_dict()."""
    snapshot = _make_snapshot()
    captured = StringIO()

    with (
        patch(
            "warpt.commands.monitor_cmd.SystemMonitorDaemon"
        ) as mock_daemon_cls,
        patch("sys.stdout", captured),
    ):
        daemon_instance = mock_daemon_cls.return_value
        daemon_instance.get_latest_snapshot.return_value = snapshot

        from warpt.commands.monitor_cmd import run_monitor

        run_monitor(
            interval_seconds=0.1,
            duration_seconds=0.15,
            output_json=True,
        )

    data = json.loads(captured.getvalue().strip().split("\n")[0])
    expected_keys = {
        "timestamp",
        "cpu_utilization_percent",
        "cpu_power_watts",
        "total_memory_bytes",
        "available_memory_bytes",
        "wired_memory_bytes",
        "memory_utilization_percent",
        "gpu_usage",
    }
    assert expected_keys == set(data.keys())

    gpu = data["gpu_usage"][0]
    expected_gpu_keys = {
        "index",
        "model",
        "utilization_percent",
        "memory_utilization_percent",
        "power_watts",
        "guid",
    }
    assert expected_gpu_keys == set(gpu.keys())


def test_default_output_is_not_json() -> None:
    """Without output_json, run_monitor emits human-readable text, not JSON."""
    snapshot = _make_snapshot()
    captured = StringIO()

    with (
        patch(
            "warpt.commands.monitor_cmd.SystemMonitorDaemon"
        ) as mock_daemon_cls,
        patch("sys.stdout", captured),
    ):
        daemon_instance = mock_daemon_cls.return_value
        daemon_instance.get_latest_snapshot.return_value = snapshot

        from warpt.commands.monitor_cmd import run_monitor

        run_monitor(
            interval_seconds=0.1,
            duration_seconds=0.15,
        )

    output = captured.getvalue().strip()
    assert output, "Expected text output but got nothing"

    # Default output should NOT be valid JSON
    try:
        json.loads(output.split("\n")[0])
        is_json = True
    except json.JSONDecodeError:
        is_json = False

    assert not is_json, "Default output should be human-readable text, not JSON"
    assert "CPU:" in output
