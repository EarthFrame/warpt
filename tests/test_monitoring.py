"""Tests for the live monitoring helpers."""

import time

from warpt.monitoring import SystemMonitorDaemon, get_gpu_guid


def test_system_monitor_daemon_collects_cpu_and_memory_snapshot() -> None:
    """Daemon rounds collect CPU/memory info even when GPUs are excluded."""
    daemon = SystemMonitorDaemon(interval_seconds=0.1, include_gpu=False)
    daemon.start()

    try:
        time.sleep(0.2)
        snapshot = daemon.get_latest_snapshot()
    finally:
        daemon.stop()

    assert snapshot is not None
    assert snapshot.cpu_utilization_percent is not None
    assert snapshot.total_memory_bytes > 0
    assert snapshot.memory_utilization_percent is not None


def test_system_monitor_listener_receives_snapshots() -> None:
    """Snapshot listener receives every collected sample."""
    samples: list = []
    daemon = SystemMonitorDaemon(
        interval_seconds=0.05,
        include_gpu=False,
        snapshot_listener=samples.append,
    )
    daemon.start()

    try:
        time.sleep(0.15)
    finally:
        daemon.stop()

    assert samples
    assert samples[-1].cpu_utilization_percent is not None


def test_get_gpu_guid_returns_none_for_missing_device() -> None:
    """Missing GPUs return None when GUID cannot be resolved."""
    assert get_gpu_guid(9999) is None
