"""Tests for the VitalsNurse — subprocess polling, ring buffer, heartbeats."""

from __future__ import annotations

from unittest.mock import MagicMock

from warpt.daemon.vitals_nurse import DEFAULT_GPU_THRESHOLDS, VitalsNurse


def _sample_snapshot(ts: str = "2026-03-23T14:00:00") -> dict:
    """Build a sample JSON snapshot matching ResourceSnapshot.to_dict()."""
    return {
        "timestamp": ts,
        "cpu_utilization_percent": 45.2,
        "cpu_power_watts": 125.0,
        "total_memory_bytes": 8_589_934_592,
        "available_memory_bytes": 2_684_354_560,
        "wired_memory_bytes": 1_073_741_824,
        "memory_utilization_percent": 68.7,
        "gpu_usage": [
            {
                "index": 0,
                "model": "NVIDIA RTX 4090",
                "utilization_percent": 82.3,
                "memory_utilization_percent": 45.1,
                "power_watts": 280.5,
                "guid": "GPU-aaa-bbb-ccc",
            },
        ],
    }


def test_feed_snapshot_populates_ring_buffer() -> None:
    """Feeding a snapshot adds it to the ring buffer."""
    casefile = MagicMock()
    nurse = VitalsNurse(casefile=casefile, buffer_size=60)

    nurse.feed_snapshot(_sample_snapshot())

    buf = nurse.get_buffer()
    assert len(buf) == 1
    assert buf[0]["cpu_utilization_percent"] == 45.2


def test_ring_buffer_rotates_when_full() -> None:
    """Oldest snapshot drops off when buffer exceeds max size."""
    casefile = MagicMock()
    nurse = VitalsNurse(casefile=casefile, buffer_size=3)

    for i in range(5):
        nurse.feed_snapshot(_sample_snapshot(ts=f"2026-03-23T14:00:0{i}"))

    buf = nurse.get_buffer()
    assert len(buf) == 3
    assert buf[0]["timestamp"] == "2026-03-23T14:00:02"
    assert buf[2]["timestamp"] == "2026-03-23T14:00:04"


def test_heartbeat_writes_vitals_on_first_snapshot() -> None:
    """First snapshot triggers an immediate heartbeat write."""
    casefile = MagicMock()
    nurse = VitalsNurse(casefile=casefile, heartbeat_interval=300.0)

    nurse.feed_snapshot(_sample_snapshot())

    casefile.execute.assert_called()
    # Find the vitals INSERT call (has 'heartbeat' as last param)
    vitals_calls = [
        c
        for c in casefile.execute.call_args_list
        if c.args
        and len(c.args) > 1
        and isinstance(c.args[1], list)
        and "heartbeat" in c.args[1]
    ]
    assert len(vitals_calls) == 1


def test_heartbeat_respects_interval() -> None:
    """Second snapshot within the interval does not write another heartbeat."""
    casefile = MagicMock()
    nurse = VitalsNurse(casefile=casefile, heartbeat_interval=300.0)

    nurse.feed_snapshot(_sample_snapshot(ts="2026-03-23T14:00:00"))
    heartbeat_count_after_first = len(
        [
            c
            for c in casefile.execute.call_args_list
            if c.args
            and len(c.args) > 1
            and isinstance(c.args[1], list)
            and "heartbeat" in c.args[1]
        ]
    )

    nurse.feed_snapshot(_sample_snapshot(ts="2026-03-23T14:00:05"))
    heartbeat_count_after_second = len(
        [
            c
            for c in casefile.execute.call_args_list
            if c.args
            and len(c.args) > 1
            and isinstance(c.args[1], list)
            and "heartbeat" in c.args[1]
        ]
    )

    assert heartbeat_count_after_first == 1
    assert heartbeat_count_after_second == 1  # No second heartbeat


def test_registers_new_gpu_on_first_encounter() -> None:
    """First snapshot with a GPU GUID triggers a gpu_profiles upsert."""
    casefile = MagicMock()
    nurse = VitalsNurse(casefile=casefile)

    nurse.feed_snapshot(_sample_snapshot())

    gpu_calls = [
        c
        for c in casefile.execute.call_args_list
        if c.args and "gpu_profiles" in c.args[0]
    ]
    assert len(gpu_calls) == 1
    assert gpu_calls[0].args[1][0] == "GPU-aaa-bbb-ccc"
    assert gpu_calls[0].args[1][1] == "NVIDIA RTX 4090"


def test_does_not_re_register_known_gpu() -> None:
    """Subsequent snapshots with the same GPU GUID skip registration."""
    casefile = MagicMock()
    nurse = VitalsNurse(casefile=casefile)

    nurse.feed_snapshot(_sample_snapshot(ts="2026-03-23T14:00:00"))
    nurse.feed_snapshot(_sample_snapshot(ts="2026-03-23T14:00:05"))

    gpu_calls = [
        c
        for c in casefile.execute.call_args_list
        if c.args and "gpu_profiles" in c.args[0]
    ]
    assert len(gpu_calls) == 1  # Only the first encounter


def test_heartbeat_round_trips_through_casefile() -> None:
    """Heartbeat written by VitalsNurse is queryable from DuckDB."""
    from warpt.daemon.casefile import CaseFile

    cf = CaseFile(":memory:")
    nurse = VitalsNurse(casefile=cf, heartbeat_interval=0.0)

    nurse.feed_snapshot(_sample_snapshot())

    rows = cf.query("SELECT ts, collection_type FROM vitals")
    assert len(rows) == 1
    assert rows[0][1] == "heartbeat"

    gpu_rows = cf.query("SELECT gpu_guid, model FROM gpu_profiles")
    assert len(gpu_rows) == 1
    assert gpu_rows[0][0] == "GPU-aaa-bbb-ccc"
    assert gpu_rows[0][1] == "NVIDIA RTX 4090"

    cf.close()


# ── Threshold detection tests ──────────────────────────────────────


def _breach_snapshot(
    ts: str = "2026-03-23T14:00:00",
    gpu_util: float = 95.0,
    gpu_guid: str = "GPU-aaa-bbb-ccc",
) -> dict:
    """Snapshot with a GPU utilization above the default 80% threshold."""
    return {
        "timestamp": ts,
        "cpu_utilization_percent": 45.2,
        "cpu_power_watts": 125.0,
        "total_memory_bytes": 8_589_934_592,
        "available_memory_bytes": 2_684_354_560,
        "wired_memory_bytes": 1_073_741_824,
        "memory_utilization_percent": 68.7,
        "gpu_usage": [
            {
                "index": 0,
                "model": "NVIDIA RTX 4090",
                "utilization_percent": gpu_util,
                "memory_utilization_percent": 45.1,
                "power_watts": 280.5,
                "guid": gpu_guid,
            },
        ],
    }


def test_threshold_breach_fires_after_sustained_duration() -> None:
    """Callback fires only after the breach is sustained for the configured duration."""
    sustained = DEFAULT_GPU_THRESHOLDS["utilization_percent"]["sustained_seconds"]

    casefile = MagicMock()
    callback = MagicMock()
    nurse = VitalsNurse(casefile=casefile, heartbeat_interval=9999.0)
    nurse.set_on_threshold_breach(callback)

    from unittest.mock import patch

    fake_time = [100.0]

    def mock_monotonic() -> float:
        return fake_time[0]

    with patch("warpt.daemon.vitals_nurse.time.monotonic", side_effect=mock_monotonic):
        # First snapshot — breach starts but not yet sustained
        nurse.feed_snapshot(_breach_snapshot(ts="2026-03-23T14:00:00"))
        callback.assert_not_called()

        # Second snapshot — halfway through sustained period, should not fire
        fake_time[0] = 100.0 + sustained / 2
        nurse.feed_snapshot(_breach_snapshot(ts="2026-03-23T14:00:01"))
        callback.assert_not_called()

        # Third snapshot — past sustained duration, should fire
        fake_time[0] = 100.0 + sustained + 1.0
        nurse.feed_snapshot(_breach_snapshot(ts="2026-03-23T14:00:02"))
        callback.assert_called_once()

        # Verify event data shape
        event = callback.call_args[0][0]
        assert event["metric"] == "utilization_percent"
        assert event["value"] == 95.0
        assert event["threshold"] == 80.0
        assert event["gpu_guid"] == "GPU-aaa-bbb-ccc"


def test_threshold_breach_resets_when_metric_drops() -> None:
    """Breach timer resets if metric drops, requiring fresh sustained period."""
    from unittest.mock import patch

    sustained = DEFAULT_GPU_THRESHOLDS["utilization_percent"]["sustained_seconds"]
    t0 = 100.0

    casefile = MagicMock()
    callback = MagicMock()
    nurse = VitalsNurse(casefile=casefile, heartbeat_interval=9999.0)
    nurse.set_on_threshold_breach(callback)

    fake_time = [t0]

    def mock_monotonic() -> float:
        return fake_time[0]

    with patch("warpt.daemon.vitals_nurse.time.monotonic", side_effect=mock_monotonic):
        # Breach starts (95% > 80%)
        nurse.feed_snapshot(_breach_snapshot(gpu_util=95.0))

        # Still breaching, halfway through — not enough
        fake_time[0] = t0 + sustained / 2
        nurse.feed_snapshot(_breach_snapshot(gpu_util=90.0))
        callback.assert_not_called()

        # Metric drops below threshold — reset
        fake_time[0] = t0 + sustained / 2 + 1.0
        nurse.feed_snapshot(_breach_snapshot(gpu_util=70.0))
        callback.assert_not_called()

        # Breach starts again from scratch
        t_restart = t0 + sustained / 2 + 2.0
        fake_time[0] = t_restart
        nurse.feed_snapshot(_breach_snapshot(gpu_util=95.0))

        # Halfway through new sustained period — should NOT fire
        fake_time[0] = t_restart + sustained / 2
        nurse.feed_snapshot(_breach_snapshot(gpu_util=95.0))
        callback.assert_not_called()

        # Past sustained duration since restart — should fire
        fake_time[0] = t_restart + sustained + 1.0
        nurse.feed_snapshot(_breach_snapshot(gpu_util=95.0))
        callback.assert_called_once()


def test_threshold_breach_writes_vitals_snapshot() -> None:
    """On breach, a vitals row with collection_type='threshold_breach' is written."""
    from unittest.mock import patch

    casefile = MagicMock()
    nurse = VitalsNurse(casefile=casefile, heartbeat_interval=9999.0)
    nurse.set_on_threshold_breach(MagicMock())

    fake_time = [100.0]

    def mock_monotonic() -> float:
        return fake_time[0]

    with patch("warpt.daemon.vitals_nurse.time.monotonic", side_effect=mock_monotonic):
        nurse.feed_snapshot(_breach_snapshot())
        fake_time[0] = 401.0
        nurse.feed_snapshot(_breach_snapshot(ts="2026-03-23T14:05:01"))

    # Find vitals INSERT calls with 'threshold_breach'
    breach_calls = [
        c
        for c in casefile.execute.call_args_list
        if c.args
        and len(c.args) > 1
        and isinstance(c.args[1], list)
        and "threshold_breach" in c.args[1]
    ]
    assert len(breach_calls) == 1


def test_threshold_skips_null_metrics() -> None:
    """Thresholds referencing NULL data (e.g., temperature) do not fire."""
    from unittest.mock import patch

    casefile = MagicMock()
    callback = MagicMock()
    # Only temperature threshold — but temperature is not in snapshot
    nurse = VitalsNurse(
        casefile=casefile,
        heartbeat_interval=9999.0,
        gpu_thresholds={
            "temperature_c": {"value": 80.0, "sustained_seconds": 0.0},
        },
    )
    nurse.set_on_threshold_breach(callback)

    fake_time = [100.0]

    def mock_monotonic() -> float:
        return fake_time[0]

    with patch("warpt.daemon.vitals_nurse.time.monotonic", side_effect=mock_monotonic):
        # Snapshot has no temperature_c field in gpu_usage
        nurse.feed_snapshot(_breach_snapshot())
        fake_time[0] = 9999.0
        nurse.feed_snapshot(_breach_snapshot())

    callback.assert_not_called()


def test_custom_thresholds_override_defaults() -> None:
    """User-provided thresholds replace the defaults entirely."""
    from unittest.mock import patch

    casefile = MagicMock()
    callback = MagicMock()
    # Custom threshold: utilization > 99% (above the 95% in _breach_snapshot)
    nurse = VitalsNurse(
        casefile=casefile,
        heartbeat_interval=9999.0,
        gpu_thresholds={
            "utilization_percent": {"value": 99.0, "sustained_seconds": 0.0},
        },
    )
    nurse.set_on_threshold_breach(callback)

    fake_time = [100.0]

    def mock_monotonic() -> float:
        return fake_time[0]

    with patch("warpt.daemon.vitals_nurse.time.monotonic", side_effect=mock_monotonic):
        # 95% utilization is below the custom 99% threshold — no breach
        nurse.feed_snapshot(_breach_snapshot(gpu_util=95.0))
        fake_time[0] = 9999.0
        nurse.feed_snapshot(_breach_snapshot(gpu_util=95.0))
        callback.assert_not_called()

        # 99.5% exceeds custom 99% threshold — fires immediately
        nurse.feed_snapshot(_breach_snapshot(gpu_util=99.5))
        callback.assert_called_once()


def test_threshold_breach_round_trips_through_casefile() -> None:
    """Threshold breach writes queryable vitals + fires callback with real DuckDB."""
    from unittest.mock import patch

    from warpt.daemon.casefile import CaseFile

    cf = CaseFile(":memory:")
    events_received: list[dict] = []
    nurse = VitalsNurse(
        casefile=cf,
        heartbeat_interval=9999.0,
        gpu_thresholds={
            "utilization_percent": {"value": 80.0, "sustained_seconds": 0.0},
        },
    )
    nurse.set_on_threshold_breach(lambda e: events_received.append(e))

    fake_time = [100.0]

    def mock_monotonic() -> float:
        return fake_time[0]

    with patch("warpt.daemon.vitals_nurse.time.monotonic", side_effect=mock_monotonic):
        nurse.feed_snapshot(_breach_snapshot(gpu_util=95.0))

    # Verify vitals row written with threshold_breach type
    rows = cf.query(
        "SELECT ts, collection_type FROM vitals "
        "WHERE collection_type = 'threshold_breach'"
    )
    assert len(rows) == 1
    assert rows[0][1] == "threshold_breach"

    # Verify callback received well-formed event
    assert len(events_received) == 1
    assert events_received[0]["metric"] == "utilization_percent"
    assert events_received[0]["value"] == 95.0
    assert events_received[0]["gpu_guid"] == "GPU-aaa-bbb-ccc"

    cf.close()
