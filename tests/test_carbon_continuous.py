"""Tests for the continuous energy odometer (ContinuousEnergyTracker)."""

from __future__ import annotations

import json

import pytest

from warpt.backends.power.daemon_client import PowerClientError, PowerReading
from warpt.carbon.calculator import CarbonCalculator
from warpt.carbon.continuous import (
    STATE_FILENAME,
    ContinuousEnergyTracker,
    peek_persisted_totals,
)

JOULES_PER_KWH = 3_600_000.0


def make_reading(joules: float, reset_time: float = 100.0) -> PowerReading:
    """Build a PowerReading with only the odometer-relevant fields set."""
    return PowerReading(
        timestamp=0.0,
        watts=50.0,
        joules_since_reset=joules,
        watt_hours_since_reset=joules / 3600.0,
        reset_time=reset_time,
        hostname="test-node",
    )


class StubClient:
    """PowerClient stand-in that replays a scripted sequence of readings.

    Entries may be ``PowerReading`` instances or exceptions to raise.
    The final entry is repeated once the script is exhausted.
    """

    def __init__(self, *script: object) -> None:
        self._script = list(script)

    def current(self) -> PowerReading:
        """Return (or raise) the next scripted entry."""
        item = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, PowerReading)
        return item


def make_tracker(tmp_path, *script: object) -> ContinuousEnergyTracker:
    """Build a tracker over tmp_path with a scripted stub client."""
    return ContinuousEnergyTracker(str(tmp_path), client=StubClient(*script))


def test_first_read_baselines_at_zero(tmp_path):
    """The odometer counts observed energy only — first read credits nothing."""
    tracker = make_tracker(tmp_path, make_reading(1_000_000.0))
    totals = tracker.read()
    assert totals["energy_kwh"] == 0.0
    assert totals["co2_grams"] == 0.0
    assert totals["cost_usd"] == 0.0
    assert totals["daemon_reachable"] is True


def test_accumulates_counter_deltas(tmp_path):
    """Successive reads credit the counter deltas between them."""
    tracker = make_tracker(
        tmp_path,
        make_reading(1_000.0),
        make_reading(4_000.0),
        make_reading(10_000.0),
    )
    tracker.read()  # baseline at 1000 J
    tracker.read()  # +3000 J
    totals = tracker.read()  # +6000 J
    assert totals["energy_kwh"] == pytest.approx(9_000.0 / JOULES_PER_KWH, abs=1e-6)


def test_reset_rebaselines_and_keeps_total(tmp_path):
    """A daemon restart re-baselines without losing the running total."""
    tracker = make_tracker(
        tmp_path,
        make_reading(1_000.0, reset_time=100.0),
        make_reading(5_000.0, reset_time=100.0),
        make_reading(200.0, reset_time=999.0),  # daemon restarted
        make_reading(1_200.0, reset_time=999.0),
    )
    tracker.read()  # baseline
    tracker.read()  # +4000 J
    totals = tracker.read()  # reset: re-baseline, credit nothing
    assert totals["energy_kwh"] == pytest.approx(4_000.0 / JOULES_PER_KWH, abs=1e-6)
    totals = tracker.read()  # +1000 J from the new baseline
    assert totals["energy_kwh"] == pytest.approx(5_000.0 / JOULES_PER_KWH, abs=1e-6)


def test_backward_counter_is_not_credited(tmp_path):
    """Same reset_time but a lower counter must never subtract energy."""
    tracker = make_tracker(
        tmp_path,
        make_reading(5_000.0),
        make_reading(3_000.0),  # backwards
    )
    tracker.read()
    totals = tracker.read()
    assert totals["energy_kwh"] == 0.0


def test_unreachable_daemon_returns_last_totals(tmp_path):
    """An unreachable daemon returns last totals, flagged unreachable."""
    tracker = make_tracker(
        tmp_path,
        make_reading(1_000.0),
        make_reading(8_200.0),
        PowerClientError("daemon down"),
    )
    tracker.read()
    tracker.read()  # +7200 J = 0.002 kWh
    totals = tracker.read()
    assert totals["daemon_reachable"] is False
    assert totals["energy_kwh"] == pytest.approx(7_200.0 / JOULES_PER_KWH, abs=1e-6)


def test_fresh_tracker_with_unreachable_daemon_is_zero(tmp_path):
    """A fresh tracker with no daemon reports zeros, not an error."""
    tracker = make_tracker(tmp_path, PowerClientError("daemon down"))
    totals = tracker.read()
    assert totals == {
        "energy_kwh": 0.0,
        "co2_grams": 0.0,
        "cost_usd": 0.0,
        "daemon_reachable": False,
    }


def test_persists_and_credits_gap_across_instances(tmp_path):
    """A new tracker continues the total and credits energy from the gap."""
    first = make_tracker(tmp_path, make_reading(1_000.0), make_reading(3_000.0))
    first.read()
    first.read()  # +2000 J

    # Same node, later: daemon counter advanced while we were down and the
    # daemon did NOT restart -> the gap (3000 -> 6000) is credited.
    second = make_tracker(tmp_path, make_reading(6_000.0))
    totals = second.read()
    assert totals["energy_kwh"] == pytest.approx(5_000.0 / JOULES_PER_KWH, abs=1e-6)


def test_gap_with_daemon_restart_keeps_total_but_drops_gap(tmp_path):
    """A restart during an outage drops the gap but keeps the total."""
    first = make_tracker(tmp_path, make_reading(1_000.0), make_reading(3_000.0))
    first.read()
    first.read()  # +2000 J

    # Daemon restarted during the outage -> unobserved gap is dropped,
    # cumulative total survives.
    second = make_tracker(tmp_path, make_reading(500.0, reset_time=777.0))
    totals = second.read()
    assert totals["energy_kwh"] == pytest.approx(2_000.0 / JOULES_PER_KWH, abs=1e-6)


def test_corrupt_state_file_starts_fresh(tmp_path):
    """A corrupt state file is discarded and tracking restarts clean."""
    (tmp_path / STATE_FILENAME).write_text("{not json!!")
    tracker = make_tracker(tmp_path, make_reading(1_000.0), make_reading(2_000.0))
    tracker.read()
    totals = tracker.read()
    assert totals["energy_kwh"] == pytest.approx(1_000.0 / JOULES_PER_KWH, abs=1e-6)


def test_co2_and_cost_follow_calculator(tmp_path):
    """CO2 and cost figures match CarbonCalculator conversions."""
    tracker = ContinuousEnergyTracker(
        str(tmp_path),
        region="US",
        cost_per_kwh=0.20,
        client=StubClient(make_reading(0.0), make_reading(7_200_000.0)),
    )
    tracker.read()
    totals = tracker.read()  # 7 200 000 J = 2 kWh

    calc = CarbonCalculator(region="US")
    assert totals["energy_kwh"] == pytest.approx(2.0)
    assert totals["co2_grams"] == pytest.approx(calc.co2_from_energy(2.0), abs=0.01)
    assert totals["cost_usd"] == pytest.approx(0.40)


def test_peek_persisted_totals(tmp_path):
    """Peek reads persisted totals without daemon contact or writes."""
    assert peek_persisted_totals(str(tmp_path)) is None

    tracker = make_tracker(tmp_path, make_reading(0.0), make_reading(3_600_000.0))
    tracker.read()
    tracker.read()  # 1 kWh

    peeked = peek_persisted_totals(str(tmp_path))
    assert peeked is not None
    assert peeked["energy_kwh"] == pytest.approx(1.0)
    assert "daemon_reachable" not in peeked

    # Peek must be read-only: state file unchanged afterwards.
    state = json.loads((tmp_path / STATE_FILENAME).read_text())
    assert state["cumulative_joules"] == pytest.approx(3_600_000.0)


def test_background_poller_advances_odometer(tmp_path):
    """The background poll thread advances the odometer on its own."""
    tracker = make_tracker(tmp_path, make_reading(0.0), make_reading(1_800_000.0))
    tracker.start(interval_s=0.01)
    try:
        import time

        deadline = time.time() + 2.0
        while time.time() < deadline:
            if tracker.read()["energy_kwh"] >= 0.5:
                break
            time.sleep(0.01)
    finally:
        tracker.stop()
    assert tracker.read()["energy_kwh"] == pytest.approx(0.5)
