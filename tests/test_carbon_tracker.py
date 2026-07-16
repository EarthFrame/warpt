"""Tests for CarbonTracker context manager (daemon-only).

Energy is read exclusively from the Rust power-daemon counter. There is no
native/polled fallback: the daemon is either reachable (tracking works) or it
isn't (tracking is disabled or terminated, never guessed).
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from warpt.backends.power.daemon_client import PowerClientError, PowerReading
from warpt.carbon.tracker import CarbonTracker
from warpt.models.carbon_models import CarbonSession
from warpt.models.power_models import PowerSnapshot

# Names are looked up in the tracker module at runtime, so patch them there.
_PM_PATH = "warpt.carbon.tracker.PowerMonitor"
_CLIENT_PATH = "warpt.carbon.tracker.PowerClient"
_STORE_PATH = "warpt.carbon.tracker.EnergyStore"


def _reading(joules: float, reset_time: float = 42.0, watts: float = 100.0):
    return PowerReading(
        timestamp=time.time(),
        watts=watts,
        joules_since_reset=joules,
        watt_hours_since_reset=joules / 3600.0,
        reset_time=reset_time,
        hostname="node1",
    )


def _mock_monitor(total_watts: float | None = 100.0):
    monitor = MagicMock()
    monitor.initialize.return_value = True
    monitor.get_snapshot.return_value = PowerSnapshot(
        timestamp=time.time(), total_power_watts=total_watts
    )
    return monitor


def _mock_client(readings, healthz: bool = True):
    client = MagicMock()
    client.current.side_effect = readings
    client.healthz.return_value = healthz
    return client


class TestCarbonTrackerNoop:
    """The daemon being down disables tracking but never the workload."""

    @patch(_PM_PATH)
    def test_noop_when_daemon_unreachable(self, mock_pm_cls):
        """initialize() False → no-op tracker, wrapped code still runs."""
        monitor = MagicMock()
        monitor.initialize.return_value = False
        mock_pm_cls.return_value = monitor

        with CarbonTracker(label="test") as tracker:
            assert tracker._noop is True
            result = 1 + 1
        assert result == 2

    @patch(_PM_PATH, side_effect=RuntimeError("no monitor"))
    def test_noop_when_monitor_raises(self, _mock_pm_cls):
        """A monitor construction error degrades to a no-op, not a crash."""
        with CarbonTracker(label="test") as tracker:
            assert tracker._noop is True

    @patch(_CLIENT_PATH)
    @patch(_PM_PATH)
    def test_noop_when_start_counter_unreadable(self, mock_pm_cls, mock_client_cls):
        """Daemon healthy but counter read fails at start → no-op."""
        mock_pm_cls.return_value = _mock_monitor()
        client = MagicMock()
        client.current.side_effect = PowerClientError("counter down")
        mock_client_cls.return_value = client

        with CarbonTracker(label="test") as tracker:
            assert tracker._noop is True


class TestCarbonTrackerHappyPath:
    """Daemon reachable: energy comes from the counter delta."""

    @patch(_STORE_PATH)
    @patch(_CLIENT_PATH)
    @patch(_PM_PATH)
    def test_creates_and_finalizes_session(
        self, mock_pm_cls, mock_client_cls, mock_store_cls
    ):
        """A reachable daemon yields a completed, counter-sourced session."""
        mock_pm_cls.return_value = _mock_monitor()
        mock_client_cls.return_value = _mock_client(
            [_reading(1000.0), _reading(4600.0)]
        )
        store = MagicMock()
        mock_store_cls.return_value = store

        with CarbonTracker(label="test", interval=0.05):
            time.sleep(0.15)

        assert store.create_session.called
        assert store.update_session.called
        session = store.update_session.call_args[0][0]
        assert isinstance(session, CarbonSession)
        assert session.label == "test"
        assert session.end_time is not None
        assert session.duration_s > 0
        assert session.metadata["status"] == "completed"
        assert session.metadata["energy_source"] == "daemon-counter"

    @patch(_STORE_PATH)
    @patch(_CLIENT_PATH)
    @patch(_PM_PATH)
    def test_energy_from_counter_delta(
        self, mock_pm_cls, mock_client_cls, mock_store_cls
    ):
        """3600 J consumed between start and end → 0.001 kWh."""
        mock_pm_cls.return_value = _mock_monitor()
        mock_client_cls.return_value = _mock_client(
            [_reading(1000.0), _reading(4600.0)]
        )
        mock_store_cls.return_value = MagicMock()

        with CarbonTracker(label="test", interval=0.05):
            time.sleep(0.1)

        session = mock_store_cls.return_value.update_session.call_args[0][0]
        assert session.energy_kwh == pytest.approx(0.001)
        assert session.metadata["energy_source"] == "daemon-counter"

    @patch(_STORE_PATH)
    @patch(_CLIENT_PATH)
    @patch(_PM_PATH)
    def test_samples_collected(self, mock_pm_cls, mock_client_cls, mock_store_cls):
        """Background sampling records daemon power readings for avg/peak."""
        mock_pm_cls.return_value = _mock_monitor(total_watts=120.0)
        mock_client_cls.return_value = _mock_client(
            [_reading(1000.0), _reading(2000.0)]
        )
        mock_store_cls.return_value = MagicMock()

        with CarbonTracker(label="test", interval=0.05) as tracker:
            time.sleep(0.2)
            assert len(tracker._samples) >= 2

    @patch(_STORE_PATH)
    @patch(_CLIENT_PATH)
    @patch(_PM_PATH)
    def test_finalizes_even_when_wrapped_code_raises(
        self, mock_pm_cls, mock_client_cls, mock_store_cls
    ):
        """An exception in the wrapped workload still finalizes the session."""
        mock_pm_cls.return_value = _mock_monitor()
        mock_client_cls.return_value = _mock_client(
            [_reading(1000.0), _reading(1500.0)]
        )
        store = MagicMock()
        mock_store_cls.return_value = store

        with pytest.raises(ValueError):
            with CarbonTracker(label="test", interval=0.1):
                raise ValueError("boom")

        assert store.update_session.called


class TestCarbonTrackerTermination:
    """Daemon failures record a terminated session, never a guessed number."""

    @patch(_STORE_PATH)
    @patch(_CLIENT_PATH)
    @patch(_PM_PATH)
    def test_counter_reset_is_terminated(
        self, mock_pm_cls, mock_client_cls, mock_store_cls
    ):
        """A reset_time change (daemon restart) → terminated, no energy."""
        mock_pm_cls.return_value = _mock_monitor()
        mock_client_cls.return_value = _mock_client(
            [_reading(5000.0, reset_time=42.0), _reading(100.0, reset_time=99.0)]
        )
        mock_store_cls.return_value = MagicMock()

        with CarbonTracker(label="test", interval=0.05):
            time.sleep(0.1)

        session = mock_store_cls.return_value.update_session.call_args[0][0]
        assert "terminated" in session.metadata["status"]
        assert session.energy_kwh is None

    @patch(_STORE_PATH)
    @patch(_CLIENT_PATH)
    @patch(_PM_PATH)
    def test_end_read_failure_is_terminated(
        self, mock_pm_cls, mock_client_cls, mock_store_cls
    ):
        """A failed end-of-session counter read → terminated (loss of conn.)."""
        mock_pm_cls.return_value = _mock_monitor()
        mock_client_cls.return_value = _mock_client(
            [_reading(1000.0), PowerClientError("gone")]
        )
        mock_store_cls.return_value = MagicMock()

        with CarbonTracker(label="test", interval=0.05):
            time.sleep(0.1)

        session = mock_store_cls.return_value.update_session.call_args[0][0]
        assert "loss of connection" in session.metadata["status"]
        assert session.energy_kwh is None

    @patch(_STORE_PATH)
    @patch(_CLIENT_PATH)
    @patch(_PM_PATH)
    def test_midrun_loss_terminates_tracking(
        self, mock_pm_cls, mock_client_cls, mock_store_cls
    ):
        """Snapshot stops returning data and health check fails → terminated."""
        monitor = _mock_monitor(total_watts=None)  # daemon returns nothing
        mock_pm_cls.return_value = monitor
        # Start read succeeds; healthz False so the one reconnect attempt fails.
        mock_client_cls.return_value = _mock_client([_reading(1000.0)], healthz=False)
        mock_store_cls.return_value = MagicMock()

        with CarbonTracker(label="test", interval=0.05) as tracker:
            time.sleep(0.7)  # allow the 0.5s reconnect attempt to elapse
            assert tracker._daemon_lost is True

        session = mock_store_cls.return_value.update_session.call_args[0][0]
        assert "loss of connection" in session.metadata["status"]
        assert session.energy_kwh is None
