"""Tests for CarbonTracker context manager."""

import time
from unittest.mock import MagicMock, patch

import pytest

from warpt.carbon.tracker import CarbonTracker
from warpt.models.carbon_models import CarbonSession
from warpt.models.power_models import PowerSnapshot

# PowerMonitor is imported at module level in tracker.py, so patch
# the name in the tracker module (where it's looked up at runtime).
_PM_PATH = "warpt.carbon.tracker.PowerMonitor"


class TestCarbonTrackerNoop:
    """Tests for CarbonTracker graceful degradation."""

    @patch(_PM_PATH)
    def test_noop_when_no_power_sources(self, _mock_pm_cls):
        """Tracker becomes no-op when initialize() returns False."""
        mock_monitor = MagicMock()
        mock_monitor.initialize.return_value = False
        _mock_pm_cls.return_value = mock_monitor

        with CarbonTracker(label="test") as tracker:
            assert tracker._noop is True
            result = 1 + 1

        assert result == 2

    @patch(_PM_PATH, side_effect=ImportError("no backend"))
    def test_noop_when_import_fails(self, _mock_pm_cls):
        """Tracker becomes no-op when PowerMonitor raises."""
        with CarbonTracker(label="test") as tracker:
            assert tracker._noop is True


class TestCarbonTrackerIntegration:
    """Tests for CarbonTracker with mocked power backend."""

    @patch("warpt.carbon.tracker.EnergyStore")
    @patch(_PM_PATH)
    def test_creates_and_finalizes_session(self, mock_pm_cls, mock_store_cls):
        """Tracker creates a session on enter and finalizes on exit."""
        mock_monitor = MagicMock()
        mock_monitor.initialize.return_value = True
        mock_monitor.get_available_sources.return_value = []
        mock_pm_cls.return_value = mock_monitor

        snapshot = PowerSnapshot(
            timestamp=time.time(),
            total_power_watts=50.0,
        )
        mock_monitor.get_snapshot.return_value = snapshot

        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        with CarbonTracker(label="test", interval=0.1):
            time.sleep(0.3)

        assert mock_store.create_session.called
        assert mock_store.update_session.called

        final_call = mock_store.update_session.call_args
        session = final_call[0][0]
        assert isinstance(session, CarbonSession)
        assert session.label == "test"
        assert session.end_time is not None
        assert session.duration_s is not None
        assert session.duration_s > 0

    @patch("warpt.carbon.tracker.EnergyStore")
    @patch(_PM_PATH)
    def test_samples_collected(self, mock_pm_cls, mock_store_cls):
        """Tracker collects power samples in background thread."""
        mock_monitor = MagicMock()
        mock_monitor.initialize.return_value = True
        mock_monitor.get_available_sources.return_value = []
        mock_pm_cls.return_value = mock_monitor

        call_count = 0

        def make_snapshot():
            nonlocal call_count
            call_count += 1
            return PowerSnapshot(
                timestamp=time.time(),
                total_power_watts=100.0,
            )

        mock_monitor.get_snapshot.side_effect = make_snapshot
        mock_store_cls.return_value = MagicMock()

        with CarbonTracker(label="test", interval=0.05):
            time.sleep(0.25)

        assert call_count >= 2

    @patch("warpt.carbon.tracker.EnergyStore")
    @patch(_PM_PATH)
    def test_zero_power_samples_skipped(self, mock_pm_cls, mock_store_cls):
        """Samples with None total power are not recorded."""
        mock_monitor = MagicMock()
        mock_monitor.initialize.return_value = True
        mock_monitor.get_available_sources.return_value = []
        mock_pm_cls.return_value = mock_monitor

        snapshot = PowerSnapshot(
            timestamp=time.time(),
            total_power_watts=None,
        )
        mock_monitor.get_snapshot.return_value = snapshot
        mock_store_cls.return_value = MagicMock()

        with CarbonTracker(label="test", interval=0.05) as tracker:
            time.sleep(0.15)

        assert len(tracker._samples) == 0

    @patch("warpt.carbon.tracker.EnergyStore")
    @patch(_PM_PATH)
    def test_exception_in_wrapped_code(self, mock_pm_cls, mock_store_cls):
        """Tracker finalizes session even if wrapped code raises."""
        mock_monitor = MagicMock()
        mock_monitor.initialize.return_value = True
        mock_monitor.get_available_sources.return_value = []
        mock_pm_cls.return_value = mock_monitor

        snapshot = PowerSnapshot(timestamp=time.time(), total_power_watts=50.0)
        mock_monitor.get_snapshot.return_value = snapshot

        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        with pytest.raises(ValueError):
            with CarbonTracker(label="test", interval=0.1):
                raise ValueError("boom")

        assert mock_store.update_session.called
