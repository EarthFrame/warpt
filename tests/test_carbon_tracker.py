"""Tests for CarbonTracker context manager."""

import time
from unittest.mock import MagicMock, patch

import pytest

from warpt.carbon.tracker import CarbonTracker
from warpt.models.carbon_models import CarbonSession
from warpt.models.power_models import (
    DomainPower,
    PowerDomain,
    PowerSnapshot,
    PowerSource,
)

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


class TestCarbonTrackerEnergyCounter:
    """Tests for hardware energy counter integration."""

    @patch("warpt.carbon.tracker.EnergyStore")
    @patch(_PM_PATH)
    def test_uses_counter_when_available(self, mock_pm_cls, mock_store_cls):
        """Tracker uses GPU energy counter delta when available."""
        mock_monitor = MagicMock()
        mock_monitor.initialize.return_value = True
        mock_monitor.get_available_sources.return_value = []
        mock_pm_cls.return_value = mock_monitor

        call_count = 0

        def make_snapshot():
            nonlocal call_count
            call_count += 1
            # First call: __enter__ snapshot (start counter)
            # Middle calls: sample loop
            # Last call: _get_gpu_counter_delta (end counter)
            energy_j = 1000.0 + (call_count * 500.0)
            return PowerSnapshot(
                timestamp=time.time(),
                total_power_watts=200.0,
                domains=[
                    DomainPower(
                        domain=PowerDomain.GPU,
                        power_watts=150.0,
                        energy_joules=energy_j,
                        source=PowerSource.NVML,
                        metadata={"gpu_index": 0},
                    ),
                ],
            )

        mock_monitor.get_snapshot.side_effect = make_snapshot
        mock_store_cls.return_value = MagicMock()

        with CarbonTracker(label="test", interval=0.05):
            time.sleep(0.15)

        final_session = mock_store_cls.return_value.update_session.call_args[0][0]
        assert final_session.metadata["energy_source"] == "counter"

    @patch("warpt.carbon.tracker.EnergyStore")
    @patch(_PM_PATH)
    def test_falls_back_to_polled_without_counter(self, mock_pm_cls, mock_store_cls):
        """Tracker falls back to trapezoidal integration without counters."""
        mock_monitor = MagicMock()
        mock_monitor.initialize.return_value = True
        mock_monitor.get_available_sources.return_value = []
        mock_pm_cls.return_value = mock_monitor

        # No energy_joules on domains — no counter available
        snapshot = PowerSnapshot(
            timestamp=time.time(),
            total_power_watts=100.0,
        )
        mock_monitor.get_snapshot.return_value = snapshot
        mock_store_cls.return_value = MagicMock()

        with CarbonTracker(label="test", interval=0.05):
            time.sleep(0.15)

        final_session = mock_store_cls.return_value.update_session.call_args[0][0]
        assert final_session.metadata["energy_source"] == "polled"

    @patch("warpt.carbon.tracker.EnergyStore")
    @patch(_PM_PATH)
    def test_counter_delta_captures_start_energy(self, mock_pm_cls, mock_store_cls):
        """Start energy is captured from initial snapshot in __enter__."""
        mock_monitor = MagicMock()
        mock_monitor.initialize.return_value = True
        mock_monitor.get_available_sources.return_value = []
        mock_pm_cls.return_value = mock_monitor

        snapshot = PowerSnapshot(
            timestamp=time.time(),
            total_power_watts=100.0,
            domains=[
                DomainPower(
                    domain=PowerDomain.GPU,
                    power_watts=80.0,
                    energy_joules=5000.0,
                    source=PowerSource.NVML,
                    metadata={"gpu_index": 0},
                ),
            ],
        )
        mock_monitor.get_snapshot.return_value = snapshot
        mock_store_cls.return_value = MagicMock()

        with CarbonTracker(label="test", interval=0.05) as tracker:
            assert tracker._start_gpu_energy == {0: 5000.0}
            time.sleep(0.1)


class TestCarbonTrackerCPUCounter:
    """Tests for CPU RAPL energy counter integration."""

    @patch("warpt.carbon.tracker.EnergyStore")
    @patch(_PM_PATH)
    def test_uses_cpu_counter_when_available(self, mock_pm_cls, mock_store_cls):
        """Tracker uses CPU energy counter delta when available."""
        mock_monitor = MagicMock()
        mock_monitor.initialize.return_value = True
        mock_monitor.get_available_sources.return_value = []
        mock_pm_cls.return_value = mock_monitor

        call_count = 0

        def make_snapshot():
            nonlocal call_count
            call_count += 1
            energy_j = 1000.0 + (call_count * 200.0)
            return PowerSnapshot(
                timestamp=time.time(),
                total_power_watts=100.0,
                domains=[
                    DomainPower(
                        domain=PowerDomain.PACKAGE,
                        power_watts=50.0,
                        energy_joules=energy_j,
                        source=PowerSource.RAPL,
                        metadata={"rapl_name": "package-0"},
                    ),
                ],
            )

        mock_monitor.get_snapshot.side_effect = make_snapshot
        mock_store_cls.return_value = MagicMock()

        with CarbonTracker(label="test", interval=0.05):
            time.sleep(0.15)

        final_session = mock_store_cls.return_value.update_session.call_args[0][0]
        assert final_session.metadata["energy_source"] == "counter"

    @patch("warpt.carbon.tracker.EnergyStore")
    @patch(_PM_PATH)
    def test_cpu_counter_with_gpu_counter(self, mock_pm_cls, mock_store_cls):
        """Both CPU and GPU counters present, both used."""
        mock_monitor = MagicMock()
        mock_monitor.initialize.return_value = True
        mock_monitor.get_available_sources.return_value = []
        mock_pm_cls.return_value = mock_monitor

        call_count = 0

        def make_snapshot():
            nonlocal call_count
            call_count += 1
            cpu_energy = 500.0 + (call_count * 100.0)
            gpu_energy = 2000.0 + (call_count * 300.0)
            return PowerSnapshot(
                timestamp=time.time(),
                total_power_watts=200.0,
                domains=[
                    DomainPower(
                        domain=PowerDomain.PACKAGE,
                        power_watts=60.0,
                        energy_joules=cpu_energy,
                        source=PowerSource.RAPL,
                        metadata={"rapl_name": "package-0"},
                    ),
                    DomainPower(
                        domain=PowerDomain.GPU,
                        power_watts=140.0,
                        energy_joules=gpu_energy,
                        source=PowerSource.NVML,
                        metadata={"gpu_index": 0},
                    ),
                ],
            )

        mock_monitor.get_snapshot.side_effect = make_snapshot
        mock_store_cls.return_value = MagicMock()

        with CarbonTracker(label="test", interval=0.05):
            time.sleep(0.15)

        final_session = mock_store_cls.return_value.update_session.call_args[0][0]
        assert final_session.metadata["energy_source"] == "counter"
        assert final_session.energy_kwh > 0

    @patch("warpt.carbon.tracker.EnergyStore")
    @patch(_PM_PATH)
    def test_cpu_counter_without_gpu_counter(self, mock_pm_cls, mock_store_cls):
        """CPU counter only, GPU polled."""
        mock_monitor = MagicMock()
        mock_monitor.initialize.return_value = True
        mock_monitor.get_available_sources.return_value = []
        mock_pm_cls.return_value = mock_monitor

        call_count = 0

        def make_snapshot():
            nonlocal call_count
            call_count += 1
            cpu_energy = 1000.0 + (call_count * 150.0)
            return PowerSnapshot(
                timestamp=time.time(),
                total_power_watts=120.0,
                domains=[
                    DomainPower(
                        domain=PowerDomain.PACKAGE,
                        power_watts=70.0,
                        energy_joules=cpu_energy,
                        source=PowerSource.RAPL,
                        metadata={"rapl_name": "package-0"},
                    ),
                    DomainPower(
                        domain=PowerDomain.GPU,
                        power_watts=50.0,
                        energy_joules=None,  # No GPU counter
                        source=PowerSource.NVML,
                        metadata={"gpu_index": 0},
                    ),
                ],
            )

        mock_monitor.get_snapshot.side_effect = make_snapshot
        mock_store_cls.return_value = MagicMock()

        with CarbonTracker(label="test", interval=0.05):
            time.sleep(0.15)

        final_session = mock_store_cls.return_value.update_session.call_args[0][0]
        assert final_session.metadata["energy_source"] == "counter"

    @patch("warpt.carbon.tracker.EnergyStore")
    @patch(_PM_PATH)
    def test_counter_delta_captures_start_cpu_energy(self, mock_pm_cls, mock_store_cls):
        """Start CPU energy is captured from initial snapshot in __enter__."""
        mock_monitor = MagicMock()
        mock_monitor.initialize.return_value = True
        mock_monitor.get_available_sources.return_value = []
        mock_pm_cls.return_value = mock_monitor

        snapshot = PowerSnapshot(
            timestamp=time.time(),
            total_power_watts=80.0,
            domains=[
                DomainPower(
                    domain=PowerDomain.PACKAGE,
                    power_watts=40.0,
                    energy_joules=3000.0,
                    source=PowerSource.RAPL,
                    metadata={"rapl_name": "package-0"},
                ),
            ],
        )
        mock_monitor.get_snapshot.return_value = snapshot
        mock_store_cls.return_value = MagicMock()

        with CarbonTracker(label="test", interval=0.05) as tracker:
            assert tracker._start_cpu_energy == {"package-0": 3000.0}
            time.sleep(0.1)
