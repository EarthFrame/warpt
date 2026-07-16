"""Tests for the daemon-only PowerMonitor (no native fallback)."""

from unittest.mock import MagicMock, patch

import pytest

from warpt.backends.power.factory import PowerMonitor
from warpt.models.power_models import (
    DomainPower,
    GPUPowerInfo,
    PowerDomain,
    PowerSource,
)

_DAEMON = "warpt.backends.power.factory.DaemonPowerBackend"


@pytest.fixture(autouse=True)
def _no_retry_pause(monkeypatch):
    """Skip the health-check retry pause so tests stay fast."""
    monkeypatch.setattr("warpt.backends.power.factory._HEALTH_RETRY_PAUSE_S", 0.0)


def _fake_daemon():
    daemon = MagicMock()
    daemon.is_available.return_value = True
    daemon.read_snapshot.return_value = (
        [
            DomainPower(PowerDomain.PACKAGE, 45.0, 5000.0, PowerSource.DAEMON),
            DomainPower(
                PowerDomain.GPU, 250.0, 30000.0, PowerSource.DAEMON, {"gpu_index": 0}
            ),
        ],
        [GPUPowerInfo(index=0, name="NVIDIA A100", power_watts=250.0)],
        546.0,
    )
    return daemon


class TestDaemonIsSoleSource:
    """When the daemon is reachable it is the only measurement source."""

    @patch(_DAEMON)
    def test_daemon_is_sole_source(self, mock_cls):
        """A reachable daemon is adopted as the only power source."""
        mock_cls.return_value = _fake_daemon()
        monitor = PowerMonitor()
        assert monitor.initialize() is True
        assert monitor.is_daemon_active() is True
        assert monitor.get_available_sources() == [PowerSource.DAEMON]

    @patch(_DAEMON)
    def test_snapshot_comes_from_daemon(self, mock_cls):
        """Snapshot total, domains, and GPUs all come from the daemon."""
        mock_cls.return_value = _fake_daemon()
        monitor = PowerMonitor()
        monitor.initialize()
        snapshot = monitor.get_snapshot()

        assert snapshot.total_power_watts == 546.0
        assert len(snapshot.domains) == 2
        assert len(snapshot.gpus) == 1
        assert snapshot.gpus[0].name == "NVIDIA A100"
        assert snapshot.processes == []  # per-process attribution dropped

    @patch(_DAEMON)
    def test_health_check_retries_once(self, mock_cls):
        """A first failed health check is retried once before adopting."""
        daemon = MagicMock()
        daemon.is_available.side_effect = [False, True]  # recovers on retry
        daemon.read_snapshot.return_value = ([], [], 0.0)
        mock_cls.return_value = daemon

        monitor = PowerMonitor()
        assert monitor.initialize() is True
        assert daemon.is_available.call_count == 2


class TestDaemonUnavailable:
    """No daemon → no source, no native fallback."""

    @patch(_DAEMON)
    def test_initialize_false_when_daemon_down(self, mock_cls):
        """An unreachable daemon means no source and a clear reason."""
        daemon = MagicMock()
        daemon.is_available.return_value = False
        mock_cls.return_value = daemon

        monitor = PowerMonitor()
        assert monitor.initialize() is False
        assert monitor.is_daemon_active() is False
        assert monitor.get_available_sources() == []
        assert monitor.get_unavailable_reasons()  # non-empty explanation

    @patch(_DAEMON)
    def test_snapshot_empty_when_daemon_down(self, mock_cls):
        """With no daemon, snapshots carry no readings."""
        daemon = MagicMock()
        daemon.is_available.return_value = False
        mock_cls.return_value = daemon

        monitor = PowerMonitor()
        monitor.initialize()
        snapshot = monitor.get_snapshot()
        assert snapshot.total_power_watts is None
        assert snapshot.domains == []
        assert snapshot.gpus == []
