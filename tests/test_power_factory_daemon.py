"""Tests for PowerMonitor daemon detection / preference / fallback."""

from unittest.mock import MagicMock, patch

from warpt.backends.power.factory import PowerMonitor
from warpt.models.power_models import (
    DomainPower,
    GPUPowerInfo,
    PowerDomain,
    PowerSource,
)

_DAEMON = "warpt.backends.power.factory.DaemonPowerBackend"
_NVIDIA = "warpt.backends.power.factory.NvidiaPowerBackend"


def _fake_daemon():
    daemon = MagicMock()
    daemon.is_available.return_value = True
    daemon.get_source.return_value = PowerSource.DAEMON
    daemon.get_power_readings.return_value = [
        DomainPower(PowerDomain.PACKAGE, 45.0, 5000.0, PowerSource.DAEMON),
        DomainPower(
            PowerDomain.GPU, 250.0, 30000.0, PowerSource.DAEMON, {"gpu_index": 0}
        ),
    ]
    daemon.get_gpu_power_info.return_value = [
        GPUPowerInfo(index=0, name="NVIDIA A100", power_watts=250.0),
    ]
    daemon.get_total_watts.return_value = 546.0
    return daemon


class TestDaemonPreferred:
    """When the daemon is reachable it becomes the measurement source."""

    @patch(_NVIDIA)
    @patch(_DAEMON)
    def test_daemon_is_sole_backend(self, mock_daemon_cls, mock_nvidia_cls):
        """An available daemon is the only measurement backend."""
        mock_daemon_cls.return_value = _fake_daemon()
        mock_nvidia_cls.return_value.is_available.return_value = False

        monitor = PowerMonitor(include_process_attribution=False)
        assert monitor.initialize() is True
        assert monitor.is_daemon_active() is True
        assert monitor.get_available_sources() == [PowerSource.DAEMON]
        assert monitor._backends == [monitor._daemon_backend]

    @patch(_NVIDIA)
    @patch(_DAEMON)
    def test_nvml_kept_only_for_attribution(self, mock_daemon_cls, mock_nvidia_cls):
        """NVML is retained for attribution but not as a measurement backend."""
        mock_daemon_cls.return_value = _fake_daemon()
        nvidia = MagicMock()
        nvidia.is_available.return_value = True
        mock_nvidia_cls.return_value = nvidia

        monitor = PowerMonitor(include_process_attribution=True)
        monitor.initialize()

        # NVML retained for get_process_gpu_usage(), but NOT a measurement backend.
        assert monitor._nvidia_backend is nvidia
        assert nvidia not in monitor._backends

    @patch(_NVIDIA)
    @patch(_DAEMON)
    def test_snapshot_total_and_gpus_from_daemon(
        self, mock_daemon_cls, mock_nvidia_cls
    ):
        """Snapshot total and GPU info come from the daemon when active."""
        mock_daemon_cls.return_value = _fake_daemon()
        mock_nvidia_cls.return_value.is_available.return_value = False

        monitor = PowerMonitor(include_process_attribution=False)
        monitor.initialize()
        snapshot = monitor.get_snapshot()

        assert snapshot.total_power_watts == 546.0
        assert len(snapshot.gpus) == 1
        assert snapshot.gpus[0].name == "NVIDIA A100"


class TestDaemonUnavailableFallback:
    """When the daemon is unreachable, the native path is used."""

    @patch(_DAEMON)
    def test_falls_back_to_native_when_daemon_down(self, mock_daemon_cls):
        """An unavailable daemon leaves the monitor on the native path."""
        mock_daemon_cls.return_value.is_available.return_value = False

        monitor = PowerMonitor(include_process_attribution=False)
        monitor.initialize()  # native path; may or may not find sources

        assert monitor.is_daemon_active() is False
        assert PowerSource.DAEMON not in monitor.get_available_sources()
