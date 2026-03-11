"""Tests for the Tenstorrent accelerator and power backends.

All tests are mock-based and do not require Tenstorrent hardware.
Sysfs interactions are mocked via ``unittest.mock.patch``.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from warpt.backends.factory import get_accelerator_backend
from warpt.models.power_models import PowerDomain, PowerSource

# ---------------------------------------------------------------------------
# Helpers: build fake device dicts matching _discover_devices() output
# ---------------------------------------------------------------------------

def _make_fake_devices(count=2):
    """Build a list of fake device dicts matching ``_discover_devices()`` output.

    Device 0: Blackhole n150 at PCI 0000:61:00.0
    Device 1: Wormhole n150 at PCI 0000:81:00.0
    """
    devs = []
    if count >= 1:
        devs.append({
            "name": "tenstorrent!0",
            "tt_dir": Path(
                "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
                "/tenstorrent/tenstorrent!0"
            ),
            "hwmon_dir": Path(
                "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
                "/hwmon/hwmon0"
            ),
            "pci_bdf": "0000:61:00.0",
        })
    if count >= 2:
        devs.append({
            "name": "tenstorrent!1",
            "tt_dir": Path(
                "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
                "/tenstorrent/tenstorrent!1"
            ),
            "hwmon_dir": Path(
                "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
                "/hwmon/hwmon1"
            ),
            "pci_bdf": "0000:81:00.0",
        })
    return devs


# Mapping from sysfs file paths to their raw text content.
_SYSFS_DATA: dict[str, str] = {
    # --- Device 0 (Blackhole n150) ---
    "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
    "/tenstorrent/tenstorrent!0/tt_serial": "010001851172B05C",
    "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
    "/tenstorrent/tenstorrent!0/tt_asic_id": "1279201437FE0AFE",
    "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
    "/tenstorrent/tenstorrent!0/tt_card_type": "n150",
    "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
    "/tenstorrent/tenstorrent!0/tt_aiclk": "500",
    "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
    "/tenstorrent/tenstorrent!0/tt_arcclk": "540",
    "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
    "/tenstorrent/tenstorrent!0/tt_axiclk": "900",
    "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
    "/tenstorrent/tenstorrent!0/tt_fw_bundle_ver": "80.10.0.0",
    "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
    "/tenstorrent/tenstorrent!0/tt_arc_fw_ver": "2.27.0.0",
    "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
    "/tenstorrent/tenstorrent!0/tt_eth_fw_ver": "6.9.0",
    "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
    "/tenstorrent/tenstorrent!0/tt_m3app_fw_ver": "5.9.0.0",
    # hwmon device 0
    "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
    "/hwmon/hwmon0/name": "blackhole",
    "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
    "/hwmon/hwmon0/temp1_input": "41216",
    "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
    "/hwmon/hwmon0/power1_input": "17000000",
    "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
    "/hwmon/hwmon0/power1_max": "75000000",
    "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
    "/hwmon/hwmon0/in0_input": "735",
    "/sys/devices/pci0000:60/0000:60:03.1/0000:61:00.0"
    "/hwmon/hwmon0/curr1_input": "22000",
    # --- Device 1 (Wormhole n150) ---
    "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
    "/tenstorrent/tenstorrent!1/tt_serial": "010001851172B05D",
    "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
    "/tenstorrent/tenstorrent!1/tt_card_type": "n150",
    "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
    "/tenstorrent/tenstorrent!1/tt_aiclk": "500",
    "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
    "/tenstorrent/tenstorrent!1/tt_arcclk": "540",
    "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
    "/tenstorrent/tenstorrent!1/tt_axiclk": "900",
    "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
    "/tenstorrent/tenstorrent!1/tt_fw_bundle_ver": "80.10.0.0",
    "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
    "/tenstorrent/tenstorrent!1/tt_arc_fw_ver": "2.27.0.0",
    "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
    "/tenstorrent/tenstorrent!1/tt_eth_fw_ver": "6.9.0",
    "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
    "/tenstorrent/tenstorrent!1/tt_m3app_fw_ver": "5.9.0.0",
    # hwmon device 1
    "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
    "/hwmon/hwmon1/name": "wormhole",
    "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
    "/hwmon/hwmon1/temp1_input": "38500",
    "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
    "/hwmon/hwmon1/power1_input": "15000000",
    "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
    "/hwmon/hwmon1/power1_max": "75000000",
    "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
    "/hwmon/hwmon1/in0_input": "720",
    "/sys/devices/pci0000:80/0000:80:03.1/0000:81:00.0"
    "/hwmon/hwmon1/curr1_input": "20000",
}


def _fake_read_sysfs(path):
    """Mock replacement for ``_read_sysfs``."""
    key = str(path)
    return _SYSFS_DATA.get(key)


def _fake_read_sysfs_int(path):
    """Mock replacement for ``_read_sysfs_int``."""
    raw = _fake_read_sysfs(path)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


# ===================================================================
# AcceleratorBackend tests
# ===================================================================


class TestTenstorrentBackend:
    """Tests for ``TenstorrentBackend``."""

    def _make_backend(self, device_count=2):
        """Create a backend with mocked sysfs discovery."""
        with patch(
            "warpt.backends.tenstorrent._discover_devices",
            return_value=_make_fake_devices(device_count),
        ):
            from warpt.backends.tenstorrent import TenstorrentBackend

            backend = TenstorrentBackend()

        # Patch read functions for subsequent calls
        self._patches = [
            patch(
                "warpt.backends.tenstorrent._read_sysfs",
                side_effect=_fake_read_sysfs,
            ),
            patch(
                "warpt.backends.tenstorrent._read_sysfs_int",
                side_effect=_fake_read_sysfs_int,
            ),
        ]
        for p in self._patches:
            p.start()

        return backend

    def _cleanup_patches(self):
        for p in self._patches:
            p.stop()

    def test_is_available(self):
        """Backend reports available when devices are present."""
        backend = self._make_backend()
        try:
            assert backend.is_available() is True
        finally:
            self._cleanup_patches()

    def test_get_device_count(self):
        """Backend discovers both fake devices."""
        backend = self._make_backend()
        try:
            assert backend.get_device_count() == 2
        finally:
            self._cleanup_patches()

    def test_list_devices_count(self):
        """list_devices returns one GPUInfo per device."""
        backend = self._make_backend()
        try:
            devices = backend.list_devices()
            assert len(devices) == 2
        finally:
            self._cleanup_patches()

    def test_list_devices_model_string(self):
        """Model string includes card type and chip name."""
        backend = self._make_backend()
        try:
            devices = backend.list_devices()
            assert "Tenstorrent" in devices[0].model
            assert "n150" in devices[0].model
            assert "blackhole" in devices[0].model
        finally:
            self._cleanup_patches()

    def test_list_devices_uuid_is_serial(self):
        """UUID field is populated with the board serial number."""
        backend = self._make_backend()
        try:
            devices = backend.list_devices()
            assert devices[0].uuid == "010001851172B05C"
            assert devices[1].uuid == "010001851172B05D"
        finally:
            self._cleanup_patches()

    def test_list_devices_extra_metrics(self):
        """Extra metrics contain clocks, firmware versions, PCI BDF."""
        backend = self._make_backend()
        try:
            devices = backend.list_devices()
            extras = devices[0].extra_metrics
            assert extras is not None
            assert extras["ai_clk_mhz"] == 500
            assert extras["arc_clk_mhz"] == 540
            assert extras["axi_clk_mhz"] == 900
            assert extras["fw_bundle_version"] == "80.10.0.0"
            assert extras["card_type"] == "n150"
            assert extras["chip_name"] == "blackhole"
            assert extras["pci_bdf"] == "0000:61:00.0"
        finally:
            self._cleanup_patches()

    def test_list_devices_memory_gb_zero(self):
        """Memory is 0 since sysfs does not expose it."""
        backend = self._make_backend()
        try:
            devices = backend.list_devices()
            assert devices[0].memory_gb == 0
        finally:
            self._cleanup_patches()

    def test_get_temperature(self):
        """Temperature is correctly converted from millidegrees."""
        backend = self._make_backend()
        try:
            temp = backend.get_temperature(0)
            assert temp is not None
            assert abs(temp - 41.216) < 0.001
        finally:
            self._cleanup_patches()

    def test_get_temperature_device1(self):
        """Second device reports a different temperature."""
        backend = self._make_backend()
        try:
            temp = backend.get_temperature(1)
            assert temp is not None
            assert abs(temp - 38.5) < 0.001
        finally:
            self._cleanup_patches()

    def test_get_temperature_invalid_index(self):
        """Out-of-range index returns None."""
        backend = self._make_backend()
        try:
            assert backend.get_temperature(99) is None
            assert backend.get_temperature(-1) is None
        finally:
            self._cleanup_patches()

    def test_get_power_usage(self):
        """Power is correctly converted from microwatts."""
        backend = self._make_backend()
        try:
            power = backend.get_power_usage(0)
            assert power is not None
            assert abs(power - 17.0) < 0.001
        finally:
            self._cleanup_patches()

    def test_get_power_usage_device1(self):
        """Second device reports its own power reading."""
        backend = self._make_backend()
        try:
            power = backend.get_power_usage(1)
            assert power is not None
            assert abs(power - 15.0) < 0.001
        finally:
            self._cleanup_patches()

    def test_get_memory_usage_returns_none(self):
        """Memory usage is not available via sysfs."""
        backend = self._make_backend()
        try:
            assert backend.get_memory_usage(0) is None
        finally:
            self._cleanup_patches()

    def test_get_utilization_returns_none(self):
        """Utilization is not available via sysfs."""
        backend = self._make_backend()
        try:
            assert backend.get_utilization(0) is None
        finally:
            self._cleanup_patches()

    def test_get_throttle_reasons_empty(self):
        """Throttle reasons are not exposed by sysfs."""
        backend = self._make_backend()
        try:
            assert backend.get_throttle_reasons(0) == []
        finally:
            self._cleanup_patches()

    def test_get_pytorch_device_string(self):
        """Device string follows 'tt:N' convention."""
        backend = self._make_backend()
        try:
            assert backend.get_pytorch_device_string(0) == "tt:0"
            assert backend.get_pytorch_device_string(1) == "tt:1"
        finally:
            self._cleanup_patches()

    def test_get_driver_version(self):
        """Driver version returns firmware bundle version."""
        backend = self._make_backend()
        try:
            assert backend.get_driver_version() == "80.10.0.0"
        finally:
            self._cleanup_patches()

    def test_shutdown_clears_devices(self):
        """Shutdown clears internal device cache."""
        backend = self._make_backend()
        try:
            assert backend.get_device_count() == 2
            backend.shutdown()
            assert backend.get_device_count() == 0
        finally:
            self._cleanup_patches()

    def test_no_devices_detected(self):
        """Backend reports unavailable when no devices are found."""
        with patch(
            "warpt.backends.tenstorrent._discover_devices",
            return_value=[],
        ):
            from warpt.backends.tenstorrent import TenstorrentBackend

            backend = TenstorrentBackend()
            assert backend.is_available() is False
            assert backend.get_device_count() == 0
            assert backend.list_devices() == []

    def test_build_model_string_without_chip(self):
        """Model string works without a chip name."""
        from warpt.backends.tenstorrent import TenstorrentBackend

        result = TenstorrentBackend._build_model_string("n300", None)
        assert result == "Tenstorrent n300"

    def test_build_model_string_with_chip(self):
        """Model string includes chip name in parentheses."""
        from warpt.backends.tenstorrent import TenstorrentBackend

        result = TenstorrentBackend._build_model_string("n150", "wormhole")
        assert result == "Tenstorrent n150 (wormhole)"


# ===================================================================
# PowerBackend tests
# ===================================================================


class TestTenstorrentPowerBackend:
    """Tests for ``TenstorrentPowerBackend``."""

    def _make_power_backend(self):
        """Create a power backend with mocked sysfs discovery."""
        with patch(
            "warpt.backends.tenstorrent._discover_devices",
            return_value=_make_fake_devices(2),
        ):
            from warpt.backends.power.tenstorrent_power import (
                TenstorrentPowerBackend,
            )

            backend = TenstorrentPowerBackend()
            backend.initialize()

        # Patch the read functions imported by the power module
        self._patches = [
            patch(
                "warpt.backends.power.tenstorrent_power._read_sysfs",
                side_effect=_fake_read_sysfs,
            ),
            patch(
                "warpt.backends.power.tenstorrent_power._read_sysfs_int",
                side_effect=_fake_read_sysfs_int,
            ),
        ]
        for p in self._patches:
            p.start()

        return backend

    def _cleanup_patches(self):
        for p in self._patches:
            p.stop()

    def test_is_available(self):
        """Power backend reports available with devices present."""
        with patch(
            "warpt.backends.tenstorrent._discover_devices",
            return_value=_make_fake_devices(2),
        ):
            from warpt.backends.power.tenstorrent_power import (
                TenstorrentPowerBackend,
            )

            backend = TenstorrentPowerBackend()
            assert backend.is_available() is True

    def test_get_source(self):
        """Power source is ESTIMATED (sysfs sensor readings)."""
        from warpt.backends.power.tenstorrent_power import (
            TenstorrentPowerBackend,
        )

        backend = TenstorrentPowerBackend()
        assert backend.get_source() == PowerSource.ESTIMATED

    def test_initialize(self):
        """Initialization succeeds and populates devices."""
        with patch(
            "warpt.backends.tenstorrent._discover_devices",
            return_value=_make_fake_devices(2),
        ):
            from warpt.backends.power.tenstorrent_power import (
                TenstorrentPowerBackend,
            )

            backend = TenstorrentPowerBackend()
            result = backend.initialize()
            assert result is True
            assert len(backend._devices) == 2

    def test_get_power_readings(self):
        """Power readings return one DomainPower per device."""
        backend = self._make_power_backend()
        try:
            readings = backend.get_power_readings()
            assert len(readings) == 2
            assert readings[0].domain == PowerDomain.GPU
            assert abs(readings[0].power_watts - 17.0) < 0.001
            assert abs(readings[1].power_watts - 15.0) < 0.001
        finally:
            self._cleanup_patches()

    def test_get_power_readings_source(self):
        """Power readings use ESTIMATED source."""
        backend = self._make_power_backend()
        try:
            readings = backend.get_power_readings()
            for r in readings:
                assert r.source == PowerSource.ESTIMATED
        finally:
            self._cleanup_patches()

    def test_get_power_readings_metadata(self):
        """Power reading metadata includes device name and PCI BDF."""
        backend = self._make_power_backend()
        try:
            readings = backend.get_power_readings()
            meta = readings[0].metadata
            assert "gpu_index" in meta
            assert meta["gpu_index"] == 0
            assert "pci_bdf" in meta
            assert meta["pci_bdf"] == "0000:61:00.0"
        finally:
            self._cleanup_patches()

    def test_get_gpu_power_info(self):
        """GPU power info includes power, power limit, and temperature."""
        backend = self._make_power_backend()
        try:
            gpus = backend.get_gpu_power_info()
            assert len(gpus) == 2

            gpu0 = gpus[0]
            assert abs(gpu0.power_watts - 17.0) < 0.001
            assert gpu0.power_limit_watts is not None
            assert abs(gpu0.power_limit_watts - 75.0) < 0.001
            assert gpu0.temperature_celsius is not None
            assert abs(gpu0.temperature_celsius - 41.216) < 0.001
        finally:
            self._cleanup_patches()

    def test_get_gpu_power_info_metadata(self):
        """GPU power info metadata includes voltage and current."""
        backend = self._make_power_backend()
        try:
            gpus = backend.get_gpu_power_info()
            meta = gpus[0].metadata
            assert abs(meta["voltage_v"] - 0.735) < 0.001
            assert abs(meta["current_a"] - 22.0) < 0.001
        finally:
            self._cleanup_patches()

    def test_get_gpu_power_info_name(self):
        """GPU power info name includes Tenstorrent and card type."""
        backend = self._make_power_backend()
        try:
            gpus = backend.get_gpu_power_info()
            assert "Tenstorrent" in gpus[0].name
            assert "n150" in gpus[0].name
        finally:
            self._cleanup_patches()

    def test_get_total_gpu_power(self):
        """Total GPU power sums across all devices."""
        backend = self._make_power_backend()
        try:
            total = backend.get_total_gpu_power()
            assert abs(total - 32.0) < 0.001  # 17W + 15W
        finally:
            self._cleanup_patches()

    def test_cleanup(self):
        """Cleanup resets initialization state."""
        backend = self._make_power_backend()
        self._cleanup_patches()
        assert backend._initialized is True
        backend.cleanup()
        assert backend._initialized is False
        assert backend._devices == []

    def test_not_initialized_auto_initializes(self):
        """Methods auto-initialize via _discover_devices if needed."""
        with patch(
            "warpt.backends.tenstorrent._discover_devices",
            return_value=_make_fake_devices(2),
        ), patch(
            "warpt.backends.power.tenstorrent_power._read_sysfs",
            side_effect=_fake_read_sysfs,
        ), patch(
            "warpt.backends.power.tenstorrent_power._read_sysfs_int",
            side_effect=_fake_read_sysfs_int,
        ):
            from warpt.backends.power.tenstorrent_power import (
                TenstorrentPowerBackend,
            )

            backend = TenstorrentPowerBackend()
            # Should auto-initialize on first call
            readings = backend.get_power_readings()
            assert len(readings) == 2


# ===================================================================
# Factory fallthrough tests
# ===================================================================


class TestFactoryTenstorrentFallthrough:
    """Tests for Tenstorrent in the backend factory priority chain."""

    def test_factory_returns_tenstorrent_when_nvidia_fails(self):
        """If NVIDIA is unavailable, factory falls through to Tenstorrent."""
        mock_nvidia = MagicMock()
        mock_nvidia.NvidiaBackend.side_effect = RuntimeError("No NVML")

        mock_tt = MagicMock()
        mock_tt_instance = MagicMock()
        mock_tt_instance.is_available.return_value = True
        mock_tt.TenstorrentBackend.return_value = mock_tt_instance

        with patch.dict(
            sys.modules,
            {
                "warpt.backends.nvidia": mock_nvidia,
                "warpt.backends.tenstorrent": mock_tt,
            },
        ):
            backend = get_accelerator_backend()
            assert backend == mock_tt_instance
            mock_tt_instance.is_available.assert_called_once()

    def test_factory_prefers_nvidia_over_tenstorrent(self):
        """NVIDIA has higher priority than Tenstorrent."""
        mock_nvidia = MagicMock()
        mock_nvidia_instance = MagicMock()
        mock_nvidia_instance.is_available.return_value = True
        mock_nvidia.NvidiaBackend.return_value = mock_nvidia_instance

        mock_tt = MagicMock()
        mock_tt_instance = MagicMock()
        mock_tt_instance.is_available.return_value = True
        mock_tt.TenstorrentBackend.return_value = mock_tt_instance

        with patch.dict(
            sys.modules,
            {
                "warpt.backends.nvidia": mock_nvidia,
                "warpt.backends.tenstorrent": mock_tt,
            },
        ):
            backend = get_accelerator_backend()
            assert backend == mock_nvidia_instance
            mock_tt_instance.is_available.assert_not_called()

    def test_factory_tenstorrent_unavailable_falls_through(self):
        """If Tenstorrent reports unavailable, factory tries AMD next."""
        mock_nvidia = MagicMock()
        mock_nvidia.NvidiaBackend.side_effect = RuntimeError("No NVML")

        mock_tt = MagicMock()
        mock_tt_instance = MagicMock()
        mock_tt_instance.is_available.return_value = False
        mock_tt.TenstorrentBackend.return_value = mock_tt_instance

        mock_amd = MagicMock()
        mock_amd_instance = MagicMock()
        mock_amd_instance.is_available.return_value = True
        mock_amd.AMDBackend.return_value = mock_amd_instance

        with patch.dict(
            sys.modules,
            {
                "warpt.backends.nvidia": mock_nvidia,
                "warpt.backends.tenstorrent": mock_tt,
                "warpt.backends.amd": mock_amd,
            },
        ):
            backend = get_accelerator_backend()
            assert backend == mock_amd_instance

    def test_factory_all_unavailable_raises(self):
        """RuntimeError is raised when no backends are available."""
        mock_nvidia = MagicMock()
        mock_nvidia.NvidiaBackend.side_effect = RuntimeError("No NVML")

        mock_tt = MagicMock()
        mock_tt.TenstorrentBackend.return_value.is_available.return_value = (
            False
        )

        mock_amd = MagicMock()
        mock_amd.AMDBackend.return_value.is_available.return_value = False

        mock_intel = MagicMock()
        mock_intel.IntelBackend.return_value.is_available.return_value = False

        with patch.dict(
            sys.modules,
            {
                "warpt.backends.nvidia": mock_nvidia,
                "warpt.backends.tenstorrent": mock_tt,
                "warpt.backends.amd": mock_amd,
                "warpt.backends.intel": mock_intel,
            },
        ):
            with pytest.raises(RuntimeError, match="No GPUs detected"):
                get_accelerator_backend()


# ===================================================================
# Edge case tests
# ===================================================================


class TestTenstorrentEdgeCases:
    """Edge case and error handling tests."""

    def test_driver_version_no_devices(self):
        """Driver version returns None when no devices exist."""
        with patch(
            "warpt.backends.tenstorrent._discover_devices",
            return_value=[],
        ):
            from warpt.backends.tenstorrent import TenstorrentBackend

            backend = TenstorrentBackend()
            assert backend.get_driver_version() is None

    def test_power_usage_invalid_index(self):
        """Power usage returns None for out-of-range index."""
        with patch(
            "warpt.backends.tenstorrent._discover_devices",
            return_value=_make_fake_devices(1),
        ):
            from warpt.backends.tenstorrent import TenstorrentBackend

            backend = TenstorrentBackend()
            assert backend.get_power_usage(99) is None

    def test_temperature_invalid_index(self):
        """Temperature returns None for out-of-range index."""
        with patch(
            "warpt.backends.tenstorrent._discover_devices",
            return_value=_make_fake_devices(1),
        ):
            from warpt.backends.tenstorrent import TenstorrentBackend

            backend = TenstorrentBackend()
            assert backend.get_temperature(-1) is None

    def test_list_devices_indices_are_sequential(self):
        """Device indices are 0-based and sequential."""
        with patch(
            "warpt.backends.tenstorrent._discover_devices",
            return_value=_make_fake_devices(2),
        ), patch(
            "warpt.backends.tenstorrent._read_sysfs",
            side_effect=_fake_read_sysfs,
        ), patch(
            "warpt.backends.tenstorrent._read_sysfs_int",
            side_effect=_fake_read_sysfs_int,
        ):
            from warpt.backends.tenstorrent import TenstorrentBackend

            backend = TenstorrentBackend()
            devices = backend.list_devices()
            for i, dev in enumerate(devices):
                assert dev.index == i

    def test_second_device_chip_name(self):
        """Second device correctly identifies as wormhole."""
        with patch(
            "warpt.backends.tenstorrent._discover_devices",
            return_value=_make_fake_devices(2),
        ), patch(
            "warpt.backends.tenstorrent._read_sysfs",
            side_effect=_fake_read_sysfs,
        ), patch(
            "warpt.backends.tenstorrent._read_sysfs_int",
            side_effect=_fake_read_sysfs_int,
        ):
            from warpt.backends.tenstorrent import TenstorrentBackend

            backend = TenstorrentBackend()
            devices = backend.list_devices()
            assert devices[1].extra_metrics is not None
            assert devices[1].extra_metrics["chip_name"] == "wormhole"

    def test_device_without_hwmon(self):
        """Device without hwmon dir returns None for telemetry."""
        dev_no_hwmon = [{
            "name": "tenstorrent!0",
            "tt_dir": Path("/sys/devices/fake/tenstorrent/tenstorrent!0"),
            "hwmon_dir": None,
            "pci_bdf": "0000:99:00.0",
        }]
        with patch(
            "warpt.backends.tenstorrent._discover_devices",
            return_value=dev_no_hwmon,
        ):
            from warpt.backends.tenstorrent import TenstorrentBackend

            backend = TenstorrentBackend()
            assert backend.get_temperature(0) is None
            assert backend.get_power_usage(0) is None

    def test_power_backend_no_devices_returns_empty(self):
        """Power backend returns empty results when no devices exist."""
        with patch(
            "warpt.backends.tenstorrent._discover_devices",
            return_value=[],
        ):
            from warpt.backends.power.tenstorrent_power import (
                TenstorrentPowerBackend,
            )

            backend = TenstorrentPowerBackend()
            backend.initialize()
            assert backend.get_total_gpu_power() == 0.0
            assert backend.get_power_readings() == []
            assert backend.get_gpu_power_info() == []

    def test_single_device_backend(self):
        """Backend works correctly with a single device."""
        with patch(
            "warpt.backends.tenstorrent._discover_devices",
            return_value=_make_fake_devices(1),
        ), patch(
            "warpt.backends.tenstorrent._read_sysfs",
            side_effect=_fake_read_sysfs,
        ), patch(
            "warpt.backends.tenstorrent._read_sysfs_int",
            side_effect=_fake_read_sysfs_int,
        ):
            from warpt.backends.tenstorrent import TenstorrentBackend

            backend = TenstorrentBackend()
            assert backend.get_device_count() == 1
            devices = backend.list_devices()
            assert len(devices) == 1
            assert devices[0].uuid == "010001851172B05C"
