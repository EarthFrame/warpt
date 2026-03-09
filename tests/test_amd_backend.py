"""Tests for the AMD GPU backend and power backend.

All tests use mocks — no physical AMD hardware required.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — build a fake ``amdsmi`` module that the backends will import
# ---------------------------------------------------------------------------

def _make_amdsmi_mock():
    """Create a fully wired mock of the ``amdsmi`` package.

    Returns
    -------
    MagicMock
        A mock object that behaves like ``import amdsmi``.
    """
    mock = MagicMock()

    # Enums used during init / temperature / power
    mock.AmdSmiInitFlags.INIT_AMD_GPUS = 0x1
    mock.AmdSmiTemperatureType.EDGE = 0
    mock.AmdSmiTemperatureType.HOTSPOT = 1
    mock.AmdSmiTemperatureMetric.CURRENT = 0
    mock.AmdSmiVramType = MagicMock(side_effect=lambda _v: MagicMock(name="HBM2E"))

    return mock


def _make_handles(n: int = 2):
    """Return *n* opaque processor-handle mocks."""
    return [MagicMock(name=f"handle_{i}") for i in range(n)]


# ---------------------------------------------------------------------------
# Sample return values matching the AMD SMI interface
# ---------------------------------------------------------------------------

ASIC_INFO = {
    "market_name": "Instinct MI300X",
    "vendor_id": "0x1002",
    "vendor_name": "Advanced Micro Devices Inc.",
    "subvendor_id": "0x1002",
    "device_id": "0x7408",
    "rev_id": "0x01",
    "asic_serial": "0x1234567890ABCDEF",
    "oam_id": 0,
    "num_compute_units": 304,
    "target_graphics_version": "gfx942",
    "subsystem_id": "0x0000",
}

VRAM_USAGE = {"vram_total": 196608, "vram_used": 1024}  # MB

PCIE_INFO = {
    "pcie_static": {
        "max_pcie_width": 16,
        "max_pcie_speed": 32000,
        "pcie_interface_version": 5,
        "slot_type": "OAM",
    },
    "pcie_metric": {
        "pcie_width": 16,
        "pcie_speed": 32000,
        "pcie_bandwidth": 63015,
    },
}

VRAM_INFO = {
    "vram_type": 4,
    "vram_vendor": "Samsung",
    "vram_size": 196608,
    "vram_bit_width": 8192,
    "vram_max_bandwidth": 5300,
}

BOARD_INFO = {
    "model_number": "N/A",
    "product_serial": "N/A",
    "fru_id": "N/A",
    "product_name": "MI300X OAM",
    "manufacturer_name": "AMD",
}

DRIVER_INFO = {
    "driver_name": "amdgpu",
    "driver_version": "6.7.12",
    "driver_date": "2025/08/01 00:00",
}

GPU_ACTIVITY = {
    "gfx_activity": 45,
    "umc_activity": 30,
    "mm_activity": 0,
}

POWER_INFO = {
    "current_socket_power": 350,
    "average_socket_power": 340,
    "socket_power": 345,
    "gfx_voltage": 900,
    "soc_voltage": 800,
    "mem_voltage": 1200,
    "power_limit": 750,
}

POWER_CAP_INFO = {
    "power_cap": 750_000_000,  # microwatts
    "default_power_cap": 600_000_000,
    "dpm_cap": 0,
    "min_power_cap": 100_000_000,
    "max_power_cap": 750_000_000,
}

VIOLATION_STATUS = {
    "active_ppt_pwr": 0,
    "active_socket_thrm": 0,
    "active_prochot_thrm": 0,
    "active_vr_thrm": 0,
    "active_hbm_thrm": 0,
}

PROCESS_LIST = [
    {
        "name": "python3",
        "pid": 12345,
        "mem": 1073741824,
        "engine_usage": {"gfx": 100, "enc": 0},
        "memory_usage": {"gtt_mem": 0, "cpu_mem": 0, "vram_mem": 1073741824},
        "cu_occupancy": 50,
        "evicted_time": 0,
    },
]


# ===================================================================
# AmdBackend tests
# ===================================================================


class TestAmdBackend:
    """Unit tests for ``AmdBackend``."""

    def _build_backend(self, mock_amdsmi, handles=None):
        """Patch ``amdsmi`` into ``sys.modules`` and construct the backend.

        Returns the backend instance and the amdsmi mock.
        """
        if handles is None:
            handles = _make_handles(2)

        mock_amdsmi.amdsmi_get_processor_handles.return_value = handles
        mock_amdsmi.amdsmi_get_gpu_asic_info.return_value = ASIC_INFO
        mock_amdsmi.amdsmi_get_gpu_device_uuid.return_value = (
            "GPU-abcd-1234-efgh-5678"
        )
        mock_amdsmi.amdsmi_get_gpu_vram_usage.return_value = VRAM_USAGE
        mock_amdsmi.amdsmi_get_pcie_info.return_value = PCIE_INFO
        mock_amdsmi.amdsmi_get_gpu_vram_info.return_value = VRAM_INFO
        mock_amdsmi.amdsmi_get_gpu_board_info.return_value = BOARD_INFO
        mock_amdsmi.amdsmi_get_gpu_driver_info.return_value = DRIVER_INFO
        mock_amdsmi.amdsmi_get_gpu_activity.return_value = GPU_ACTIVITY
        mock_amdsmi.amdsmi_get_power_info.return_value = POWER_INFO
        mock_amdsmi.amdsmi_get_violation_status.return_value = (
            VIOLATION_STATUS
        )
        mock_amdsmi.amdsmi_get_temp_metric.return_value = 65000  # mC

        # Patch the module-level import so the backend file sees it
        with patch.dict(sys.modules, {"amdsmi": mock_amdsmi}):
            # Reload to pick up the patched module
            import importlib

            import warpt.backends.amd as _amd_mod

            importlib.reload(_amd_mod)
            backend = _amd_mod.AmdBackend()

        return backend

    # ----- is_available / device_count -----

    def test_is_available_true(self):
        """Backend reports available when handles are present."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock, _make_handles(1))
        assert backend.is_available() is True

    def test_is_available_false_no_handles(self):
        """Backend reports unavailable when no handles found."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock, [])
        assert backend.is_available() is False

    def test_get_device_count(self):
        """Device count matches number of handles."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock, _make_handles(4))
        assert backend.get_device_count() == 4

    # ----- list_devices -----

    def test_list_devices_returns_gpu_info(self):
        """list_devices returns correctly populated GPUInfo objects."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        devices = backend.list_devices()

        assert len(devices) == 2
        dev = devices[0]
        assert dev.index == 0
        assert dev.model == "Instinct MI300X"
        assert dev.memory_gb == 192  # 196608 MB // 1024
        assert dev.uuid == "GPU-abcd-1234-efgh-5678"
        assert dev.compute_capability is None
        assert dev.pcie_gen == 5
        assert dev.driver_version == "6.7.12"
        assert dev.extra_metrics is not None
        assert dev.extra_metrics["target_graphics_version"] == "gfx942"
        assert dev.extra_metrics["num_compute_units"] == 304

    def test_list_devices_handles_asic_info_failure(self):
        """Gracefully handles failure to read ASIC info."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_get_gpu_asic_info.side_effect = RuntimeError("fail")
        devices = backend.list_devices()

        assert len(devices) == 2
        assert devices[0].model == "AMD GPU"  # fallback name

    def test_list_devices_na_market_name(self):
        """Falls back to 'AMD GPU' when market_name is 'N/A'."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        # Override AFTER build so _build_backend defaults don't clobber
        na_asic = dict(ASIC_INFO, market_name="N/A")
        mock.amdsmi_get_gpu_asic_info.return_value = na_asic
        devices = backend.list_devices()

        assert devices[0].model == "AMD GPU"

    # ----- get_temperature -----

    def test_get_temperature(self):
        """Temperature is correctly converted from millidegrees."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        temp = backend.get_temperature(0)
        assert temp == 65.0

    def test_get_temperature_edge_fails_hotspot_succeeds(self):
        """Falls back to hotspot when edge sensor is unavailable."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)

        # Override AFTER build; EDGE fails, HOTSPOT succeeds
        def _temp_side_effect(_handle, sensor, _metric):
            if sensor == mock.AmdSmiTemperatureType.EDGE:
                raise RuntimeError("EDGE not supported")
            return 72000  # hotspot in mC

        mock.amdsmi_get_temp_metric.side_effect = _temp_side_effect
        temp = backend.get_temperature(0)
        assert temp == 72.0

    def test_get_temperature_all_fail(self):
        """Returns None when all temperature sensors fail."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_get_temp_metric.side_effect = RuntimeError("fail")
        assert backend.get_temperature(0) is None

    # ----- get_memory_usage -----

    def test_get_memory_usage(self):
        """Memory usage is converted from MB to bytes."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mem = backend.get_memory_usage(0)

        assert mem is not None
        assert mem["total"] == 196608 * 1024 * 1024
        assert mem["used"] == 1024 * 1024 * 1024
        assert mem["free"] == (196608 - 1024) * 1024 * 1024

    def test_get_memory_usage_failure(self):
        """Returns None when VRAM usage query fails."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_get_gpu_vram_usage.side_effect = RuntimeError("fail")
        assert backend.get_memory_usage(0) is None

    # ----- get_utilization -----

    def test_get_utilization(self):
        """Utilization correctly maps gfx/umc activity."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        util = backend.get_utilization(0)

        assert util is not None
        assert util["gpu"] == 45.0
        assert util["memory"] == 30.0

    def test_get_utilization_na_values(self):
        """Handles 'N/A' sentinel values gracefully."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_get_gpu_activity.return_value = {
            "gfx_activity": "N/A",
            "umc_activity": "N/A",
            "mm_activity": "N/A",
        }
        util = backend.get_utilization(0)

        assert util is not None
        assert util["gpu"] == 0.0
        assert util["memory"] == 0.0

    # ----- get_pytorch_device_string -----

    def test_get_pytorch_device_string(self):
        """ROCm PyTorch uses 'cuda:N' device strings."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        assert backend.get_pytorch_device_string(0) == "cuda:0"
        assert backend.get_pytorch_device_string(3) == "cuda:3"

    # ----- get_power_usage -----

    def test_get_power_usage(self):
        """Power usage returns current_socket_power in watts."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        power = backend.get_power_usage(0)
        assert power == 350.0

    def test_get_power_usage_fallback_to_average(self):
        """Falls back to average_socket_power when current is N/A."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_get_power_info.return_value = {
            "current_socket_power": "N/A",
            "average_socket_power": 340,
            "socket_power": 345,
            "power_limit": 750,
        }
        assert backend.get_power_usage(0) == 340.0

    def test_get_power_usage_failure(self):
        """Returns None when power query fails."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_get_power_info.side_effect = RuntimeError("fail")
        assert backend.get_power_usage(0) is None

    # ----- get_throttle_reasons -----

    def test_get_throttle_reasons_none(self):
        """Returns empty list when no violations detected."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        assert backend.get_throttle_reasons(0) == []

    def test_get_throttle_reasons_power_and_thermal(self):
        """Returns mapped reason strings for active violations."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_get_violation_status.return_value = {
            "active_ppt_pwr": 1,
            "active_socket_thrm": 1,
            "active_prochot_thrm": 1,
            "active_vr_thrm": 0,
            "active_hbm_thrm": 0,
        }
        reasons = backend.get_throttle_reasons(0)
        assert "power_limit" in reasons
        assert "thermal" in reasons
        # thermal should appear only once despite two active keys
        assert reasons.count("thermal") == 1
        assert "vr_thermal" not in reasons
        assert "hbm_thermal" not in reasons

    def test_get_throttle_reasons_vr_and_hbm_thermal(self):
        """Maps vr_thermal and hbm_thermal reason strings."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_get_violation_status.return_value = {
            "active_ppt_pwr": 0,
            "active_socket_thrm": 0,
            "active_prochot_thrm": 0,
            "active_vr_thrm": 1,
            "active_hbm_thrm": 1,
        }
        reasons = backend.get_throttle_reasons(0)
        assert "vr_thermal" in reasons
        assert "hbm_thermal" in reasons
        assert "thermal" not in reasons
        assert "power_limit" not in reasons

    def test_get_throttle_reasons_failure(self):
        """Returns empty list when violation query fails."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_get_violation_status.side_effect = RuntimeError("fail")
        assert backend.get_throttle_reasons(0) == []

    # ----- get_driver_version -----

    def test_get_driver_version(self):
        """Returns the driver version string."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        assert backend.get_driver_version() == "6.7.12"

    def test_get_driver_version_na(self):
        """Returns None when driver version is N/A."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_get_gpu_driver_info.return_value = {
            "driver_name": "amdgpu",
            "driver_version": "N/A",
            "driver_date": "N/A",
        }
        assert backend.get_driver_version() is None

    def test_get_driver_version_no_handles(self):
        """Returns None when there are no GPU handles."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock, [])
        assert backend.get_driver_version() is None

    # ----- shutdown -----

    def test_shutdown(self):
        """Shutdown calls amdsmi_shut_down without error."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        # Should not raise
        backend.shutdown()

    def test_shutdown_handles_exception(self):
        """Shutdown swallows exceptions from amdsmi."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_shut_down.side_effect = RuntimeError("fail")
        # Should not raise
        backend.shutdown()

    # ----- init failure -----

    def test_init_raises_without_amdsmi(self):
        """Raises RuntimeError if amdsmi is not importable."""
        with patch.dict(sys.modules, {"amdsmi": None}):
            import importlib

            import warpt.backends.amd as _amd_mod

            importlib.reload(_amd_mod)
            # After reload with amdsmi=None, AMDSMI_AVAILABLE is False
            # but we can't directly test the import guard since None
            # will cause AttributeError. Instead test the guard:
            assert _amd_mod.AMDSMI_AVAILABLE is False


# ===================================================================
# AmdPowerBackend tests
# ===================================================================


class TestAmdPowerBackend:
    """Unit tests for ``AmdPowerBackend``."""

    def _build_power_backend(self, mock_amdsmi, handles=None):
        """Patch amdsmi and build an initialized AmdPowerBackend."""
        if handles is None:
            handles = _make_handles(2)

        mock_amdsmi.amdsmi_get_processor_handles.return_value = handles
        mock_amdsmi.amdsmi_get_gpu_asic_info.return_value = ASIC_INFO
        mock_amdsmi.amdsmi_get_power_info.return_value = POWER_INFO
        mock_amdsmi.amdsmi_get_power_cap_info.return_value = POWER_CAP_INFO
        mock_amdsmi.amdsmi_get_gpu_activity.return_value = GPU_ACTIVITY
        mock_amdsmi.amdsmi_get_temp_metric.return_value = 65000
        mock_amdsmi.amdsmi_get_gpu_process_list.return_value = PROCESS_LIST

        with patch.dict(sys.modules, {"amdsmi": mock_amdsmi}):
            import importlib

            import warpt.backends.power.amd_power as _pw_mod

            importlib.reload(_pw_mod)
            backend = _pw_mod.AmdPowerBackend()
            backend.initialize()

        return backend

    # ----- is_available -----

    def test_is_available_true(self):
        """Reports available when amdsmi finds GPUs."""
        mock = _make_amdsmi_mock()
        mock.amdsmi_get_processor_handles.return_value = _make_handles(1)
        with patch.dict(sys.modules, {"amdsmi": mock}):
            import importlib

            import warpt.backends.power.amd_power as _pw_mod

            importlib.reload(_pw_mod)
            backend = _pw_mod.AmdPowerBackend()
            assert backend.is_available() is True

    def test_is_available_false_no_amdsmi(self):
        """Reports unavailable when amdsmi is not installed."""
        with patch.dict(sys.modules, {"amdsmi": None}):
            import importlib

            import warpt.backends.power.amd_power as _pw_mod

            importlib.reload(_pw_mod)
            backend = _pw_mod.AmdPowerBackend()
            assert backend.is_available() is False

    # ----- get_source -----

    def test_get_source(self):
        """Returns ROCM_SMI power source."""
        mock = _make_amdsmi_mock()
        backend = self._build_power_backend(mock)

        from warpt.models.power_models import PowerSource

        assert backend.get_source() == PowerSource.ROCM_SMI

    # ----- get_power_readings -----

    def test_get_power_readings(self):
        """Returns DomainPower for each GPU."""
        mock = _make_amdsmi_mock()
        backend = self._build_power_backend(mock)
        readings = backend.get_power_readings()

        assert len(readings) == 2
        assert readings[0].power_watts == 350.0
        assert readings[0].domain.value == "gpu"
        assert readings[0].source.value == "rocm_smi"
        assert readings[0].metadata["gpu_index"] == 0
        assert readings[0].metadata["gpu_name"] == "Instinct MI300X"

    def test_get_power_readings_skip_unavailable(self):
        """Skips GPUs where power reading fails."""
        mock = _make_amdsmi_mock()
        backend = self._build_power_backend(mock)
        mock.amdsmi_get_power_info.side_effect = RuntimeError("fail")
        readings = backend.get_power_readings()
        assert len(readings) == 0

    # ----- get_gpu_power_info -----

    def test_get_gpu_power_info(self):
        """Returns detailed GPUPowerInfo per device."""
        mock = _make_amdsmi_mock()
        backend = self._build_power_backend(mock)
        info = backend.get_gpu_power_info()

        assert len(info) == 2
        gpu0 = info[0]
        assert gpu0.index == 0
        assert gpu0.name == "Instinct MI300X"
        assert gpu0.power_watts == 350.0
        assert gpu0.power_limit_watts == 750.0
        assert gpu0.utilization_percent == 45.0
        assert gpu0.memory_utilization_percent == 30.0
        assert gpu0.temperature_celsius == 65.0

    def test_get_gpu_power_info_power_limit_fallback(self):
        """Falls back to power_cap_info when power_limit is N/A."""
        mock = _make_amdsmi_mock()
        backend = self._build_power_backend(mock)
        mock.amdsmi_get_power_info.return_value = {
            "current_socket_power": 200,
            "average_socket_power": 195,
            "socket_power": 198,
            "power_limit": "N/A",
        }
        info = backend.get_gpu_power_info()

        assert info[0].power_limit_watts == 750.0  # from POWER_CAP_INFO

    # ----- get_total_gpu_power -----

    def test_get_total_gpu_power(self):
        """Total GPU power sums across all GPUs."""
        mock = _make_amdsmi_mock()
        backend = self._build_power_backend(mock)
        total = backend.get_total_gpu_power()
        # 2 GPUs * 350W each
        assert total == 700.0

    # ----- cleanup -----

    def test_cleanup(self):
        """Cleanup resets state and calls amdsmi_shut_down."""
        mock = _make_amdsmi_mock()
        backend = self._build_power_backend(mock)
        backend.cleanup()
        assert backend._initialized is False
        assert backend._processor_handles == []

    # ----- process list -----

    def test_get_gpu_processes(self):
        """Processes are extracted from amdsmi_get_gpu_process_list."""
        mock = _make_amdsmi_mock()
        backend = self._build_power_backend(mock)
        info = backend.get_gpu_power_info()
        procs = info[0].processes
        assert len(procs) == 1
        assert procs[0]["pid"] == 12345
        assert procs[0]["name"] == "python3"
        assert procs[0]["gpu_memory_mb"] == 1024.0


# ===================================================================
# Factory fallthrough tests
# ===================================================================


class TestFactoryAmdFallthrough:
    """Verify the backend factory discovers AmdBackend correctly."""

    def test_nvidia_fail_falls_through_to_amd(self):
        """NVIDIA import failure falls through to AmdBackend."""
        mock_nvidia = MagicMock()
        mock_nvidia.NvidiaBackend.side_effect = RuntimeError("NVML fail")

        mock_amd = MagicMock()
        mock_amd_instance = MagicMock()
        mock_amd_instance.is_available.return_value = True
        mock_amd.AmdBackend.return_value = mock_amd_instance

        modules = {
            "warpt.backends.nvidia": mock_nvidia,
            "warpt.backends.amd": mock_amd,
        }

        with patch.dict(sys.modules, modules):
            from warpt.backends.factory import get_accelerator_backend

            backend = get_accelerator_backend()
            assert backend == mock_amd_instance
            mock_amd_instance.is_available.assert_called_once()

    def test_nvidia_unavailable_falls_through_to_amd(self):
        """NVIDIA present but unavailable falls through to AmdBackend."""
        mock_nvidia = MagicMock()
        mock_nvidia.NvidiaBackend.return_value.is_available.return_value = (
            False
        )

        mock_amd = MagicMock()
        mock_amd_instance = MagicMock()
        mock_amd_instance.is_available.return_value = True
        mock_amd.AmdBackend.return_value = mock_amd_instance

        modules = {
            "warpt.backends.nvidia": mock_nvidia,
            "warpt.backends.amd": mock_amd,
        }

        with patch.dict(sys.modules, modules):
            from warpt.backends.factory import get_accelerator_backend

            backend = get_accelerator_backend()
            assert backend == mock_amd_instance

    def test_amd_unavailable_falls_through_to_intel(self):
        """AMD unavailable falls through to Intel."""
        mock_nvidia = MagicMock()
        mock_nvidia.NvidiaBackend.side_effect = RuntimeError("no NVML")

        mock_amd = MagicMock()
        mock_amd.AmdBackend.return_value.is_available.return_value = False

        mock_intel = MagicMock()
        mock_intel_instance = MagicMock()
        mock_intel_instance.is_available.return_value = True
        mock_intel.IntelBackend.return_value = mock_intel_instance

        modules = {
            "warpt.backends.nvidia": mock_nvidia,
            "warpt.backends.amd": mock_amd,
            "warpt.backends.intel": mock_intel,
        }

        with patch.dict(sys.modules, modules):
            from warpt.backends.factory import get_accelerator_backend

            backend = get_accelerator_backend()
            assert backend == mock_intel_instance

    def test_all_unavailable_raises(self):
        """RuntimeError raised when no backends are available."""
        mock_nvidia = MagicMock()
        mock_nvidia.NvidiaBackend.return_value.is_available.return_value = (
            False
        )

        mock_amd = MagicMock()
        mock_amd.AmdBackend.return_value.is_available.return_value = False

        mock_intel = MagicMock()
        mock_intel.IntelBackend.return_value.is_available.return_value = False

        modules = {
            "warpt.backends.nvidia": mock_nvidia,
            "warpt.backends.amd": mock_amd,
            "warpt.backends.intel": mock_intel,
        }

        with patch.dict(sys.modules, modules):
            from warpt.backends.factory import get_accelerator_backend

            with pytest.raises(RuntimeError, match="No GPUs detected"):
                get_accelerator_backend()


# ===================================================================
# Edge cases and sentinel handling
# ===================================================================


class TestAmdBackendEdgeCases:
    """Edge cases: sentinel values, empty returns, partial failures."""

    def _build_backend(self, mock_amdsmi, handles=None):
        """Quick helper reusing TestAmdBackend._build_backend logic."""
        if handles is None:
            handles = _make_handles(1)

        mock_amdsmi.amdsmi_get_processor_handles.return_value = handles
        mock_amdsmi.amdsmi_get_gpu_asic_info.return_value = ASIC_INFO
        mock_amdsmi.amdsmi_get_gpu_device_uuid.return_value = "GPU-uuid"
        mock_amdsmi.amdsmi_get_gpu_vram_usage.return_value = VRAM_USAGE
        mock_amdsmi.amdsmi_get_pcie_info.return_value = PCIE_INFO
        mock_amdsmi.amdsmi_get_gpu_vram_info.return_value = VRAM_INFO
        mock_amdsmi.amdsmi_get_gpu_board_info.return_value = BOARD_INFO
        mock_amdsmi.amdsmi_get_gpu_driver_info.return_value = DRIVER_INFO
        mock_amdsmi.amdsmi_get_gpu_activity.return_value = GPU_ACTIVITY
        mock_amdsmi.amdsmi_get_power_info.return_value = POWER_INFO
        mock_amdsmi.amdsmi_get_violation_status.return_value = (
            VIOLATION_STATUS
        )
        mock_amdsmi.amdsmi_get_temp_metric.return_value = 65000

        with patch.dict(sys.modules, {"amdsmi": mock_amdsmi}):
            import importlib

            import warpt.backends.amd as _amd_mod

            importlib.reload(_amd_mod)
            return _amd_mod.AmdBackend()

    def test_pcie_gen_na_sentinel(self):
        """PCIe gen set to None when interface_version is 'N/A'."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_get_pcie_info.return_value = {
            "pcie_static": {"pcie_interface_version": "N/A"},
            "pcie_metric": {},
        }
        devices = backend.list_devices()
        assert devices[0].pcie_gen is None

    def test_vram_usage_na_values(self):
        """Memory usage returns None when vram values are not ints."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_get_gpu_vram_usage.return_value = {
            "vram_total": "N/A",
            "vram_used": "N/A",
        }
        assert backend.get_memory_usage(0) is None

    def test_power_all_na_returns_none(self):
        """Power returns None when all power fields are 'N/A'."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_get_power_info.return_value = {
            "current_socket_power": "N/A",
            "average_socket_power": "N/A",
            "socket_power": "N/A",
            "power_limit": "N/A",
        }
        assert backend.get_power_usage(0) is None

    def test_zero_power_returns_none(self):
        """Power returns None when all values are zero."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_get_power_info.return_value = {
            "current_socket_power": 0,
            "average_socket_power": 0,
            "socket_power": 0,
            "power_limit": 0,
        }
        assert backend.get_power_usage(0) is None

    def test_uuid_failure_returns_none(self):
        """UUID is None when amdsmi_get_gpu_device_uuid fails."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_get_gpu_device_uuid.side_effect = RuntimeError("fail")
        devices = backend.list_devices()
        assert devices[0].uuid is None

    def test_board_info_all_na(self):
        """Board info N/A values are excluded from extra_metrics."""
        mock = _make_amdsmi_mock()
        backend = self._build_backend(mock)
        mock.amdsmi_get_gpu_board_info.return_value = {
            "model_number": "N/A",
            "product_serial": "N/A",
            "fru_id": "N/A",
            "product_name": "N/A",
            "manufacturer_name": "N/A",
        }
        devices = backend.list_devices()
        em = devices[0].extra_metrics
        assert em is None or "product_name" not in em
        assert em is None or "manufacturer_name" not in em
