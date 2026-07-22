"""Tests for the Intel Level Zero sysman backend and power backend.

All tests are mock-based and require no Intel hardware or the Level Zero
loader. The ctypes FFI surface is isolated behind ``_IntelSysman``, so the
backend logic is exercised by mocking that wrapper, while pure helpers and the
library loader are tested directly.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from warpt.backends.factory import get_accelerator_backend
from warpt.backends.intel import (
    IntelBackend,
    _clamp_percent,
    _decode,
    _decode_throttle,
    _energy_to_watts,
    _IntelSysman,
    _load_library,
    _ZeError,
)
from warpt.backends.power.intel_power import IntelPowerBackend
from warpt.models.list_models import GPUInfo
from warpt.models.power_models import DomainPower, GPUPowerInfo, PowerSource


def _fake_sysman(num_devices: int = 1) -> MagicMock:
    """Build a mock ``_IntelSysman`` with healthy default return values."""
    sysman = MagicMock()
    sysman.get_devices.return_value = [object() for _ in range(num_devices)]
    sysman.get_device_properties.return_value = {
        "model": "Intel Arc A770",
        "brand": "Intel Arc",
        "vendor": "Intel Corporation",
        "serial": "ABC123",
        "board": "BOARD-1",
        "driver_version": "1.3.26241",
        "num_subdevices": 0,
    }
    sysman.get_pci_properties.return_value = {"gen": 4, "width": 16}
    sysman.get_memory.return_value = {
        "total": 16 * 1024**3,
        "used": 4 * 1024**3,
        "free": 12 * 1024**3,
    }
    sysman.get_temperature.return_value = 55.0
    sysman.get_compute_utilization.return_value = 42.0
    sysman.get_power_watts.return_value = 120.5
    sysman.get_power_limit_watts.return_value = 225.0
    sysman.get_throttle_reasons.return_value = ["thermal"]
    return sysman


def _build_backend(sysman: MagicMock) -> IntelBackend:
    """Construct an ``IntelBackend`` wired to a mock sysman wrapper."""
    with patch("warpt.backends.intel._load_library", return_value=MagicMock()), patch(
        "warpt.backends.intel._IntelSysman", return_value=sysman
    ):
        return IntelBackend()


def _build_power_backend(sysman: MagicMock) -> IntelPowerBackend:
    """Construct and initialize an ``IntelPowerBackend`` with a mock sysman."""
    backend = IntelPowerBackend()
    with patch(
        "warpt.backends.power.intel_power._load_library", return_value=MagicMock()
    ), patch(
        "warpt.backends.power.intel_power._IntelSysman", return_value=sysman
    ), patch("warpt.backends.power.intel_power.LEVEL_ZERO_AVAILABLE", True):
        assert backend.initialize() is True
    return backend


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_decode_strips_null_and_whitespace():
    """``_decode`` handles bytes, NULL padding and surrounding whitespace."""
    assert _decode(b"Intel Arc A770\x00\x00") == "Intel Arc A770"
    assert _decode("  spaced  ") == "spaced"


def test_decode_throttle_maps_flags():
    """``_decode_throttle`` maps the bitmask to reason strings."""
    assert _decode_throttle(0) == []
    assert _decode_throttle(0x8) == ["thermal"]
    assert _decode_throttle(0x1 | 0x8) == ["power_limit", "thermal"]


def test_energy_to_watts_and_guard():
    """Energy (uJ) over time (us) yields Watts; non-positive dt is guarded."""
    # 100 J over 1 s => 100 W (100_000_000 uJ over 1_000_000 us).
    assert _energy_to_watts(100_000_000, 1_000_000) == pytest.approx(100.0)
    assert _energy_to_watts(50, 0) is None
    assert _energy_to_watts(50, -5) is None


def test_clamp_percent_bounds():
    """``_clamp_percent`` bounds the value to [0, 100]."""
    assert _clamp_percent(-5.0) == 0.0
    assert _clamp_percent(150.0) == 100.0
    assert _clamp_percent(42.0) == 42.0


def test_load_library_success():
    """``_load_library`` returns the first loadable shared object."""
    fake = object()
    with patch("warpt.backends.intel.ctypes.CDLL", return_value=fake) as cdll:
        assert _load_library() is fake
    cdll.assert_called_once()


def test_load_library_raises_when_missing():
    """``_load_library`` raises OSError when no loader can be opened."""
    with patch(
        "warpt.backends.intel.ctypes.CDLL", side_effect=OSError("missing")
    ):
        with pytest.raises(OSError):
            _load_library()


# ---------------------------------------------------------------------------
# _IntelSysman wrapper
# ---------------------------------------------------------------------------


def test_sysman_check_raises_on_error():
    """``_check`` passes on success and raises ``_ZeError`` otherwise."""
    assert _IntelSysman._check(0) is None
    with pytest.raises(_ZeError):
        _IntelSysman._check(0x78000001)


def test_sysman_init_failure():
    """``init`` raises ``_ZeError`` when ``zesInit`` returns non-success."""
    lib = MagicMock()
    lib.zesInit.return_value = 1
    with pytest.raises(_ZeError):
        _IntelSysman(lib).init()


def test_sysman_get_devices_without_driver():
    """``get_devices`` is empty before a driver has been resolved."""
    assert _IntelSysman(MagicMock()).get_devices() == []


# ---------------------------------------------------------------------------
# IntelBackend
# ---------------------------------------------------------------------------


def test_backend_init_raises_without_library():
    """Construction propagates the loader OSError for the factory to catch."""
    with patch("warpt.backends.intel._load_library", side_effect=OSError):
        with pytest.raises(OSError):
            IntelBackend()


def test_device_count_and_availability():
    """Device count reflects the wrapper; availability follows the count."""
    backend = _build_backend(_fake_sysman(2))
    assert backend.get_device_count() == 2
    assert backend.is_available() is True


def test_is_available_false_when_no_devices():
    """A backend with no devices is not available."""
    backend = _build_backend(_fake_sysman(0))
    assert backend.get_device_count() == 0
    assert backend.is_available() is False


def test_list_devices_populates_gpuinfo():
    """``list_devices`` returns a fully-populated ``GPUInfo``."""
    backend = _build_backend(_fake_sysman(1))
    devices = backend.list_devices()
    assert len(devices) == 1
    info = devices[0]
    assert isinstance(info, GPUInfo)
    assert info.index == 0
    assert info.model == "Intel Arc A770"
    assert info.memory_gb == 16
    assert info.uuid == "ABC123"
    assert info.compute_capability is None
    assert info.pcie_gen == 4
    assert info.driver_version == "1.3.26241"
    assert info.extra_metrics["vendor_name"] == "Intel Corporation"
    assert info.extra_metrics["pcie_width"] == 16
    assert info.extra_metrics["backend"] == "level_zero_sysman"


def test_list_devices_unknown_values_fall_back():
    """Unknown identity fields degrade to safe defaults."""
    sysman = _fake_sysman(1)
    sysman.get_device_properties.return_value = {
        "model": "unknown",
        "brand": "unknown",
        "vendor": "unknown",
        "serial": "unknown",
        "board": "unknown",
        "driver_version": "unknown",
        "num_subdevices": 0,
    }
    info = _build_backend(sysman).list_devices()[0]
    assert info.model == "Intel GPU"
    assert info.uuid is None
    assert info.driver_version is None


def test_list_devices_handles_missing_memory_and_pci():
    """Missing memory / PCI data yields zero memory and a None PCIe gen."""
    sysman = _fake_sysman(1)
    sysman.get_memory.return_value = None
    sysman.get_pci_properties.return_value = {"gen": None, "width": None}
    info = _build_backend(sysman).list_devices()[0]
    assert info.memory_gb == 0
    assert info.pcie_gen is None


def test_get_temperature_paths():
    """Temperature reads from the wrapper, guards range and swallows errors."""
    backend = _build_backend(_fake_sysman(1))
    assert backend.get_temperature(0) == 55.0
    assert backend.get_temperature(5) is None
    backend._sysman.get_temperature.side_effect = RuntimeError("boom")
    assert backend.get_temperature(0) is None


def test_get_memory_usage():
    """Memory usage is passed through from the wrapper."""
    backend = _build_backend(_fake_sysman(1))
    usage = backend.get_memory_usage(0)
    assert usage["total"] == 16 * 1024**3
    assert usage["used"] == 4 * 1024**3
    assert backend.get_memory_usage(99) is None


def test_get_utilization_combines_compute_and_memory():
    """Utilization reports compute % and derived memory-allocation %."""
    backend = _build_backend(_fake_sysman(1))
    util = backend.get_utilization(0)
    assert util == {"gpu": 42.0, "memory": 25.0}


def test_get_utilization_none_when_compute_unavailable():
    """When compute utilization is unavailable the whole reading is None."""
    sysman = _fake_sysman(1)
    sysman.get_compute_utilization.return_value = None
    backend = _build_backend(sysman)
    assert backend.get_utilization(0) is None
    assert backend.get_utilization(7) is None


def test_get_power_usage():
    """Power usage is passed through and guards the index."""
    backend = _build_backend(_fake_sysman(1))
    assert backend.get_power_usage(0) == 120.5
    assert backend.get_power_usage(3) is None


def test_get_throttle_reasons():
    """Throttle reasons are passed through; out-of-range yields an empty list."""
    backend = _build_backend(_fake_sysman(1))
    assert backend.get_throttle_reasons(0) == ["thermal"]
    assert backend.get_throttle_reasons(9) == []


def test_get_driver_version_variants():
    """Driver version handles the happy path, unknown and no-device cases."""
    backend = _build_backend(_fake_sysman(1))
    assert backend.get_driver_version() == "1.3.26241"

    unknown = _fake_sysman(1)
    unknown.get_device_properties.return_value = {"driver_version": "unknown"}
    assert _build_backend(unknown).get_driver_version() is None

    empty = _build_backend(_fake_sysman(0))
    assert empty.get_driver_version() is None


def test_static_identifiers():
    """PyTorch device string, topology and distributed backend are static."""
    backend = _build_backend(_fake_sysman(1))
    assert backend.get_pytorch_device_string(0) == "xpu:0"
    assert backend.get_pytorch_device_string(3) == "xpu:3"
    assert backend.get_topology() == "PCIe"
    assert backend.get_distributed_backend() == "ccl"


def test_shutdown_swallows_errors():
    """Shutdown delegates to the wrapper and never raises."""
    backend = _build_backend(_fake_sysman(1))
    backend.shutdown()
    backend._sysman.shutdown.assert_called_once()

    backend._sysman.shutdown.side_effect = RuntimeError("boom")
    backend.shutdown()  # must not raise


# ---------------------------------------------------------------------------
# IntelPowerBackend
# ---------------------------------------------------------------------------


def test_power_source_is_estimated():
    """The power source is reported as ESTIMATED (no Level Zero enum member)."""
    assert IntelPowerBackend().get_source() is PowerSource.ESTIMATED


def test_power_is_available_true():
    """``is_available`` is True when the loader and a device are present."""
    sysman = _fake_sysman(1)
    with patch(
        "warpt.backends.power.intel_power._load_library", return_value=MagicMock()
    ), patch(
        "warpt.backends.power.intel_power._IntelSysman", return_value=sysman
    ), patch("warpt.backends.power.intel_power.LEVEL_ZERO_AVAILABLE", True):
        assert IntelPowerBackend().is_available() is True


def test_power_is_available_false_without_level_zero():
    """``is_available`` is False when Level Zero could not be imported."""
    with patch("warpt.backends.power.intel_power.LEVEL_ZERO_AVAILABLE", False):
        assert IntelPowerBackend().is_available() is False


def test_power_initialize_failure_returns_false():
    """A loader failure makes ``initialize`` return False, not raise."""
    with patch(
        "warpt.backends.power.intel_power._load_library", side_effect=OSError
    ), patch("warpt.backends.power.intel_power.LEVEL_ZERO_AVAILABLE", True):
        assert IntelPowerBackend().initialize() is False


def test_power_readings():
    """``get_power_readings`` returns one DomainPower per reporting GPU."""
    backend = _build_power_backend(_fake_sysman(2))
    readings = backend.get_power_readings()
    assert len(readings) == 2
    assert all(isinstance(r, DomainPower) for r in readings)
    assert readings[0].power_watts == 120.5
    assert readings[0].source is PowerSource.ESTIMATED


def test_power_readings_skip_none():
    """GPUs that report no power are skipped in the readings."""
    sysman = _fake_sysman(2)
    sysman.get_power_watts.return_value = None
    backend = _build_power_backend(sysman)
    assert backend.get_power_readings() == []


def test_gpu_power_info():
    """``get_gpu_power_info`` produces populated GPUPowerInfo objects."""
    backend = _build_power_backend(_fake_sysman(1))
    infos = backend.get_gpu_power_info()
    assert len(infos) == 1
    info = infos[0]
    assert isinstance(info, GPUPowerInfo)
    assert info.name == "Intel Arc A770"
    assert info.power_watts == 120.5
    assert info.power_limit_watts == 225.0
    assert info.utilization_percent == 42.0
    assert info.memory_utilization_percent == pytest.approx(25.0)
    assert info.temperature_celsius == 55.0


def test_gpu_power_info_tolerates_errors():
    """Wrapper failures degrade to safe defaults in GPUPowerInfo."""
    sysman = _fake_sysman(1)
    sysman.get_power_watts.side_effect = RuntimeError("boom")
    sysman.get_compute_utilization.return_value = None
    sysman.get_temperature.return_value = None
    backend = _build_power_backend(sysman)
    info = backend.get_gpu_power_info()[0]
    assert info.power_watts == 0.0
    assert info.utilization_percent == 0.0
    assert info.temperature_celsius is None


def test_total_gpu_power():
    """Total power sums each GPU's reading."""
    backend = _build_power_backend(_fake_sysman(3))
    assert backend.get_total_gpu_power() == pytest.approx(120.5 * 3)


def test_power_cleanup_resets_state():
    """Cleanup calls shutdown and clears cached state."""
    sysman = _fake_sysman(1)
    backend = _build_power_backend(sysman)
    backend.cleanup()
    sysman.shutdown.assert_called_once()
    assert backend._initialized is False
    assert backend._devices == []
    assert backend._sysman is None


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


def test_factory_returns_intel_when_others_unavailable():
    """Factory falls through NVIDIA/AMD to the available Intel backend."""
    mock_nvidia = MagicMock()
    mock_nvidia.NvidiaBackend.return_value.is_available.return_value = False
    mock_amd = MagicMock()
    mock_amd.AMDBackend.side_effect = NotImplementedError

    mock_intel = MagicMock()
    intel_instance = MagicMock()
    intel_instance.is_available.return_value = True
    mock_intel.IntelBackend.return_value = intel_instance

    modules = {
        "warpt.backends.nvidia": mock_nvidia,
        "warpt.backends.amd": mock_amd,
        "warpt.backends.intel": mock_intel,
    }
    with patch.dict(sys.modules, modules):
        backend = get_accelerator_backend()
    assert backend is intel_instance
    intel_instance.is_available.assert_called_once()


def test_factory_raises_when_intel_unavailable():
    """Factory raises RuntimeError when no vendor (including Intel) is present."""
    mock_nvidia = MagicMock()
    mock_nvidia.NvidiaBackend.return_value.is_available.return_value = False
    mock_amd = MagicMock()
    mock_amd.AMDBackend.side_effect = NotImplementedError
    mock_intel = MagicMock()
    mock_intel.IntelBackend.return_value.is_available.return_value = False

    modules = {
        "warpt.backends.nvidia": mock_nvidia,
        "warpt.backends.amd": mock_amd,
        "warpt.backends.intel": mock_intel,
    }
    with patch.dict(sys.modules, modules):
        with pytest.raises(RuntimeError):
            get_accelerator_backend()
