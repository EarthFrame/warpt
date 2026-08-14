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
    _CACHE_TTL_S,
    _MEASURE_WINDOW_S,
    _RESUME_SETTLE_S,
    IntelBackend,
    _clamp_percent,
    _counters_incoherent,
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
        "integrated": False,
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
    ), patch(
        "warpt.backends.power.intel_power.LEVEL_ZERO_AVAILABLE", True
    ):
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
    with patch("warpt.backends.intel.ctypes.CDLL", side_effect=OSError("missing")):
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


def _sysman_with_energy(readings: list[tuple[int, int]]) -> _IntelSysman:
    """Build a sysman whose energy counter yields ``readings`` in order.

    Each reading is ``(energy_uj, timestamp_us)``. One power domain is
    enumerated so ``get_power_watts`` has something to sample.
    """
    remaining = list(readings)

    def _enum(_handle, count_ref, array):
        count_ref._obj.value = 1
        if array is not None:
            array[0] = 0xABCD
        return 0

    def _counter(_domain, ref):
        energy, timestamp = remaining.pop(0)
        ref._obj.energy = energy
        ref._obj.timestamp = timestamp
        return 0

    lib = MagicMock()
    lib.zesDeviceEnumPowerDomains.side_effect = _enum
    lib.zesPowerGetEnergyCounter.side_effect = _counter
    sysman = _IntelSysman(lib)
    sysman._reads_left = remaining  # exposed for assertions
    return sysman


def _sysman_drawing(clock: list[float], watts: float) -> _IntelSysman:
    """Build a sysman whose energy counter accrues ``watts`` against ``clock``.

    Driving the counter from the fake clock rather than a fixed list of
    readings means the number of reads does not have to be predicted: the
    measurement loop polls as often as it likes and still sees a device drawing
    exactly ``watts``.
    """

    def _enum(_handle, count_ref, array):
        count_ref._obj.value = 1
        if array is not None:
            array[0] = 0xABCD
        return 0

    def _counter(_domain, ref):
        ref._obj.energy = int(watts * clock[0] * 1_000_000)  # microjoules
        ref._obj.timestamp = int(clock[0] * 1_000_000)  # microseconds
        return 0

    lib = MagicMock()
    lib.zesDeviceEnumPowerDomains.side_effect = _enum
    lib.zesPowerGetEnergyCounter.side_effect = _counter
    return _IntelSysman(lib)


def _advancing_clock(clock: list[float]):
    """Patch context where ``time.sleep`` advances the fake monotonic clock.

    The measurement loop polls until a deadline, so a frozen clock would spin
    forever. Letting sleep drive the clock keeps the loop terminating while
    still making time deterministic.
    """
    return (
        patch("warpt.backends.intel.time.monotonic", side_effect=lambda: clock[0]),
        patch(
            "warpt.backends.intel.time.sleep",
            side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        ),
    )


def _measure(sysman, clock, *, status="active", pm_stays_awake=True):
    """Run one ``get_power_watts`` with runtime-PM state stubbed out."""
    sysman.get_pci_bdf = lambda _handle: "0000:03:00.0"
    # (active_ms, suspended_ms); a device that napped has suspended_ms advance.
    pm_values = [(1000, 500), (1000, 500 if pm_stays_awake else 1500)]
    monotonic, sleep = _advancing_clock(clock)
    with monotonic, sleep, patch(
        "warpt.backends.intel._runtime_pm_status", return_value=status
    ), patch(
        "warpt.backends.intel._read_runtime_pm",
        side_effect=lambda _bdf: pm_values.pop(0) if pm_values else (1000, 500),
    ):
        return sysman.get_power_watts(MagicMock())


def test_power_watts_measures_device_draw():
    """The reported figure is the device's actual draw over the window."""
    clock = [100.0]
    sysman = _sysman_drawing(clock, 5.3)
    assert _measure(sysman, clock) == pytest.approx(5.3)


def test_power_watts_skips_settle_when_device_already_active():
    """An already-running device is measured immediately, with no settle cost.

    This is the workload case: adding seconds of latency to every reading
    while a job is running would be unacceptable.
    """
    clock = [100.0]
    sysman = _sysman_drawing(clock, 42.0)
    start = clock[0]
    assert _measure(sysman, clock, status="active") == pytest.approx(42.0)
    elapsed = clock[0] - start
    assert elapsed == pytest.approx(_MEASURE_WINDOW_S, abs=0.3)
    assert elapsed < _RESUME_SETTLE_S


def test_power_watts_settles_before_measuring_a_suspended_device():
    """A suspended device is woken and the post-resume dip discarded."""
    clock = [100.0]
    sysman = _sysman_drawing(clock, 5.3)
    start = clock[0]
    assert _measure(sysman, clock, status="suspended") == pytest.approx(5.3)
    assert clock[0] - start >= _RESUME_SETTLE_S + _MEASURE_WINDOW_S


def test_power_watts_none_when_device_suspends_mid_window():
    """A window the device slept through is discarded, not reported.

    The counter accrues a large fictitious figure across runtime suspend, so
    such a window must yield None rather than an inflated number.
    """
    clock = [100.0]
    sysman = _sysman_drawing(clock, 5.3)
    assert _measure(sysman, clock, pm_stays_awake=False) is None


def test_power_watts_measures_without_runtime_pm_interface():
    """Off Linux there is no runtime-PM interface; measurement still works.

    Without it the settle is applied unconditionally, which costs latency but
    never accuracy, and the stayed-awake check is skipped.
    """
    clock = [100.0]
    sysman = _sysman_drawing(clock, 5.3)
    sysman.get_pci_bdf = lambda _handle: None
    monotonic, sleep = _advancing_clock(clock)
    start = clock[0]
    with monotonic, sleep, patch(
        "warpt.backends.intel._read_runtime_pm", return_value=None
    ):
        assert sysman.get_power_watts(MagicMock()) == pytest.approx(5.3)
    assert clock[0] - start >= _RESUME_SETTLE_S


def _sysman_with_two_domains(
    clock: list[float], card_watts: float, package_watts: float
) -> _IntelSysman:
    """Build a sysman exposing a card domain and a package domain.

    Each accrues at its own rate against the fake clock, so the coherence
    cross-check can be driven into either state.
    """
    card_handle, package_handle = 0xCA, 0xBB
    rates = {card_handle: card_watts, package_handle: package_watts}

    def _enum(_handle, count_ref, array):
        count_ref._obj.value = 2
        if array is not None:
            array[0] = card_handle
            array[1] = package_handle
        return 0

    def _counter(domain, ref):
        rate = rates[domain.value]
        ref._obj.energy = int(rate * clock[0] * 1_000_000)
        ref._obj.timestamp = int(clock[0] * 1_000_000)
        return 0

    lib = MagicMock()
    lib.zesDeviceEnumPowerDomains.side_effect = _enum
    lib.zesPowerGetEnergyCounter.side_effect = _counter
    return _IntelSysman(lib)


def test_counters_incoherent_flags_impossible_package_reading():
    """Package power above card power is impossible; near-equal is not.

    Values are those measured on Battlemage: the corrupted state reports
    ~43 W card / ~220 W package, while healthy readings sit close together at
    low power and comfortably apart under load.
    """
    assert _counters_incoherent(42.9, 219.7) is True  # observed corrupted state
    assert _counters_incoherent(1.59, 2.24) is False  # healthy, near-equal
    assert _counters_incoherent(221.0, 164.0) is False  # healthy under load
    assert _counters_incoherent(5.27, 1.96) is False  # healthy at idle


def test_power_watts_rejects_incoherent_counter_state():
    """A window where package exceeds card is discarded rather than reported.

    Without this the corrupted state surfaces as a plausible-looking ~43 W,
    which is roughly 8x the device's true idle draw.
    """
    clock = [100.0]
    sysman = _sysman_with_two_domains(clock, card_watts=42.9, package_watts=219.7)
    assert _measure(sysman, clock) is None


def test_power_watts_accepts_coherent_two_domain_reading():
    """A healthy card/package pair measures normally."""
    clock = [100.0]
    sysman = _sysman_with_two_domains(clock, card_watts=5.3, package_watts=2.0)
    assert _measure(sysman, clock) == pytest.approx(5.3)


def test_power_watts_is_cached_within_ttl():
    """The two calls making up one snapshot share a single measurement window."""
    clock = [100.0]
    sysman = _sysman_drawing(clock, 5.3)
    assert _measure(sysman, clock) == pytest.approx(5.3)
    reads_after_first = sysman._lib.zesPowerGetEnergyCounter.call_count

    monotonic, sleep = _advancing_clock(clock)
    with monotonic, sleep:
        clock[0] += _CACHE_TTL_S / 2
        assert sysman.get_power_watts(MagicMock()) == pytest.approx(5.3)
    assert sysman._lib.zesPowerGetEnergyCounter.call_count == reads_after_first


def test_get_energy_joules_is_one_read():
    """The cumulative counter is exposed directly, at a cost of one read."""
    sysman = _sysman_with_energy([(7_500_000, 1_000_000)])
    assert sysman.get_energy_joules(MagicMock()) == pytest.approx(7.5)
    assert sysman._lib.zesPowerGetEnergyCounter.call_count == 1


def test_power_watts_no_domains_returns_none():
    """A device exposing no power domain yields None, not an exception."""
    lib = MagicMock()
    lib.zesDeviceEnumPowerDomains.return_value = 1  # non-success
    assert _IntelSysman(lib).get_power_watts(MagicMock()) is None


def test_shutdown_clears_cached_power():
    """Cached figures are dropped on shutdown; handles die with the driver."""
    clock = [100.0]
    sysman = _sysman_drawing(clock, 5.3)
    _measure(sysman, clock)
    assert sysman._energy_watts
    sysman.shutdown()
    assert sysman._energy_watts == {}


def _sysman_with_device_flags(flags: int) -> _IntelSysman:
    """Build a sysman whose ``zesDeviceGetProperties`` reports ``core.flags``.

    Writes through the ``byref`` pointer the wrapper passes, so the real
    ctypes struct marshalling in ``get_device_properties`` is exercised.
    """

    def _fill(_handle, ref):
        props = ref._obj
        props.core.flags = flags
        props.modelName = b"Intel Test GPU"
        props.brandName = b"Intel"
        props.vendorName = b"Intel Corporation"
        props.serialNumber = b"SER-1"
        props.boardNumber = b"BRD-1"
        props.driverVersion = b"1.3.0"
        props.numSubdevices = 0
        return 0

    lib = MagicMock()
    lib.zesDeviceGetProperties.side_effect = _fill
    return _IntelSysman(lib)


def test_device_properties_reports_integrated_flag():
    """``integrated`` is True only when the INTEGRATED bit is set in flags."""
    # 0x1 == ZE_DEVICE_PROPERTY_FLAG_INTEGRATED.
    props = _sysman_with_device_flags(0x1).get_device_properties(MagicMock())
    assert props["integrated"] is True
    assert props["model"] == "Intel Test GPU"


def test_device_properties_discrete_when_integrated_bit_clear():
    """Other flag bits must not be mistaken for the INTEGRATED bit.

    0x8 (ONDEMANDPAGING) is what a real discrete Arc card reports.
    """
    assert (
        _sysman_with_device_flags(0x8).get_device_properties(MagicMock())["integrated"]
        is False
    )
    assert (
        _sysman_with_device_flags(0x0).get_device_properties(MagicMock())["integrated"]
        is False
    )
    # Integrated alongside other bits is still integrated.
    assert (
        _sysman_with_device_flags(0x9).get_device_properties(MagicMock())["integrated"]
        is True
    )


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


def test_power_source_is_level_zero():
    """The power source is reported as LEVEL_ZERO (hardware energy counter)."""
    assert IntelPowerBackend().get_source() is PowerSource.LEVEL_ZERO


def test_power_is_available_true():
    """``is_available`` is True when the loader and a device are present."""
    sysman = _fake_sysman(1)
    with patch(
        "warpt.backends.power.intel_power._load_library", return_value=MagicMock()
    ), patch(
        "warpt.backends.power.intel_power._IntelSysman", return_value=sysman
    ), patch(
        "warpt.backends.power.intel_power.LEVEL_ZERO_AVAILABLE", True
    ):
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
    assert readings[0].source is PowerSource.LEVEL_ZERO


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
    assert backend._static == []
    assert backend._sysman is None


# ---------------------------------------------------------------------------
# Vendor tagging and integrated-GPU reporting
# ---------------------------------------------------------------------------


def test_readings_are_tagged_with_vendor():
    """Every reading carries the 'intel' vendor tag.

    Both NVML and Level Zero number devices from 0, so the vendor tag is what
    keeps an Intel GPU 0 distinct from an NVIDIA GPU 0 downstream.
    """
    backend = _build_power_backend(_fake_sysman(2))
    assert [r.metadata["vendor"] for r in backend.get_power_readings()] == [
        "intel",
        "intel",
    ]
    assert [g.vendor for g in backend.get_gpu_power_info()] == ["intel", "intel"]
    assert [g.index for g in backend.get_gpu_power_info()] == [0, 1]


def test_discrete_gpu_reported_as_not_integrated():
    """A discrete card is flagged integrated=False on both reading paths."""
    backend = _build_power_backend(_fake_sysman(1))
    assert backend.get_power_readings()[0].metadata["integrated"] is False
    assert backend.get_gpu_power_info()[0].metadata["integrated"] is False


def test_integrated_gpu_flag_propagates_to_readings():
    """An integrated GPU is flagged so PowerMonitor can skip double-counting."""
    sysman = _fake_sysman(1)
    sysman.get_device_properties.return_value = {
        "model": "Intel Iris Xe Graphics",
        "brand": "Intel",
        "vendor": "Intel Corporation",
        "serial": "ABC123",
        "board": "BOARD-1",
        "driver_version": "1.3.26241",
        "num_subdevices": 0,
        "integrated": True,
    }
    backend = _build_power_backend(sysman)
    assert backend.get_power_readings()[0].metadata["integrated"] is True
    info = backend.get_gpu_power_info()[0]
    assert info.metadata["integrated"] is True
    assert info.name == "Intel Iris Xe Graphics"


def test_integrated_defaults_false_when_property_missing():
    """A driver that omits the flag degrades to discrete, not integrated.

    Treating a discrete GPU as integrated would silently drop its power from
    the system total, so False is the safer default.
    """
    sysman = _fake_sysman(1)
    sysman.get_device_properties.return_value = {"model": "Intel Arc A770"}
    backend = _build_power_backend(sysman)
    assert backend.get_gpu_power_info()[0].metadata["integrated"] is False


def test_integrated_defaults_false_when_properties_unavailable():
    """A raising properties call still yields a usable, discrete-tagged device."""
    sysman = _fake_sysman(1)
    sysman.get_device_properties.side_effect = RuntimeError("boom")
    backend = _build_power_backend(sysman)
    info = backend.get_gpu_power_info()[0]
    assert info.name == "Intel GPU"
    assert info.metadata["integrated"] is False


def test_static_identity_is_read_once_per_device():
    """Identity is cached at initialize, not re-read on every snapshot.

    The carbon sampling loop calls get_snapshot on an interval, so identity
    lookups must not add FFI calls per sample.
    """
    sysman = _fake_sysman(2)
    backend = _build_power_backend(sysman)
    assert sysman.get_device_properties.call_count == 2

    for _ in range(3):
        backend.get_gpu_power_info()
        backend.get_power_readings()
    assert sysman.get_device_properties.call_count == 2


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
