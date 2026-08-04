"""Intel GPU backend using the oneAPI Level Zero Sysman API.

Intel GPUs (integrated Xe, Arc, Flex, and Max) expose telemetry through the
Level Zero System Resource Management (sysman) interface. There is no
first-party Python binding, so this backend binds directly to
``libze_loader.so.1`` via :mod:`ctypes`.

The low-level ctypes marshalling is isolated in :class:`_IntelSysman` so that
:class:`IntelBackend` reads cleanly and mirrors the NVIDIA reference backend.
The same wrapper is reused by the Intel power backend.
"""

from __future__ import annotations

import ctypes
import glob
import os
import time
from typing import Any

from warpt.backends.base import AcceleratorBackend
from warpt.models.list_models import GPUInfo

# ``ze_result_t`` success sentinel. Every zes* call returns an int status.
_ZE_RESULT_SUCCESS = 0

# Fixed-size character buffers used by the sysman structs.
_ZES_STRING_PROPERTY_SIZE = 64
_ZE_MAX_DEVICE_NAME = 256

# ``zes_structure_type_t`` values (derived from the documented enum ordering,
# which starts at 0x1). Every output struct must have ``stype`` set before use.
_ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x1
_ZES_STRUCTURE_TYPE_PCI_PROPERTIES = 0x2
_ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES = 0x5
_ZES_STRUCTURE_TYPE_FREQ_PROPERTIES = 0x9
_ZES_STRUCTURE_TYPE_POWER_PROPERTIES = 0xD
_ZES_STRUCTURE_TYPE_TEMP_PROPERTIES = 0x14
_ZES_STRUCTURE_TYPE_FREQ_STATE = 0x1B
_ZES_STRUCTURE_TYPE_MEM_STATE = 0x1E

# ``ze_device_property_flag_t`` bits set on ``ze_device_properties_t.flags``.
# INTEGRATED marks a GPU fused into the CPU package, whose power is already
# accounted for by the CPU package (RAPL) domain — see
# ``IntelPowerBackend.get_gpu_power_info``.
_ZE_DEVICE_PROPERTY_FLAG_INTEGRATED = 0x1

# ``zes_engine_group_t`` selectors we care about.
_ZES_ENGINE_GROUP_ALL = 0
_ZES_ENGINE_GROUP_COMPUTE_ALL = 1

# ``zes_temp_sensors_t`` selectors.
_ZES_TEMP_SENSORS_GLOBAL = 0
_ZES_TEMP_SENSORS_GPU = 1

# ``zes_freq_domain_t`` selectors.
_ZES_FREQ_DOMAIN_GPU = 0

# ``zes_freq_throttle_reason_flag_t`` bit flags mapped to warpt-style strings.
_THROTTLE_FLAGS: list[tuple[int, str]] = [
    (0x1, "power_limit"),
    (0x2, "burst_power_limit"),
    (0x4, "current_limit"),
    (0x8, "thermal"),
    (0x10, "psu_alert"),
    (0x20, "sw_range"),
    (0x40, "hw_range"),
]

# Seconds between the two snapshots required to derive an instantaneous
# utilization or power value from the monotonic activity/energy counters.
_SAMPLE_INTERVAL_S = 0.05

# The energy counter misbehaves around the device's *runtime sleep*, which is
# what these constants exist to work around. Measured on Battlemage (Arc Pro
# B70) under the ``xe`` driver, with the card headless and therefore suspended
# ~86% of the time:
#
#   * For ~2 s after a runtime resume the counter under-reports badly: the
#     first second reads ~1.7 W and the second ~0.1 W, against a true ~5.3 W.
#     A GPU that is fully powered up cannot draw 0.1 W, so this is the counter
#     catching up, not a real measurement.
#   * While the device is suspended the counter accrues ~40 W, versus ~5 W
#     measured awake-and-idle. Any window spanning a nap is inflated.
#
# Reading the counter is *not* itself harmful: 602 reads and 2 reads over the
# same 30 s window agree to 0.04 W, and sweeping the poll rate from 0.033 Hz to
# 100 Hz moves the figure by 0.4%. What matters is only whether the device slept
# during the window.
#
# That gives the strategy below: skip the post-resume settling period, and poll
# often enough during the measurement that the device cannot autosuspend
# underneath us (each read resets its autosuspend timer).

# Settling period discarded after a runtime resume, before measuring.
_RESUME_SETTLE_S = 2.0

# Window the reported power figure is averaged over.
_MEASURE_WINDOW_S = 1.0

# How often the counter is touched inside a window. Must stay below the
# device's autosuspend delay (1 s on this hardware) so that polling keeps the
# device awake for the duration of the measurement.
_POLL_INTERVAL_S = 0.25

# One snapshot calls :meth:`get_power_watts` twice (once for the domain reading
# and once for the per-GPU detail). Caching for a beat means the measurement
# window is paid once and both calls agree with each other.
_CACHE_TTL_S = 1.0

# The counter also sometimes resumes into a corrupted state, where it reports a
# steady but wrong figure for as long as it lasts — measured at ~43 W on the
# card domain against a true ~5 W, on 3 of 16 resumes. It is not transient, so
# neither a longer settling period nor averaging removes it.
#
# It is detectable: the package domain is a subset of the card domain, so a
# package reading far above the card reading is physically impossible. In the
# corrupted state the package domain reads ~220 W against the card's ~43 W,
# while healthy readings have package comfortably below card (~0.9 W vs ~1.7 W
# idle, 164 W vs 221 W under load). Both a ratio and an absolute margin are
# required so that near-equal readings at very low power are not rejected.
_INCOHERENCE_RATIO = 1.5
_INCOHERENCE_MARGIN_W = 5.0

# Candidate shared-object names for the Level Zero loader.
_LIBRARY_NAMES = ("libze_loader.so.1", "libze_loader.so")

# Preferred hwmon temperature-sensor labels, most-preferred group first. Used
# to pick the GPU/package sensor over memory (VRAM) when Level Zero itself
# reports no temperature (a known gap on some Intel drivers/firmware, e.g.
# Battlemage under the ``xe`` driver). Matching is case-insensitive substring.
_HWMON_TEMP_LABEL_PRIORITY: tuple[tuple[str, ...], ...] = (
    ("gpu",),
    ("pkg", "package"),
    ("gt",),
    ("core",),
)
# Labels describing memory rather than the GPU die; used only as a last resort
# so a VRAM-only card still reports something instead of None.
_HWMON_TEMP_LABEL_DEPRIORITIZE: tuple[str, ...] = ("vram", "mem", "hbm")


def _read_sysfs(path: str) -> str | None:
    """Read and strip a sysfs text file, returning None on any error."""
    try:
        with open(path, encoding="ascii") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _counters_incoherent(card_watts: float, package_watts: float) -> bool:
    """Report whether a package reading implausibly exceeds its card reading.

    The package power domain is contained within the card domain, so package
    power can never meaningfully exceed card power. When it does, the energy
    counter is in the corrupted state described alongside
    ``_INCOHERENCE_RATIO`` and neither figure can be trusted.

    Parameters
    ----------
    card_watts : float
        Power derived from the card-level domain.
    package_watts : float
        Power derived from the package-level domain.

    Returns
    -------
    bool
        True if the pair is physically impossible and should be discarded.
    """
    return (
        package_watts > card_watts * _INCOHERENCE_RATIO
        and package_watts - card_watts > _INCOHERENCE_MARGIN_W
    )


def _runtime_pm_status(pci_bdf: str) -> str | None:
    """Return the device's runtime-PM state, e.g. ``active`` or ``suspended``.

    Read from the kernel's generic PCI runtime-power interface, not from the
    device, so this does **not** wake it. Returns None where the interface does
    not exist (non-Linux), leaving the caller to assume nothing.

    Parameters
    ----------
    pci_bdf : str
        The device's PCI address in sysfs ``DDDD:BB:DD.F`` form.

    Returns
    -------
    str or None
        The lowercase runtime-PM status, or None if unavailable.
    """
    raw = _read_sysfs(f"/sys/bus/pci/devices/{pci_bdf}/power/runtime_status")
    return raw.lower() if raw else None


def _read_runtime_pm(pci_bdf: str) -> tuple[int, int] | None:
    """Return ``(active_ms, suspended_ms)`` from the kernel's runtime-PM counters.

    These are kernel-side bookkeeping, so — unlike every other reading in this
    module — sampling them does not wake or otherwise disturb the device. That
    makes them the one honest witness to whether the device stayed awake across
    a measurement window.

    Parameters
    ----------
    pci_bdf : str
        The device's PCI address in sysfs ``DDDD:BB:DD.F`` form.

    Returns
    -------
    tuple[int, int] or None
        Cumulative milliseconds spent active and suspended, or None where the
        interface does not exist (non-Linux) or either value is unreadable.
    """
    base = f"/sys/bus/pci/devices/{pci_bdf}/power"
    active = _read_sysfs(f"{base}/runtime_active_time")
    suspended = _read_sysfs(f"{base}/runtime_suspended_time")
    if active is None or suspended is None:
        return None
    try:
        return int(active), int(suspended)
    except ValueError:
        return None


def _rank_hwmon_label(label: str) -> int:
    """Rank an hwmon temperature label; lower is more preferred.

    Preferred GPU/package labels sort first, unknown labels next, and memory
    (VRAM) labels last so they are only chosen when nothing better exists.
    """
    lowered = label.lower()
    for score, keys in enumerate(_HWMON_TEMP_LABEL_PRIORITY):
        if any(key in lowered for key in keys):
            return score
    if any(key in lowered for key in _HWMON_TEMP_LABEL_DEPRIORITIZE):
        return len(_HWMON_TEMP_LABEL_PRIORITY) + 1
    return len(_HWMON_TEMP_LABEL_PRIORITY)


def _read_hwmon_temperature(pci_bdf: str) -> float | None:
    """Read a GPU temperature from the kernel hwmon interface.

    Level Zero reports no temperature on some Intel drivers/firmware, but the
    kernel still exposes the sensor under the card's PCI device. This reads
    that sensor, keyed strictly by the device's PCI address so it can never
    pick up an unrelated (e.g. CPU package) sensor. It scans every
    ``hwmon*/temp*_input`` under the device, ranks them by their ``_label``
    sibling, and returns the best-available reading.

    On non-Linux platforms (or a card with no hwmon temperature) the glob
    matches nothing and this returns None.

    Parameters
    ----------
    pci_bdf : str
        The device's PCI address in sysfs ``DDDD:BB:DD.F`` form.

    Returns
    -------
    float or None
        The best-available temperature in Celsius, or None if the device
        exposes no usable hwmon temperature sensor.
    """
    base = f"/sys/bus/pci/devices/{pci_bdf}/hwmon"
    candidates: list[tuple[int, int, float]] = []
    for input_path in sorted(glob.glob(os.path.join(base, "hwmon*", "temp*_input"))):
        raw = _read_sysfs(input_path)
        if raw is None:
            continue
        try:
            celsius = int(raw) / 1000.0
        except ValueError:
            continue
        if not 0.0 < celsius < 150.0:
            continue
        label = _read_sysfs(input_path.replace("_input", "_label")) or ""
        candidates.append((_rank_hwmon_label(label), len(candidates), celsius))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _temperature_with_fallback(sysman: Any, handle: Any) -> float | None:
    """Return a device temperature, trying Level Zero then kernel hwmon.

    ``sysman`` is an :class:`_IntelSysman` (or compatible) exposing
    ``get_temperature`` and ``get_pci_bdf``. Some Intel drivers/firmware
    report no temperature through Level Zero; the kernel still exposes the
    sensor under the card's PCI device, so this falls back to hwmon keyed by
    the device's PCI address. Shared by the accelerator and power backends so
    both surface the same value.
    """
    try:
        temperature = sysman.get_temperature(handle)
    except Exception:
        temperature = None
    if temperature is not None:
        return temperature
    try:
        bdf = sysman.get_pci_bdf(handle)
    except Exception:
        bdf = None
    if isinstance(bdf, str) and bdf:
        return _read_hwmon_temperature(bdf)
    return None


def _decode(raw: bytes | str) -> str:
    """Decode a NULL-terminated ctypes character buffer to a clean string.

    Parameters
    ----------
    raw : bytes or str
        The raw value read from a ctypes ``c_char`` array.

    Returns
    -------
    str
        The decoded value with NULL padding and surrounding whitespace removed.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "ignore")
    return raw.replace("\x00", "").strip()


def _decode_throttle(flags: int) -> list[str]:
    """Translate a throttle-reason bitmask into a list of reason strings.

    Parameters
    ----------
    flags : int
        The ``throttleReasons`` bitmask from ``zes_freq_state_t``.

    Returns
    -------
    list[str]
        Active throttle reasons, empty if the frequency is not throttled.
    """
    return [name for bit, name in _THROTTLE_FLAGS if flags & bit]


def _energy_to_watts(delta_energy_uj: int, delta_time_us: int) -> float | None:
    """Convert an energy-counter delta into average power in Watts.

    The Level Zero energy counter is in microjoules and its timestamp is in
    microseconds, so ``uJ / us`` yields Watts directly.

    Parameters
    ----------
    delta_energy_uj : int
        Change in the energy counter between two snapshots, in microjoules.
    delta_time_us : int
        Change in the timestamp between the two snapshots, in microseconds.

    Returns
    -------
    float or None
        Average power in Watts, or None if the time delta is not positive.
    """
    if delta_time_us <= 0:
        return None
    return delta_energy_uj / delta_time_us


def _clamp_percent(value: float) -> float:
    """Clamp a value into the inclusive ``[0.0, 100.0]`` percentage range.

    Parameters
    ----------
    value : float
        The raw percentage value.

    Returns
    -------
    float
        The value bounded to ``[0.0, 100.0]``.
    """
    return max(0.0, min(100.0, value))


class _ZeDeviceUuid(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint8 * 16)]


class _ZeDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_int),
        ("vendorId", ctypes.c_uint32),
        ("deviceId", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("subdeviceId", ctypes.c_uint32),
        ("coreClockRate", ctypes.c_uint32),
        ("maxMemAllocSize", ctypes.c_uint64),
        ("maxHardwareContexts", ctypes.c_uint32),
        ("maxCommandQueuePriority", ctypes.c_uint32),
        ("numThreadsPerEU", ctypes.c_uint32),
        ("physicalEUSimdWidth", ctypes.c_uint32),
        ("numEUsPerSubslice", ctypes.c_uint32),
        ("numSubslicesPerSlice", ctypes.c_uint32),
        ("numSlices", ctypes.c_uint32),
        ("timerResolution", ctypes.c_uint64),
        ("timestampValidBits", ctypes.c_uint32),
        ("kernelTimestampValidBits", ctypes.c_uint32),
        ("uuid", _ZeDeviceUuid),
        ("name", ctypes.c_char * _ZE_MAX_DEVICE_NAME),
    ]


class _ZesDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("core", _ZeDeviceProperties),
        ("numSubdevices", ctypes.c_uint32),
        ("serialNumber", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
        ("boardNumber", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
        ("brandName", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
        ("modelName", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
        ("vendorName", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
        ("driverVersion", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
    ]


class _ZesPciAddress(ctypes.Structure):
    _fields_ = [
        ("domain", ctypes.c_uint32),
        ("bus", ctypes.c_uint32),
        ("device", ctypes.c_uint32),
        ("function", ctypes.c_uint32),
    ]


class _ZesPciSpeed(ctypes.Structure):
    _fields_ = [
        ("gen", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("maxBandwidth", ctypes.c_int64),
    ]


class _ZesPciProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("address", _ZesPciAddress),
        ("maxSpeed", _ZesPciSpeed),
        ("haveBandwidthCounters", ctypes.c_uint8),
        ("havePacketCounters", ctypes.c_uint8),
        ("haveReplayCounters", ctypes.c_uint8),
    ]


class _ZesEngineProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_int),
        ("onSubdevice", ctypes.c_uint8),
        ("subdeviceId", ctypes.c_uint32),
    ]


class _ZesEngineStats(ctypes.Structure):
    _fields_ = [
        ("activeTime", ctypes.c_uint64),
        ("timestamp", ctypes.c_uint64),
    ]


class _ZesFreqProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_int),
        ("onSubdevice", ctypes.c_uint8),
        ("subdeviceId", ctypes.c_uint32),
        ("canControl", ctypes.c_uint8),
        ("isThrottleEventSupported", ctypes.c_uint8),
        ("min", ctypes.c_double),
        ("max", ctypes.c_double),
    ]


class _ZesFreqState(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("currentVoltage", ctypes.c_double),
        ("request", ctypes.c_double),
        ("tdp", ctypes.c_double),
        ("efficient", ctypes.c_double),
        ("actual", ctypes.c_double),
        ("throttleReasons", ctypes.c_uint32),
    ]


class _ZesMemState(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("health", ctypes.c_int),
        ("free", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
    ]


class _ZesPowerProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("onSubdevice", ctypes.c_uint8),
        ("subdeviceId", ctypes.c_uint32),
        ("canControl", ctypes.c_uint8),
        ("isEnergyThresholdSupported", ctypes.c_uint8),
        ("defaultLimit", ctypes.c_int32),
        ("minLimit", ctypes.c_int32),
        ("maxLimit", ctypes.c_int32),
    ]


class _ZesPowerEnergyCounter(ctypes.Structure):
    _fields_ = [
        ("energy", ctypes.c_uint64),
        ("timestamp", ctypes.c_uint64),
    ]


class _ZesTempProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_int),
        ("onSubdevice", ctypes.c_uint8),
        ("subdeviceId", ctypes.c_uint32),
        ("maxTemperature", ctypes.c_double),
        ("isCriticalTempSupported", ctypes.c_uint8),
        ("isThreshold1Supported", ctypes.c_uint8),
        ("isThreshold2Supported", ctypes.c_uint8),
    ]


class _ZeError(Exception):
    """Raised when a Level Zero sysman call returns a non-success status."""

    def __init__(self, result: int) -> None:
        """Store the raw ``ze_result_t`` status.

        Parameters
        ----------
        result : int
            The non-zero status code returned by the sysman call.
        """
        self.result = result & 0xFFFFFFFF
        super().__init__(f"Level Zero call failed: 0x{self.result:08x}")


def _load_library() -> ctypes.CDLL:
    """Load the Level Zero loader shared object.

    Returns
    -------
    ctypes.CDLL
        A handle to ``libze_loader``.

    Raises
    ------
    OSError
        If none of the candidate loader names can be opened.
    """
    last_error: OSError | None = None
    for name in _LIBRARY_NAMES:
        try:
            return ctypes.CDLL(name)
        except OSError as error:
            last_error = error
    raise OSError(
        f"Level Zero loader ({', '.join(_LIBRARY_NAMES)}) not found: {last_error}"
    )


class _IntelSysman:
    """Thin ctypes wrapper over the Level Zero sysman (``zes*``) API.

    Every method returns Python-native values and raises :class:`_ZeError` on a
    non-success status so that callers can degrade gracefully.
    """

    def __init__(self, lib: ctypes.CDLL) -> None:
        """Bind to an already-loaded Level Zero loader.

        Parameters
        ----------
        lib : ctypes.CDLL
            A handle returned by :func:`_load_library`.
        """
        self._lib = lib
        self._driver: ctypes.c_void_p | None = None
        # Last derived Watts per power domain, keyed by domain handle address,
        # as {key: (watts, measured_at_monotonic)}. A measurement costs a real
        # wall-clock window, so the two calls making up a single snapshot share
        # one. See :meth:`get_power_watts`.
        self._energy_watts: dict[int, tuple[float | None, float]] = {}

    @staticmethod
    def _check(result: int) -> None:
        """Raise :class:`_ZeError` unless ``result`` is a success status."""
        if result != _ZE_RESULT_SUCCESS:
            raise _ZeError(result)

    def init(self) -> None:
        """Initialize sysman and cache the first driver handle.

        Raises
        ------
        _ZeError
            If ``zesInit`` fails or no sysman driver is available.
        """
        self._check(self._lib.zesInit(0))
        self._driver = self._get_driver()

    def _get_driver(self) -> ctypes.c_void_p:
        """Return the first sysman driver handle."""
        count = ctypes.c_uint32(0)
        self._check(self._lib.zesDriverGet(ctypes.byref(count), None))
        if count.value == 0:
            raise _ZeError(_ZE_RESULT_SUCCESS)
        drivers = (ctypes.c_void_p * count.value)()
        self._check(self._lib.zesDriverGet(ctypes.byref(count), drivers))
        return ctypes.c_void_p(drivers[0])

    def get_devices(self) -> list[ctypes.c_void_p]:
        """Return handles for every sysman device on the cached driver.

        Returns
        -------
        list[ctypes.c_void_p]
            One handle per detected Intel device (may be empty).
        """
        if self._driver is None:
            return []
        count = ctypes.c_uint32(0)
        self._check(self._lib.zesDeviceGet(self._driver, ctypes.byref(count), None))
        if count.value == 0:
            return []
        devices = (ctypes.c_void_p * count.value)()
        self._check(self._lib.zesDeviceGet(self._driver, ctypes.byref(count), devices))
        return [ctypes.c_void_p(devices[i]) for i in range(count.value)]

    def _enumerate(
        self, enum_fn: Any, handle: ctypes.c_void_p
    ) -> list[ctypes.c_void_p]:
        """Run a two-call ``zesDeviceEnum*`` pattern and return the handles."""
        count = ctypes.c_uint32(0)
        if enum_fn(handle, ctypes.byref(count), None) != _ZE_RESULT_SUCCESS:
            return []
        if count.value == 0:
            return []
        array = (ctypes.c_void_p * count.value)()
        if enum_fn(handle, ctypes.byref(count), array) != _ZE_RESULT_SUCCESS:
            return []
        return [ctypes.c_void_p(array[i]) for i in range(count.value)]

    def get_device_properties(self, handle: ctypes.c_void_p) -> dict[str, Any]:
        """Return identity properties for a device.

        Parameters
        ----------
        handle : ctypes.c_void_p
            A sysman device handle.

        Returns
        -------
        dict[str, Any]
            Model, brand, vendor, serial, board, driver version, subdevice
            count and whether the GPU is integrated into the CPU package.
        """
        props = _ZesDeviceProperties()
        props.stype = _ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES
        props.pNext = None
        self._check(self._lib.zesDeviceGetProperties(handle, ctypes.byref(props)))
        flags = int(props.core.flags)
        return {
            "model": _decode(props.modelName),
            "brand": _decode(props.brandName),
            "vendor": _decode(props.vendorName),
            "serial": _decode(props.serialNumber),
            "board": _decode(props.boardNumber),
            "driver_version": _decode(props.driverVersion),
            "num_subdevices": int(props.numSubdevices),
            "integrated": bool(flags & _ZE_DEVICE_PROPERTY_FLAG_INTEGRATED),
        }

    def get_pci_properties(self, handle: ctypes.c_void_p) -> dict[str, int | None]:
        """Return PCIe generation and width for a device.

        Parameters
        ----------
        handle : ctypes.c_void_p
            A sysman device handle.

        Returns
        -------
        dict[str, int or None]
            ``gen`` (clamped to a valid PCIe generation) and ``width``. Either
            may be None when the driver reports the value as unknown.
        """
        props = _ZesPciProperties()
        props.stype = _ZES_STRUCTURE_TYPE_PCI_PROPERTIES
        props.pNext = None
        self._check(self._lib.zesDevicePciGetProperties(handle, ctypes.byref(props)))
        gen = int(props.maxSpeed.gen)
        width = int(props.maxSpeed.width)
        return {
            "gen": gen if 1 <= gen <= 5 else None,
            "width": width if width > 0 else None,
        }

    def get_pci_bdf(self, handle: ctypes.c_void_p) -> str | None:
        """Return the device's PCI address in sysfs ``DDDD:BB:DD.F`` form.

        Parameters
        ----------
        handle : ctypes.c_void_p
            A sysman device handle.

        Returns
        -------
        str or None
            The lowercase PCI address (e.g. ``0000:03:00.0``) used to locate
            the device under ``/sys``. None only if the call fails upstream.
        """
        props = _ZesPciProperties()
        props.stype = _ZES_STRUCTURE_TYPE_PCI_PROPERTIES
        props.pNext = None
        self._check(self._lib.zesDevicePciGetProperties(handle, ctypes.byref(props)))
        addr = props.address
        return (
            f"{int(addr.domain):04x}:{int(addr.bus):02x}:"
            f"{int(addr.device):02x}.{int(addr.function):x}"
        )

    def get_memory(self, handle: ctypes.c_void_p) -> dict[str, int] | None:
        """Return aggregated memory usage across all memory modules.

        Parameters
        ----------
        handle : ctypes.c_void_p
            A sysman device handle.

        Returns
        -------
        dict[str, int] or None
            ``total``, ``used`` and ``free`` in bytes, or None if no memory
            module reports a usable size.
        """
        modules = self._enumerate(self._lib.zesDeviceEnumMemoryModules, handle)
        if not modules:
            return None
        total = 0
        free = 0
        for module in modules:
            state = _ZesMemState()
            state.stype = _ZES_STRUCTURE_TYPE_MEM_STATE
            state.pNext = None
            if self._lib.zesMemoryGetState(module, ctypes.byref(state)) != (
                _ZE_RESULT_SUCCESS
            ):
                continue
            total += int(state.size)
            free += int(state.free)
        if total == 0:
            return None
        return {"total": total, "used": total - free, "free": free}

    def get_temperature(self, handle: ctypes.c_void_p) -> float | None:
        """Return the GPU (or global) temperature in Celsius.

        Parameters
        ----------
        handle : ctypes.c_void_p
            A sysman device handle.

        Returns
        -------
        float or None
            Temperature in Celsius, or None if no sensor could be read.
        """
        sensors = self._enumerate(self._lib.zesDeviceEnumTemperatureSensors, handle)
        if not sensors:
            return None
        sensor = self._pick_temperature_sensor(sensors)
        if sensor is None:
            return None
        temperature = ctypes.c_double(0.0)
        if self._lib.zesTemperatureGetState(sensor, ctypes.byref(temperature)) != (
            _ZE_RESULT_SUCCESS
        ):
            return None
        return float(temperature.value)

    def _pick_temperature_sensor(
        self, sensors: list[ctypes.c_void_p]
    ) -> ctypes.c_void_p | None:
        """Choose the GPU sensor, falling back to global, then the first."""
        gpu = None
        global_sensor = None
        for sensor in sensors:
            props = _ZesTempProperties()
            props.stype = _ZES_STRUCTURE_TYPE_TEMP_PROPERTIES
            props.pNext = None
            if self._lib.zesTemperatureGetProperties(
                sensor, ctypes.byref(props)
            ) != _ZE_RESULT_SUCCESS:
                continue
            if props.type == _ZES_TEMP_SENSORS_GPU:
                gpu = sensor
            elif props.type == _ZES_TEMP_SENSORS_GLOBAL:
                global_sensor = sensor
        return gpu or global_sensor or sensors[0]

    def get_compute_utilization(self, handle: ctypes.c_void_p) -> float | None:
        """Return compute-engine utilization as a percentage.

        Two activity snapshots are taken ``_SAMPLE_INTERVAL_S`` apart and the
        percentage is derived from the active-time delta.

        Parameters
        ----------
        handle : ctypes.c_void_p
            A sysman device handle.

        Returns
        -------
        float or None
            Utilization percentage in ``[0.0, 100.0]``, or None on failure.
        """
        engines = self._enumerate(self._lib.zesDeviceEnumEngineGroups, handle)
        if not engines:
            return None
        engine = self._pick_engine(engines)
        if engine is None:
            return None
        first = _ZesEngineStats()
        if self._lib.zesEngineGetActivity(engine, ctypes.byref(first)) != (
            _ZE_RESULT_SUCCESS
        ):
            return None
        time.sleep(_SAMPLE_INTERVAL_S)
        second = _ZesEngineStats()
        if self._lib.zesEngineGetActivity(engine, ctypes.byref(second)) != (
            _ZE_RESULT_SUCCESS
        ):
            return None
        delta_time = int(second.timestamp) - int(first.timestamp)
        if delta_time <= 0:
            return 0.0
        delta_active = int(second.activeTime) - int(first.activeTime)
        return _clamp_percent(delta_active / delta_time * 100.0)

    def _pick_engine(
        self, engines: list[ctypes.c_void_p]
    ) -> ctypes.c_void_p | None:
        """Choose the compute-all engine group, falling back sensibly."""
        compute_all = None
        all_group = None
        first = None
        for engine in engines:
            props = _ZesEngineProperties()
            props.stype = _ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES
            props.pNext = None
            if self._lib.zesEngineGetProperties(
                engine, ctypes.byref(props)
            ) != _ZE_RESULT_SUCCESS:
                continue
            if first is None:
                first = engine
            if props.type == _ZES_ENGINE_GROUP_COMPUTE_ALL:
                compute_all = engine
            elif props.type == _ZES_ENGINE_GROUP_ALL:
                all_group = engine
        return compute_all or all_group or first

    def _read_energy(self, domain: ctypes.c_void_p) -> tuple[int, int] | None:
        """Read one energy-counter snapshot as ``(energy_uj, timestamp_us)``.

        Returns None if the driver rejects the query.
        """
        counter = _ZesPowerEnergyCounter()
        if self._lib.zesPowerGetEnergyCounter(domain, ctypes.byref(counter)) != (
            _ZE_RESULT_SUCCESS
        ):
            return None
        return int(counter.energy), int(counter.timestamp)

    def _poll_energy(self, domain: ctypes.c_void_p, seconds: float) -> None:
        """Touch the energy counter repeatedly for ``seconds``.

        Each read resets the device's autosuspend timer, so polling faster than
        that delay holds the device awake. Used both to burn off the
        post-resume settling period and to keep the device from suspending
        underneath a measurement window. Reads are cheap and harmless.
        """
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(min(_POLL_INTERVAL_S, max(0.0, deadline - time.monotonic())))
            self._read_energy(domain)

    def get_power_watts(self, handle: ctypes.c_void_p) -> float | None:
        """Return power draw in watts, measured over a short controlled window.

        Purpose: derive instantaneous power from the monotonic energy counter,
        controlling the device's runtime-PM state so the reading is trustworthy
        (see the notes on ``_RESUME_SETTLE_S`` and ``_INCOHERENCE_RATIO``).
        Input:   ``handle`` — a sysman device handle.
        Output:  power draw in watts, or None if unavailable or if the device
                 suspended during the measurement window.
        """
        domains = self._enumerate(self._lib.zesDeviceEnumPowerDomains, handle)
        if not domains:
            return None
        domain = domains[0]
        # Second domain, where present, is the package contained within the
        # card domain. Used only to sanity-check the reading; see
        # :func:`_counters_incoherent`.
        package = domains[1] if len(domains) > 1 else None
        key = domain.value or 0

        cached = self._energy_watts.get(key)
        if cached is not None and time.monotonic() - cached[1] < _CACHE_TTL_S:
            return cached[0]

        try:
            bdf = self.get_pci_bdf(handle)
        except Exception:
            bdf = None

        # Wake the device and let the counter settle. With no runtime-PM
        # interface to consult we cannot tell whether it was asleep, so settle
        # unconditionally.
        if bdf is None or _runtime_pm_status(bdf) != "active":
            self._poll_energy(domain, _RESUME_SETTLE_S)

        pm_before = _read_runtime_pm(bdf) if bdf else None
        first = self._read_energy(domain)
        first_package = self._read_energy(package) if package else None
        if first is None:
            return None
        self._poll_energy(domain, _MEASURE_WINDOW_S)
        second = self._read_energy(domain)
        second_package = self._read_energy(package) if package else None
        if second is None:
            return None
        pm_after = _read_runtime_pm(bdf) if bdf else None

        def _reject() -> None:
            self._energy_watts[key] = (None, time.monotonic())

        # Polling should have held the device awake; if it suspended anyway the
        # counter accrues a large fictitious figure over that period, so the
        # window has to be thrown away.
        if (
            pm_before is not None
            and pm_after is not None
            and pm_after[1] != pm_before[1]
        ):
            _reject()
            return None

        watts = _energy_to_watts(second[0] - first[0], second[1] - first[1])
        if watts is None:
            return None

        # Cross-check against the package domain, which cannot exceed the card
        # domain. When it does, the counter has resumed into its corrupted
        # state and this window is fiction.
        if first_package is not None and second_package is not None:
            package_watts = _energy_to_watts(
                second_package[0] - first_package[0],
                second_package[1] - first_package[1],
            )
            if package_watts is not None and _counters_incoherent(
                watts, package_watts
            ):
                _reject()
                return None

        self._energy_watts[key] = (watts, time.monotonic())
        return watts

    def get_energy_joules(self, handle: ctypes.c_void_p) -> float | None:
        """Return the device's cumulative energy counter in Joules.

        This is the raw monotonic total, not a rate, and costs exactly one
        counter read. Taking the difference between a reading at the start of
        an interval and one at the end gives that interval's energy without the
        sampling error of integrating :meth:`get_power_watts` figures.

        .. warning::
           Not yet safe for totalling energy over long intervals on this
           hardware. The counter accrues a large fictitious amount while the
           device is runtime-suspended (see the notes on ``_RESUME_SETTLE_S``),
           and a start/end pair cannot tell how much of the interval was spent
           asleep. Using this for accounting needs sleep-time tracking across
           the interval, which is not implemented.

        Parameters
        ----------
        handle : ctypes.c_void_p
            A sysman device handle.

        Returns
        -------
        float or None
            Cumulative energy in Joules, or None if unavailable.
        """
        domains = self._enumerate(self._lib.zesDeviceEnumPowerDomains, handle)
        if not domains:
            return None
        domain = domains[0]
        current = self._read_energy(domain)
        if current is None:
            return None
        return current[0] / 1_000_000.0

    def get_power_limit_watts(self, handle: ctypes.c_void_p) -> float | None:
        """Return the sustained power limit (TDP) in Watts, if known.

        Parameters
        ----------
        handle : ctypes.c_void_p
            A sysman device handle.

        Returns
        -------
        float or None
            Power limit in Watts, or None if unavailable/unknown.
        """
        domains = self._enumerate(self._lib.zesDeviceEnumPowerDomains, handle)
        if not domains:
            return None
        props = _ZesPowerProperties()
        props.stype = _ZES_STRUCTURE_TYPE_POWER_PROPERTIES
        props.pNext = None
        if self._lib.zesPowerGetProperties(domains[0], ctypes.byref(props)) != (
            _ZE_RESULT_SUCCESS
        ):
            return None
        limit = int(props.defaultLimit)
        if limit < 0:
            return None
        return limit / 1000.0

    def get_throttle_reasons(self, handle: ctypes.c_void_p) -> list[str]:
        """Return active throttle reasons for the GPU frequency domain.

        Parameters
        ----------
        handle : ctypes.c_void_p
            A sysman device handle.

        Returns
        -------
        list[str]
            Active throttle reasons, empty if not throttling or on failure.
        """
        domains = self._enumerate(self._lib.zesDeviceEnumFrequencyDomains, handle)
        if not domains:
            return []
        frequency = self._pick_frequency_domain(domains)
        if frequency is None:
            return []
        state = _ZesFreqState()
        state.stype = _ZES_STRUCTURE_TYPE_FREQ_STATE
        state.pNext = None
        if self._lib.zesFrequencyGetState(frequency, ctypes.byref(state)) != (
            _ZE_RESULT_SUCCESS
        ):
            return []
        return _decode_throttle(int(state.throttleReasons))

    def _pick_frequency_domain(
        self, domains: list[ctypes.c_void_p]
    ) -> ctypes.c_void_p | None:
        """Choose the GPU-core frequency domain, falling back to the first."""
        gpu = None
        first = None
        for domain in domains:
            props = _ZesFreqProperties()
            props.stype = _ZES_STRUCTURE_TYPE_FREQ_PROPERTIES
            props.pNext = None
            if self._lib.zesFrequencyGetProperties(
                domain, ctypes.byref(props)
            ) != _ZE_RESULT_SUCCESS:
                continue
            if first is None:
                first = domain
            if props.type == _ZES_FREQ_DOMAIN_GPU:
                gpu = domain
        return gpu or first

    def shutdown(self) -> None:
        """Release cached state.

        The Level Zero sysman API has no explicit teardown call, so this simply
        drops the cached driver handle and the cached power figures (whose
        domain handles do not outlive the driver).
        """
        self._driver = None
        self._energy_watts.clear()


class IntelBackend(AcceleratorBackend):
    """Backend for Intel GPU information via the Level Zero sysman API."""

    def __init__(self) -> None:
        """Load Level Zero, initialize sysman and enumerate devices.

        Raises
        ------
        OSError
            If the Level Zero loader cannot be found.
        _ZeError
            If sysman initialization fails.
        """
        self._lib = _load_library()
        self._sysman = _IntelSysman(self._lib)
        self._sysman.init()
        self._devices = self._sysman.get_devices()

    def _valid(self, index: int) -> bool:
        """Return True if ``index`` refers to a known device."""
        return 0 <= index < len(self._devices)

    def _bytes_to_gb(self, bytes_value: int) -> int:
        """Convert a byte count to whole gigabytes."""
        return int(bytes_value / (1024**3))

    def is_available(self) -> bool:
        """Check if Intel GPUs are available.

        Returns
        -------
        bool
            True if at least one Intel GPU was detected.
        """
        try:
            return self.get_device_count() > 0
        except Exception:
            return False

    def get_device_count(self) -> int:
        """Get the number of Intel GPUs.

        Returns
        -------
        int
            Number of Intel GPUs detected.
        """
        return len(self._devices)

    def list_devices(self) -> list[GPUInfo]:
        """List all Intel GPUs with their specifications.

        Returns
        -------
        list[GPUInfo]
            One entry per detected Intel GPU.
        """
        devices: list[GPUInfo] = []
        for index, handle in enumerate(self._devices):
            props = self._safe(self._sysman.get_device_properties, handle, default={})
            pci = self._safe(self._sysman.get_pci_properties, handle, default={})
            memory = self._safe(self._sysman.get_memory, handle, default=None)

            model = props.get("model") or ""
            if not model or model.lower() == "unknown":
                brand = props.get("brand") or ""
                model = brand if brand and brand.lower() != "unknown" else "Intel GPU"

            serial = props.get("serial") or ""
            uuid = serial if serial and serial.lower() != "unknown" else None

            driver_version = props.get("driver_version") or None
            if driver_version and driver_version.lower() == "unknown":
                driver_version = None

            memory_gb = self._bytes_to_gb(memory["total"]) if memory else 0

            devices.append(
                GPUInfo(
                    index=index,
                    model=model,
                    memory_gb=memory_gb,
                    uuid=uuid,
                    compute_capability=None,
                    pcie_gen=pci.get("gen"),
                    driver_version=driver_version,
                    extra_metrics={
                        "vendor_name": props.get("vendor") or None,
                        "board_number": props.get("board") or None,
                        "serial_number": serial or None,
                        "num_subdevices": props.get("num_subdevices"),
                        "pcie_width": pci.get("width"),
                        "backend": "level_zero_sysman",
                    },
                )
            )
        return devices

    @staticmethod
    def _safe(func: Any, *args: Any, default: Any) -> Any:
        """Call ``func`` returning ``default`` if it raises."""
        try:
            return func(*args)
        except Exception:
            return default

    def get_temperature(self, index: int) -> float | None:
        """Get GPU temperature in degrees Celsius.

        Tries the Level Zero sensor first. When the driver reports no
        temperature (a known gap on some Intel firmware/drivers), falls back
        to the kernel hwmon sensor for this device, located by its PCI
        address. Returns None only if neither source has a reading.

        Parameters
        ----------
        index : int
            GPU index (0-based).

        Returns
        -------
        float or None
            Temperature in Celsius, or None if unavailable.
        """
        if not self._valid(index):
            return None
        return _temperature_with_fallback(self._sysman, self._devices[index])

    def get_memory_usage(self, index: int) -> dict | None:
        """Get current GPU memory usage.

        Parameters
        ----------
        index : int
            GPU index (0-based).

        Returns
        -------
        dict or None
            ``total``, ``used`` and ``free`` in bytes, or None if unavailable.
        """
        if not self._valid(index):
            return None
        return self._safe(
            self._sysman.get_memory, self._devices[index], default=None
        )

    def get_utilization(self, index: int) -> dict | None:
        """Get GPU utilization percentages.

        Parameters
        ----------
        index : int
            GPU index (0-based).

        Returns
        -------
        dict or None
            ``gpu`` (compute engine %) and ``memory`` (allocated %), both in
            ``[0, 100]``, or None if unavailable.
        """
        if not self._valid(index):
            return None
        handle = self._devices[index]
        gpu = self._safe(
            self._sysman.get_compute_utilization, handle, default=None
        )
        if gpu is None:
            return None
        memory = self._safe(self._sysman.get_memory, handle, default=None)
        memory_percent = 0.0
        if memory and memory["total"] > 0:
            memory_percent = _clamp_percent(
                memory["used"] / memory["total"] * 100.0
            )
        return {"gpu": float(gpu), "memory": float(memory_percent)}

    def get_pytorch_device_string(self, device_id: int) -> str:
        """Get PyTorch device string for Intel GPUs.

        Parameters
        ----------
        device_id : int
            GPU index (0-based).

        Returns
        -------
        str
            PyTorch device string (e.g., ``'xpu:0'``).
        """
        return f"xpu:{device_id}"

    def get_power_usage(self, index: int) -> float | None:
        """Get current GPU power usage in Watts.

        Parameters
        ----------
        index : int
            GPU index (0-based).

        Returns
        -------
        float or None
            Power usage in Watts, or None if unavailable.
        """
        if not self._valid(index):
            return None
        return self._safe(
            self._sysman.get_power_watts, self._devices[index], default=None
        )

    def get_throttle_reasons(self, index: int) -> list[str]:
        """Get current GPU throttling reasons.

        Parameters
        ----------
        index : int
            GPU index (0-based).

        Returns
        -------
        list[str]
            Active throttle reasons, empty list if not throttling.
        """
        if not self._valid(index):
            return []
        return self._safe(
            self._sysman.get_throttle_reasons, self._devices[index], default=[]
        )

    def get_driver_version(self) -> str | None:
        """Get the Intel GPU driver version.

        Returns
        -------
        str or None
            Driver version string, or None if unavailable.
        """
        if not self._devices:
            return None
        props = self._safe(
            self._sysman.get_device_properties, self._devices[0], default={}
        )
        version = props.get("driver_version") or None
        if version and version.lower() == "unknown":
            return None
        return version

    def get_topology(self) -> str:
        """Get GPU interconnect topology.

        Returns
        -------
        str
            Interconnect type. Intel discrete GPUs use PCIe (Max-series parts
            may additionally use Xe Link, not exposed by this trimmed API).
        """
        return "PCIe"

    def get_distributed_backend(self) -> str:
        """Return the torch.distributed backend for Intel GPUs.

        Returns
        -------
        str
            ``'ccl'`` (oneCCL, via the Intel torch-ccl bindings).
        """
        return "ccl"

    def shutdown(self) -> None:
        """Cleanup and shutdown the Intel backend."""
        try:
            self._sysman.shutdown()
        except Exception:
            pass
