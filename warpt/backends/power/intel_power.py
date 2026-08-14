"""Intel GPU power monitoring backend using Level Zero sysman.

Provides per-GPU power information for Intel GPUs (integrated Xe, Arc, Flex and
Max) derived from the hardware energy counter exposed by the Level Zero sysman
API. It reuses the :class:`~warpt.backends.intel._IntelSysman` ctypes wrapper to
keep the FFI layer in one place.

Readings are reported with ``PowerSource.LEVEL_ZERO``.

Each reading is tagged with the vendor and with whether the GPU is integrated
into the CPU package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from warpt.backends.power.base import PowerBackend
from warpt.models.power_models import (
    DomainPower,
    GPUPowerInfo,
    PowerDomain,
    PowerSource,
)

_T = TypeVar("_T")

# Vendor tag stamped on every reading so consumers can distinguish an Intel
# GPU 0 from an NVIDIA GPU 0 (both backends number devices from 0).
_VENDOR = "intel"

try:
    from warpt.backends.intel import (
        _IntelSysman,
        _load_library,
        _temperature_with_fallback,
    )

    LEVEL_ZERO_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    _IntelSysman = None  # type: ignore[assignment,misc]
    _load_library = None  # type: ignore[assignment]
    _temperature_with_fallback = None  # type: ignore[assignment]
    LEVEL_ZERO_AVAILABLE = False


class IntelPowerBackend(PowerBackend):
    """Backend for Intel GPU power monitoring via Level Zero sysman.

    Provides per-GPU power readings and detailed power information sourced from
    the device energy counters.
    """

    def __init__(self) -> None:
        """Initialize the Intel power backend (lazy; see :meth:`initialize`)."""
        self._initialized = False
        self._sysman: _IntelSysman | None = None
        self._devices: list = []
        # Per-device static identity, parallel to ``_devices``. Cached at
        # initialize() so the sampling loop makes no extra FFI calls per
        # snapshot for values that cannot change.
        self._static: list[dict[str, Any]] = []

    def is_available(self) -> bool:
        """Check if Intel GPUs are available.

        Returns
        -------
        bool
            True if Level Zero is loadable and at least one device is detected.
        """
        if not LEVEL_ZERO_AVAILABLE:
            return False
        try:
            lib = _load_library()
            sysman = _IntelSysman(lib)
            sysman.init()
            return len(sysman.get_devices()) > 0
        except Exception:
            return False

    def get_source(self) -> PowerSource:
        """Get the power source type.

        Returns
        -------
        PowerSource
            ``PowerSource.LEVEL_ZERO`` — the reading comes from the Level Zero
            sysman hardware energy counter.
        """
        return PowerSource.LEVEL_ZERO

    def initialize(self) -> bool:
        """Initialize Level Zero sysman and enumerate devices.

        Returns
        -------
        bool
            True if initialization succeeded.
        """
        if self._initialized:
            return True
        if not LEVEL_ZERO_AVAILABLE:
            return False
        try:
            lib = _load_library()
            sysman = _IntelSysman(lib)
            sysman.init()
            self._sysman = sysman
            self._devices = sysman.get_devices()
            self._static = [self._static_info(handle) for handle in self._devices]
            self._initialized = True
            return True
        except Exception:
            return False

    def get_power_readings(self) -> list[DomainPower]:
        """Get power readings from all Intel GPUs.

        Returns
        -------
        list[DomainPower]
            One entry per GPU that reports a power value.
        """
        if not self._initialized and not self.initialize():
            return []
        assert self._sysman is not None
        readings: list[DomainPower] = []
        for idx, handle in enumerate(self._devices):
            try:
                watts = self._sysman.get_power_watts(handle)
            except Exception:
                watts = None
            if watts is None:
                continue
            readings.append(
                DomainPower(
                    domain=PowerDomain.GPU,
                    power_watts=watts,
                    energy_joules=None,
                    source=self.get_source(),
                    metadata={
                        "gpu_index": idx,
                        "vendor": _VENDOR,
                        "backend": "level_zero_sysman",
                        "integrated": self._integrated(idx),
                    },
                )
            )
        return readings

    def get_gpu_power_info(self) -> list[GPUPowerInfo]:
        """Get detailed power information for all Intel GPUs.

        Each entry carries ``metadata["integrated"]``, which
        :class:`~warpt.backends.power.factory.PowerMonitor` uses to avoid
        double-counting an integrated GPU whose power is already inside the CPU
        package (RAPL) reading.

        Returns
        -------
        list[GPUPowerInfo]
            Comprehensive per-GPU power, utilization and temperature data.
        """
        if not self._initialized and not self.initialize():
            return []
        assert self._sysman is not None
        gpus: list[GPUPowerInfo] = []
        for idx, handle in enumerate(self._devices):
            watts = self._safe(self._sysman.get_power_watts, handle, default=None)
            name = self._device_name(idx)
            limit = self._safe(self._sysman.get_power_limit_watts, handle, default=None)
            gpu_util = self._safe(
                self._sysman.get_compute_utilization, handle, default=None
            )
            memory = self._safe(self._sysman.get_memory, handle, default=None)
            memory_util = 0.0
            if memory and memory["total"] > 0:
                memory_util = memory["used"] / memory["total"] * 100.0
            temperature = _temperature_with_fallback(self._sysman, handle)
            gpus.append(
                GPUPowerInfo(
                    index=idx,
                    name=name,
                    vendor=_VENDOR,
                    power_watts=watts if watts is not None else 0.0,
                    power_limit_watts=limit,
                    utilization_percent=gpu_util if gpu_util is not None else 0.0,
                    memory_utilization_percent=memory_util,
                    temperature_celsius=temperature,
                    processes=[],
                    metadata={
                        "backend": "level_zero_sysman",
                        "integrated": self._integrated(idx),
                    },
                )
            )
        return gpus

    def get_total_gpu_power(self) -> float:
        """Get total power consumption across all Intel GPUs.

        Returns
        -------
        float
            Total GPU power in Watts.
        """
        if not self._initialized and not self.initialize():
            return 0.0
        assert self._sysman is not None
        total = 0.0
        for handle in self._devices:
            watts = self._safe(self._sysman.get_power_watts, handle, default=None)
            if watts:
                total += watts
        return total

    def cleanup(self) -> None:
        """Release Level Zero sysman state."""
        if self._sysman is not None:
            try:
                self._sysman.shutdown()
            except Exception:
                pass
        self._initialized = False
        self._devices = []
        self._static = []
        self._sysman = None

    def _static_info(self, handle: object) -> dict[str, Any]:
        """Read the immutable identity of one device.

        Called once per device from :meth:`initialize`. Both values are fixed
        for the life of the process, so caching them keeps the per-snapshot
        path free of identity FFI calls.

        Returns
        -------
        dict[str, Any]
            ``name`` (display name, defaulting to 'Intel GPU') and
            ``integrated`` (True when the GPU is fused into the CPU package).
        """
        assert self._sysman is not None
        props: dict[str, Any] = self._safe(
            self._sysman.get_device_properties, handle, default={}
        )
        name = props.get("model") or props.get("brand") or ""
        if not name or name.lower() == "unknown":
            name = "Intel GPU"
        return {"name": name, "integrated": bool(props.get("integrated", False))}

    def _device_name(self, index: int) -> str:
        """Return the cached display name for a device index."""
        if 0 <= index < len(self._static):
            return str(self._static[index]["name"])
        return "Intel GPU"

    def _integrated(self, index: int) -> bool:
        """Return whether a device index is an integrated (in-package) GPU.

        Defaults to False for an unknown index: treating a discrete GPU as
        integrated would silently drop its power from the system total, which
        is the worse failure of the two.
        """
        if 0 <= index < len(self._static):
            return bool(self._static[index]["integrated"])
        return False

    @staticmethod
    def _safe(func: Callable[..., _T], *args: object, default: _T) -> _T:
        """Call ``func`` returning ``default`` if it raises."""
        try:
            return func(*args)
        except Exception:
            return default
