"""Intel GPU power monitoring backend using Level Zero sysman.

Provides per-GPU power information for Intel GPUs (integrated Xe, Arc, Flex and
Max) derived from the hardware energy counter exposed by the Level Zero sysman
API. It reuses the :class:`~warpt.backends.intel._IntelSysman` ctypes wrapper so
that the FFI layer lives in exactly one place.

Note: the Level Zero energy counter is a true hardware measurement, but the
:class:`~warpt.models.power_models.PowerSource` enum has no Level Zero / Intel
member, so :meth:`IntelPowerBackend.get_source` reports ``ESTIMATED``. See
``questions.yaml`` for the follow-up on adding a dedicated source.
"""

from __future__ import annotations

from warpt.backends.power.base import PowerBackend
from warpt.models.power_models import (
    DomainPower,
    GPUPowerInfo,
    PowerDomain,
    PowerSource,
)

try:
    from warpt.backends.intel import _IntelSysman, _load_library

    LEVEL_ZERO_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    _IntelSysman = None  # type: ignore[assignment,misc]
    _load_library = None  # type: ignore[assignment]
    LEVEL_ZERO_AVAILABLE = False


class IntelPowerBackend(PowerBackend):
    """Backend for Intel GPU power monitoring via Level Zero sysman.

    Provides per-GPU power readings and detailed power information sourced from
    the device energy counters.
    """

    def __init__(self) -> None:
        """Initialize the Intel power backend (lazy; see :meth:`initialize`)."""
        self._initialized = False
        self._sysman: object | None = None
        self._devices: list = []

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
            ``PowerSource.ESTIMATED`` — the enum lacks a Level Zero member even
            though the reading comes from a hardware energy counter.
        """
        return PowerSource.ESTIMATED

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
                    metadata={"gpu_index": idx, "backend": "level_zero_sysman"},
                )
            )
        return readings

    def get_gpu_power_info(self) -> list[GPUPowerInfo]:
        """Get detailed power information for all Intel GPUs.

        Returns
        -------
        list[GPUPowerInfo]
            Comprehensive per-GPU power, utilization and temperature data.
        """
        if not self._initialized and not self.initialize():
            return []
        gpus: list[GPUPowerInfo] = []
        for idx, handle in enumerate(self._devices):
            watts = self._safe(self._sysman.get_power_watts, handle, default=None)
            name = self._device_name(handle)
            limit = self._safe(
                self._sysman.get_power_limit_watts, handle, default=None
            )
            gpu_util = self._safe(
                self._sysman.get_compute_utilization, handle, default=None
            )
            memory = self._safe(self._sysman.get_memory, handle, default=None)
            memory_util = 0.0
            if memory and memory["total"] > 0:
                memory_util = memory["used"] / memory["total"] * 100.0
            temperature = self._safe(
                self._sysman.get_temperature, handle, default=None
            )
            gpus.append(
                GPUPowerInfo(
                    index=idx,
                    name=name,
                    power_watts=watts if watts is not None else 0.0,
                    power_limit_watts=limit,
                    utilization_percent=gpu_util if gpu_util is not None else 0.0,
                    memory_utilization_percent=memory_util,
                    temperature_celsius=temperature,
                    processes=[],
                    metadata={"backend": "level_zero_sysman"},
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
        self._sysman = None

    def _device_name(self, handle: object) -> str:
        """Return a display name for a device, falling back to 'Intel GPU'."""
        props = self._safe(self._sysman.get_device_properties, handle, default={})
        name = props.get("model") or props.get("brand") or "Intel GPU"
        if not name or name.lower() == "unknown":
            return "Intel GPU"
        return name

    @staticmethod
    def _safe(func: object, *args: object, default: object) -> object:
        """Call ``func`` returning ``default`` if it raises."""
        try:
            return func(*args)  # type: ignore[operator]
        except Exception:
            return default
