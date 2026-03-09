"""AMD GPU power monitoring backend using amdsmi.

Provides detailed power information for AMD GPUs including:
- Per-GPU power consumption
- Power limits and constraints
- Per-process GPU utilization and memory usage
- Temperature information

This enables GPU power attribution for AMD ROCm systems.
"""

from __future__ import annotations

from typing import Any

from warpt.backends.power.base import PowerBackend
from warpt.models.power_models import (
    DomainPower,
    GPUPowerInfo,
    PowerDomain,
    PowerSource,
)

# Optional amdsmi import
try:
    import amdsmi

    AMDSMI_AVAILABLE = True
except ImportError:
    amdsmi = None  # type: ignore[assignment]
    AMDSMI_AVAILABLE = False


class AmdPowerBackend(PowerBackend):
    """Backend for AMD GPU power monitoring via amdsmi.

    Provides accurate GPU power readings and per-process GPU usage
    for power attribution on AMD ROCm systems.
    """

    def __init__(self) -> None:
        """Initialize the AMD power backend."""
        self._initialized = False
        self._processor_handles: list = []

    def is_available(self) -> bool:
        """Check if AMD GPUs are available for power monitoring.

        Returns
        -------
        bool
            True if amdsmi is available and GPUs are detected.
        """
        if not AMDSMI_AVAILABLE:
            return False

        try:
            amdsmi.amdsmi_init(amdsmi.AmdSmiInitFlags.INIT_AMD_GPUS)
            handles = amdsmi.amdsmi_get_processor_handles()
            amdsmi.amdsmi_shut_down()
            return len(handles) > 0
        except Exception:
            return False

    def get_source(self) -> PowerSource:
        """Get the power source type.

        Returns
        -------
        PowerSource
            ``PowerSource.ROCM_SMI``.
        """
        return PowerSource.ROCM_SMI

    def initialize(self) -> bool:
        """Initialize amdsmi for power monitoring.

        Returns
        -------
        bool
            True if initialization succeeded.
        """
        if self._initialized:
            return True

        if not AMDSMI_AVAILABLE:
            return False

        try:
            amdsmi.amdsmi_init(amdsmi.AmdSmiInitFlags.INIT_AMD_GPUS)
            self._processor_handles = (
                amdsmi.amdsmi_get_processor_handles()
            )
            self._initialized = True
            return True
        except Exception:
            return False

    def get_power_readings(self) -> list[DomainPower]:
        """Get power readings from all AMD GPUs.

        Returns
        -------
        list[DomainPower]
            One ``DomainPower`` per GPU with current power draw.
        """
        if not self._initialized:
            if not self.initialize():
                return []

        readings: list[DomainPower] = []

        for idx, handle in enumerate(self._processor_handles):
            try:
                power_info = amdsmi.amdsmi_get_power_info(handle)
                power_watts = self._extract_power_watts(power_info)
                if power_watts is None:
                    continue

                # Get GPU name for metadata
                name = self._get_gpu_name(handle)

                readings.append(
                    DomainPower(
                        domain=PowerDomain.GPU,
                        power_watts=power_watts,
                        source=PowerSource.ROCM_SMI,
                        metadata={
                            "gpu_index": idx,
                            "gpu_name": name,
                        },
                    )
                )
            except Exception:
                continue

        return readings

    def get_gpu_power_info(self) -> list[GPUPowerInfo]:
        """Get detailed power information for all AMD GPUs.

        Returns
        -------
        list[GPUPowerInfo]
            Comprehensive power data per GPU.
        """
        if not self._initialized:
            if not self.initialize():
                return []

        gpus: list[GPUPowerInfo] = []

        for idx, handle in enumerate(self._processor_handles):
            try:
                name = self._get_gpu_name(handle)

                # Power draw and limit
                power_watts = 0.0
                power_limit_watts = None
                try:
                    power_info = amdsmi.amdsmi_get_power_info(handle)
                    extracted = self._extract_power_watts(power_info)
                    if extracted is not None:
                        power_watts = extracted
                    limit = power_info.get("power_limit", "N/A")
                    if isinstance(limit, int | float) and limit > 0:
                        power_limit_watts = float(limit)
                except Exception:
                    pass

                # Fallback: power cap info (microwatts)
                if power_limit_watts is None:
                    try:
                        cap_info = amdsmi.amdsmi_get_power_cap_info(
                            handle
                        )
                        cap_uw = cap_info.get("power_cap", 0)
                        if isinstance(cap_uw, int) and cap_uw > 0:
                            power_limit_watts = cap_uw / 1_000_000.0
                    except Exception:
                        pass

                # Utilization
                gpu_util = 0.0
                mem_util = 0.0
                try:
                    activity = amdsmi.amdsmi_get_gpu_activity(handle)
                    gfx = activity.get("gfx_activity", "N/A")
                    umc = activity.get("umc_activity", "N/A")
                    if isinstance(gfx, int | float):
                        gpu_util = float(gfx)
                    if isinstance(umc, int | float):
                        mem_util = float(umc)
                except Exception:
                    pass

                # Temperature (millidegrees C -> degrees C)
                temp = self._get_temperature(handle)

                # Per-process info
                processes = self._get_gpu_processes(handle)

                gpus.append(
                    GPUPowerInfo(
                        index=idx,
                        name=name,
                        power_watts=power_watts,
                        power_limit_watts=power_limit_watts,
                        utilization_percent=gpu_util,
                        memory_utilization_percent=mem_util,
                        temperature_celsius=temp,
                        processes=processes,
                    )
                )
            except Exception:
                continue

        return gpus

    def get_total_gpu_power(self) -> float:
        """Get total power consumption across all AMD GPUs.

        Returns
        -------
        float
            Total GPU power in watts.
        """
        if not self._initialized:
            if not self.initialize():
                return 0.0

        total_power = 0.0
        for handle in self._processor_handles:
            try:
                power_info = amdsmi.amdsmi_get_power_info(handle)
                extracted = self._extract_power_watts(power_info)
                if extracted is not None:
                    total_power += extracted
            except Exception:
                continue

        return total_power

    def cleanup(self) -> None:
        """Shutdown amdsmi and clean up resources."""
        if self._initialized and AMDSMI_AVAILABLE:
            try:
                amdsmi.amdsmi_shut_down()
            except Exception:
                pass
        self._initialized = False
        self._processor_handles = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_power_watts(
        power_info: dict[str, Any],
    ) -> float | None:
        """Extract power in watts from an amdsmi power_info dict.

        Parameters
        ----------
        power_info : dict
            Dict returned by ``amdsmi_get_power_info``.

        Returns
        -------
        float or None
            Power in watts, or None if no valid reading found.
        """
        for key in (
            "current_socket_power",
            "average_socket_power",
            "socket_power",
        ):
            val = power_info.get(key, "N/A")
            if isinstance(val, int | float) and val > 0:
                return float(val)
        return None

    @staticmethod
    def _get_gpu_name(handle: Any) -> str:
        """Get the marketing name for an AMD GPU.

        Parameters
        ----------
        handle
            AMD SMI processor handle.

        Returns
        -------
        str
            GPU marketing name, or ``'AMD GPU'`` as fallback.
        """
        try:
            asic_info = amdsmi.amdsmi_get_gpu_asic_info(handle)
            market_name = asic_info.get("market_name", "N/A")
            if market_name and market_name != "N/A":
                return str(market_name)
        except Exception:
            pass
        return "AMD GPU"

    @staticmethod
    def _get_temperature(handle: Any) -> float | None:
        """Read edge or hotspot temperature in degrees Celsius.

        Parameters
        ----------
        handle
            AMD SMI processor handle.

        Returns
        -------
        float or None
            Temperature in degrees Celsius, or None.
        """
        for sensor in (
            amdsmi.AmdSmiTemperatureType.EDGE,
            amdsmi.AmdSmiTemperatureType.HOTSPOT,
        ):
            try:
                temp_mc = amdsmi.amdsmi_get_temp_metric(
                    handle,
                    sensor,
                    amdsmi.AmdSmiTemperatureMetric.CURRENT,
                )
                return float(temp_mc) / 1000.0
            except Exception:
                continue
        return None

    @staticmethod
    def _get_gpu_processes(
        handle: Any,
    ) -> list[dict[str, Any]]:
        """Get processes using a specific AMD GPU.

        Parameters
        ----------
        handle
            AMD SMI processor handle.

        Returns
        -------
        list[dict[str, Any]]
            List of process information dictionaries.
        """
        processes: list[dict[str, Any]] = []
        try:
            proc_list = amdsmi.amdsmi_get_gpu_process_list(handle)
            for proc in proc_list:
                try:
                    pid = proc.get("pid", 0)
                    name = proc.get("name", f"pid_{pid}")
                    if name == "N/A":
                        name = f"pid_{pid}"
                    mem_bytes = proc.get("mem", 0)
                    memory_mb = (
                        mem_bytes / (1024 * 1024) if mem_bytes else 0
                    )
                    processes.append(
                        {
                            "pid": pid,
                            "name": name,
                            "gpu_memory_mb": round(memory_mb, 1),
                        }
                    )
                except Exception:
                    continue
        except Exception:
            pass
        return processes
