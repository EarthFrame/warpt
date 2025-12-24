"""NVIDIA GPU power monitoring backend using NVML.

Provides detailed power information for NVIDIA GPUs including:
- Per-GPU power consumption
- Power limits and constraints
- Per-process GPU utilization and memory usage
- Temperature and throttling information

This enables accurate GPU power attribution to running processes.
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

# Optional pynvml import
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None
    PYNVML_AVAILABLE = False


class NvidiaPowerBackend(PowerBackend):
    """Backend for NVIDIA GPU power monitoring via NVML.

    Provides accurate GPU power readings and per-process GPU usage
    for power attribution.
    """

    def __init__(self) -> None:
        """Initialize the NVIDIA power backend."""
        self._initialized = False
        self._device_count = 0

    def is_available(self) -> bool:
        """Check if NVIDIA GPUs are available.

        Returns:
            True if NVML is available and GPUs are detected.
        """
        if not PYNVML_AVAILABLE:
            return False

        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            return bool(count > 0)
        except Exception:
            return False

    def get_source(self) -> PowerSource:
        """Get the power source type.

        Returns:
            PowerSource.NVML
        """
        return PowerSource.NVML

    def initialize(self) -> bool:
        """Initialize NVML.

        Returns:
            True if initialization succeeded.
        """
        if self._initialized:
            return True

        if not PYNVML_AVAILABLE:
            return False

        try:
            pynvml.nvmlInit()
            self._device_count = pynvml.nvmlDeviceGetCount()
            self._initialized = True
            return True
        except Exception:
            return False

    def get_power_readings(self) -> list[DomainPower]:
        """Get power readings from all NVIDIA GPUs.

        Returns:
            List of DomainPower objects, one per GPU.
        """
        if not self._initialized:
            if not self.initialize():
                return []

        readings: list[DomainPower] = []

        for idx in range(self._device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_watts = power_mw / 1000.0

                # Get GPU name for metadata
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode(errors="ignore")

                readings.append(
                    DomainPower(
                        domain=PowerDomain.GPU,
                        power_watts=power_watts,
                        source=PowerSource.NVML,
                        metadata={
                            "gpu_index": idx,
                            "gpu_name": name,
                            "raw_mw": power_mw,
                        },
                    )
                )
            except Exception:
                continue

        return readings

    def get_gpu_power_info(self) -> list[GPUPowerInfo]:
        """Get detailed power information for all GPUs.

        Returns:
            List of GPUPowerInfo objects with comprehensive GPU data.
        """
        if not self._initialized:
            if not self.initialize():
                return []

        gpus: list[GPUPowerInfo] = []

        for idx in range(self._device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

                # Basic info
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode(errors="ignore")

                # Power
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_watts = power_mw / 1000.0

                # Power limit
                try:
                    power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                    power_limit_watts = power_limit_mw / 1000.0
                except Exception:
                    power_limit_watts = None

                # Utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = float(util.gpu)
                    mem_util = float(util.memory)
                except Exception:
                    gpu_util = 0.0
                    mem_util = 0.0

                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temp = None

                # Per-process info
                processes = self._get_gpu_processes(handle, idx)

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

    def _get_gpu_processes(self, handle: Any, _gpu_index: int) -> list[dict[str, Any]]:
        """Get processes using a specific GPU.

        Args:
            handle: NVML device handle.
            gpu_index: GPU index.

        Returns:
            List of process information dictionaries.
        """
        processes: list[dict[str, Any]] = []

        try:
            # Get compute processes (CUDA, etc.)
            compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for proc in compute_procs:
                proc_info = self._get_process_info(proc, "compute")
                if proc_info:
                    processes.append(proc_info)
        except Exception:
            pass

        try:
            # Get graphics processes
            graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            for proc in graphics_procs:
                # Avoid duplicates
                if not any(p["pid"] == proc.pid for p in processes):
                    proc_info = self._get_process_info(proc, "graphics")
                    if proc_info:
                        processes.append(proc_info)
        except Exception:
            pass

        return processes

    def _get_process_info(self, proc: Any, proc_type: str) -> dict[str, Any] | None:
        """Extract process information from NVML process object.

        Args:
            proc: NVML process info object.
            proc_type: Type of process ("compute" or "graphics").

        Returns:
            Dictionary with process info or None.
        """
        try:
            # Get process name
            try:
                import psutil

                process = psutil.Process(proc.pid)
                name = process.name()
            except Exception:
                name = f"pid_{proc.pid}"

            memory_mb = proc.usedGpuMemory / (1024 * 1024) if proc.usedGpuMemory else 0

            return {
                "pid": proc.pid,
                "name": name,
                "type": proc_type,
                "gpu_memory_mb": round(memory_mb, 1),
            }
        except Exception:
            return None

    def get_process_gpu_usage(self) -> dict[int, dict[str, Any]]:
        """Get GPU usage per process across all GPUs.

        Returns:
            Dictionary mapping PID to GPU usage info.
        """
        if not self._initialized:
            if not self.initialize():
                return {}

        process_usage: dict[int, dict[str, Any]] = {}

        for idx in range(self._device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                total_power_watts = power_mw / 1000.0

                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                total_gpu_util = float(util.gpu)

                # Get processes on this GPU
                processes = self._get_gpu_processes(handle, idx)
                if not processes:
                    continue

                # Estimate power per process based on memory usage
                # (rough approximation - memory is a proxy for GPU work)
                total_memory = sum(p.get("gpu_memory_mb", 0) for p in processes)
                if total_memory == 0:
                    continue

                for proc in processes:
                    pid = proc["pid"]
                    memory_mb = proc.get("gpu_memory_mb", 0)

                    # Estimate GPU utilization fraction
                    if total_memory > 0:
                        memory_fraction = memory_mb / total_memory
                    else:
                        memory_fraction = 0
                    estimated_util = total_gpu_util * memory_fraction
                    estimated_power = total_power_watts * memory_fraction

                    if pid in process_usage:
                        # Aggregate across GPUs
                        process_usage[pid]["gpu_memory_mb"] += memory_mb
                        process_usage[pid]["estimated_gpu_util"] += estimated_util
                        process_usage[pid]["estimated_power_watts"] += estimated_power
                        process_usage[pid]["gpus"].append(idx)
                    else:
                        process_usage[pid] = {
                            "pid": pid,
                            "name": proc["name"],
                            "gpu_memory_mb": memory_mb,
                            "estimated_gpu_util": estimated_util,
                            "estimated_power_watts": estimated_power,
                            "gpus": [idx],
                        }

            except Exception:
                continue

        return process_usage

    def get_total_gpu_power(self) -> float:
        """Get total power consumption across all GPUs.

        Returns:
            Total GPU power in watts.
        """
        if not self._initialized:
            if not self.initialize():
                return 0.0

        total_power = 0.0
        for idx in range(self._device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                total_power += power_mw / 1000.0
            except Exception:
                continue

        return total_power

    def cleanup(self) -> None:
        """Shutdown NVML and clean up resources."""
        if self._initialized and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        self._initialized = False
        self._device_count = 0
