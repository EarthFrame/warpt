"""NVIDIA GPU backend using pynvml (NVIDIA Management Library).

This backend collects GPU information for the list command.
"""
from typing import Any, Dict, List, Optional

import pynvml

from warpt.backends.base import GPUBackend


class NvidiaBackend(GPUBackend):
    """Backend for NVIDIA GPU information using pynvml."""

    def __init__(self):
        """Initialize NVML library."""
        pynvml.nvmlInit()

    def _get_compute_capability(self, device_handle: "pynvml.nvmlDevice_t"):
        """Get CUDA compute capability for a GPU.

        The GPU architecture generation and what is the GPU's feature level.

        Args:
            device_handle: NVML device handle

        Returns
        -------
            str: Compute capability (e.g., "8.9")
        """
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(device_handle)
        return f"{major}.{minor}"

    def _get_pcie_generation(self, device_handle: "pynvml.nvmlDevice_t"):
        """Get PCIe generation for a GPU.

        Debugging slow data transfers. PCIe generation affects bandwidth
        between CPU and GPU. Gen 3 = 32 GB/s, Gen 4 = 64 GB/s, Gen 5 = 128 GB/s

        Args:
            device_handle: NVML device handle

        Returns
        -------
            int or None: PCIe generation (3, 4, 5) or None if unavailable
        """
        try:
            return pynvml.nvmlDeviceGetMaxPcielinkGeneration(device_handle)
        except pynvml.NVMLError:
            return None  # TODO - want to look into standardized logging for errors

    def _bytes_to_gb(self, bytes_value):
        """Convert bytes to gigabytes (GB).

        Args:
            bytes_value: Memory in bytes

        Returns
        -------
            int: Memory in gigabytes
        """
        return int(bytes_value / (1024**3))

    def list_devices(self):
        """List all NVIDIA GPUs with information for the list command.

        Returns
        -------
            list[dict]: List of GPU information dictionaries matching GPUInfo model
        """
        device_count = self.get_device_count()

        # Get driver version (system-wide, same for all NVIDIA GPUs)
        driver_version = pynvml.nvmlSystemGetDriverVersion()

        device_info = []

        for i in range(device_count):
            device_handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            device_info.append(
                {
                    "index": i,
                    "model": pynvml.nvmlDeviceGetName(device_handle),
                    "memory_gb": self._bytes_to_gb(
                        pynvml.nvmlDeviceGetMemoryInfo(device_handle).total
                    ),
                    "compute_capability": self._get_compute_capability(device_handle),
                    "pcie_gen": self._get_pcie_generation(device_handle),
                    "driver_version": driver_version,
                }
            )

        return device_info

    def get_temperature(self, index: int) -> float | None:
        """Get GPU temperature in degrees Celsius.

        Args:
            index: GPU index (0-based)

        Returns
        -------
            float: GPU temperature in degrees Celsius
        """
        try:
            device_handle = self._get_device_handle(index)
            return pynvml.nvmlDeviceGetTemperature(  # type: ignore[no-any-return]
                device_handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except pynvml.NVMLError:
            return None

    def is_available(self) -> bool:
        """Check if NVIDIA GPUs are available.

        Returns:
            bool: True if at least one NVIDIA GPU is detected
        """
        try:
            return self.get_device_count() > 0
        except pynvml.NVMLError:
            return False

    def get_device_count(self) -> int:
        """Get the number of NVIDIA GPUs.

        Returns:
            int: Number of NVIDIA GPUs detected
        """
        try:
            return pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            return 0

    def _get_device_handle(self, index: int):
        """
        Get NVML device handle for a GPU (internal use only).

        Args:
            index: GPU index (0-based)

        Returns:
            pynvml.nvmlDevice_t: NVML device handle
        """
        return pynvml.nvmlDeviceGetHandleByIndex(index)

    def get_pytorch_device_string(self, device_id: int) -> str:
        """Get PyTorch device string for NVIDIA GPUs.

        Args:
            device_id: GPU index (0-based)

        Returns:
            str: PyTorch device string (e.g., 'cuda:0')
        """
        return f"cuda:{device_id}"

    def get_memory_usage(self, index: int) -> Optional[Dict]:
        """
        Get current GPU memory usage.

        Args:
            index: GPU index (0-based)

        Returns:
            dict with keys: total, used, free (all in bytes), or None if unavailable
        """
        try:
            device_handle = self._get_device_handle(index)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
            return {
                "total": mem_info.total,
                "used": mem_info.used,
                "free": mem_info.free,
            }
        except pynvml.NVMLError:
            return None

    def get_utilization(self, index: int) -> Optional[Dict]:
        """
        Get GPU utilization percentage.

        Args:
            index: GPU index (0-based)

        Returns:
            dict with keys: gpu, memory (both 0-100), or None if unavailable
        """
        try:
            device_handle = self._get_device_handle(index)
            util = pynvml.nvmlDeviceGetUtilizationRates(device_handle)
            return {
                "gpu": float(util.gpu),
                "memory": float(util.memory),
            }
        except pynvml.NVMLError:
            return None

    def get_power_usage(self, index: int) -> Optional[float]:
        """
        Get current GPU power usage in Watts.

        Args:
            index: GPU index (0-based)

        Returns:
            float: Power usage in Watts, or None if unavailable
        """
        try:
            device_handle = self._get_device_handle(index)
            # pynvml returns power in milliwatts
            power_mw = pynvml.nvmlDeviceGetPowerUsage(device_handle)
            return power_mw / 1000.0  # Convert to Watts
        except pynvml.NVMLError:
            return None

    def get_throttle_reasons(self, index: int) -> List[str]:
        """
        Get current GPU throttling reasons.

        Args:
            index: GPU index (0-based)

        Returns:
            List[str]: List of active throttle reasons, empty list if not throttling
        """
        try:
            device_handle = self._get_device_handle(index)
            throttle_reasons = []
            clocks_throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(
                device_handle
            )

            # Check each throttle reason bit flag
            if clocks_throttle_reasons & pynvml.nvmlClocksThrottleReasonGpuIdle:
                throttle_reasons.append("gpu_idle")
            if (
                clocks_throttle_reasons
                & pynvml.nvmlClocksThrottleReasonApplicationsClocksSetting
            ):
                throttle_reasons.append("applications_clocks_setting")
            if clocks_throttle_reasons & pynvml.nvmlClocksThrottleReasonSwPowerCap:
                throttle_reasons.append("sw_power_cap")
            if clocks_throttle_reasons & pynvml.nvmlClocksThrottleReasonHwSlowdown:
                throttle_reasons.append("hw_slowdown")
            if clocks_throttle_reasons & pynvml.nvmlClocksThrottleReasonSyncBoost:
                throttle_reasons.append("sync_boost")
            if (
                clocks_throttle_reasons
                & pynvml.nvmlClocksThrottleReasonSwThermalSlowdown
            ):
                throttle_reasons.append("sw_thermal_slowdown")
            if (
                clocks_throttle_reasons
                & pynvml.nvmlClocksThrottleReasonHwThermalSlowdown
            ):
                throttle_reasons.append("hw_thermal_slowdown")
            if (
                clocks_throttle_reasons
                & pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown
            ):
                throttle_reasons.append("hw_power_brake_slowdown")

            return throttle_reasons
        except pynvml.NVMLError:
            return []

    def shutdown(self):
        """Cleanup and shutdown NVML."""
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass  # Ignore errors during shutdown
