"""
NVIDIA GPU backend using pynvml (NVIDIA Management Library).

This backend collects GPU information for the list command.
"""

import pynvml


class NvidiaBackend:
    def __init__(self):
        """Initialize NVML library."""
        pynvml.nvmlInit()

    def _get_compute_capability(self, handle):
        """
        Get CUDA compute capability for a GPU; the GPU architecture generation
        and what is the GPU's feature level

        Args:
            handle: NVML device handle

        Returns:
            str: Compute capability (e.g., "8.9")
        """
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        return f"{major}.{minor}"

    def _get_pcie_generation(self, handle):
        """
        Get PCIe generation for a GPU. Debugging slow data transfers

        PCIe generation affects bandwidth between CPU and GPU.
             Gen 3 = 32 GB/s, Gen 4 = 64 GB/s, Gen 5 = 128 GB/s

        Args:
            handle: NVML device handle

        Returns:
            int or None: PCIe generation (3, 4, 5) or None if unavailable
        """
        try:
            return pynvml.nvmlDeviceGetMaxPcielinkGeneration(handle)
        except pynvml.NVMLError:
            return None # TODO - want to look into standardized logging for errors
            

    def _bytes_to_gb(self, bytes_value):
        """
        Convert bytes to gigabytes (GB).

        Args:
            bytes_value: Memory in bytes

        Returns:
            int: Memory in gigabytes
        """
        return int(bytes_value / (1024**3))

    def list_devices(self):
        """
        List all NVIDIA GPUs with information for the list command.

        Returns:
            list[dict]: List of GPU information dictionaries matching GPUInfo model
        """
        device_count = pynvml.nvmlDeviceGetCount()
        device_info = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            device_info.append({
                'index': i,
                'model': pynvml.nvmlDeviceGetName(handle),
                'memory_gb': self._bytes_to_gb(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
                'compute_capability': self._get_compute_capability(handle),
                'pcie_gen': self._get_pcie_generation(handle)
            })

        return device_info
    
    def get_temperature(self, handle) -> float:
        """
        # TODO - currently unused and will be used for stress testing, monitoring
        Get GPU temperature in degrees Celsius

        Args:
            handle: NVML device handle

        Returns:
            float: GPU temperature in degrees Celsius
        """
        try:
            return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError:
            return -1
    

    