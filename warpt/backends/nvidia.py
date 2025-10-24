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