"""
Abstract base classes for hardware backends.

interfaces that all vendor-specific backends must implement

- each vendor implements the same interface
- Graceful degradation when hardware is unavailable

- NVIDIA: nvidia-ml-py
- AMD: amdsmi (ROCm system management)
- Intel: Level Zero or oneAPI libraries
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from warpt.models.list_models import GPUInfo


class GPUBackend(ABC):
    """
    Abstract base class for GPU vendor backends (NVIDIA, AMD, Intel, etc.)

    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the GPU backend

        Should handle vendor-specific library initialization (e.g., nvmlInit()).
        May raise exceptions if the vendor's libraries are not available or
        if initialization fails.

        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this GPU vendor's hardware is available on the system.
        Used by the factory pattern to determine which backends to use.

        Returns:
            bool: True if at least one GPU from this vendor is detected
        """
        pass

    @abstractmethod
    def get_device_count(self) -> int:
        """
        Get the number of GPUs from this vendor.

        Returns:
            int: Number of GPUs detected
        """
        pass

    @abstractmethod
    def list_devices(self) -> List[GPUInfo]:
        """
        List all GPUs from this vendor with their specifications

        Used by the 'list' command

        Example:
            return [GPUInfo(
                index=0,
                model='NVIDIA RTX 4090',
                memory_gb=24,
                compute_capability='8.9',
                pcie_gen=4,
                
                extra_metrics={'cuda_cores': 16384, 'sm_count': 128}
            )]
        """
        pass

    @abstractmethod
    def get_device_handle(self, index: int):
        """
        Get a vendor specific device handle for a GPU

        This handle is used for monitoring
        The type of handle is vendor specific

        Args:
            index: GPU index (0-based)

        Returns:
            Vendor-specific device handle object

        """
        pass

    @abstractmethod
    def get_temperature(self, handle) -> Optional[float]:
        """
        Get GPU temperature in degrees C

        Used for monitoring and stress testing, and for detecting
        thermal throttling and ensuring system health

        Args:
            handle: vendor specific device handle from get_device_handle()

        Returns:
            float: temperature in C, or None if unavailable
        """
        pass

    @abstractmethod
    def get_memory_usage(self, handle) -> Optional[Dict]:
        """
        Get current GPU memory usage

        Used for monitoring memory pressure during stress tests and
        benchmarks
        
        detecting memory leaks or over-allocation

        Args:
            handle: Vendor specific device handle from get_device_handle()

        Returns:
            dict with keys:
                - total (int): Total memory in bytes
                - used (int): Used memory in bytes
                - free (int): Free memory in bytes
            Or None if unavailable
        """
        pass

    @abstractmethod
    def get_utilization(self, handle) -> Optional[Dict]:
        """
        Get GPU util %

        Used for real time monitoring and stress testing 
        identify if GPU is being fully utilized or if there are bottlenecks

        Args:
            handle: Vendor specific device handle from get_device_handle()

        Returns:
            dict with keys:
                - gpu (float): GPU compute utilization percentage (0-100)
                - memory (float): Memory bandwidth utilization percentage (0-100)
            Or None if unavailable

        """
        pass

    @abstractmethod
    def shutdown(self):
        """
        Cleanup and shutdown the GPU backend

        Should handle vendor specific library cleanup (nvmlShutdown())
        """
        pass


class CPUBackend(ABC):
    """
    Abstract base class for CPU information backends.

    TODO: implement this interface when CPU stress testing is added
    """

    @abstractmethod
    def list_devices(self) -> Dict:
        """
        Get CPU info

        Returns:
            dict with keys:
                - model (str): CPU model name
                - cores (int): Physical core count
                - threads (int): Logical thread count
                - features (list[str]): CPU features (AVX, SSE, etc.)
        """
        pass
