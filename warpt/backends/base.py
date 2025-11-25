"""Abstract base classes for hardware backends.

interfaces that all vendor-specific backends must implement

- each vendor implements the same interface
- Graceful degradation when hardware is unavailable

- NVIDIA: nvidia-ml-py
- AMD: amdsmi (ROCm system management)
- Intel: Level Zero or oneAPI libraries
"""

from abc import ABC, abstractmethod
from typing import Any

from warpt.models.list_models import GPUInfo


class GPUBackend(ABC):
    """Abstract base class for GPU vendor backends (NVIDIA, AMD, Intel, etc.)."""

    @abstractmethod
    def __init__(self):
        """Initialize the GPU backend.

        Should handle vendor-specific library initialization (e.g., nvmlInit()).
        May raise exceptions if the vendor's libraries are not available or
        if initialization fails.

        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this GPU vendor's hardware is available on the system.

        Used by the factory pattern to determine which backends to use.

        Returns
        -------
            bool: True if at least one GPU from this vendor is detected
        """
        pass

    @abstractmethod
    def get_device_count(self) -> int:
        """Get the number of GPUs from this vendor.

        Returns
        -------
            int: Number of GPUs detected
        """
        pass

    @abstractmethod
    def list_devices(self) -> list[GPUInfo]:
        """List all GPUs from this vendor with their specifications.

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
    def get_temperature(self, index: int) -> float | None:
        """Get GPU temperature in degrees C.

        Used for monitoring and stress testing, and for detecting
        thermal throttling and ensuring system health

        Args:
            index: GPU index (0-based)

        Returns
        -------
            float: temperature in C, or None if unavailable
        """
        pass

    @abstractmethod
    def get_memory_usage(self, index: int) -> dict | None:
        """Get current GPU memory usage.

        Used for monitoring memory pressure during stress tests and
        benchmarks

        detecting memory leaks or over-allocation

        Args:
            index: GPU index (0-based)

        Returns
        -------
            dict with keys:
                - total (int): Total memory in bytes
                - used (int): Used memory in bytes
                - free (int): Free memory in bytes
            Or None if unavailable
        """
        pass

    @abstractmethod
    def get_utilization(self, index: int) -> dict | None:
        """Get GPU util %.

        Used for real time monitoring and stress testing
        identify if GPU is being fully utilized or if there are bottlenecks

        Args:
            index: GPU index (0-based)

        Returns
        -------
            dict with keys:
                - gpu (float): GPU compute utilization percentage (0-100)
                - memory (float): Memory bandwidth utilization percentage (0-100)
            Or None if unavailable

        """
        pass

    @abstractmethod
    def get_pytorch_device_string(self, device_id: int) -> str:
        """
        Get PyTorch device string for this vendor.

        Used by stress tests to select the correct device in PyTorch.
        Different vendors use different device strings:
        - NVIDIA: 'cuda:0', 'cuda:1', etc.
        - AMD (ROCm): 'cuda:0' (same as NVIDIA when using ROCm-enabled PyTorch)
        - Intel: 'xpu:0', 'xpu:1', etc.
        - Apple: 'mps:0'

        Args:
            device_id: GPU index (0-based)

        Returns:
            str: PyTorch device string (e.g., 'cuda:0')
        """
        pass

    @abstractmethod
    def get_power_usage(self, index: int) -> Optional[float]:
        """
        Get current GPU power usage in Watts.

        Used for monitoring power consumption during stress tests and
        detecting power throttling.

        Args:
            index: GPU index (0-based)

        Returns:
            float: Power usage in Watts, or None if unavailable
        """
        pass

    @abstractmethod
    def get_throttle_reasons(self, index: int) -> List[str]:
        """
        Get current GPU throttling reasons.

        Used for detecting performance degradation during stress tests.
        Common throttle reasons:
        - 'thermal' - Temperature limit reached
        - 'power_limit' - Power limit reached
        - 'sw_power_cap' - Software-imposed power cap
        - 'hw_slowdown' - Hardware slowdown
        - 'sync_boost' - Sync boost limit

        Args:
            index: GPU index (0-based)

        Returns:
            List[str]: List of active throttle reasons, empty list if not throttling
        """
        pass

    @abstractmethod
    def shutdown(self):
        """Cleanup and shutdown the GPU backend.

        Should handle vendor specific library cleanup (nvmlShutdown())
        """
        pass


class CPUBackend(ABC):
    """Abstract base class for CPU information backends.

    TODO: implement this interface when CPU stress testing is added
    """

    @abstractmethod
    def list_devices(self) -> dict:
        """Get CPU info.

        Returns
        -------
            dict with keys:
                - model (str): CPU model name
                - cores (int): Physical core count
                - threads (int): Logical thread count
                - features (list[str]): CPU features (AVX, SSE, etc.)
        """
        pass
