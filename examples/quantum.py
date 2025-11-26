"""Quantum computing backend placeholder.

This backend is a placeholder for future quantum computing accelerators.

Currently raises NotImplementedError for all methods.
"""

from typing import Any

from warpt.backends.base import GPUBackend
from warpt.models.list_models import GPUInfo


class QuantumBackend(GPUBackend):
    """Backend for quantum computing accelerators."""

    def __init__(self):
        """Initialize Quantum backend."""
        raise NotImplementedError(
            "Quantum backend not yet implemented. "
            "This is a placeholder for future quantum computing support."
        )

    def is_available(self) -> bool:
        """Check if quantum accelerators are available."""
        return False

    def get_device_count(self) -> int:
        """Get the number of quantum processors."""
        raise NotImplementedError("Quantum backend not yet implemented")

    def list_devices(self) -> list[GPUInfo]:
        """List all quantum processors."""
        raise NotImplementedError("Quantum backend not yet implemented")

    def get_temperature(self, index: int) -> Optional[float]:
        """Get quantum processor temperature."""
        raise NotImplementedError("Quantum backend not yet implemented")

    def get_memory_usage(self, index: int) -> Optional[Dict]:
        """Get quantum processor memory usage."""
        raise NotImplementedError("Quantum backend not yet implemented")

    def get_utilization(self, index: int) -> Optional[Dict]:
        """Get quantum processor utilization."""
        raise NotImplementedError("Quantum backend not yet implemented")

    def get_pytorch_device_string(self, device_id: int) -> str:
        """Get PyTorch device string for quantum processors."""
        raise NotImplementedError("Quantum backend not yet implemented")

    def get_power_usage(self, index: int) -> Optional[float]:
        """Get quantum processor power usage."""
        raise NotImplementedError("Quantum backend not yet implemented")

    def get_throttle_reasons(self, index: int) -> List[str]:
        """Get quantum processor throttling reasons."""
        raise NotImplementedError("Quantum backend not yet implemented")

    def shutdown(self):
        """Cleanup and shutdown Quantum backend."""
        raise NotImplementedError("Quantum backend not yet implemented")
