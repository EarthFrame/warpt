"""AMD GPU backend placeholder.

This backend will use AMD's ROCm system management interface (amdsmi)
when implemented.

Currently raises NotImplementedError for all methods.
"""

from warpt.backends.base import GPUBackend
from warpt.models.list_models import GPUInfo


class AMDBackend(GPUBackend):
    """Backend for AMD GPU information (placeholder - not yet implemented)."""

    def __init__(self):
        """Initialize AMD backend."""
        raise NotImplementedError(
            "AMD backend not yet implemented. "
            "NVIDIA GPUs are currently supported via NvidiaBackend."
        )

    def is_available(self) -> bool:
        """Check if AMD GPUs are available."""
        return False

    def get_device_count(self) -> int:
        """Get the number of AMD GPUs."""
        raise NotImplementedError("AMD backend not yet implemented")

    def list_devices(self) -> list[GPUInfo]:
        """List all AMD GPUs."""
        raise NotImplementedError("AMD backend not yet implemented")

    def get_temperature(self, index: int) -> float | None:
        """Get GPU temperature."""
        raise NotImplementedError("AMD backend not yet implemented")

    def get_memory_usage(self, index: int) -> dict | None:
        """Get GPU memory usage."""
        raise NotImplementedError("AMD backend not yet implemented")

    def get_utilization(self, index: int) -> dict | None:
        """Get GPU utilization."""
        raise NotImplementedError("AMD backend not yet implemented")

    def get_pytorch_device_string(self, device_id: int) -> str:
        """Get PyTorch device string for AMD GPUs."""
        raise NotImplementedError("AMD backend not yet implemented")

    def get_power_usage(self, index: int) -> float | None:
        """Get GPU power usage."""
        raise NotImplementedError("AMD backend not yet implemented")

    def get_throttle_reasons(self, index: int) -> list[str]:
        """Get GPU throttling reasons."""
        raise NotImplementedError("AMD backend not yet implemented")

    def shutdown(self):
        """Cleanup and shutdown AMD backend."""
        raise NotImplementedError("AMD backend not yet implemented")
