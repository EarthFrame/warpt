"""Base class for power monitoring sources."""

from abc import ABC, abstractmethod


class PowerSource(ABC):
    """Abstract base class for a power monitoring source."""

    @abstractmethod
    def get_power_w(self) -> float | None:
        """Get current power consumption in Watts.

        Returns
        -------
            Power consumption in Watts, or None if unavailable.
        """
        pass

    @abstractmethod
    def check_permissions(self) -> bool:
        """Check if we have the necessary permissions to use this power source.

        Returns
        -------
            True if permissions are sufficient, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the power source (e.g., 'cpu', 'gpu_0')."""
        pass
