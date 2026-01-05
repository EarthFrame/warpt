"""Base classes for power monitoring sources and backends.

This module provides the abstract interfaces for both individual power sources
(like a specific GPU or CPU package) and complete power backends (like RAPL
or NVML).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from warpt.models.power_models import DomainPower, PowerSource


class BasePowerSource(ABC):
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


class PowerBackend(ABC):
    """Abstract base class for power monitoring backends.

    Each platform-specific backend (RAPL, powermetrics, NVML) implements
    this interface to provide power readings.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this power source is available on the current system.

        Returns:
            True if the power source can be used.
        """
        ...

    @abstractmethod
    def get_power_readings(self) -> list[DomainPower]:
        """Get current power readings from this backend.

        Returns:
            List of DomainPower objects with current measurements.
        """
        ...

    @abstractmethod
    def get_source(self) -> PowerSource:
        """Get the power source type for this backend.

        Returns:
            PowerSource enum value.
        """
        ...

    def initialize(self) -> bool:
        """Initialize the backend. Called once before use.

        Returns:
            True if initialization succeeded.
        """
        return True

    def cleanup(self) -> None:  # noqa: B027
        """Clean up resources. Called when done with the backend."""
        pass
