"""Base class for CPU monitoring - to be implemented per platform."""

from abc import ABC, abstractmethod


class CPUMonitor(ABC):
    """Abstract base class for CPU monitoring."""

    @abstractmethod
    def get_temperature(self) -> float | None:
        """
        Get CPU temperature in Celsius.

        Returns:
            Temperature in Celsius or None if unavailable
        """
        pass

    @abstractmethod
    def get_frequency(self) -> float | None:
        """
        Get current CPU frequency in MHz.

        Returns:
            Frequency in MHz or None if unavailable
        """
        pass

    @abstractmethod
    def get_utilization(self) -> float | None:
        """
        Get CPU utilization percentage.

        Returns:
            Utilization percentage (0-100) or None if unavailable
        """
        pass

    @abstractmethod
    def is_throttling(self) -> bool:
        """
        Check if CPU is currently throttling due to thermal or power limits.

        Returns:
            True if throttling detected, False otherwise
        """
        pass

    @abstractmethod
    def get_power(self) -> float | None:
        """
        Get CPU power consumption in Watts.

        Returns:
            Power consumption in Watts or None if unavailable
        """
        pass

    @abstractmethod
    def get_throttle_reason(self) -> str | None:
        """
        Get reason for CPU throttling.

        Returns:
            Throttle reason string (e.g., 'thermal', 'power', 'tdp_limit') or None
        """
        pass

    @abstractmethod
    def get_core_count(self) -> int:
        """
        Get number of active CPU cores.

        Returns:
            Number of active cores
        """
        pass

    @abstractmethod
    def get_base_frequency(self) -> float | None:
        """
        Get CPU base frequency in MHz.

        Returns:
            Base frequency in MHz or None if unavailable
        """
        pass

    @abstractmethod
    def get_boost_frequency(self) -> float | None:
        """
        Get CPU boost (turbo) frequency in MHz.

        Returns:
            Boost frequency in MHz or None if unavailable
        """
        pass
