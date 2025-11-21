"""Base class for stress tests."""

from abc import ABC, abstractmethod


class StressTest(ABC):
    """Abstract base class for all stress tests."""

    @abstractmethod
    def run(self, duration: int) -> dict:
        """
        Run the stress test.

        Args:
            duration: Test duration in seconds

        Returns:
            Dictionary containing test results
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the test name for display.

        Returns:
            Human-readable test name
        """
        pass
