"""CPU compute stress tests."""

from warpt.stress.base import StressTest


class CPUMatMulTest(StressTest):
    """Matrix multiplication stress test for CPU."""

    def run(self, duration: int) -> dict:
        """
        Run CPU matrix multiplication test.

        Args:
            duration: Test duration in seconds

        Returns:
            Dictionary containing test results (TFLOPS, etc.)
        """
        # TODO: Implement matrix multiplication test
        pass

    def get_name(self) -> str:
        """Get test name."""
        return "CPU Matrix Multiplication"
