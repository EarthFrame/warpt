"""CPU compute stress tests."""

import time
from typing import Any

from warpt.backends.system import CPU
from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.stress.base import StressTest, TestCategory
from warpt.stress.utils import calculate_tflops


class CPUMatMulTest(StressTest):
    """Matrix multiplication stress test for CPU.

    Uses NumPy's BLAS-backed matmul to stress all CPU cores. On macOS,
    this uses the Accelerate framework; on Linux, typically OpenBLAS or MKL.
    """

    _PARAM_FIELDS = ("matrix_size", "burnin_seconds")

    def __init__(
        self,
        matrix_size: int = 4096,
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
    ):
        """Initialize CPU matmul test.

        Args:
            matrix_size: Size of square matrices (NxN). Default 4096.
            burnin_seconds: Warmup duration before measurement.
        """
        self.matrix_size = matrix_size
        self.burnin_seconds = burnin_seconds
        self._cpu: CPU | None = None
        self._cpu_info: Any = None

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "CPU Matrix Multiplication"

    def get_description(self) -> str:
        """Return one-line description."""
        return "Measures CPU compute throughput via FP64 matrix multiplication"

    def get_category(self) -> TestCategory:
        """Return test category."""
        return TestCategory.CPU

    # -------------------------------------------------------------------------
    # Configuration & Parameters
    # -------------------------------------------------------------------------
    # Configuration is managed by _PARAM_FIELDS. get_parameters() and
    # set_parameters() are inherited from StressTest base class.

    # -------------------------------------------------------------------------
    # Hardware & Availability
    # -------------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if NumPy is available."""
        try:
            import numpy  # noqa: F401

            return True
        except ImportError:
            return False

    def validate_configuration(self) -> None:
        """Validate test configuration."""
        if not self.is_available():
            raise RuntimeError("NumPy is not installed")
        if self.matrix_size < 64:
            raise ValueError("matrix_size must be >= 64")
        if self.burnin_seconds < 0:
            raise ValueError("burnin_seconds must be >= 0")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize CPU info."""
        self._cpu = CPU()
        self._cpu_info = self._cpu.get_cpu_info()

    def teardown(self) -> None:
        """Clean up resources."""
        self._cpu = None
        self._cpu_info = None

    def warmup(self, duration_seconds: int = 0, iterations: int = 3) -> None:
        """Run warmup iterations to let CPU/cache warm up.

        Args:
            duration_seconds: Warmup duration. If 0, uses self.burnin_seconds.
            iterations: Number of iterations if both duration_seconds and
                burnin_seconds are 0.
        """
        import numpy as np

        # Use burnin_seconds if no duration specified
        if duration_seconds == 0:
            duration_seconds = self.burnin_seconds

        if duration_seconds > 0:
            self.logger.debug(f"Warming up for {duration_seconds}s...")
            start = time.time()
            while (time.time() - start) < duration_seconds:
                a = np.random.rand(self.matrix_size, self.matrix_size)
                b = np.random.rand(self.matrix_size, self.matrix_size)
                _ = np.matmul(a, b)
                del a, b
        else:
            self.logger.debug(f"Warming up for {iterations} iterations...")
            for _ in range(iterations):
                a = np.random.rand(self.matrix_size, self.matrix_size)
                b = np.random.rand(self.matrix_size, self.matrix_size)
                _ = np.matmul(a, b)
                del a, b

    # -------------------------------------------------------------------------
    # Core Test
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> dict:
        """Execute the CPU matrix multiplication test.

        Args:
            duration: Test duration in seconds.
            iterations: Ignored for CPU test (runs for duration).

        Returns:
            Dictionary containing test results (TFLOPS, etc.)
        """
        import numpy as np

        del iterations  # Unused; test runs for duration
        start_time = time.time()
        iter_count = 0

        while (time.time() - start_time) < duration:
            a = np.random.rand(self.matrix_size, self.matrix_size)
            b = np.random.rand(self.matrix_size, self.matrix_size)
            _ = np.matmul(a, b)
            iter_count += 1
            del a, b

        elapsed = time.time() - start_time

        # Calculate TFLOPS (2*N^3 - N^2 ops per matmul)
        ops_per_matmul = 2 * (self.matrix_size**3) - (self.matrix_size**2)
        total_ops = iter_count * ops_per_matmul
        tflops = calculate_tflops(total_ops, elapsed)

        self.logger.info(f"Result: {tflops:.2f} TFLOPS ({iter_count} iterations)")

        return {
            "test_name": self.get_name(),
            "tflops": tflops,
            "duration": elapsed,
            "iterations": iter_count,
            "matrix_size": self.matrix_size,
            "total_operations": total_ops,
            "burnin_seconds": self.burnin_seconds,
            "cpu_physical_cores": self._cpu_info.total_physical_cores,
            "cpu_logical_cores": self._cpu_info.total_logical_cores,
        }
