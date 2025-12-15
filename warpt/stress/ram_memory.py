"""RAM memory stress tests."""

from typing import Any

from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.models.stress_models import RAMMemoryStressResult
from warpt.stress.base import StressTest, TestCategory


class RAMBandwidthTest(StressTest):
    """Memory bandwidth stress test for system RAM.

    Tests sequential read and write bandwidth under normal conditions
    (no swap pressure). Uses NumPy for memory operations.
    """

    _PARAM_FIELDS = ("allocation_percent", "burnin_seconds")

    def __init__(
        self,
        allocation_percent: float = 0.5,
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
    ):
        """Initialize RAM bandwidth test.

        Args:
            allocation_percent: Percentage of available RAM to allocate (0.0-1.0).
                Default 0.5 (50%) to avoid forcing swap.
            burnin_seconds: Warmup duration before measurement.
        """
        self.allocation_percent = allocation_percent
        self.burnin_seconds = burnin_seconds
        self._array: Any | None = None
        self._source_array_for_writes: Any | None = None
        self._allocated_gb = 0.0
        self._total_ram_gb = 0.0
        self._available_ram_gb = 0.0

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "RAM Memory Bandwidth"

    def get_description(self) -> str:
        """Return one-line description."""
        return (
            "Measures system RAM sequential read/write bandwidth "
            "(duration split between both)"
        )

    def get_category(self) -> TestCategory:
        """Return test category."""
        return TestCategory.RAM

    # -------------------------------------------------------------------------
    # Hardware & Availability
    # -------------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if NumPy and psutil are available."""
        try:
            import numpy  # noqa: F401
            import psutil  # noqa: F401

            return True
        except ImportError:
            return False

    def validate_configuration(self) -> None:
        """Validate test configuration."""
        if not self.is_available():
            raise RuntimeError("NumPy and psutil are required for RAM tests")
        if not 0.0 < self.allocation_percent <= 1.0:
            raise ValueError(
                f"allocation_percent must be between 0 and 1, "
                f"got {self.allocation_percent}"
            )
        if self.burnin_seconds < 0:
            raise ValueError("burnin_seconds must be >= 0")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """Allocate memory for testing."""
        import numpy as np
        import psutil

        # Get system RAM info
        mem = psutil.virtual_memory()
        self._total_ram_gb = mem.total / (1024**3)
        self._available_ram_gb = mem.available / (1024**3)

        # Calculate allocation size (total for both arrays)
        total_allocation_gb = self._available_ram_gb * self.allocation_percent
        # Split evenly between read array and write source array
        self._allocated_gb = total_allocation_gb / 2

        self.logger.info(f"Total RAM: {self._total_ram_gb:.2f} GB")
        self.logger.info(f"Available RAM: {self._available_ram_gb:.2f} GB")
        self.logger.info(
            f"Allocating: {total_allocation_gb:.2f} GB total "
            f"({self._allocated_gb:.2f} GB x 2 arrays, "
            f"{self.allocation_percent*100:.0f}%)"
        )

        # Allocate memory arrays (each is half of total allocation)
        num_elements = int(self._allocated_gb * (1024**3) / 8)  # 8 bytes per float64
        self._array = np.random.rand(num_elements)
        self._source_array_for_writes = np.random.rand(num_elements)
        self.logger.debug(f"Allocated {num_elements:,} float64 elements per array")

    def teardown(self) -> None:
        """Clean up allocated memory."""
        if self._array is not None:
            del self._array
            self._array = None
        if self._source_array_for_writes is not None:
            del self._source_array_for_writes
            self._source_array_for_writes = None
        self.logger.debug("Memory cleaned up")

    def warmup(self, duration_seconds: int = 0, iterations: int = 3) -> None:
        """Run warmup to stabilize memory subsystem.

        Args:
            duration_seconds: Warmup duration. If 0, uses self.burnin_seconds.
            iterations: Number of iterations if both duration_seconds and
                burnin_seconds are 0.
        """
        import time

        import numpy as np

        if self._array is None:
            return

        # Use burnin_seconds if no duration specified
        if duration_seconds == 0:
            duration_seconds = self.burnin_seconds

        if duration_seconds > 0:
            self.logger.debug(f"Warming up for {duration_seconds}s...")
            start = time.time()
            while (time.time() - start) < duration_seconds:
                _ = np.sum(self._array)
        else:
            self.logger.debug(f"Warming up for {iterations} iterations...")
            for _ in range(iterations):
                _ = np.sum(self._array)

    # -------------------------------------------------------------------------
    # Core Test
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> RAMMemoryStressResult:
        """Execute the RAM memory bandwidth test.

        Args:
            duration: Test duration in seconds (split between read and write).
            iterations: Ignored for RAM test.

        Returns:
            RAMMemoryStressResult with baseline bandwidth measurements.
        """
        del iterations  # Unused

        # Split duration between read and write tests
        per_test_duration = duration // 2

        # Run read and write bandwidth benchmarks
        read_bandwidth_gbps = self._benchmark_sequential_read(per_test_duration)
        write_bandwidth_gbps = self._benchmark_sequential_write(per_test_duration)

        self.logger.info(f"Read: {read_bandwidth_gbps:.2f} GB/s")
        self.logger.info(f"Write: {write_bandwidth_gbps:.2f} GB/s")

        # Return results (baseline only, no swap pressure)
        return RAMMemoryStressResult(
            # System info
            total_ram_gb=self._total_ram_gb,
            available_ram_gb=self._available_ram_gb,
            allocated_memory_gb=self._allocated_gb * 2,  # Total for both arrays
            # Test metadata
            duration=float(duration),
            burnin_seconds=self.burnin_seconds,
            # Baseline metrics (actual measurements)
            baseline_read_gbps=read_bandwidth_gbps,
            baseline_write_gbps=write_bandwidth_gbps,
            baseline_latency_ms=0.0,  # Not measuring latency
            # Pressure metrics (not tested, set to 0)
            pressure_read_gbps=0.0,
            pressure_write_gbps=0.0,
            pressure_latency_ms=0.0,
            # Performance degradation (no degradation = 1.0)
            read_slowdown_factor=1.0,
            write_slowdown_factor=1.0,
            latency_increase_factor=1.0,
            # Swap metrics (no swap testing)
            swap_occurred=False,
            swap_in_mb=None,
            swap_out_mb=None,
            peak_swap_usage_mb=None,
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _benchmark_sequential_read(self, duration: int) -> float:
        """Benchmark sequential read bandwidth.

        Args:
            duration: Test duration in seconds.

        Returns:
            Read bandwidth in GB/s.
        """
        import time

        import numpy as np

        assert self._array is not None, "Array not allocated in setup()"

        start_time = time.time()
        bytes_read = 0
        iter_count = 0

        while (time.time() - start_time) < duration:
            # Sequential read: sum forces reading all elements
            _ = np.sum(self._array)
            bytes_read += self._array.nbytes
            iter_count += 1

        elapsed = time.time() - start_time
        gb_read = bytes_read / (1024**3)
        bandwidth_gbps = gb_read / elapsed

        self.logger.debug(
            f"Read: {iter_count} iterations, {gb_read:.2f} GB in {elapsed:.2f}s"
        )

        return bandwidth_gbps

    def _benchmark_sequential_write(self, duration: int) -> float:
        """Benchmark sequential write bandwidth.

        Args:
            duration: Test duration in seconds.

        Returns:
            Write bandwidth in GB/s.
        """
        import time

        assert self._array is not None, "Array not allocated in setup()"
        assert (
            self._source_array_for_writes is not None
        ), "Source array not allocated in setup()"

        start_time = time.time()
        bytes_written = 0
        iter_count = 0

        while (time.time() - start_time) < duration:
            # Sequential write: copy from pre-allocated source array
            # This measures pure write bandwidth without RNG overhead
            self._array[:] = self._source_array_for_writes
            bytes_written += self._array.nbytes
            iter_count += 1

        elapsed = time.time() - start_time
        gb_written = bytes_written / (1024**3)
        bandwidth_gbps = gb_written / elapsed

        self.logger.debug(
            f"Write: {iter_count} iterations, {gb_written:.2f} GB in {elapsed:.2f}s"
        )

        return bandwidth_gbps
