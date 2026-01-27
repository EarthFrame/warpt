"""RAM swap pressure stress test."""

from typing import Any

from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.models.stress_models import RAMMemoryStressResult
from warpt.stress.base import StressTest, TestCategory


class RAMSwapPressureTest(StressTest):
    """Memory stress test that forces swap to measure performance degradation.

    Tests both baseline (normal RAM) and pressure (forced swap) conditions
    to measure how much performance degrades when the system runs out of
    physical RAM and must use swap space.
    """

    _PARAM_FIELDS = (
        "allocation_percent_baseline",
        "allocation_percent_pressure",
        "burnin_seconds",
    )

    def __init__(
        self,
        allocation_percent_baseline: float = 0.5,
        allocation_percent_pressure: float = 1.5,
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
    ):
        """Initialize RAM swap pressure test.

        Args:
            allocation_percent_baseline: Percentage of available RAM for
                baseline test (0.0-1.0). Default 0.5 (50%).
            allocation_percent_pressure: Percentage of available RAM for
                pressure test (>1.0 to force swap). Default 1.5 (150%).
            burnin_seconds: Warmup duration before measurement.
        """
        self.allocation_percent_baseline = allocation_percent_baseline
        self.allocation_percent_pressure = allocation_percent_pressure
        self.burnin_seconds = burnin_seconds

        # Runtime state (set in setup/execute_test)
        self._array: Any | None = None
        self._source_array_for_writes: Any | None = None
        self._total_ram_gb = 0.0
        self._available_ram_gb = 0.0

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "RAM Swap Pressure Test"

    def get_description(self) -> str:
        """Return one-line description."""
        return (
            "Measures RAM performance degradation under memory pressure "
            "(baseline vs forced swap)"
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
        if not 0.0 < self.allocation_percent_baseline <= 1.0:
            raise ValueError(
                f"allocation_percent_baseline must be between 0 and 1, "
                f"got {self.allocation_percent_baseline}"
            )
        if self.allocation_percent_pressure <= 1.0:
            raise ValueError(
                f"allocation_percent_pressure must be > 1.0 to force swap, "
                f"got {self.allocation_percent_pressure}"
            )
        if self.burnin_seconds < 0:
            raise ValueError("burnin_seconds must be >= 0")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """Get system RAM info (arrays allocated per-phase in execute_test)."""
        import psutil

        mem = psutil.virtual_memory()
        self._total_ram_gb = mem.total / (1024**3)
        self._available_ram_gb = mem.available / (1024**3)

        self.logger.info(f"Total RAM: {self._total_ram_gb:.2f} GB")
        self.logger.info(f"Available RAM: {self._available_ram_gb:.2f} GB")

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

        Note: Actual warmup happens per-phase in execute_test after allocating
        arrays. This just satisfies the base class requirement.

        Args:
            duration_seconds: Warmup duration. If 0, uses self.burnin_seconds.
            iterations: Number of iterations if both duration_seconds and
                burnin_seconds are 0.
        """
        import time

        if duration_seconds == 0:
            duration_seconds = self.burnin_seconds

        if duration_seconds > 0:
            self.logger.debug(f"Global warmup: {duration_seconds}s...")
            time.sleep(duration_seconds)
        else:
            self.logger.debug(f"Global warmup: {iterations} iterations...")
            for _ in range(iterations):
                time.sleep(0.01)

    # -------------------------------------------------------------------------
    # Core Test
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> RAMMemoryStressResult:
        """Execute the RAM swap pressure test.

        Runs two phases:
        1. Baseline: Measure performance with normal RAM allocation
        2. Pressure: Measure performance when forcing swap

        Args:
            duration: Test duration in seconds (split between both phases).
            iterations: Ignored for RAM test.

        Returns:
            RAMMemoryStressResult with both baseline and pressure metrics.
        """
        del iterations  # Unused

        # Split duration between baseline and pressure phases
        phase_duration = duration // 2
        per_test_duration = phase_duration // 2  # Split each phase: read + write

        self.logger.info("=== Phase 1: Baseline (No Swap) ===")
        # Allocate baseline arrays
        self._allocate_arrays(self.allocation_percent_baseline)

        # Measure baseline performance
        baseline_read_gbps = self._benchmark_sequential_read(per_test_duration)
        baseline_write_gbps = self._benchmark_sequential_write(per_test_duration)

        self.logger.info(f"Baseline Read: {baseline_read_gbps:.2f} GB/s")
        self.logger.info(f"Baseline Write: {baseline_write_gbps:.2f} GB/s")

        # Free baseline arrays
        self._free_arrays()

        self.logger.info("=== Phase 2: Pressure (Force Swap) ===")
        # Get initial swap stats
        swap_before = self._get_swap_stats()

        # Allocate pressure arrays (force swap)
        self._allocate_arrays(self.allocation_percent_pressure)
        assert self._array is not None, "Array not allocated"
        allocated_pressure_gb = self._array.nbytes / (1024**3) * 2

        # Measure pressure performance
        pressure_read_gbps = self._benchmark_sequential_read(per_test_duration)
        pressure_write_gbps = self._benchmark_sequential_write(per_test_duration)

        self.logger.info(f"Pressure Read: {pressure_read_gbps:.2f} GB/s")
        self.logger.info(f"Pressure Write: {pressure_write_gbps:.2f} GB/s")

        # Get final swap stats
        swap_after = self._get_swap_stats()

        # Free pressure arrays
        self._free_arrays()

        # Calculate degradation factors
        read_slowdown = (
            baseline_read_gbps / pressure_read_gbps if pressure_read_gbps > 0 else 1.0
        )
        write_slowdown = (
            baseline_write_gbps / pressure_write_gbps
            if pressure_write_gbps > 0
            else 1.0
        )

        # Detect if swapping occurred
        swap_occurred = swap_after["used"] > swap_before["used"]
        swap_delta_mb = (swap_after["used"] - swap_before["used"]) / (1024**2)

        # Return results
        return RAMMemoryStressResult(
            # System info
            total_ram_gb=self._total_ram_gb,
            available_ram_gb=self._available_ram_gb,
            allocated_memory_gb=allocated_pressure_gb,  # Report max allocation
            # Test metadata
            duration=float(duration),
            burnin_seconds=self.burnin_seconds,
            # Baseline metrics
            baseline_read_gbps=baseline_read_gbps,
            baseline_write_gbps=baseline_write_gbps,
            baseline_latency_ms=0.0,  # Not measuring latency
            # Pressure metrics
            pressure_read_gbps=pressure_read_gbps,
            pressure_write_gbps=pressure_write_gbps,
            pressure_latency_ms=0.0,  # Not measuring latency
            # Performance degradation
            read_slowdown_factor=read_slowdown,
            write_slowdown_factor=write_slowdown,
            latency_increase_factor=1.0,  # Not measuring latency
            # Swap metrics
            swap_occurred=swap_occurred,
            swap_in_mb=None,  # psutil doesn't provide sin/sout on all platforms
            swap_out_mb=None,
            peak_swap_usage_mb=swap_delta_mb if swap_occurred else None,
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _allocate_arrays(self, allocation_percent: float) -> None:
        """Allocate memory arrays for testing.

        Args:
            allocation_percent: Percentage of available RAM to allocate.
        """
        import numpy as np

        # Calculate total allocation (for both arrays)
        total_allocation_gb = self._available_ram_gb * allocation_percent
        # Split evenly between read array and write source array
        per_array_gb = total_allocation_gb / 2

        self.logger.info(
            f"Allocating: {total_allocation_gb:.2f} GB total "
            f"({per_array_gb:.2f} GB x 2 arrays, "
            f"{allocation_percent * 100:.0f}%)"
        )

        # Allocate arrays
        num_elements = int(per_array_gb * (1024**3) / 8)  # 8 bytes per float64
        self._array = np.random.rand(num_elements)
        self._source_array_for_writes = np.random.rand(num_elements)

    def _free_arrays(self) -> None:
        """Free allocated memory arrays."""
        if self._array is not None:
            del self._array
            self._array = None
        if self._source_array_for_writes is not None:
            del self._source_array_for_writes
            self._source_array_for_writes = None

    def _get_swap_stats(self) -> dict[str, int]:
        """Get current swap memory statistics.

        Returns:
            Dictionary with 'total' and 'used' swap in bytes.
        """
        import psutil

        swap = psutil.swap_memory()
        return {"total": swap.total, "used": swap.used}

    def _benchmark_sequential_read(self, duration: int) -> float:
        """Benchmark sequential read bandwidth.

        Args:
            duration: Test duration in seconds.

        Returns:
            Read bandwidth in GB/s.
        """
        import time

        import numpy as np

        assert self._array is not None, "Array not allocated"

        start_time = time.time()
        bytes_read = 0
        iter_count = 0

        while (time.time() - start_time) < duration:
            # Sequential read: sum forces reading all elements
            # TODO: Consider random access to defeat OS/CPU prefetching
            # and guarantee severe swap degradation
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

        assert self._array is not None, "Array not allocated"
        assert self._source_array_for_writes is not None, "Source array not allocated"

        start_time = time.time()
        bytes_written = 0
        iter_count = 0

        while (time.time() - start_time) < duration:
            # Sequential write: copy from pre-allocated source array
            # TODO: Consider random access to defeat OS/CPU prefetching
            # and guarantee severe swap degradation
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
