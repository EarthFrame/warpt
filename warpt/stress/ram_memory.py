"""RAM memory stress tests."""

from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.models.stress_models import RAMMemoryStressResult
from warpt.stress.ram_base import RAMBaseTest


class RAMBandwidthTest(RAMBaseTest):
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
        super().__init__()
        self.allocation_percent = allocation_percent
        self.burnin_seconds = burnin_seconds
        self._allocated_gb = 0.0
        self._memory_speed_mt_s: int | None = None

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "RAM Memory Bandwidth"

    def get_description(self) -> str:
        """Return one-line description."""
        return (
            "Single-threaded sequential read/write bandwidth. Does not "
            "saturate all memory channels — results reflect what one "
            "thread can achieve, not peak hardware capability."
        )

    # -------------------------------------------------------------------------
    # Hardware & Availability
    # -------------------------------------------------------------------------

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
            f"{self.allocation_percent * 100:.0f}%)"
        )

        # Check headroom: leave at least 1 GB free to avoid OOM killer
        remaining_gb = self._available_ram_gb - total_allocation_gb
        if remaining_gb < 1.0:
            raise RuntimeError(
                f"Allocation of {total_allocation_gb:.2f} GB would leave only "
                f"{remaining_gb:.2f} GB free (need >= 1 GB headroom). "
                f"Reduce allocation_percent or free system memory."
            )

        # Allocate memory arrays (each is half of total allocation)
        num_elements = int(self._allocated_gb * (1024**3) / 8)  # 8 bytes per float64
        try:
            self._array = np.random.rand(num_elements)
            self._source_array_for_writes = np.random.rand(num_elements)
        except MemoryError as exc:
            self._array = None
            self._source_array_for_writes = None
            raise RuntimeError(
                f"Failed to allocate {total_allocation_gb:.2f} GB — "
                f"system ran out of memory. Reduce allocation_percent "
                f"(currently {self.allocation_percent * 100:.0f}%)."
            ) from exc
        self.logger.debug(f"Allocated {num_elements:,} float64 elements per array")

        # Detect memory hardware specs
        self._detect_memory_type()

        if self._memory_type:
            self.logger.info(f"Memory type: {self._memory_type}")
        self.logger.info(
            "Note: single-threaded test — measures bandwidth available to "
            "one thread, not peak hardware capability across all channels"
        )

    def teardown(self) -> None:
        """Clean up allocated memory."""
        self._free_arrays()
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
        self.logger.info(f"Phase 1/2: Sequential Read ({per_test_duration}s)")
        read_bandwidth_gbps = self._benchmark_sequential_read(per_test_duration)
        self.logger.info(f"Phase 2/2: Sequential Write ({per_test_duration}s)")
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
            mode="single-threaded",
            burnin_seconds=self.burnin_seconds,
            # Memory hardware info
            memory_type=self._memory_type,
            # Baseline metrics (actual measurements)
            baseline_read_gbps=read_bandwidth_gbps,
            baseline_write_gbps=write_bandwidth_gbps,
        )
