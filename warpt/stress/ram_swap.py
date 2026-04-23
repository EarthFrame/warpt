"""RAM swap pressure stress test."""

import time

from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.models.stress_models import RAMMemoryStressResult
from warpt.stress.ram_base import RAMBaseTest


class RAMSwapPressureTest(RAMBaseTest):
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
        super().__init__()
        self.allocation_percent_baseline = allocation_percent_baseline
        self.allocation_percent_pressure = allocation_percent_pressure
        self.burnin_seconds = burnin_seconds
        self._pressure_chunks: list = []

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

    # -------------------------------------------------------------------------
    # Hardware & Availability
    # -------------------------------------------------------------------------

    def validate_configuration(self) -> None:
        """Validate test configuration."""
        if not self.is_available():
            raise RuntimeError("NumPy and psutil are required for RAM tests")
        if not 0.0 < self.allocation_percent_baseline <= 1.0:
            raise ValueError(
                f"allocation_percent_baseline must be between 0 and 1, "
                f"got {self.allocation_percent_baseline}"
            )
        if self.allocation_percent_pressure <= self.allocation_percent_baseline:
            raise ValueError(
                f"allocation_percent_pressure ({self.allocation_percent_pressure}) "
                f"must be greater than allocation_percent_baseline "
                f"({self.allocation_percent_baseline})"
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

        self._detect_memory_type()
        if self._memory_type:
            self.logger.info(f"Memory type: {self._memory_type}")

    def teardown(self) -> None:
        """Clean up allocated memory."""
        self._free_arrays()
        self.logger.debug("Memory cleaned up")

    def _free_arrays(self) -> None:
        """Free pressure chunks, then delegate to base for shared arrays."""
        if self._pressure_chunks:
            del self._pressure_chunks[:]
            self._pressure_chunks = []
        super()._free_arrays()

    def warmup(self, duration_seconds: int = 0, iterations: int = 3) -> None:
        """Run warmup to stabilize memory subsystem.

        Note: Actual warmup happens per-phase in execute_test after allocating
        arrays. This just satisfies the base class requirement.

        Args:
            duration_seconds: Warmup duration. If 0, uses self.burnin_seconds.
            iterations: Number of iterations if both duration_seconds and
                burnin_seconds are 0.
        """
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

        start = time.time()

        # Split duration between baseline and pressure phases
        phase_duration = duration // 2
        per_test_duration = phase_duration // 2  # Split each phase: read + write

        self.logger.info("=== Phase 1: Baseline (No Swap) ===")
        self._allocate_baseline()
        assert self._array is not None, "Array not allocated"

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

        # Allocate incrementally until swap is engaged
        self._allocate_pressure()
        assert self._array is not None, "Array not allocated"
        chunks_gb = sum(c.nbytes for c in self._pressure_chunks) / (1024**3)
        allocated_pressure_gb = self._array.nbytes / (1024**3) * 2 + chunks_gb

        # Measure pressure performance
        pressure_read_gbps = self._benchmark_sequential_read(per_test_duration)
        pressure_write_gbps = self._benchmark_sequential_write(per_test_duration)

        self.logger.info(f"Pressure Read: {pressure_read_gbps:.2f} GB/s")
        self.logger.info(f"Pressure Write: {pressure_write_gbps:.2f} GB/s")

        # Get final swap stats
        swap_after = self._get_swap_stats()

        # Free pressure arrays
        self._free_arrays()

        # Calculate degradation factors (clamped to 1.0 — noise can make
        # pressure slightly faster than baseline, which isn't real degradation)
        read_slowdown = max(
            baseline_read_gbps / pressure_read_gbps if pressure_read_gbps > 0 else 1.0,
            1.0,
        )
        write_slowdown = max(
            (
                baseline_write_gbps / pressure_write_gbps
                if pressure_write_gbps > 0
                else 1.0
            ),
            1.0,
        )

        # Detect if swapping occurred
        swap_occurred = swap_after["used"] > swap_before["used"]
        swap_delta_mb = (swap_after["used"] - swap_before["used"]) / (1024**2)

        actual_elapsed = time.time() - start

        # Return results
        return RAMMemoryStressResult(
            # System info
            total_ram_gb=self._total_ram_gb,
            available_ram_gb=self._available_ram_gb,
            allocated_memory_gb=allocated_pressure_gb,
            # Test metadata
            duration=actual_elapsed,
            burnin_seconds=self.burnin_seconds,
            # Memory hardware info
            memory_type=self._memory_type,
            # Baseline metrics
            baseline_read_gbps=baseline_read_gbps,
            baseline_write_gbps=baseline_write_gbps,
            # Pressure metrics
            pressure_read_gbps=pressure_read_gbps,
            pressure_write_gbps=pressure_write_gbps,
            # Performance degradation
            read_slowdown_factor=read_slowdown,
            write_slowdown_factor=write_slowdown,
            # Swap metrics
            swap_occurred=swap_occurred,
            swap_in_mb=None,  # psutil doesn't provide sin/sout on all platforms
            swap_out_mb=None,
            peak_swap_usage_mb=swap_delta_mb if swap_occurred else None,
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _allocate_baseline(self) -> None:
        """Allocate arrays for baseline test (fits in RAM)."""
        import numpy as np

        total_gb = self._available_ram_gb * self.allocation_percent_baseline
        per_array_gb = total_gb / 2
        num_elements = int(per_array_gb * (1024**3) / 8)

        self.logger.info(
            f"Allocating: {total_gb:.2f} GB total "
            f"({per_array_gb:.2f} GB x 2 arrays, "
            f"{self.allocation_percent_baseline * 100:.0f}%)"
        )

        try:
            self._array = np.random.rand(num_elements)
            self._source_array_for_writes = np.random.rand(num_elements)
        except MemoryError as exc:
            self._array = None
            self._source_array_for_writes = None
            raise RuntimeError(
                f"Failed to allocate {total_gb:.2f} GB for baseline — "
                f"system ran out of memory. Reduce allocation_percent_baseline "
                f"(currently {self.allocation_percent_baseline * 100:.0f}%)."
            ) from exc

    def _allocate_pressure(self) -> None:
        """Allocate arrays incrementally until swap is engaged.

        Starts at the baseline size and grows in chunks, checking
        psutil.swap_memory().used after each chunk. Stops when swap
        usage increases or when available memory is exhausted.
        """
        import numpy as np
        import psutil

        # Start with baseline-sized arrays
        baseline_gb = self._available_ram_gb * self.allocation_percent_baseline
        target_gb = self._available_ram_gb * self.allocation_percent_pressure
        per_array_gb = baseline_gb / 2
        num_elements = int(per_array_gb * (1024**3) / 8)

        self._array = np.random.rand(num_elements)
        self._source_array_for_writes = np.random.rand(num_elements)
        current_gb = baseline_gb

        self.logger.info(
            f"Starting at {current_gb:.2f} GB, " f"growing until swap is engaged..."
        )

        # Grow in chunks (~5% of available RAM each)
        chunk_gb = max(self._available_ram_gb * 0.05, 0.5)
        chunk_elements = int(chunk_gb * (1024**3) / 8)
        chunks: list[np.ndarray] = []

        swap_before = psutil.swap_memory().used

        while current_gb < target_gb:
            # Check if we'd exceed what the system can provide
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            free_total = mem.available + swap.free

            # Stop if less than 2 GB headroom to avoid OOM killer
            if free_total < 2 * (1024**3):
                self.logger.info(
                    f"Stopping growth at {current_gb:.2f} GB — "
                    f"only {free_total / (1024**3):.1f} GB free (RAM+swap)"
                )
                break

            try:
                chunk = np.random.rand(chunk_elements)
                chunks.append(chunk)
                current_gb += chunk_gb
            except MemoryError:
                self.logger.info(
                    f"Stopping growth at {current_gb:.2f} GB — MemoryError"
                )
                break

            swap_now = psutil.swap_memory().used
            if swap_now > swap_before:
                swap_delta_mb = (swap_now - swap_before) / (1024**2)
                self.logger.info(
                    f"Swap engaged at {current_gb:.2f} GB "
                    f"(swap used: +{swap_delta_mb:.0f} MB)"
                )

        # Consolidate: resize main arrays to include chunk data
        # We keep chunks alive to maintain memory pressure, and benchmark
        # using the main arrays.
        self.logger.info(
            f"Pressure allocation: {current_gb:.2f} GB total "
            f"({current_gb / self._available_ram_gb * 100:.0f}% of available RAM)"
        )

        # Store chunks so they stay allocated during benchmarks
        self._pressure_chunks = chunks

    def _get_swap_stats(self) -> dict[str, int]:
        """Get current swap memory statistics.

        Returns:
            Dictionary with 'total' and 'used' swap in bytes.
        """
        import psutil

        swap = psutil.swap_memory()
        return {"total": swap.total, "used": swap.used}
