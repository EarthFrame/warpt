"""RAM latency stress test - measures memory hierarchy latency.

This test measures memory access latency across the cache hierarchy
(L1/L2/L3 caches and main memory) using pointer-chasing to defeat
hardware prefetchers.
"""

import time
from typing import Any

from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.stress.base import StressTest, TestCategory


class RAMLatencyTest(StressTest):
    """Memory latency stress test for cache hierarchy profiling.

    Measures memory access latency at different array sizes to detect:
    - L1 cache latency (typically 1-4 cycles, ~1-2 ns)
    - L2 cache latency (typically 10-20 cycles, ~5-10 ns)
    - L3 cache latency (typically 40-80 cycles, ~20-40 ns)
    - Main memory latency (typically 200-300 cycles, ~80-120 ns)

    Uses pointer-chasing (dependent loads) to defeat hardware prefetchers,
    ensuring we measure true random-access latency.
    """

    _PARAM_FIELDS = ("iterations_per_size", "burnin_seconds")

    def __init__(
        self,
        iterations_per_size: int = 10000000,
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
    ):
        """Initialize RAM latency test.

        Args:
            iterations_per_size: Number of pointer-chasing iterations per
                array size. Default 10M (takes ~1-2s per size).
            burnin_seconds: Warmup duration before measurement.
        """
        self.iterations_per_size = iterations_per_size
        self.burnin_seconds = burnin_seconds
        self._total_ram_gb = 0.0
        self._available_ram_gb = 0.0

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "RAM Latency Test"

    def get_description(self) -> str:
        """Return one-line description."""
        return (
            "Measures memory access latency across cache hierarchy "
            "(L1/L2/L3/main memory)"
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
        if self.iterations_per_size < 100000:
            raise ValueError(
                "iterations_per_size must be >= 100,000 for statistical accuracy"
            )
        if self.iterations_per_size > 100000000:
            raise ValueError(
                "iterations_per_size must be <= 100,000,000 (avoid excessive runtime)"
            )
        if self.burnin_seconds < 0:
            raise ValueError("burnin_seconds must be >= 0")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize system info."""
        import psutil

        mem = psutil.virtual_memory()
        self._total_ram_gb = mem.total / (1024**3)
        self._available_ram_gb = mem.available / (1024**3)

        self.logger.info(f"Total RAM: {self._total_ram_gb:.2f} GB")
        self.logger.info(f"Available RAM: {self._available_ram_gb:.2f} GB")

    def teardown(self) -> None:
        """Clean up resources."""
        pass

    def warmup(self, duration_seconds: int = 0, iterations: int = 3) -> None:
        """Run warmup to stabilize memory subsystem.

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
                # Warmup with small array (fits in cache)
                arr = np.arange(1000, dtype=np.int64)
                _ = arr.sum()
                del arr
        else:
            self.logger.debug(f"Warming up for {iterations} iterations...")
            for _ in range(iterations):
                arr = np.arange(1000, dtype=np.int64)
                _ = arr.sum()
                del arr

    # -------------------------------------------------------------------------
    # Core Test
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> dict[str, Any]:
        """Execute the RAM latency test.

        Args:
            duration: Ignored (test runs fixed iterations per size).
            iterations: Ignored (uses self.iterations_per_size).

        Returns:
            Dictionary containing latency measurements.
        """
        del duration, iterations  # Unused

        self.logger.info("Measuring memory latency across cache hierarchy...")
        self.logger.info(f"Iterations per size: {self.iterations_per_size:,}")

        # Test array sizes: 1KB to 256MB (powers of 2)
        # This sweeps from L1 cache to main memory
        test_sizes_kb = [
            1,
            2,
            4,
            8,  # L1 cache (typically 32-64 KB per core)
            16,
            32,
            64,
            128,  # L2 cache (typically 256-512 KB per core)
            256,
            512,
            1024,
            2048,  # L3 cache (typically 2-32 MB shared)
            4096,
            8192,
            16384,
            32768,  # Main memory
            65536,
            131072,
            262144,  # 256 MB
        ]

        results = []
        start_time = time.time()

        for size_kb in test_sizes_kb:
            latency_ns = self._measure_latency_for_size(size_kb)
            results.append({"size_kb": size_kb, "latency_ns": latency_ns})

            self.logger.info(f"  {size_kb:>6} KB: {latency_ns:6.2f} ns/access")

        elapsed = time.time() - start_time

        # Identify cache levels (simple heuristic: look for latency jumps)
        l1_latency = results[0]["latency_ns"]  # Smallest size
        l2_latency = self._find_cache_level_latency(results, l1_latency * 1.5)
        l3_latency = self._find_cache_level_latency(results, l2_latency * 1.5)
        mem_latency = results[-1]["latency_ns"]  # Largest size

        # Calculate min/max across all measurements
        all_latencies = [r["latency_ns"] for r in results]
        min_latency = min(all_latencies)
        max_latency = max(all_latencies)

        self.logger.info("\nEstimated cache latencies:")
        self.logger.info(f"  L1 cache: {l1_latency:.2f} ns")
        self.logger.info(f"  L2 cache: {l2_latency:.2f} ns")
        self.logger.info(f"  L3 cache: {l3_latency:.2f} ns")
        self.logger.info(f"  Main memory: {mem_latency:.2f} ns")
        self.logger.info(f"  Range: min={min_latency:.2f} ns, max={max_latency:.2f} ns")

        return {
            "test_name": self.get_name(),
            "duration": elapsed,
            "iterations_per_size": self.iterations_per_size,
            "burnin_seconds": self.burnin_seconds,
            "total_ram_gb": self._total_ram_gb,
            "available_ram_gb": self._available_ram_gb,
            # Estimated cache latencies
            "l1_latency_ns": l1_latency,
            "l2_latency_ns": l2_latency,
            "l3_latency_ns": l3_latency,
            "main_memory_latency_ns": mem_latency,
            # Min/max across all measurements
            "min_latency_ns": min_latency,
            "max_latency_ns": max_latency,
            # Full latency curve
            "latency_curve": results,
        }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _measure_latency_for_size(self, size_kb: int) -> float:
        """Measure memory latency for a specific array size.

        Uses pointer-chasing technique: create a random permutation where
        each element points to the next, forcing dependent loads that
        defeat hardware prefetchers.

        Args:
            size_kb: Array size in KB.

        Returns:
            Average latency per access in nanoseconds.
        """
        import numpy as np

        # Calculate number of int64 elements (8 bytes each)
        num_elements = (size_kb * 1024) // 8

        # Ensure at least 2 elements
        if num_elements < 2:
            num_elements = 2

        # Create random permutation for pointer chasing
        # permutation[i] contains the index of the next element to access
        permutation = np.arange(num_elements, dtype=np.int64)
        np.random.shuffle(permutation)

        # Ensure it's a single cycle (last element points to first)
        permutation[-1] = permutation[0]

        # Warm up the array (load into cache at appropriate level)
        _ = permutation.sum()

        # Pointer chasing loop: each access depends on previous
        # This defeats prefetchers and measures true random-access latency
        iterations = self.iterations_per_size
        index = 0

        start_time = time.perf_counter_ns()

        for _ in range(iterations):
            index = permutation[index]

        elapsed_ns = time.perf_counter_ns() - start_time

        # Calculate average latency per access
        latency_ns = elapsed_ns / iterations

        # Use index to prevent compiler from optimizing away the loop
        if index < 0:
            self.logger.debug(f"Impossible: index={index}")

        return latency_ns

    def _find_cache_level_latency(
        self, results: list[dict[str, Any]], threshold_ns: float
    ) -> float:
        """Find latency at a cache level using threshold heuristic.

        Args:
            results: List of {size_kb, latency_ns} dicts.
            threshold_ns: Latency threshold to look for.

        Returns:
            Latency in nanoseconds at the detected cache level.
        """
        for result in results:
            if result["latency_ns"] >= threshold_ns:
                return float(result["latency_ns"])

        # If no threshold found, return last measurement
        return float(results[-1]["latency_ns"])
