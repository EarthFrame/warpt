"""RAM latency stress test - measures memory hierarchy latency.

This test measures memory access latency across the cache hierarchy
(L1/L2/L3 caches and main memory) using pointer-chasing to defeat
hardware prefetchers.
"""

import itertools
import time
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.stress.base import StressTest, TestCategory

# ---------------------------------------------------------------------------
# Sattolo's algorithm  (guaranteed single-cycle permutation)
# ---------------------------------------------------------------------------


def sattolo_permutation(
    n: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Return an int64 array of length *n* that forms a single cycle.

    Sattolo's algorithm swaps ``arr[i]`` with a uniformly-random element
    from ``arr[0..i-1]`` (exclusive of *i*), guaranteeing exactly one
    cycle of length *n* and no fixed points for n >= 2.
    """
    if n < 2:
        return np.zeros(1, dtype=np.int64)

    if rng is None:
        rng = np.random.default_rng()

    arr = np.arange(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))  # [0, i) — never equals i
        arr[i], arr[j] = arr[j], arr[i]
    return arr


# ---------------------------------------------------------------------------
# Gradient-based cache-level detection
# ---------------------------------------------------------------------------


def detect_cache_boundaries(
    results: list[dict[str, Any]],
    min_ratio: float = 1.05,
    max_boundaries: int = 3,
) -> dict[str, float]:
    """Detect cache-level boundaries from a latency curve.

    Computes pairwise ratios between consecutive latency measurements
    and picks the top *max_boundaries* jumps that exceed *min_ratio*.
    Uses the median of each plateau for noise resistance.

    Returns dict with keys ``l1_latency_ns``, ``l2_latency_ns``,
    ``l3_latency_ns``, ``main_memory_latency_ns``.
    """
    latencies = [r["latency_ns"] for r in results]
    n = len(latencies)

    if n < 2:
        v = latencies[0] if latencies else 0.0
        return {
            "l1_latency_ns": v,
            "l2_latency_ns": v,
            "l3_latency_ns": v,
            "main_memory_latency_ns": v,
        }

    # Compute consecutive ratios
    ratios: list[tuple[float, int]] = []
    for i in range(1, n):
        prev = latencies[i - 1]
        if prev > 0:
            ratios.append((latencies[i] / prev, i))

    # Keep only ratios above the noise floor and sort descending
    significant = [(r, idx) for r, idx in ratios if r >= min_ratio]
    significant.sort(key=lambda t: t[0], reverse=True)

    # Pick up to max_boundaries boundary indices (sorted by position)
    boundary_indices = sorted([idx for _, idx in significant[:max_boundaries]])

    # Build plateaus (segments between boundaries)
    cuts = [0, *boundary_indices, n]
    plateaus: list[float] = []
    for s, e in itertools.pairwise(cuts):
        segment = latencies[s:e]
        plateaus.append(float(np.median(segment)))

    # Pad / truncate to exactly 4 levels: L1, L2, L3, main memory
    while len(plateaus) < 4:
        plateaus.append(plateaus[-1])

    return {
        "l1_latency_ns": plateaus[0],
        "l2_latency_ns": plateaus[1],
        "l3_latency_ns": plateaus[2],
        "main_memory_latency_ns": plateaus[3],
    }


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
        self._backend = "python_fallback"

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

        # Resolve pointer-chase backend
        from warpt.stress._pointer_chase import get_pointer_chase

        cpp_fn = get_pointer_chase()
        if cpp_fn is not None:
            self._chase_fn = cpp_fn
            self._backend = "cpp"
            self.logger.info("Using C++ pointer-chase backend")
        else:
            from warpt.stress._pointer_chase import pointer_chase_python

            self._chase_fn = pointer_chase_python
            self._backend = "python_fallback"
            warnings.warn(
                "C++ pointer-chase unavailable; falling back to Python "
                "(latencies will include ~60 ns interpreter overhead)",
                RuntimeWarning,
                stacklevel=2,
            )

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
        # Use burnin_seconds if no duration specified
        if duration_seconds == 0:
            duration_seconds = self.burnin_seconds

        if duration_seconds > 0:
            self.logger.debug(f"Warming up for {duration_seconds}s...")
            start = time.time()
            while (time.time() - start) < duration_seconds:
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

        # Detect cache levels via gradient analysis
        cache_levels = detect_cache_boundaries(results)

        l1_latency = cache_levels["l1_latency_ns"]
        l2_latency = cache_levels["l2_latency_ns"]
        l3_latency = cache_levels["l3_latency_ns"]
        mem_latency = cache_levels["main_memory_latency_ns"]

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
            # Backend used
            "backend": self._backend,
        }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _measure_latency_for_size(self, size_kb: int) -> float:
        """Measure memory latency for a specific array size.

        Uses pointer-chasing with Sattolo's permutation (guaranteed
        single cycle) and a C++ chase loop to eliminate interpreter
        overhead.

        Args:
            size_kb: Array size in KB.

        Returns:
            Average latency per access in nanoseconds.
        """
        # Calculate number of int64 elements (8 bytes each)
        num_elements = (size_kb * 1024) // 8

        # Ensure at least 2 elements
        if num_elements < 2:
            num_elements = 2

        # Create single-cycle permutation via Sattolo's algorithm
        permutation = sattolo_permutation(num_elements)

        # Warm up the array (load into cache at appropriate level)
        _ = permutation.sum()

        iters = self.iterations_per_size

        start_time = time.perf_counter_ns()
        _ = self._chase_fn(permutation, iters)
        elapsed_ns = time.perf_counter_ns() - start_time

        return elapsed_ns / iters
