"""RAM workload simulation test - simulates real application memory usage patterns.

This test simulates different workload types (ML training, database, key-value store)
at various RAM allocation levels to determine minimum RAM requirements for
acceptable performance.
"""

import random
import time
from typing import Any

import numpy as np

from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.stress.base import StressTest, TestCategory


class RAMWorkloadSimulationTest(StressTest):
    """Workload simulation test for RAM performance analysis.

    Simulates real application memory usage patterns to determine how
    performance degrades as available RAM decreases. Tests three workload types:

    1. ML Training: Large sequential reads (batch loading) +
       random writes (weight updates)
    2. Database: Random reads (queries) + random writes (inserts/updates)
    3. Key-Value Store: Heavy random reads (cache hits) +
       light writes (Zipf distribution)

    The test measures operations/second at different RAM allocation levels
    to find the "knee point" where performance degrades significantly.
    """

    _PARAM_FIELDS = (
        "workload_mode",
        "operations_per_level",
        "test_ram_percentages",
        "burnin_seconds",
    )

    def __init__(
        self,
        workload_mode: str = "database",
        operations_per_level: int = 100000,
        test_ram_percentages: str = "0.3,0.5,0.7,0.9",
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
    ):
        """Initialize RAM workload simulation test.

        Args:
            workload_mode: Workload type to simulate.
                - "ml": ML training (80% seq read, 20% random write)
                - "database": Database (50% random read, 50% random write)
                - "keyvalue": Key-value (90% random read, 10% write, Zipf)
            operations_per_level: Number of operations to run at each RAM level.
                Default 100,000 (takes ~1-2s per level).
            test_ram_percentages: Comma-separated RAM allocation percentages.
                Default "0.3,0.5,0.7,0.9" tests at 30%, 50%, 70%, 90%.
            burnin_seconds: Warmup duration before measurement.
        """
        self.workload_mode = workload_mode.lower()
        self.operations_per_level = operations_per_level
        self.burnin_seconds = burnin_seconds

        # Parse test percentages
        self.test_ram_percentages = [
            float(p.strip()) for p in test_ram_percentages.split(",")
        ]

        self._total_ram_gb = 0.0
        self._available_ram_gb = 0.0

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "RAM Workload Simulation Test"

    def get_description(self) -> str:
        """Return one-line description."""
        return (
            "Simulates real workloads (ML/database/key-value) at different "
            "RAM levels to find performance degradation points"
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

        valid_modes = ["ml", "database", "keyvalue"]
        if self.workload_mode not in valid_modes:
            raise ValueError(
                f"workload_mode must be one of {valid_modes}, "
                f"got '{self.workload_mode}'"
            )

        if self.operations_per_level < 10000:
            raise ValueError(
                "operations_per_level must be >= 10,000 for statistical accuracy"
            )
        if self.operations_per_level > 10000000:
            raise ValueError(
                "operations_per_level must be <= 10,000,000 (avoid excessive runtime)"
            )

        for pct in self.test_ram_percentages:
            if not 0.1 <= pct <= 1.0:
                raise ValueError(
                    f"RAM percentages must be between 0.1 and 1.0, got {pct}"
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
        self.logger.info(f"Workload mode: {self.workload_mode}")

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
                # Warmup with small array
                arr = np.arange(10000, dtype=np.int64)
                _ = arr.sum()
                del arr
        else:
            self.logger.debug(f"Warming up for {iterations} iterations...")
            for _ in range(iterations):
                arr = np.arange(10000, dtype=np.int64)
                _ = arr.sum()
                del arr

    # -------------------------------------------------------------------------
    # Core Test
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> dict[str, Any]:
        """Execute the RAM workload simulation test.

        Args:
            duration: Ignored (test runs fixed operations per level).
            iterations: Ignored (uses self.operations_per_level).

        Returns:
            Dictionary containing performance measurements at each RAM level.
        """
        del duration, iterations  # Unused

        self.logger.info(
            f"Simulating {self.workload_mode} workload at different RAM levels..."
        )
        self.logger.info(f"Operations per level: {self.operations_per_level:,}")
        percentages = ", ".join(f"{p * 100:.0f}%" for p in self.test_ram_percentages)
        self.logger.info(f"Testing at: {percentages}")

        results = []
        start_time = time.time()

        for pct in sorted(self.test_ram_percentages):
            result = self._test_at_ram_level(pct)
            results.append(result)

            self.logger.info(
                f"  {pct * 100:>3.0f}% RAM: {result['ops_per_sec']:>8.0f} ops/sec, "
                f"avg latency: {result['avg_latency_us']:.2f} μs"
            )

        elapsed = time.time() - start_time

        # Find performance degradation
        max_ops = max(r["ops_per_sec"] for r in results)
        degradation_threshold = max_ops * 0.8  # 20% degradation

        recommended_pct = None
        for result in sorted(results, key=lambda r: r["ram_pct"]):
            if result["ops_per_sec"] >= degradation_threshold:
                recommended_pct = result["ram_pct"]
                break

        if recommended_pct:
            recommended_gb = self._available_ram_gb * recommended_pct
            self.logger.info(
                f"\nRecommendation: Maintain at least {recommended_pct * 100:.0f}% "
                f"RAM ({recommended_gb:.1f} GB) for acceptable performance"
            )
        else:
            self.logger.info("\nWarning: Performance degraded at all tested RAM levels")

        return {
            "test_name": self.get_name(),
            "duration": elapsed,
            "workload_mode": self.workload_mode,
            "operations_per_level": self.operations_per_level,
            "burnin_seconds": self.burnin_seconds,
            "total_ram_gb": self._total_ram_gb,
            "available_ram_gb": self._available_ram_gb,
            # Performance curve
            "performance_curve": results,
            # Recommendation
            "recommended_ram_pct": recommended_pct,
            "recommended_ram_gb": (
                self._available_ram_gb * recommended_pct if recommended_pct else None
            ),
        }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _test_at_ram_level(self, ram_pct: float) -> dict[str, Any]:
        """Test workload at a specific RAM allocation level.

        Args:
            ram_pct: Percentage of available RAM to use (0.0-1.0).

        Returns:
            Dictionary with performance metrics at this RAM level.
        """
        # Allocate array at specified RAM percentage
        ram_gb = self._available_ram_gb * ram_pct
        num_elements = int((ram_gb * (1024**3)) / 8)  # 8 bytes per int64

        # Create working set
        data = np.random.randint(0, 1000000, size=num_elements, dtype=np.int64)

        # Run workload simulation
        latencies = []
        start_time = time.perf_counter()

        for _ in range(self.operations_per_level):
            op_start = time.perf_counter_ns()

            if self.workload_mode == "ml":
                self._ml_operation(data)
            elif self.workload_mode == "database":
                self._database_operation(data)
            elif self.workload_mode == "keyvalue":
                self._keyvalue_operation(data)

            op_elapsed_ns = time.perf_counter_ns() - op_start
            latencies.append(op_elapsed_ns / 1000.0)  # Convert to microseconds

        elapsed = time.perf_counter() - start_time

        # Clean up
        del data

        # Calculate metrics
        ops_per_sec = self.operations_per_level / elapsed if elapsed > 0 else 0
        avg_latency_us = sum(latencies) / len(latencies) if latencies else 0

        return {
            "ram_pct": ram_pct,
            "ram_gb": ram_gb,
            "ops_per_sec": ops_per_sec,
            "avg_latency_us": avg_latency_us,
            "duration": elapsed,
        }

    def _ml_operation(self, data: np.ndarray) -> None:
        """Simulate ML training operation: 80% sequential read, 20% random write."""
        if random.random() < 0.8:
            # Sequential read (batch loading)
            batch_size = min(1000, len(data))
            start_idx = random.randint(0, len(data) - batch_size)
            _ = data[start_idx : start_idx + batch_size].sum()
        else:
            # Random write (weight update)
            idx = random.randint(0, len(data) - 1)
            data[idx] = random.randint(0, 1000000)

    def _database_operation(self, data: np.ndarray) -> None:
        """Simulate database operation: 50% random read, 50% random write."""
        idx = random.randint(0, len(data) - 1)

        if random.random() < 0.5:
            # Random read (query)
            _ = data[idx]
        else:
            # Random write (insert/update)
            data[idx] = random.randint(0, 1000000)

    def _keyvalue_operation(self, data: np.ndarray) -> None:
        """Simulate key-value store: 90% random read, 10% random write (Zipf)."""
        # Zipf distribution: some keys accessed much more frequently
        # Using simple approximation: 80% of accesses hit 20% of keys
        if random.random() < 0.8:
            # Hot keys (first 20% of array)
            idx = random.randint(0, len(data) // 5)
        else:
            # Cold keys (remaining 80% of array)
            idx = random.randint(len(data) // 5, len(data) - 1)

        if random.random() < 0.9:
            # Read (cache hit)
            _ = data[idx]
        else:
            # Write (cache update)
            data[idx] = random.randint(0, 1000000)
