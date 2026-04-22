"""Tests for RAMLatencyTest, Sattolo permutation, and cache detection."""

from io import StringIO
from unittest.mock import patch

import numpy as np
import pytest

from warpt.stress._pointer_chase import (
    get_pointer_chase,
    pointer_chase_python,
    reset_cache,
)
from warpt.stress.ram_latency import (
    RAMLatencyTest,
    detect_cache_boundaries,
    sattolo_permutation,
)
from warpt.utils.logger import Logger


@pytest.fixture(autouse=True)
def _configure_logger():
    """Ensure the warpt logger is configured for every test."""
    if not Logger.is_configured():
        Logger.configure(level="WARNING", output=StringIO())


# =========================================================================
# Sattolo Permutation
# =========================================================================


class TestSattoloPermutation:
    """Tests for sattolo_permutation()."""

    def test_single_cycle(self):
        """Every element must be reachable from index 0."""
        n = 200
        perm = sattolo_permutation(n, rng=np.random.default_rng(42))
        visited = set()
        idx = 0
        for _ in range(n):
            idx = int(perm[idx])
            visited.add(idx)
        assert visited == set(range(n)), "Not all elements visited — not a single cycle"

    def test_visits_all_elements(self):
        """Cycle length must equal n (visits every element exactly once)."""
        n = 500
        perm = sattolo_permutation(n, rng=np.random.default_rng(7))
        idx = 0
        for step in range(1, n + 1):
            idx = int(perm[idx])
            if idx == 0:
                assert step == n, f"Cycle closed after {step} steps, expected {n}"
                break
        else:
            pytest.fail("Cycle did not close after n steps")

    def test_no_fixed_points(self):
        """No element should map to itself (perm[i] != i for all i)."""
        n = 300
        perm = sattolo_permutation(n, rng=np.random.default_rng(99))
        indices = np.arange(n, dtype=np.int64)
        assert not np.any(perm == indices), "Found fixed point(s)"

    def test_dtype_is_int64(self):
        perm = sattolo_permutation(50)
        assert perm.dtype == np.int64

    def test_small_n(self):
        """n=2 should produce [1, 0]."""
        perm = sattolo_permutation(2)
        assert list(perm) == [1, 0]

    def test_n_one(self):
        """n=1 degenerate case returns [0]."""
        perm = sattolo_permutation(1)
        assert list(perm) == [0]


# =========================================================================
# Cache Boundary Detection
# =========================================================================


class TestCacheDetection:
    """Tests for detect_cache_boundaries()."""

    def test_clear_boundaries(self):
        """Three clear jumps should give L1 < L2 < L3 < main memory."""
        # Simulate: L1 ~2ns (4 pts), L2 ~8ns (4 pts), L3 ~25ns (4 pts), mem ~80ns (4 pts)
        results = (
            [{"size_kb": 2**i, "latency_ns": 2.0} for i in range(4)]
            + [{"size_kb": 2 ** (i + 4), "latency_ns": 8.0} for i in range(4)]
            + [{"size_kb": 2 ** (i + 8), "latency_ns": 25.0} for i in range(4)]
            + [{"size_kb": 2 ** (i + 12), "latency_ns": 80.0} for i in range(4)]
        )
        levels = detect_cache_boundaries(results)

        assert levels["l1_latency_ns"] < levels["l2_latency_ns"]
        assert levels["l2_latency_ns"] < levels["l3_latency_ns"]
        assert levels["l3_latency_ns"] <= levels["main_memory_latency_ns"]

    def test_monotonic_ordering(self):
        """Output latencies must be non-decreasing (L1 <= L2 <= L3 <= mem)."""
        results = [
            {"size_kb": 1, "latency_ns": 1.5},
            {"size_kb": 4, "latency_ns": 1.6},
            {"size_kb": 16, "latency_ns": 5.0},
            {"size_kb": 64, "latency_ns": 5.2},
            {"size_kb": 256, "latency_ns": 20.0},
            {"size_kb": 1024, "latency_ns": 21.0},
            {"size_kb": 4096, "latency_ns": 70.0},
            {"size_kb": 16384, "latency_ns": 72.0},
        ]
        levels = detect_cache_boundaries(results)
        assert levels["l1_latency_ns"] <= levels["l2_latency_ns"]
        assert levels["l2_latency_ns"] <= levels["l3_latency_ns"]
        assert levels["l3_latency_ns"] <= levels["main_memory_latency_ns"]

    def test_compressed_range_no_crash(self):
        """All-equal latencies should not crash or produce negative values."""
        results = [{"size_kb": 2**i, "latency_ns": 5.0} for i in range(10)]
        levels = detect_cache_boundaries(results)
        assert levels["l1_latency_ns"] == pytest.approx(5.0)
        assert levels["main_memory_latency_ns"] == pytest.approx(5.0)

    def test_two_elements(self):
        """Minimal input with 2 data points."""
        results = [
            {"size_kb": 1, "latency_ns": 2.0},
            {"size_kb": 1024, "latency_ns": 80.0},
        ]
        levels = detect_cache_boundaries(results)
        assert levels["l1_latency_ns"] <= levels["main_memory_latency_ns"]

    def test_single_element(self):
        results = [{"size_kb": 1, "latency_ns": 3.0}]
        levels = detect_cache_boundaries(results)
        assert levels["l1_latency_ns"] == pytest.approx(3.0)

    def test_min_ratio_filters_noise(self):
        """Tiny jumps below min_ratio should not be treated as boundaries."""
        results = [{"size_kb": 2**i, "latency_ns": 5.0 + i * 0.01} for i in range(10)]
        levels = detect_cache_boundaries(results, min_ratio=1.05)
        # All values within noise — should collapse to one plateau
        assert levels["l1_latency_ns"] == pytest.approx(
            levels["main_memory_latency_ns"], abs=0.1
        )


# =========================================================================
# Pointer Chase
# =========================================================================


class TestPointerChase:
    """Tests for pointer chase backends."""

    def test_python_fallback_correctness(self):
        """Python fallback should follow the chain correctly."""
        # Simple cycle: 0->1->2->3->0
        arr = np.array([1, 2, 3, 0], dtype=np.int64)
        result = pointer_chase_python(arr, 4)
        assert result == 0  # back to start after full cycle

    def test_python_fallback_partial(self):
        arr = np.array([1, 2, 3, 0], dtype=np.int64)
        result = pointer_chase_python(arr, 2)
        assert result == 2  # 0->1->2

    def test_cpp_extension_available(self):
        """C++ extension should load and produce same results as Python."""
        reset_cache()
        cpp_fn = get_pointer_chase()
        if cpp_fn is None:
            pytest.skip("C++ extension not built")

        arr = np.array([1, 2, 3, 0], dtype=np.int64)
        assert cpp_fn(arr, 4) == 0
        assert cpp_fn(arr, 2) == 2
        reset_cache()

    def test_fallback_when_extension_missing(self):
        """get_pointer_chase returns None when extension is not importable."""
        reset_cache()
        with patch.dict("sys.modules", {"warpt.stress._pointer_chase_ext": None}):
            result = get_pointer_chase()
            assert result is None
        reset_cache()


# =========================================================================
# Integration
# =========================================================================


class TestRAMLatencyTestIntegration:
    """Integration tests for the full RAMLatencyTest."""

    def test_expected_result_keys(self):
        """Result dict should contain all expected keys."""
        test = RAMLatencyTest(iterations_per_size=100_000, burnin_seconds=0)
        test.validate_configuration()
        test.setup()
        try:
            result = test.execute_test(duration=0, iterations=0)
        finally:
            test.teardown()

        expected_keys = {
            "test_name",
            "duration",
            "iterations_per_size",
            "burnin_seconds",
            "total_ram_gb",
            "available_ram_gb",
            "l1_latency_ns",
            "l2_latency_ns",
            "l3_latency_ns",
            "main_memory_latency_ns",
            "min_latency_ns",
            "max_latency_ns",
            "latency_curve",
            "backend",
        }
        assert expected_keys.issubset(result.keys())

    def test_backend_field(self):
        """Backend should be 'cpp' or 'python_fallback'."""
        test = RAMLatencyTest(iterations_per_size=100_000, burnin_seconds=0)
        test.validate_configuration()
        test.setup()
        try:
            result = test.execute_test(duration=0, iterations=0)
        finally:
            test.teardown()

        assert result["backend"] in ("cpp", "python_fallback")

    def test_latency_ordering(self):
        """Detected cache latencies should be non-decreasing."""
        test = RAMLatencyTest(iterations_per_size=100_000, burnin_seconds=0)
        test.validate_configuration()
        test.setup()
        try:
            result = test.execute_test(duration=0, iterations=0)
        finally:
            test.teardown()

        assert result["l1_latency_ns"] <= result["l2_latency_ns"]
        assert result["l2_latency_ns"] <= result["l3_latency_ns"]
        assert result["l3_latency_ns"] <= result["main_memory_latency_ns"]

    def test_latency_curve_length(self):
        """Latency curve should have one entry per test size (19 sizes)."""
        test = RAMLatencyTest(iterations_per_size=100_000, burnin_seconds=0)
        test.validate_configuration()
        test.setup()
        try:
            result = test.execute_test(duration=0, iterations=0)
        finally:
            test.teardown()

        assert len(result["latency_curve"]) == 19
