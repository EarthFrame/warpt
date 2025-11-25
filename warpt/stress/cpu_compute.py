"""CPU compute stress tests."""

import time

import numpy as np

from warpt.backends.system import CPU
from warpt.models.constants import (
    CPU_STRESS_TEST,
    DEFAULT_BURNIN_SECONDS,
)
from warpt.stress.base import StressTest
from warpt.stress.utils import calculate_tflops


class CPUMatMulTest(StressTest):
    """Matrix multiplication stress test for CPU."""

    def __init__(
        self, matrix_size: int = 4096, burnin_seconds: int = DEFAULT_BURNIN_SECONDS
    ):
        """Initialize CPU matmul test.

        Args:
            matrix_size: Size of square matrices (NxN)
            burnin_seconds: Warmup duration before measurement
        """
        self.matrix_size = matrix_size
        self.burnin_seconds = burnin_seconds

    def run(self, duration: int) -> dict:
        """Run CPU matrix multiplication test.

        Args:
            duration: Test duration in seconds

        Returns
        -------
            Dictionary containing test results (TFLOPS, etc.)
        """
        # TODO: Monitor CPU utilization during test to verify all cores are
        #       being used. Use psutil.cpu_percent(percpu=True) to check
        #       per-core utilization. NumPy's BLAS (Accelerate on macOS) should
        #       auto-parallelize, but we should verify

        # Get CPU info for core count
        cpu = CPU()
        cpu_info = cpu.get_cpu_info()

        # Burnin/warmup phase - let CPU/cache warm up
        print(f"  Warming up for {self.burnin_seconds}s...")
        burnin_start = time.time()
        while (time.time() - burnin_start) < self.burnin_seconds:
            a = np.random.rand(self.matrix_size, self.matrix_size)
            b = np.random.rand(self.matrix_size, self.matrix_size)
            c = np.matmul(a, b)
            del a, b, c  # Free memory

        # Actual test phase - measured performance
        print(f"  Running test for {duration}s...")
        start_time = time.time()
        iterations = 0

        while (time.time() - start_time) < duration:
            a = np.random.rand(self.matrix_size, self.matrix_size)
            b = np.random.rand(self.matrix_size, self.matrix_size)
            c = np.matmul(a, b)
            iterations += 1
            del a, b, c  # Free memory

        elapsed = time.time() - start_time

        # Calculate TFLOPS
        # Matrix multiplication: 2*N^3 - N^2 operations
        ops_per_matmul = 2 * (self.matrix_size**3) - (self.matrix_size**2)
        total_ops = iterations * ops_per_matmul
        tflops = calculate_tflops(total_ops, elapsed)

        return {
            "test_name": self.get_name(),
            "tflops": tflops,
            "duration": elapsed,
            "iterations": iterations,
            "matrix_size": self.matrix_size,
            "total_operations": total_ops,
            "burnin_seconds": self.burnin_seconds,
            "cpu_physical_cores": cpu_info.total_physical_cores,
            "cpu_logical_cores": cpu_info.total_logical_cores,
        }

    def get_name(self) -> str:
        """Get test name."""
        return CPU_STRESS_TEST
