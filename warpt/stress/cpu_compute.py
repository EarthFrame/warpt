"""CPU compute stress tests."""

import time

import numpy as np

from warpt.models.constants import DEFAULT_BURNIN_DURATION
from warpt.stress.base import StressTest


class CPUMatMulTest(StressTest):
    """Matrix multiplication stress test for CPU."""

    def __init__(self, matrix_size: int = 4096, burnin_seconds: int = DEFAULT_BURNIN_DURATION):
        """
        Initialize CPU matmul test.

        Args:
            matrix_size: Size of square matrices (NxN)
            burnin_seconds: Warmup duration before measurement
        """
        self.matrix_size = matrix_size
        self.burnin_seconds = burnin_seconds

    def run(self, duration: int) -> dict:
        """
        Run CPU matrix multiplication test.

        Args:
            duration: Test duration in seconds

        Returns:
            Dictionary containing test results (TFLOPS, etc.)
        """
        # Burnin/warmup phase - let CPU/cache warm up
        print(f"  Warming up for {self.burnin_seconds}s...")
        burnin_start = time.time()
        while (time.time() - burnin_start) < self.burnin_seconds:
            A = np.random.rand(self.matrix_size, self.matrix_size)
            B = np.random.rand(self.matrix_size, self.matrix_size)
            C = np.matmul(A, B)
            del A, B, C  # Free memory

        # Actual test phase - measured performance
        print(f"  Running test for {duration}s...")
        start_time = time.time()
        iterations = 0

        while (time.time() - start_time) < duration:
            A = np.random.rand(self.matrix_size, self.matrix_size)
            B = np.random.rand(self.matrix_size, self.matrix_size)
            C = np.matmul(A, B)
            iterations += 1
            del A, B, C  # Free memory

        elapsed = time.time() - start_time

        # Calculate TFLOPS
        # Matrix multiplication: 2*N^3 - N^2 operations
        ops_per_matmul = 2 * (self.matrix_size ** 3) - (self.matrix_size ** 2)
        total_ops = iterations * ops_per_matmul
        tflops = (total_ops / elapsed) / 1e12

        return {
            'test_name': self.get_name(),
            'tflops': tflops,
            'duration': elapsed,
            'iterations': iterations,
            'matrix_size': self.matrix_size,
            'total_operations': total_ops,
            'burnin_seconds': self.burnin_seconds,
        }

    def get_name(self) -> str:
        """Get test name."""
        return "CPU Matrix Multiplication"
