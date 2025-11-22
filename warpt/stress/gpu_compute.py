"""GPU compute stress tests."""

import time
from typing import Optional

from warpt.backends.base import GPUBackend
from warpt.models.constants import GPU_STRESS_TEST
from warpt.stress.base import StressTest
from warpt.stress.utils import calculate_tflops


class GPUMatMulTest(StressTest):
    """Matrix multiplication stress test for GPU."""

    def __init__(
        self,
        device_id: int,
        burnin_seconds: int,
        backend: Optional[GPUBackend] = None,
        matrix_size: int = 8192
    ):
        """
        Initialize GPU matmul test.

        Args:
            device_id: GPU device ID (0, 1, 2, etc.)
            burnin_seconds: Warmup duration before measurement
            backend: GPU backend (NvidiaBackend, AMDBackend, etc.). If None, defaults to NVIDIA.
            matrix_size: Size of square matrices (NxN). Default 8192 for GPU.
        """
        self.device_id = device_id
        self.burnin_seconds = burnin_seconds
        self.backend = backend
        self.matrix_size = matrix_size

    def run(self, duration: int) -> dict:
        """
        Run GPU matrix multiplication test.

        Args:
            duration: Test duration in seconds

        Returns:
            Dictionary containing test results
        """
        # Import PyTorch (lazy import to avoid dependency issues)
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is not installed. Install with: pip install torch")

        # Get PyTorch device string from backend or default to NVIDIA
        if self.backend:
            # Use backend to get device string (supports multiple vendors)
            device_str = self.backend.get_pytorch_device_string(self.device_id)
        else:
            # Backward compatibility: default to NVIDIA/CUDA
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Cannot run GPU stress test.")
            if self.device_id >= torch.cuda.device_count():
                raise ValueError(
                    f"GPU device {self.device_id} not found. "
                    f"Available devices: 0-{torch.cuda.device_count() - 1}"
                )
            device_str = f"cuda:{self.device_id}"

        # Set device
        device = torch.device(device_str)
        torch.cuda.set_device(device)

        # Get GPU properties
        gpu_name = torch.cuda.get_device_name(device)
        gpu_props = torch.cuda.get_device_properties(device)
        gpu_memory_total = gpu_props.total_memory / (1024**3)  # convert to GB

        # Burnin/warmup phase - let GPU warm up
        print(f"  Warming up GPU {self.device_id} ({gpu_name}) for {self.burnin_seconds}s...")
        burnin_start = time.time()
        while (time.time() - burnin_start) < self.burnin_seconds:
            A = torch.randn(self.matrix_size, self.matrix_size, dtype=torch.float32, device=device)
            B = torch.randn(self.matrix_size, self.matrix_size, dtype=torch.float32, device=device)
            C = torch.matmul(A, B)
            torch.cuda.synchronize()  # Wait for GPU to finish
            del A, B, C

        # Clear cache after burnin, before actual test
        torch.cuda.empty_cache()

        # test phase - measured performance
        print(f"  Running test on GPU {self.device_id} for {duration}s...")
        start_time = time.time()
        iterations = 0

        # Track memory usage
        torch.cuda.reset_peak_memory_stats(device)

        while (time.time() - start_time) < duration:
            A = torch.randn(self.matrix_size, self.matrix_size, dtype=torch.float32, device=device)
            B = torch.randn(self.matrix_size, self.matrix_size, dtype=torch.float32, device=device)
            C = torch.matmul(A, B)
            torch.cuda.synchronize()  # Ensure GPU completes before timing
            iterations += 1
            del A, B, C

        elapsed = time.time() - start_time

        # Get memory stats
        memory_used = torch.cuda.max_memory_allocated(device) / (1024**3)  # Convert to GB

        # Calculate TFLOPS
        # Matrix multiplication: 2*N^3 - N^2 operations
        ops_per_matmul = 2 * (self.matrix_size ** 3) - (self.matrix_size ** 2)
        total_ops = iterations * ops_per_matmul
        tflops = calculate_tflops(total_ops, elapsed)

        # clean up GPU memory
        torch.cuda.empty_cache()

        return {
            'test_name': self.get_name(),
            'device_id': self.device_id,
            'gpu_name': gpu_name,
            'tflops': tflops,
            'duration': elapsed,
            'iterations': iterations,
            'matrix_size': self.matrix_size,
            'total_operations': total_ops,
            'burnin_seconds': self.burnin_seconds,
            'memory_used_gb': memory_used,
            'memory_total_gb': gpu_memory_total,
            'precision': 'fp32',
        }

    def get_name(self) -> str:
        """Get test name."""
        return GPU_STRESS_TEST
