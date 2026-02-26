"""GPU FP64 compute stress test.

This test measures double-precision floating point (FP64) performance,
which is critical for scientific computing workloads.
"""

import time
from typing import Any

from warpt.backends.base import AcceleratorBackend
from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.stress.base import StressTest, TestCategory


class GPUFP64ComputeTest(StressTest):
    """GPU FP64 compute performance test.

    Measures double-precision (FP64) floating point performance using
    matrix operations (GEMM).

    Many GPUs have severely limited FP64 performance:
    - Gaming GPUs (RTX): 1/32 of FP32 speed
    - Professional GPUs (A100, H100): 1/2 of FP32 speed
    - Scientific GPUs (MI250X): Full FP64 support

    This test helps identify GPUs suitable for scientific computing.
    """

    _PARAM_FIELDS = ("matrix_size", "device_id", "burnin_seconds")

    def __init__(
        self,
        matrix_size: int = 8192,
        device_id: int = 0,
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
        backend: AcceleratorBackend | None = None,
    ):
        """Initialize GPU FP64 compute test.

        Args:
            matrix_size: Size of square matrices for GEMM operations.
                Default 8192 (8K x 8K = ~512MB per matrix in FP64).
            device_id: GPU device ID to test. Default 0.
            burnin_seconds: Warmup duration before measurement.
            backend: GPU backend (NvidiaBackend, AMDBackend, etc.).
                If None, defaults to NVIDIA/CUDA.
        """
        self.matrix_size = matrix_size
        self.device_id = device_id
        self.burnin_seconds = burnin_seconds
        self.backend = backend
        self._device = None
        self._gpu_name = None

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "GPU FP64 Compute Test"

    def get_description(self) -> str:
        """Return one-line description."""
        return (
            "Measures double-precision (FP64) floating point performance "
            "for scientific computing workloads"
        )

    def get_category(self) -> TestCategory:
        """Return test category."""
        return TestCategory.ACCELERATOR

    # -------------------------------------------------------------------------
    # Hardware & Availability
    # -------------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if PyTorch with CUDA is available."""
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def validate_configuration(self) -> None:
        """Validate test configuration."""
        import torch

        if not self.is_available():
            raise RuntimeError("CUDA-capable GPU and PyTorch are required")

        if self.device_id < 0 or self.device_id >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid device_id {self.device_id}. "
                f"Available devices: 0-{torch.cuda.device_count() - 1}"
            )

        if self.matrix_size < 512:
            raise ValueError("matrix_size must be >= 512 for meaningful results")
        if self.matrix_size > 16384:
            raise ValueError(
                "matrix_size must be <= 16384 (avoid excessive memory usage)"
            )

        if self.burnin_seconds < 0:
            raise ValueError("burnin_seconds must be >= 0")

        # Check if matrix will fit in GPU memory
        bytes_per_matrix = self.matrix_size * self.matrix_size * 8  # 8 bytes per FP64
        total_bytes_needed = bytes_per_matrix * 3  # A, B, C matrices
        gpu_memory = torch.cuda.get_device_properties(self.device_id).total_memory

        if total_bytes_needed > gpu_memory * 0.8:  # Leave 20% headroom
            raise ValueError(
                f"matrix_size {self.matrix_size} requires "
                f"{total_bytes_needed / (1024**3):.2f} GB "
                f"but GPU only has {gpu_memory / (1024**3):.2f} GB"
            )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize GPU and get device info."""
        import torch

        if self.backend:
            self._device_str = self.backend.get_pytorch_device_string(self.device_id)
        else:
            self._device_str = f"cuda:{self.device_id}"

        self._device = torch.device(self._device_str)
        self._gpu_name = torch.cuda.get_device_name(self.device_id)

        # Get GPU properties
        props = torch.cuda.get_device_properties(self.device_id)
        memory_gb = props.total_memory / (1024**3)

        self.logger.info(f"GPU: {self._gpu_name}")
        self.logger.info(f"Device ID: {self.device_id}")
        self.logger.info(f"Memory: {memory_gb:.2f} GB")
        self.logger.info(f"Compute Capability: {props.major}.{props.minor}")
        self.logger.info(f"Matrix size: {self.matrix_size}x{self.matrix_size}")

    def teardown(self) -> None:
        """Clean up GPU resources."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._device = None
        self._gpu_name = None

    def warmup(self, duration_seconds: int = 0, iterations: int = 3) -> None:
        """Run warmup iterations to stabilize GPU clocks.

        Args:
            duration_seconds: Warmup duration. If 0, uses self.burnin_seconds.
            iterations: Number of iterations if both duration_seconds and
                burnin_seconds are 0.
        """
        import torch

        # Use burnin_seconds if no duration specified
        if duration_seconds == 0:
            duration_seconds = self.burnin_seconds

        # Create small warmup matrices
        warmup_size = min(1024, self.matrix_size)

        try:
            a = torch.randn(
                warmup_size, warmup_size, dtype=torch.float64, device=self._device
            )
            b = torch.randn(
                warmup_size, warmup_size, dtype=torch.float64, device=self._device
            )

            if duration_seconds > 0:
                self.logger.debug(f"Warming up for {duration_seconds}s...")
                start = time.time()
                while (time.time() - start) < duration_seconds:
                    _ = torch.matmul(a, b)
                    torch.cuda.synchronize(self._device)
            else:
                self.logger.debug(f"Warming up for {iterations} iterations...")
                for _ in range(iterations):
                    _ = torch.matmul(a, b)
                    torch.cuda.synchronize(self._device)

            del a, b
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.warning(
                    "GPU out of memory during warmup. Try reducing matrix_size."
                )
                torch.cuda.empty_cache()
            raise

    # -------------------------------------------------------------------------
    # Core Test
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> dict[str, Any]:
        """Execute the FP64 compute test.

        Args:
            duration: Test duration in seconds.
            iterations: Ignored (test runs for duration).

        Returns:
            Dictionary containing FP64 performance metrics.
        """
        del iterations  # Unused; test runs for duration

        import torch

        self.logger.info(f"Running FP64 compute test for {duration}s...")
        self.logger.info(
            f"Matrix multiply: C = A @ B ({self.matrix_size}x{self.matrix_size})"
        )

        # Create test matrices in FP64
        self.logger.info("Allocating matrices on GPU...")
        try:
            a = torch.randn(
                self.matrix_size,
                self.matrix_size,
                dtype=torch.float64,
                device=self._device,
            )
            b = torch.randn(
                self.matrix_size,
                self.matrix_size,
                dtype=torch.float64,
                device=self._device,
            )
            torch.cuda.synchronize(self._device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                mem_needed_gb = (self.matrix_size * self.matrix_size * 8 * 3) / (
                    1024**3
                )
                raise RuntimeError(
                    f"GPU out of memory. Matrix size "
                    f"{self.matrix_size}x{self.matrix_size} "
                    f"requires ~{mem_needed_gb:.2f} GB. "
                    f"Try reducing matrix_size parameter."
                ) from e
            raise

        # Calculate FLOPs per matrix multiply
        # GEMM: C = A @ B requires 2*N^3 FLOPs (N^3 multiplies + N^3 adds)
        flops_per_matmul = 2 * (self.matrix_size**3)

        # Run test for specified duration
        matmul_count = 0
        test_start = time.time()
        iteration_times = []

        self.logger.info("Starting FP64 matrix multiplications...")

        # Check first iteration for extremely slow GPUs
        first_iter_start = time.perf_counter()
        try:
            c = torch.matmul(a, b)
            torch.cuda.synchronize(self._device)
        except RuntimeError as e:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"GPU computation failed: {e}. This may indicate driver issues "
                f"or insufficient GPU memory."
            ) from e

        first_iter_time = time.perf_counter() - first_iter_start
        iteration_times.append(first_iter_time)
        matmul_count += 1

        # Warn if GPU is extremely slow (gaming GPU with crippled FP64)
        if first_iter_time > 30.0:
            self.logger.warning(
                f"First iteration took {first_iter_time:.1f}s. "
                f"This GPU likely has severely limited FP64 performance. "
                f"Consider using a professional/scientific GPU for FP64 workloads."
            )

        # Continue test
        while (time.time() - test_start) < duration:
            iter_start = time.perf_counter()

            try:
                c = torch.matmul(a, b)
                torch.cuda.synchronize(self._device)
            except RuntimeError:
                self.logger.error(f"GPU computation failed at iteration {matmul_count}")
                torch.cuda.empty_cache()
                raise

            iter_elapsed = time.perf_counter() - iter_start
            iteration_times.append(iter_elapsed)
            matmul_count += 1

            # Prevent compiler optimization by occasionally modifying data
            if matmul_count % 10 == 0:
                a[0, 0] = c[0, 0]

        test_elapsed = time.time() - test_start

        # Clean up
        del a, b, c
        torch.cuda.empty_cache()

        # Calculate metrics
        total_flops = flops_per_matmul * matmul_count
        avg_tflops = (total_flops / test_elapsed) / 1e12

        if iteration_times:
            avg_iter_time = sum(iteration_times) / len(iteration_times)
            min_iter_time = min(iteration_times)
            max_iter_time = max(iteration_times)
            peak_tflops = flops_per_matmul / (min_iter_time * 1e12)

            # Calculate percentiles
            sorted_times = sorted(iteration_times)
            p50_time = sorted_times[len(sorted_times) // 2]
            p95_time = sorted_times[int(len(sorted_times) * 0.95)]
            p99_time = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            avg_iter_time = 0
            min_iter_time = 0
            max_iter_time = 0
            peak_tflops = 0
            p50_time = 0
            p95_time = 0
            p99_time = 0

        # Logging
        self.logger.info(f"Completed {matmul_count} matrix multiplications")
        self.logger.info(f"Average FP64 performance: {avg_tflops:.3f} TFLOPS")
        self.logger.info(f"Peak FP64 performance: {peak_tflops:.3f} TFLOPS")
        self.logger.info(
            f"Iteration time: avg={avg_iter_time * 1000:.2f}ms, "
            f"min={min_iter_time * 1000:.2f}ms, max={max_iter_time * 1000:.2f}ms"
        )
        self.logger.info(
            f"Percentiles: p50={p50_time * 1000:.2f}ms, "
            f"p95={p95_time * 1000:.2f}ms, p99={p99_time * 1000:.2f}ms"
        )

        return {
            "test_name": self.get_name(),
            "duration": test_elapsed,
            "matrix_size": self.matrix_size,
            "device_id": self.device_id,
            "gpu_name": self._gpu_name,
            "burnin_seconds": self.burnin_seconds,
            # Performance metrics
            "matmul_count": matmul_count,
            "total_flops": total_flops,
            "avg_fp64_tflops": avg_tflops,
            "peak_fp64_tflops": peak_tflops,
            # Timing metrics
            "avg_iteration_time_ms": avg_iter_time * 1000,
            "min_iteration_time_ms": min_iter_time * 1000,
            "max_iteration_time_ms": max_iter_time * 1000,
            "p50_iteration_time_ms": p50_time * 1000,
            "p95_iteration_time_ms": p95_time * 1000,
            "p99_iteration_time_ms": p99_time * 1000,
        }
