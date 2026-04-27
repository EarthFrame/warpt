"""GPU FP64 compute stress test.

This test measures double-precision floating point (FP64) performance,
which is critical for scientific computing workloads.
"""

import time
from typing import Any

from warpt.backends.base import AcceleratorBackend
from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.stress.base import StressTest, TestCategory
from warpt.stress.utils import measure_loop


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
        self.logger.info("Starting FP64 matrix multiplications...")

        matmul_count = 0
        c = None

        def work():
            nonlocal matmul_count, c
            c = torch.matmul(a, b)
            matmul_count += 1
            # Prevent compiler optimization by occasionally modifying data
            if matmul_count % 10 == 0:
                a[0, 0] = c[0, 0]

        test_elapsed, _ = measure_loop(
            duration=duration,
            work_fn=work,
            sync_fn=torch.cuda.synchronize,
            device=self._device,
        )

        # Clean up
        del a, b, c
        torch.cuda.empty_cache()

        # Calculate metrics
        total_flops = flops_per_matmul * matmul_count
        avg_tflops = (total_flops / test_elapsed) / 1e12

        # Logging
        self.logger.info(f"Completed {matmul_count} matrix multiplications")
        self.logger.info(f"Average FP64 performance: {avg_tflops:.3f} TFLOPS")

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
            "peak_fp64_tflops": None,  # TODO: add percentile metrics later
            # Timing metrics
            "avg_iteration_time_ms": None,  # TODO: add percentile metrics later
            "min_iteration_time_ms": None,
            "max_iteration_time_ms": None,
            "p50_iteration_time_ms": None,
            "p95_iteration_time_ms": None,
            "p99_iteration_time_ms": None,
        }
