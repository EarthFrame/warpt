"""GPU compute stress tests."""

import time
from typing import Any

from warpt.backends.base import GPUBackend
from warpt.models.constants import DEFAULT_BURNIN_SECONDS, GPU_STRESS_TEST
from warpt.stress.base import StressTest, TestCategory
from warpt.stress.utils import calculate_tflops


class GPUMatMulTest(StressTest):
    """Matrix multiplication stress test for GPU.

    Uses PyTorch's matmul to stress GPU compute units. Measures TFLOPS
    throughput using FP32 operations (with optional TF32 acceleration).
    """

    _PARAM_FIELDS = ("device_id", "burnin_seconds", "matrix_size", "allow_tf32")

    def __init__(
        self,
        device_id: int = 0,
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
        backend: GPUBackend | None = None,
        matrix_size: int = 8192,
        allow_tf32: bool = True,
    ):
        """Initialize GPU matmul test.

        Args:
            device_id: GPU device ID (0, 1, 2, etc.)
            burnin_seconds: Warmup duration before measurement.
            backend: GPU backend (NvidiaBackend, AMDBackend, etc.).
                If None, defaults to NVIDIA/CUDA.
            matrix_size: Size of square matrices (NxN). Default 8192.
            allow_tf32: Enable TF32 for FP32 operations (default True).
        """
        self.device_id = device_id
        self.burnin_seconds = burnin_seconds
        self.backend = backend
        self.matrix_size = matrix_size
        self.allow_tf32 = allow_tf32

        # Runtime state (set in setup)
        self._device: Any = None  # torch.device, set in setup()
        self._device_str: str | None = None
        self._gpu_name: str | None = None
        self._gpu_memory_total: float | None = None

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_name(self) -> str:
        """Return internal test name."""
        return GPU_STRESS_TEST

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "GPU Matrix Multiplication"

    def get_description(self) -> str:
        """Return one-line description."""
        return "Measures GPU compute throughput via FP32 matrix multiplication"

    def get_category(self) -> TestCategory:
        """Return test category."""
        return TestCategory.ACCELERATOR

    # -------------------------------------------------------------------------
    # Configuration & Parameters
    # -------------------------------------------------------------------------
    # Configuration is managed by _PARAM_FIELDS. get_parameters() and
    # set_parameters() are inherited from StressTest base class.
    #
    # Note: allow_tf32 uses int() -> bool conversion via the base class,
    # which treats 0/1 as False/True. For explicit bool values in config,
    # you can override set_parameters() here if needed.

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set test parameters with custom bool conversion for allow_tf32."""
        super().set_parameters(params)
        # allow_tf32 needs bool conversion instead of int
        if "allow_tf32" in params:
            self.allow_tf32 = bool(params["allow_tf32"])

    # -------------------------------------------------------------------------
    # Hardware & Availability
    # -------------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if PyTorch and CUDA are available."""
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def validate_configuration(self) -> None:
        """Validate test configuration."""
        if not self.is_available():
            raise RuntimeError("CUDA is not available")

        import torch

        if self.device_id >= torch.cuda.device_count():
            raise ValueError(
                f"GPU device {self.device_id} not found. "
                f"Available: 0-{torch.cuda.device_count() - 1}"
            )
        if self.matrix_size < 64:
            raise ValueError("matrix_size must be >= 64")
        if self.burnin_seconds < 0:
            raise ValueError("burnin_seconds must be >= 0")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize GPU device and get properties."""
        import torch

        # Get device string from backend or default to CUDA
        if self.backend:
            self._device_str = self.backend.get_pytorch_device_string(self.device_id)
        else:
            self._device_str = f"cuda:{self.device_id}"

        self._device = torch.device(self._device_str)
        torch.cuda.set_device(self._device)

        # Configure TF32
        if self._device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32

        # Get GPU properties
        self._gpu_name = torch.cuda.get_device_name(self._device)
        gpu_props = torch.cuda.get_device_properties(self._device)
        self._gpu_memory_total = gpu_props.total_memory / (1024**3)

        tf32_status = "Enabled" if self.allow_tf32 else "Disabled"
        self.logger.info(f"GPU {self.device_id}: {self._gpu_name}")
        self.logger.info(f"TF32: {tf32_status}")

    def teardown(self) -> None:
        """Clean up GPU resources."""
        import torch

        torch.cuda.empty_cache()
        self._device = None
        self._device_str = None

    def warmup(self, duration_seconds: int = 0, iterations: int = 3) -> None:
        """Run warmup iterations to let GPU warm up.

        Args:
            duration_seconds: Warmup duration. If 0, uses self.burnin_seconds.
            iterations: Number of iterations if both duration_seconds and
                burnin_seconds are 0.
        """
        import torch

        # Use burnin_seconds if no duration specified
        if duration_seconds == 0:
            duration_seconds = self.burnin_seconds

        if duration_seconds > 0:
            self.logger.debug(f"Warming up for {duration_seconds}s...")
            start = time.time()
            while (time.time() - start) < duration_seconds:
                a = torch.randn(
                    self.matrix_size,
                    self.matrix_size,
                    dtype=torch.float32,
                    device=self._device,
                )
                b = torch.randn(
                    self.matrix_size,
                    self.matrix_size,
                    dtype=torch.float32,
                    device=self._device,
                )
                _ = torch.matmul(a, b)
                torch.cuda.synchronize()
                del a, b
        else:
            self.logger.debug(f"Warming up for {iterations} iterations...")
            for _ in range(iterations):
                a = torch.randn(
                    self.matrix_size,
                    self.matrix_size,
                    dtype=torch.float32,
                    device=self._device,
                )
                b = torch.randn(
                    self.matrix_size,
                    self.matrix_size,
                    dtype=torch.float32,
                    device=self._device,
                )
                _ = torch.matmul(a, b)
                torch.cuda.synchronize()
                del a, b

        import torch

        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Core Test
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> dict[Any, Any]:
        """Execute the GPU matrix multiplication test.

        Args:
            duration: Test duration in seconds.
            iterations: Ignored for GPU test (runs for duration).

        Returns:
            Dictionary containing test results.
        """
        del iterations
        import torch

        torch.cuda.reset_peak_memory_stats(self._device)
        start_time = time.time()
        iter_count = 0

        while (time.time() - start_time) < duration:
            a = torch.randn(
                self.matrix_size,
                self.matrix_size,
                dtype=torch.float32,
                device=self._device,
            )
            b = torch.randn(
                self.matrix_size,
                self.matrix_size,
                dtype=torch.float32,
                device=self._device,
            )
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            iter_count += 1
            del a, b

        elapsed = time.time() - start_time

        # Get memory stats
        memory_used = torch.cuda.max_memory_allocated(self._device) / (1024**3)

        # Calculate TFLOPS (2*N^3 - N^2 ops per matmul)
        ops_per_matmul = 2 * (self.matrix_size**3) - (self.matrix_size**2)
        total_ops = iter_count * ops_per_matmul
        tflops = calculate_tflops(total_ops, elapsed)

        self.logger.info(f"Result: {tflops:.2f} TFLOPS ({iter_count} iterations)")

        return {
            "test_name": self.get_name(),
            "device_id": self.device_id,
            "gpu_name": self._gpu_name,
            "tflops": tflops,
            "duration": elapsed,
            "iterations": iter_count,
            "matrix_size": self.matrix_size,
            "total_operations": total_ops,
            "burnin_seconds": self.burnin_seconds,
            "memory_used_gb": memory_used,
            "memory_total_gb": self._gpu_memory_total,
            "precision": "fp32",
            "tf32_enabled": self.allow_tf32,
        }
