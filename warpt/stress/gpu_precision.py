"""GPU mixed precision testing - profile hardware acceleration capabilities."""

import time
from typing import Any

from warpt.backends.base import AcceleratorBackend
from warpt.models.constants import MIXED_PRECISION_TEST, Precision
from warpt.models.stress_models import MixedPrecisionResults, PrecisionResult
from warpt.stress.base import StressTest, TestCategory
from warpt.stress.utils import calculate_tflops


class GPUPrecisionTest(StressTest):
    """Mixed precision profiling test for GPU.

    Tests FP32, FP16, BF16, and INT8 to detect specialized hardware
    (Tensor Cores) and measure relative speedups.
    """

    def __init__(
        self,
        device_id: int = 0,
        burnin_seconds: int = 0,
        backend: AcceleratorBackend | None = None,
        matrix_size: int = 2048,
        test_duration: int = 5,
        allow_tf32: bool = True,
        precisions: list[Precision] | None = None,
    ):
        """Initialize GPU precision test.

        Args:
            device_id: GPU device ID (0, 1, 2, etc.)
            burnin_seconds: Warmup duration (0 = use 3 iterations).
            backend: GPU backend (NvidiaBackend, AMDBackend, etc.).
            matrix_size: Size of square matrices (NxN). Default 2048.
            test_duration: Duration per precision test in seconds.
            allow_tf32: Enable TF32 for FP32 operations.
            precisions: Precisions to test. Default: [FP32, FP16, BF16].
        """
        self.device_id = device_id
        self.burnin_seconds = burnin_seconds
        self.backend = backend
        self.matrix_size = matrix_size
        self.test_duration = test_duration
        self.allow_tf32 = allow_tf32

        # Default precisions
        if precisions is None:
            self.precisions = [Precision.FP32, Precision.FP16, Precision.BF16]
        else:
            self.precisions = list(precisions)
            # Ensure FP32 is always included as baseline
            if Precision.FP32 not in self.precisions:
                self.precisions.insert(0, Precision.FP32)

        # Runtime state
        self._device: Any = None  # torch.device, set in setup()
        self._device_str: str | None = None

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_name(self) -> str:
        """Return internal test name."""
        return MIXED_PRECISION_TEST

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "GPU Mixed Precision Profile"

    def get_description(self) -> str:
        """Return one-line description."""
        return "Profiles GPU precision support and Tensor Core acceleration"

    def get_category(self) -> TestCategory:
        """Return test category."""
        return TestCategory.ACCELERATOR

    # -------------------------------------------------------------------------
    # Configuration & Parameters
    # -------------------------------------------------------------------------

    def get_parameters(self) -> dict[str, Any]:
        """Return current test parameters."""
        return {
            "device_id": self.device_id,
            "burnin_seconds": self.burnin_seconds,
            "matrix_size": self.matrix_size,
            "test_duration": self.test_duration,
            "allow_tf32": self.allow_tf32,
            "precisions": [p.value for p in self.precisions],
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set test parameters from dictionary."""
        if "device_id" in params:
            self.device_id = int(params["device_id"])
        if "burnin_seconds" in params:
            self.burnin_seconds = int(params["burnin_seconds"])
        if "matrix_size" in params:
            self.matrix_size = int(params["matrix_size"])
        if "test_duration" in params:
            self.test_duration = int(params["test_duration"])
        if "allow_tf32" in params:
            self.allow_tf32 = bool(params["allow_tf32"])
        if "precisions" in params:
            self.precisions = [Precision(p) for p in params["precisions"]]

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
        if self.test_duration < 1:
            raise ValueError("test_duration must be >= 1")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize GPU device."""
        import torch

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

        self.logger.info(f"GPU {self.device_id}: Matrix size {self.matrix_size}")
        self.logger.info(f"Duration per precision: {self.test_duration}s")
        precision_names = ", ".join(p.value.upper() for p in self.precisions)
        self.logger.info(f"Precisions: {precision_names}")
        tf32_status = "Enabled" if self.allow_tf32 else "Disabled"
        self.logger.info(f"TF32: {tf32_status}")

    def teardown(self) -> None:
        """Clean up GPU resources."""
        import torch

        torch.cuda.empty_cache()
        self._device = None
        self._device_str = None

    def warmup(self, duration_seconds: int = 0, iterations: int = 3) -> None:
        """Warmup is handled per-precision."""
        pass

    # -------------------------------------------------------------------------
    # Precision Testing
    # -------------------------------------------------------------------------

    def _check_hardware_support(self, dtype) -> bool:
        """Check if GPU has hardware acceleration for this dtype."""
        import torch

        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(self.device_id)
                compute_capability = props.major + (props.minor / 10.0)

                # Tensor Cores: Volta 7.0+ for FP16, Ampere 8.0+ for BF16
                if dtype == torch.float16:
                    return bool(compute_capability >= 7.0)
                elif dtype == torch.bfloat16:
                    return bool(compute_capability >= 8.0)
                elif dtype == torch.float32:
                    return True
            return False
        except Exception:
            return False

    def _test_precision(
        self, dtype, precision_name: str, dtype_str: str, test_duration: int
    ) -> PrecisionResult:
        """Test a specific precision."""
        import torch

        self.logger.info(f"Testing {precision_name}...")

        hardware_supported = self._check_hardware_support(dtype)

        try:
            # Warmup
            if self.burnin_seconds > 0:
                start = time.time()
                while (time.time() - start) < self.burnin_seconds:
                    a = torch.randn(
                        self.matrix_size,
                        self.matrix_size,
                        dtype=dtype,
                        device=self._device,
                    )
                    b = torch.randn(
                        self.matrix_size,
                        self.matrix_size,
                        dtype=dtype,
                        device=self._device,
                    )
                    _ = torch.matmul(a, b)
                    torch.cuda.synchronize()
                    del a, b
            else:
                for _ in range(3):
                    a = torch.randn(
                        self.matrix_size,
                        self.matrix_size,
                        dtype=dtype,
                        device=self._device,
                    )
                    b = torch.randn(
                        self.matrix_size,
                        self.matrix_size,
                        dtype=dtype,
                        device=self._device,
                    )
                    _ = torch.matmul(a, b)
                    torch.cuda.synchronize()
                    del a, b

            torch.cuda.empty_cache()

            # Benchmark
            start_time = time.time()
            iterations = 0

            while (time.time() - start_time) < test_duration:
                a = torch.randn(
                    self.matrix_size,
                    self.matrix_size,
                    dtype=dtype,
                    device=self._device,
                )
                b = torch.randn(
                    self.matrix_size,
                    self.matrix_size,
                    dtype=dtype,
                    device=self._device,
                )
                _ = torch.matmul(a, b)
                torch.cuda.synchronize()
                iterations += 1
                del a, b

            elapsed = time.time() - start_time
            torch.cuda.empty_cache()

            # Calculate metrics
            ops_per_matmul = 2 * (self.matrix_size**3) - (self.matrix_size**2)
            total_ops = iterations * ops_per_matmul
            tflops = calculate_tflops(total_ops, elapsed)
            avg_time = elapsed / iterations

            self.logger.info(
                f"{precision_name}: {tflops:.2f} TFLOPS ({iterations} iters)"
            )

            return PrecisionResult(
                supported=True,
                iterations=iterations,
                avg_time_per_iter=avg_time,
                dtype=dtype_str,
                tflops=tflops,
                matrix_size=self.matrix_size,
                hardware_supported=hardware_supported,
                runtime_supported=True,
                method=None,
                note=None,
            )

        except Exception as e:
            self.logger.warning(f"{precision_name}: Not supported ({e})")
            return PrecisionResult(
                supported=False,
                iterations=None,
                avg_time_per_iter=None,
                dtype=dtype_str,
                tflops=None,
                matrix_size=self.matrix_size,
                hardware_supported=False,
                runtime_supported=False,
                method=None,
                note=f"GPU {self.device_id} does not support {precision_name}",
            )

    # -------------------------------------------------------------------------
    # Core Test
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> MixedPrecisionResults:
        """Execute mixed precision profiling.

        Args:
            duration: Total test duration - split across precisions.
            iterations: Ignored (uses test_duration per precision).

        Returns:
            MixedPrecisionResults with all precision results and speedups.
        """
        del iterations
        import torch

        # Precision to dtype mapping
        dtype_map = {
            Precision.FP32: torch.float32,
            Precision.FP16: torch.float16,
            Precision.BF16: torch.bfloat16,
        }

        # Calculate per-precision duration
        per_precision_duration = max(
            duration // len(self.precisions), self.test_duration
        )

        # Test each precision
        precision_results: dict[Precision, PrecisionResult] = {}
        for precision in self.precisions:
            if precision not in dtype_map:
                self.logger.info(
                    f"Skipping {precision.value.upper()}: Not yet implemented"
                )
                continue

            torch_dtype = dtype_map[precision]
            precision_results[precision] = self._test_precision(
                torch_dtype,
                precision.value.upper(),
                f"torch.{torch_dtype}",
                per_precision_duration,
            )

        # Calculate speedups vs FP32
        fp32_baseline = precision_results.get(Precision.FP32)
        fp32_tflops = fp32_baseline.tflops if fp32_baseline else 0.0

        speedups: dict[Precision, float | None] = {}
        for precision, result in precision_results.items():
            if precision != Precision.FP32:
                if result.tflops and fp32_tflops:
                    speedups[precision] = result.tflops / fp32_tflops
                else:
                    speedups[precision] = None

        # Check if mixed precision ready (>1.5x speedup)
        mixed_precision_ready = any(s and s > 1.5 for s in speedups.values())

        # Log summary
        self.logger.info("--- Speedup Summary ---")
        for precision, speedup in speedups.items():
            if speedup:
                self.logger.info(f"{precision.value.upper()} vs FP32: {speedup:.2f}x")

        hw_info = (
            "Tensor Cores detected" if mixed_precision_ready else "No acceleration"
        )
        ready_str = "Yes" if mixed_precision_ready else "No"
        self.logger.info(f"Mixed Precision Ready: {ready_str} ({hw_info})")

        # Build result
        fp32_result = precision_results.get(Precision.FP32)
        if not fp32_result:
            raise RuntimeError("FP32 baseline test failed")

        return MixedPrecisionResults(
            fp32=fp32_result,
            fp16=precision_results.get(Precision.FP16),
            bf16=precision_results.get(Precision.BF16),
            int8=precision_results.get(Precision.INT8),
            fp16_speedup=speedups.get(Precision.FP16),
            bf16_speedup=speedups.get(Precision.BF16),
            int8_speedup=speedups.get(Precision.INT8),
            mixed_precision_ready=mixed_precision_ready,
            tf32_enabled=self.allow_tf32,
        )
