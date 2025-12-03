"""GPU mixed precision testing - profile hardware acceleration capabilities."""

import time

from warpt.backends.base import GPUBackend
from warpt.models.constants import MIXED_PRECISION_TEST
from warpt.models.stress_models import MixedPrecisionResults, PrecisionResult
from warpt.stress.base import StressTest
from warpt.stress.utils import calculate_tflops


class GPUPrecisionTest(StressTest):
    """Mixed precision profiling test for GPU.

    Tests FP32, FP16, BF16, and INT8 to detect specialized hardware
    (Tensor Cores) and measure relative speedups.
    """

    def __init__(
        self,
        device_id: int,
        burnin_seconds: int = 0,
        backend: GPUBackend | None = None,
        matrix_size: int = 2048,
        test_duration: int = 5,
        allow_tf32: bool = True,
    ):
        """Initialize GPU precision test.

        Args:
            device_id: GPU device ID (0, 1, 2, etc.)
            burnin_seconds: Warmup duration before measurement (0 = use default 3 iters)
            backend: GPU backend (NvidiaBackend, AMDBackend, etc.).
                If None, defaults to NVIDIA.
            matrix_size: Size of square matrices (NxN). Default 2048 for precision.
            test_duration: Default duration per precision test in seconds
            allow_tf32: Enable TF32 for FP32 operations (default True).
        """
        self.device_id = device_id
        self.burnin_seconds = burnin_seconds
        self.backend = backend
        self.matrix_size = matrix_size
        self.test_duration = test_duration
        self.allow_tf32 = allow_tf32

    def get_name(self) -> str:
        """Return test name."""
        return MIXED_PRECISION_TEST

    def run(self, duration: int) -> MixedPrecisionResults:
        """Run mixed precision profiling.

        Args:
            duration: Total test duration - split across precisions
                (min 5s per precision)

        Returns:
            MixedPrecisionResults with all precision results and speedups
        """
        # Import PyTorch (lazy import)
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "PyTorch is not installed. Install with: pip install warpt[stress]"
            ) from None

        # Get PyTorch device string
        if self.backend:
            device_str = self.backend.get_pytorch_device_string(self.device_id)
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available") from None
            if self.device_id >= torch.cuda.device_count():
                raise ValueError(
                    f"GPU device {self.device_id} not found. "
                    f"Available: 0-{torch.cuda.device_count() - 1}"
                ) from None
            device_str = f"cuda:{self.device_id}"

        device = torch.device(device_str)
        torch.cuda.set_device(device)

        # Configure TF32 (TensorFloat-32) for FP32 operations
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32

        # Calculate per-precision duration
        # 3 precisions: FP32, FP16, BF16 (INT8 TODO)
        num_precisions = 3
        per_precision_duration = max(duration // num_precisions, 5)  # Min 5s each

        print(f"\n=== Mixed Precision Profile: GPU {self.device_id} ===\n")
        print(f"Matrix Size: {self.matrix_size}x{self.matrix_size}")
        print(f"Duration per Precision: {per_precision_duration}s")
        if self.burnin_seconds > 0:
            print(f"Burnin per Precision: {self.burnin_seconds}s")
        else:
            print("Burnin per Precision: 3 iterations (default)")

        # Display TF32 status
        if self.allow_tf32:
            print("TF32: Enabled (use --no-tf32 to disable)")
        else:
            print("TF32: Disabled")
        print()

        # Test each precision
        fp32_result = self._test_precision(
            device, torch.float32, "FP32", "torch.float32", per_precision_duration
        )
        fp16_result = self._test_precision(
            device, torch.float16, "FP16", "torch.float16", per_precision_duration
        )
        bf16_result = self._test_precision(
            device, torch.bfloat16, "BF16", "torch.bfloat16", per_precision_duration
        )

        # INT8 is more complex - skip for now
        # int8_result = self._test_int8_precision(device, per_precision_duration)

        # Calculate speedups relative to FP32 baseline
        fp32_tflops = fp32_result.tflops or 0.0
        fp16_speedup = (
            (fp16_result.tflops / fp32_tflops) if fp16_result.tflops else None
        )
        bf16_speedup = (
            (bf16_result.tflops / fp32_tflops) if bf16_result.tflops else None
        )

        # Determine if GPU is mixed precision ready
        # (has meaningful speedup in FP16 or BF16)
        mixed_precision_ready = False
        if fp16_speedup and fp16_speedup > 1.1:  # At least 10% speedup
            mixed_precision_ready = True
        if bf16_speedup and bf16_speedup > 1.1:
            mixed_precision_ready = True

        print("\n--- Speedup Summary ---")
        if fp16_speedup:
            print(f"FP16 vs FP32: {fp16_speedup:.2f}x")
        if bf16_speedup:
            print(f"BF16 vs FP32: {bf16_speedup:.2f}x")
        hw_info = (
            "Tensor Cores detected" if mixed_precision_ready else "No acceleration"
        )
        print(
            f"Mixed Precision Ready: {'Yes' if mixed_precision_ready else 'No'} "
            f"({hw_info})"
        )

        return MixedPrecisionResults(
            fp32=fp32_result,
            fp16=fp16_result,
            bf16=bf16_result,
            int8=None,  # TODO: Implement INT8
            fp16_speedup=fp16_speedup,
            bf16_speedup=bf16_speedup,
            int8_speedup=None,
            mixed_precision_ready=mixed_precision_ready,
            tf32_enabled=self.allow_tf32,
        )

    def _test_precision(
        self,
        device,
        dtype,
        precision_name: str,
        dtype_str: str,
        test_duration: int,
    ) -> PrecisionResult:
        """Test a specific precision.

        Args:
            device: PyTorch device
            dtype: PyTorch dtype (e.g., torch.float16)
            precision_name: Human-readable name (e.g., "FP16")
            dtype_str: String representation (e.g., "torch.float16")
            test_duration: Test duration in seconds

        Returns:
            PrecisionResult with test metrics
        """
        import torch

        print(f"Testing {precision_name}...", end=" ", flush=True)

        # Check hardware/runtime support
        hardware_supported = self._check_hardware_support(dtype)
        runtime_supported = True  # If we get here, runtime supports it

        try:
            # Burnin/warmup phase
            if self.burnin_seconds > 0:
                # User-specified burnin duration
                burnin_start = time.time()
                while (time.time() - burnin_start) < self.burnin_seconds:
                    a = torch.randn(
                        self.matrix_size, self.matrix_size, dtype=dtype, device=device
                    )
                    b = torch.randn(
                        self.matrix_size, self.matrix_size, dtype=dtype, device=device
                    )
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                    del a, b, c
            else:
                # Default: 3 warmup iterations
                for _ in range(3):
                    a = torch.randn(
                        self.matrix_size, self.matrix_size, dtype=dtype, device=device
                    )
                    b = torch.randn(
                        self.matrix_size, self.matrix_size, dtype=dtype, device=device
                    )
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                    del a, b, c

            torch.cuda.empty_cache()

            # Benchmark phase
            start_time = time.time()
            iterations = 0

            while (time.time() - start_time) < test_duration:
                a = torch.randn(
                    self.matrix_size, self.matrix_size, dtype=dtype, device=device
                )
                b = torch.randn(
                    self.matrix_size, self.matrix_size, dtype=dtype, device=device
                )
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                iterations += 1
                del a, b, c

            elapsed = time.time() - start_time
            torch.cuda.empty_cache()

            # Calculate metrics
            ops_per_matmul = 2 * (self.matrix_size**3) - (self.matrix_size**2)
            total_ops = iterations * ops_per_matmul
            tflops = calculate_tflops(total_ops, elapsed)
            avg_time_per_iter = elapsed / iterations

            print(f"{tflops:.2f} TFLOPS ({iterations} iters)")

            return PrecisionResult(
                supported=True,
                iterations=iterations,
                avg_time_per_iter=avg_time_per_iter,
                dtype=dtype_str,
                tflops=tflops,
                matrix_size=self.matrix_size,
                hardware_supported=hardware_supported,
                runtime_supported=runtime_supported,
                method=None,
                note=None,
            )

        except Exception as e:
            print(f"Failed: {e}")
            return PrecisionResult(
                supported=False,
                iterations=None,
                avg_time_per_iter=None,
                dtype=dtype_str,
                tflops=None,
                matrix_size=self.matrix_size,
                hardware_supported=None,
                runtime_supported=None,
                method=None,
                note=f"Test failed: {e!s}",
            )

    def _check_hardware_support(self, dtype) -> bool:
        """Check if GPU has hardware support for this dtype.

        For NVIDIA: Check compute capability for Tensor Cores.

        Args:
            dtype: PyTorch dtype

        Returns:
            True if hardware acceleration available
        """
        import torch

        try:
            # For NVIDIA GPUs, check compute capability
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(self.device_id)
                major, minor = props.major, props.minor
                compute_capability = major + (minor / 10.0)

                # Tensor Cores for FP16/BF16:
                # - Volta (7.0+): FP16 Tensor Cores
                # - Ampere (8.0+): FP16 + BF16 Tensor Cores
                if dtype == torch.float16:
                    return bool(compute_capability >= 7.0)
                elif dtype == torch.bfloat16:
                    return bool(compute_capability >= 8.0)
                elif dtype == torch.float32:
                    return True  # Always supported

            return False
        except Exception:
            return False
