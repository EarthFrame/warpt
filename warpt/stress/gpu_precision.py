"""GPU mixed precision testing - profile hardware acceleration capabilities."""

import time

from warpt.backends.base import GPUBackend
from warpt.models.constants import MIXED_PRECISION_TEST, Precision
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
        precisions: list[Precision] | None = None,
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
            precisions: List of precisions to test. If None, defaults to
                [FP32, FP16, BF16]. Unsupported precisions will be tested
                and marked as not supported in results and command line.
        """
        self.device_id = device_id
        self.burnin_seconds = burnin_seconds
        self.backend = backend
        self.matrix_size = matrix_size
        self.test_duration = test_duration
        self.allow_tf32 = allow_tf32

        # Default precisions: FP32, FP16, BF16
        # INT8 is available in enum but not included by default
        if precisions is None:
            self.precisions = [Precision.FP32, Precision.FP16, Precision.BF16]
        else:
            self.precisions = list(precisions)
            # Ensure FP32 is always included as baseline
            if Precision.FP32 not in self.precisions:
                self.precisions.insert(0, Precision.FP32)

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

        # Precision to torch dtype mapping
        precision_dtype_map = {
            Precision.FP32: torch.float32,
            Precision.FP16: torch.float16,
            Precision.BF16: torch.bfloat16,
            # INT8 will be added when implemented
        }

        # Calculate per-precision duration based on user-selected precisions
        num_precisions = len(self.precisions)
        per_precision_duration = max(duration // num_precisions, 5)  # Min 5s each

        print(f"\n=== Mixed Precision Profile: GPU {self.device_id} ===\n")
        print(f"Matrix Size: {self.matrix_size}x{self.matrix_size}")
        print(f"Duration per Precision: {per_precision_duration}s")
        print(
            f"Precisions to Test: {', '.join(p.value.upper() for p in self.precisions)}"
        )
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

        # Test each precision in the list
        precision_results: dict[Precision, PrecisionResult] = {}
        for precision in self.precisions:
            if precision not in precision_dtype_map:
                # Precision not yet implemented (e.g., INT8)
                print(f"Skipping {precision.value.upper()}: Not yet implemented\n")
                continue

            torch_dtype = precision_dtype_map[precision]
            precision_results[precision] = self._test_precision(
                device,
                torch_dtype,
                precision.value.upper(),
                f"torch.{torch_dtype}",
                per_precision_duration,
            )

        # Calculate speedups relative to FP32 baseline
        fp32_baseline = precision_results.get(Precision.FP32)
        fp32_tflops = fp32_baseline.tflops if fp32_baseline else 0.0

        # Calculate speedup for each non-FP32 precision
        speedups: dict[Precision, float | None] = {}
        for precision, result in precision_results.items():
            if precision != Precision.FP32:  # Only calculate for non-baseline
                if result.tflops and fp32_tflops:
                    speedups[precision] = result.tflops / fp32_tflops
                else:
                    speedups[precision] = None

        # Determine if GPU is mixed precision ready
        # (has meaningful speedup in any non-FP32 precision)
        mixed_precision_ready = False
        for speedup in speedups.values():
            if speedup and speedup > 1.5:  # At least 50% speedup
                mixed_precision_ready = True
                break

        # Print speedup summary
        print("\n--- Speedup Summary ---")
        for precision, speedup in speedups.items():
            if speedup:
                print(f"{precision.value.upper()} vs FP32: {speedup:.2f}x")
        hw_info = (
            "Tensor Cores detected" if mixed_precision_ready else "No acceleration"
        )
        print(
            f"Mixed Precision Ready: {'Yes' if mixed_precision_ready else 'No'} "
            f"({hw_info})"
        )

        # Build MixedPrecisionResults dynamically
        # FP32 is required as baseline (guaranteed by __init__)
        fp32_result = precision_results.get(Precision.FP32)
        if not fp32_result:
            raise RuntimeError("FP32 baseline test failed or was not run")

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
            error_msg = f"GPU {self.device_id} does not support {precision_name}"
            print(f"Not supported ({e})")
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
                note=error_msg,
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
