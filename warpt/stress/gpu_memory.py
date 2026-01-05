"""GPU memory bandwidth testing - measure GPU memory and PCIe bandwidth."""

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from warpt.backends.base import GPUBackend
from warpt.models.constants import GPU_MEMORY_TEST, MIN_MEMORY_TEST_DURATION
from warpt.models.stress_models import GPUMemoryBandwidthResult
from warpt.stress.base import StressTest, TestCategory


class GPUMemoryBandwidthTest(StressTest):
    """Memory bandwidth profiling test for GPU.

    Tests device-to-device (GPU memory), host-to-device (PCIe upload),
    and device-to-host (PCIe download) bandwidth.
    """

    _PARAM_FIELDS = ("device_id", "burnin_seconds", "data_size_gb", "use_pinned_memory")

    def __init__(
        self,
        device_id: int,
        burnin_seconds: int = 5,
        backend: GPUBackend | None = None,
        data_size_gb: float = 1.0,
        test_types: list[str] | None = None,
        use_pinned_memory: bool = True,
    ):
        """Initialize GPU memory bandwidth test.

        Args:
            device_id: GPU device ID (0, 1, 2, etc.)
            burnin_seconds: Warmup duration before measurement
            backend: GPU backend (NvidiaBackend, AMDBackend, etc.).
                If None, defaults to NVIDIA.
            data_size_gb: Size of data to transfer in GB (default 1.0).
                Same size used for all test types (D2D, H2D, D2H).
            test_types: Which tests to run. Options: ["d2d", "h2d", "d2h"]
                If None, runs all three by default.
            use_pinned_memory: Use pinned memory for H2D/D2H transfers
                (faster but uses more RAM). Default True.
        """
        self.device_id = device_id
        self.burnin_seconds = burnin_seconds
        self.backend = backend
        self.data_size_gb = data_size_gb
        self.use_pinned_memory = use_pinned_memory
        self._device_str: str | None = None
        self._device: torch.device | None = None

        # TODO: Consider adding different default sizes for D2D vs H2D/D2H
        # D2D tests GPU memory bandwidth (could use larger sizes like 2GB)
        # H2D/D2H test PCIe bandwidth (PCIe is bottleneck, smaller sizes OK)
        # Also consider multi-size sweep mode (like NVIDIA bandwidthTest shmoo)

        # Default to all tests if not specified
        if test_types is None:
            self.test_types = ["d2d", "h2d", "d2h"]
        else:
            self.test_types = test_types
            # Validate
            valid = {"d2d", "h2d", "d2h"}
            for t in self.test_types:
                if t not in valid:
                    raise ValueError(
                        f"Invalid test type: {t}. Valid: {', '.join(sorted(valid))}"
                    )
            # D2D is always required as it measures GPU memory bandwidth
            if "d2d" not in self.test_types:
                self.test_types.insert(0, "d2d")

    def get_name(self) -> str:
        """Return test name."""
        return GPU_MEMORY_TEST

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "GPU Memory Bandwidth Test"

    def get_description(self) -> str:
        """Return test description."""
        return (
            "Measures GPU memory bandwidth via device-to-device transfers, "
            "and optionally PCIe bandwidth via host-to-device and "
            "device-to-host transfers"
        )

    def get_category(self) -> TestCategory:
        """Return test category."""
        return TestCategory.ACCELERATOR

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
        if self.data_size_gb <= 0:
            raise ValueError("data_size_gb must be > 0")
        if self.burnin_seconds < 0:
            raise ValueError("burnin_seconds must be >= 0")

    def setup(self) -> None:
        """Initialize GPU device."""
        import torch

        if self.backend:
            self._device_str = self.backend.get_pytorch_device_string(self.device_id)
        else:
            self._device_str = f"cuda:{self.device_id}"

        self._device = torch.device(self._device_str)
        torch.cuda.set_device(self._device)

        self._gpu_name = torch.cuda.get_device_name(self._device)
        gpu_props = torch.cuda.get_device_properties(self._device)
        self._gpu_memory_total = gpu_props.total_memory / (1024**3)

        # Check pinned memory availability for H2D/D2H tests
        if self.use_pinned_memory and (
            "h2d" in self.test_types or "d2h" in self.test_types
        ):
            import psutil

            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            if self.data_size_gb > available_ram_gb * 0.5:
                self.logger.warning(
                    f"Low RAM ({available_ram_gb:.1f} GB available), "
                    f"requested {self.data_size_gb} GB pinned memory. "
                    "Falling back to non-pinned memory."
                )
                self.use_pinned_memory = False

        self.logger.info(f"GPU {self.device_id}: {self._gpu_name}")

    def teardown(self) -> None:
        """Clean up GPU resources."""
        import torch

        torch.cuda.empty_cache()
        self._device = None
        self._device_str = None

    def _warmup_transfer(self, src_tensor, dst_tensor) -> None:
        """Warmup the memory transfer to stabilize performance.

        Args:
            src_tensor: Source tensor
            dst_tensor: Destination tensor
        """
        import torch

        if self.burnin_seconds > 0:
            burnin_start = time.time()
            while (time.time() - burnin_start) < self.burnin_seconds:
                dst_tensor.copy_(src_tensor)
                torch.cuda.synchronize()
        else:
            # Default: 3 warmup iterations
            for _ in range(3):
                dst_tensor.copy_(src_tensor)
                torch.cuda.synchronize()

        torch.cuda.empty_cache()

    def _benchmark_transfer(
        self, src_tensor, dst_tensor, test_duration: int
    ) -> tuple[float, int]:
        """Run timed transfer benchmark with CUDA events.

        Args:
            src_tensor: Source tensor
            dst_tensor: Destination tensor
            test_duration: Test duration in seconds

        Returns:
            Tuple of (elapsed_seconds, iterations)
        """
        import torch

        # Create CUDA events for precise GPU-side timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        iterations = 0
        start_event.record()
        benchmark_start = time.time()

        while (time.time() - benchmark_start) < test_duration:
            dst_tensor.copy_(src_tensor)
            iterations += 1

        end_event.record()
        torch.cuda.synchronize()

        # Get elapsed time in milliseconds, convert to seconds
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_seconds = elapsed_ms / 1000.0

        return elapsed_seconds, iterations

    def _calculate_bandwidth(
        self, data_size_bytes: int, elapsed_seconds: float, iterations: int
    ) -> float:
        """Calculate bandwidth in GB/s from transfer metrics.

        Args:
            data_size_bytes: Size of data transferred per iteration
            elapsed_seconds: Total elapsed time
            iterations: Number of iterations completed

        Returns:
            Bandwidth in GB/s
        """
        total_bytes = data_size_bytes * iterations
        bandwidth_gbps = (total_bytes / (1024**3)) / elapsed_seconds
        return bandwidth_gbps

    def _cleanup_tensors(self, *tensors) -> None:
        """Clean up tensors and GPU cache.

        Args:
            *tensors: Variable number of tensors to clean up
        """
        import torch

        for tensor in tensors:
            del tensor
        torch.cuda.empty_cache()

    def _run_memory_transfer_benchmark(
        self,
        src_tensor,
        dst_tensor,
        data_size_bytes: int,
        test_duration: int,
    ) -> dict:
        """Orchestrate the full memory transfer benchmark.

        This is the core benchmark logic shared by all transfer types (D2D, H2D, D2H).
        Each test type differs only in tensor allocation (which device, pinned or not).

        Flow: warmup → benchmark → calculate → cleanup

        Args:
            src_tensor: Source tensor (already allocated on correct device)
            dst_tensor: Destination tensor (already allocated on correct device)
            data_size_bytes: Size of data being transferred per iteration
            test_duration: Test duration in seconds

        Returns:
            Dict with bandwidth_gbps and iterations
        """
        # Step 1: Warmup to stabilize performance
        self._warmup_transfer(src_tensor, dst_tensor)

        # Step 2: Run benchmark and get timing
        elapsed_seconds, iterations = self._benchmark_transfer(
            src_tensor, dst_tensor, test_duration
        )

        # Step 3: Calculate bandwidth
        bandwidth_gbps = self._calculate_bandwidth(
            data_size_bytes, elapsed_seconds, iterations
        )

        # Step 4: Cleanup
        self._cleanup_tensors(src_tensor, dst_tensor)

        return {"bandwidth_gbps": bandwidth_gbps, "iterations": iterations}

    def execute_test(self, duration: int, iterations: int) -> GPUMemoryBandwidthResult:
        """Execute memory bandwidth tests.

        Args:
            duration: Total test duration - split across test types
                (min 5s per test type)
            iterations: Ignored (uses duration for each test type).

        Returns:
            GPUMemoryBandwidthResult with all bandwidth measurements
        """
        del iterations
        import pynvml

        device = self._device

        # Get GPU UUID
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
        except Exception:
            gpu_uuid = f"gpu_{self.device_id}_unknown"

        print(f"\n=== GPU {self.device_id} Memory Bandwidth Test ===\n")
        print(f"GPU: {self._gpu_name}")
        print(f"Data Size: {self.data_size_gb} GB")

        # Calculate per-test duration
        num_tests = len(self.test_types)
        per_test_duration = max(duration // num_tests, MIN_MEMORY_TEST_DURATION)
        print(f"Duration per test: {per_test_duration}s")
        print(f"Tests to run: {', '.join(t.upper() for t in self.test_types)}")
        print(f"Pinned Memory: {'Enabled' if self.use_pinned_memory else 'Disabled'}")
        if self.burnin_seconds > 0:
            print(f"Burnin: {self.burnin_seconds}s per test")
        print()

        # Run tests
        # D2D is always required (enforced in __init__)
        data_size_bytes = int(self.data_size_gb * (1024**3))

        result = self._test_d2d_bandwidth(device, data_size_bytes, per_test_duration)
        d2d_bandwidth = result["bandwidth_gbps"]
        d2d_iterations = result["iterations"]

        h2d_bandwidth = None
        h2d_iterations = None
        d2h_bandwidth = None
        d2h_iterations = None

        if "h2d" in self.test_types:
            result = self._test_h2d_bandwidth(
                device, data_size_bytes, per_test_duration
            )
            h2d_bandwidth = result["bandwidth_gbps"]
            h2d_iterations = result["iterations"]

        if "d2h" in self.test_types:
            result = self._test_d2h_bandwidth(
                device, data_size_bytes, per_test_duration
            )
            d2h_bandwidth = result["bandwidth_gbps"]
            d2h_iterations = result["iterations"]

        # Build result
        return GPUMemoryBandwidthResult(
            device_id=f"gpu_{self.device_id}",
            gpu_uuid=gpu_uuid,
            gpu_name=self._gpu_name,
            duration=per_test_duration * num_tests,
            burnin_seconds=self.burnin_seconds,
            data_size_gb=self.data_size_gb,
            used_pinned_memory=self.use_pinned_memory,
            d2d_bandwidth_gbps=d2d_bandwidth,
            d2d_iterations=d2d_iterations,
            h2d_bandwidth_gbps=h2d_bandwidth,
            h2d_iterations=h2d_iterations,
            d2h_bandwidth_gbps=d2h_bandwidth,
            d2h_iterations=d2h_iterations,
        )

    def _test_d2d_bandwidth(
        self, device, data_size_bytes: int, test_duration: int
    ) -> dict:
        """Test device-to-device memory bandwidth (GPU memory).

        This measures GPU memory bandwidth (HBM2/HBM3), not PCIe.
        Both source and destination tensors are on the GPU.

        Args:
            device: PyTorch device
            data_size_bytes: Size of data to copy in bytes
            test_duration: Test duration in seconds

        Returns:
            Dict with bandwidth_gbps and iterations
        """
        import torch

        print("Testing Device-to-Device...", end=" ", flush=True)

        # Calculate tensor size (use float32 for 4 bytes per element)
        num_elements = data_size_bytes // 4

        # Allocate tensors on GPU (both src and dst on same device)
        src = torch.randn(num_elements, dtype=torch.float32, device=device)
        dst = torch.empty_like(src)

        # Run benchmark using shared helper
        result = self._run_memory_transfer_benchmark(
            src, dst, data_size_bytes, test_duration
        )

        print(
            f"{result['bandwidth_gbps']:.1f} GB/s ({result['iterations']} iterations)"
        )

        return result

    def _test_h2d_bandwidth(
        self, device, data_size_bytes: int, test_duration: int
    ) -> dict:
        """Test host-to-device memory bandwidth (PCIe upload).

        This measures PCIe bandwidth (CPU → GPU), not GPU memory bandwidth.
        The bottleneck is the PCIe/SXM bus, not the GPU memory itself.

        Args:
            device: PyTorch device
            data_size_bytes: Size of data to copy in bytes
            test_duration: Test duration in seconds

        Returns:
            Dict with bandwidth_gbps and iterations
        """
        import torch

        print("Testing Host-to-Device...", end=" ", flush=True)

        # Calculate tensor size (use float32 for 4 bytes per element)
        num_elements = data_size_bytes // 4

        # Allocate source on CPU (with optional pinned memory for faster DMA)
        src = torch.randn(num_elements, dtype=torch.float32, device="cpu")
        if self.use_pinned_memory:
            src = src.pin_memory()

        # Allocate destination on GPU
        dst = torch.empty(num_elements, dtype=torch.float32, device=device)

        # Run benchmark using shared helper
        result = self._run_memory_transfer_benchmark(
            src, dst, data_size_bytes, test_duration
        )

        pinned_str = "pinned" if self.use_pinned_memory else "non-pinned"
        print(
            f"{result['bandwidth_gbps']:.1f} GB/s "
            f"({result['iterations']} iterations, {pinned_str})"
        )

        return result

    def _test_d2h_bandwidth(
        self, device, data_size_bytes: int, test_duration: int
    ) -> dict:
        """Test device-to-host memory bandwidth (PCIe download).

        This measures PCIe bandwidth (GPU → CPU), not GPU memory bandwidth.
        The bottleneck is the PCIe/SXM bus, not the GPU memory itself.

        Args:
            device: PyTorch device
            data_size_bytes: Size of data to copy in bytes
            test_duration: Test duration in seconds

        Returns:
            Dict with bandwidth_gbps and iterations
        """
        import torch

        print("Testing Device-to-Host...", end=" ", flush=True)

        # Calculate tensor size (use float32 for 4 bytes per element)
        num_elements = data_size_bytes // 4

        # Allocate source on GPU
        src = torch.randn(num_elements, dtype=torch.float32, device=device)

        # Allocate destination on CPU (with optional pinned memory for faster DMA)
        dst = torch.empty(num_elements, dtype=torch.float32, device="cpu")
        if self.use_pinned_memory:
            dst = dst.pin_memory()

        # Run benchmark using shared helper
        result = self._run_memory_transfer_benchmark(
            src, dst, data_size_bytes, test_duration
        )

        pinned_str = "pinned" if self.use_pinned_memory else "non-pinned"
        print(
            f"{result['bandwidth_gbps']:.1f} GB/s "
            f"({result['iterations']} iterations, {pinned_str})"
        )

        return result
