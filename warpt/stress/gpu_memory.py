"""GPU memory bandwidth testing - measure GPU memory and PCIe bandwidth."""

import time

from warpt.backends.base import GPUBackend
from warpt.models.constants import GPU_MEMORY_TEST
from warpt.models.stress_models import GPUMemoryBandwidthResult
from warpt.stress.base import StressTest


class GPUMemoryBandwidthTest(StressTest):
    """Memory bandwidth profiling test for GPU.

    Tests device-to-device (GPU memory), host-to-device (PCIe upload),
    and device-to-host (PCIe download) bandwidth.
    """

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

    def run(self, duration: int) -> GPUMemoryBandwidthResult:
        """Run memory bandwidth tests.

        Args:
            duration: Total test duration - split across test types
                (min 5s per test type)

        Returns:
            GPUMemoryBandwidthResult with all bandwidth measurements
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

        # Safety check for pinned memory
        if self.use_pinned_memory and (
            "h2d" in self.test_types or "d2h" in self.test_types
        ):
            import psutil

            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            if self.data_size_gb > available_ram_gb * 0.5:
                print(f"⚠️  Warning: Low RAM ({available_ram_gb:.1f} GB available)")
                print(f"   Requested {self.data_size_gb} GB pinned memory")
                print("   Falling back to non-pinned memory")
                self.use_pinned_memory = False

        # Calculate per-test duration
        num_tests = len(self.test_types)
        per_test_duration = max(duration // num_tests, 5)  # Min 5s each

        # Get GPU info
        gpu_props = torch.cuda.get_device_properties(device)
        gpu_name = gpu_props.name

        # Get GPU UUID
        import pynvml

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
        except Exception:
            gpu_uuid = f"gpu_{self.device_id}_unknown"

        print(f"\n=== GPU {self.device_id} Memory Bandwidth Test ===\n")
        print(f"GPU: {gpu_name}")
        print(f"Data Size: {self.data_size_gb} GB")
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
            gpu_name=gpu_name,
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
        """Test device-to-device memory bandwidth.

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

        # Allocate 2 tensors on GPU
        src = torch.randn(num_elements, dtype=torch.float32, device=device)
        dst = torch.empty_like(src)

        # Warmup/burnin phase
        if self.burnin_seconds > 0:
            burnin_start = time.time()
            while (time.time() - burnin_start) < self.burnin_seconds:
                dst.copy_(src)
                torch.cuda.synchronize()
        else:
            # Default: 3 warmup iterations
            for _ in range(3):
                dst.copy_(src)
                torch.cuda.synchronize()

        torch.cuda.empty_cache()

        # Benchmark phase - use CUDA events for precise timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        iterations = 0
        start_event.record()
        benchmark_start = time.time()

        while (time.time() - benchmark_start) < test_duration:
            dst.copy_(src)
            iterations += 1

        end_event.record()
        torch.cuda.synchronize()

        # Get elapsed time in milliseconds, convert to seconds
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed = elapsed_ms / 1000.0

        # Calculate bandwidth
        # Total data transferred = data_size_bytes * iterations
        total_bytes = data_size_bytes * iterations
        bandwidth_gbps = (total_bytes / (1024**3)) / elapsed

        print(f"{bandwidth_gbps:.1f} GB/s ({iterations} iterations)")

        # Cleanup
        del src, dst
        torch.cuda.empty_cache()

        return {"bandwidth_gbps": bandwidth_gbps, "iterations": iterations}

    def _test_h2d_bandwidth(
        self, device, data_size_bytes: int, test_duration: int
    ) -> dict:
        """Test host-to-device memory bandwidth.

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

        # Allocate tensor on CPU (pinned if enabled)
        src = torch.randn(num_elements, dtype=torch.float32, device="cpu")
        if self.use_pinned_memory:
            src = src.pin_memory()

        # Allocate tensor on GPU
        dst = torch.empty(num_elements, dtype=torch.float32, device=device)

        # Warmup/burnin phase
        if self.burnin_seconds > 0:
            burnin_start = time.time()
            while (time.time() - burnin_start) < self.burnin_seconds:
                dst.copy_(src)
                torch.cuda.synchronize()
        else:
            # Default: 3 warmup iterations
            for _ in range(3):
                dst.copy_(src)
                torch.cuda.synchronize()

        torch.cuda.empty_cache()

        # Benchmark phase - use CUDA events for precise timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        iterations = 0
        start_event.record()
        benchmark_start = time.time()

        while (time.time() - benchmark_start) < test_duration:
            dst.copy_(src)
            iterations += 1

        end_event.record()
        torch.cuda.synchronize()

        # Get elapsed time in milliseconds, convert to seconds
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed = elapsed_ms / 1000.0

        # Calculate bandwidth
        # Total data transferred = data_size_bytes * iterations
        total_bytes = data_size_bytes * iterations
        bandwidth_gbps = (total_bytes / (1024**3)) / elapsed

        pinned_str = "pinned" if self.use_pinned_memory else "non-pinned"
        print(f"{bandwidth_gbps:.1f} GB/s ({iterations} iterations, {pinned_str})")

        # Cleanup
        del src, dst
        torch.cuda.empty_cache()

        return {"bandwidth_gbps": bandwidth_gbps, "iterations": iterations}

    def _test_d2h_bandwidth(
        self, device, data_size_bytes: int, test_duration: int
    ) -> dict:
        """Test device-to-host memory bandwidth.

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

        # Allocate tensor on GPU
        src = torch.randn(num_elements, dtype=torch.float32, device=device)

        # Allocate tensor on CPU (pinned if enabled)
        dst = torch.empty(num_elements, dtype=torch.float32, device="cpu")
        if self.use_pinned_memory:
            dst = dst.pin_memory()

        # Warmup/burnin phase
        if self.burnin_seconds > 0:
            burnin_start = time.time()
            while (time.time() - burnin_start) < self.burnin_seconds:
                dst.copy_(src)
                torch.cuda.synchronize()
        else:
            # Default: 3 warmup iterations
            for _ in range(3):
                dst.copy_(src)
                torch.cuda.synchronize()

        torch.cuda.empty_cache()

        # Benchmark phase - use CUDA events for precise timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        iterations = 0
        start_event.record()
        benchmark_start = time.time()

        while (time.time() - benchmark_start) < test_duration:
            dst.copy_(src)
            iterations += 1

        end_event.record()
        torch.cuda.synchronize()

        # Get elapsed time in milliseconds, convert to seconds
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed = elapsed_ms / 1000.0

        # Calculate bandwidth
        # Total data transferred = data_size_bytes * iterations
        total_bytes = data_size_bytes * iterations
        bandwidth_gbps = (total_bytes / (1024**3)) / elapsed

        pinned_str = "pinned" if self.use_pinned_memory else "non-pinned"
        print(f"{bandwidth_gbps:.1f} GB/s ({iterations} iterations, {pinned_str})")

        # Cleanup
        del src, dst
        torch.cuda.empty_cache()

        return {"bandwidth_gbps": bandwidth_gbps, "iterations": iterations}
