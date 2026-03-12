"""Multi-GPU scaling stress test.

Tests how well GPU infrastructure scales across multiple GPUs by measuring
inter-GPU communication overhead (halo exchange, residual reductions) and
comparing multi-GPU throughput to single-GPU baselines. Identifies interconnect
bottlenecks (NVLink vs PCIe).
"""

import os
import socket
import time
from typing import Any

from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.models.stress_models import GPUSystemResult
from warpt.stress.base import StressTest, TestCategory

# VRAM-based mesh defaults per GPU (GB threshold → cells).
# Uses 70% factor to leave room for communication buffers.
_VRAM_MESH_DEFAULTS: dict[float, int] = {
    3.5: 1_500_000,
    7.0: 3_500_000,
    14.0: 10_000_000,
    28.0: 20_000_000,
    56.0: 40_000_000,
}


def _find_free_port() -> int:
    """Find a free TCP port for distributed communication."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Compute percentile from a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def _timing_stats(times_ms: list[float]) -> dict[str, float]:
    """Compute timing statistics from a list of millisecond values."""
    if not times_ms:
        return {
            "count": 0,
            "avg_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }
    sorted_t = sorted(times_ms)
    return {
        "count": len(times_ms),
        "avg_ms": sum(times_ms) / len(times_ms),
        "min_ms": sorted_t[0],
        "max_ms": sorted_t[-1],
        "p50_ms": _percentile(sorted_t, 0.50),
        "p95_ms": _percentile(sorted_t, 0.95),
        "p99_ms": _percentile(sorted_t, 0.99),
    }


# ---------------------------------------------------------------------------
# Worker function (module-level, required by mp.spawn)
# ---------------------------------------------------------------------------


def _worker_fn(
    rank: int,
    world_size: int,
    shared_dict: dict,
    config: dict[str, Any],
) -> None:
    """Worker process for multi-GPU scaling test.

    Each worker runs on a single GPU and communicates via NCCL.

    Args:
        rank: Worker rank (0..world_size-1).
        world_size: Total number of workers.
        shared_dict: Manager dict for returning results to orchestrator.
        config: Test configuration dict.
    """
    import torch
    import torch.distributed as dist

    master_port = config["master_port"]
    mesh_size = config["mesh_size_per_gpu"]
    solver_iterations = config["solver_iterations"]
    halo_depth = config["halo_depth"]
    duration = config["duration"]
    dist_backend = config["distributed_backend"]
    device_prefix = config["device_prefix"]

    # ------------------------------------------------------------------
    # A. Initialize distributed
    # ------------------------------------------------------------------
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    if dist_backend == "nccl":
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    dist.init_process_group(
        backend=dist_backend,
        rank=rank,
        world_size=world_size,
        timeout=__import__("datetime").timedelta(minutes=5),
    )
    torch.cuda.set_device(rank)
    device = torch.device(f"{device_prefix}:{rank}")

    results: dict[str, Any] = {"rank": rank, "device": f"{device_prefix}:{rank}"}

    try:
        gpu_name = torch.cuda.get_device_name(rank)
        results["gpu_name"] = gpu_name

        # Duration allocation: 30% single-GPU, 60% multi-GPU, 10% comm bench
        single_duration = duration * 0.30
        multi_duration = duration * 0.60
        comm_duration = duration * 0.10

        # ------------------------------------------------------------------
        # B. Single-GPU baseline (30% of duration)
        # ------------------------------------------------------------------
        pressure = torch.randn(mesh_size, dtype=torch.float64, device=device)
        velocity = torch.randn(mesh_size, 3, dtype=torch.float64, device=device)
        residual = torch.zeros(mesh_size, dtype=torch.float64, device=device)

        single_solver_times: list[float] = []
        single_start = time.time()

        while (time.time() - single_start) < single_duration:
            iter_start = time.perf_counter()

            x = pressure.clone()
            r = residual.clone()
            p = r.clone()

            for _ in range(solver_iterations):
                ap = (
                    6.0 * p
                    - torch.roll(p, 1, 0)
                    - torch.roll(p, -1, 0)
                    - torch.roll(p, 100, 0)
                    - torch.roll(p, -100, 0)
                    - torch.roll(p, 10000, 0)
                    - torch.roll(p, -10000, 0)
                )
                rtr = torch.dot(r, r)
                alpha = rtr / (torch.dot(p, ap) + 1e-30)
                x = x + alpha * p
                r_new = r - alpha * ap
                beta = torch.dot(r_new, r_new) / (rtr + 1e-30)
                p = r_new + beta * p
                r = r_new

            torch.cuda.synchronize(device)
            single_solver_times.append((time.perf_counter() - iter_start) * 1000)

        single_elapsed = time.time() - single_start

        # Estimate TFLOPS: each solver iteration does ~14 FLOPs/cell (stencil + CG)
        total_single_flops = (
            len(single_solver_times) * solver_iterations * mesh_size * 14
        )
        single_tflops = (
            (total_single_flops / single_elapsed / 1e12) if single_elapsed > 0 else 0.0
        )

        results["single_gpu_tflops"] = single_tflops
        results["single_gpu_solver_stats"] = _timing_stats(single_solver_times)
        results["single_gpu_solves"] = len(single_solver_times)

        # ------------------------------------------------------------------
        # C. Multi-GPU CFD with halo exchange (60% of duration)
        # ------------------------------------------------------------------
        # Determine neighbors for 1D ring topology
        left_rank = (rank - 1) % world_size
        right_rank = (rank + 1) % world_size

        # Halo buffers: boundary cells to exchange
        halo_size = halo_depth * int(mesh_size**0.5)  # Scale with sqrt(mesh)
        halo_size = max(halo_size, 1)

        send_buf_left = torch.zeros(halo_size, dtype=torch.float64, device=device)
        send_buf_right = torch.zeros(halo_size, dtype=torch.float64, device=device)
        recv_buf_left = torch.zeros(halo_size, dtype=torch.float64, device=device)
        recv_buf_right = torch.zeros(halo_size, dtype=torch.float64, device=device)

        multi_compute_times: list[float] = []
        multi_halo_times: list[float] = []
        multi_allreduce_times: list[float] = []
        multi_total_times: list[float] = []

        multi_start = time.time()

        while (time.time() - multi_start) < multi_duration:
            total_iter_start = time.perf_counter()

            # --- Compute phase: solver + gradients + flux ---
            compute_start = time.perf_counter()

            # Solver (7-point stencil CG)
            x = pressure.clone()
            r = residual.clone()
            p = r.clone()

            for _ in range(solver_iterations):
                ap = (
                    6.0 * p
                    - torch.roll(p, 1, 0)
                    - torch.roll(p, -1, 0)
                    - torch.roll(p, 100, 0)
                    - torch.roll(p, -100, 0)
                    - torch.roll(p, 10000, 0)
                    - torch.roll(p, -10000, 0)
                )
                rtr = torch.dot(r, r)
                alpha = rtr / (torch.dot(p, ap) + 1e-30)
                x = x + alpha * p
                r_new = r - alpha * ap
                beta = torch.dot(r_new, r_new) / (rtr + 1e-30)
                p = r_new + beta * p
                r = r_new

            # Gradients (central difference) — compute-bound work
            _ = (torch.roll(pressure, -1, 0) - torch.roll(pressure, 1, 0)) / 2.0
            _ = (torch.roll(pressure, -100, 0) - torch.roll(pressure, 100, 0)) / 2.0

            # Flux (memory-bound)
            mass_flux = pressure * velocity[:, 0]
            pressure[:] = pressure + 0.01 * mass_flux

            torch.cuda.synchronize(device)
            compute_elapsed = (time.perf_counter() - compute_start) * 1000

            # --- Halo exchange phase ---
            halo_start = time.perf_counter()

            # Pack boundary cells into send buffers
            send_buf_left[:] = pressure[:halo_size]
            send_buf_right[:] = pressure[-halo_size:]

            # Async send/recv with neighbors
            ops = []
            ops.append(dist.isend(send_buf_right, dst=right_rank))
            ops.append(dist.irecv(recv_buf_left, src=left_rank))
            ops.append(dist.isend(send_buf_left, dst=left_rank))
            ops.append(dist.irecv(recv_buf_right, src=right_rank))

            for op in ops:
                op.wait()

            torch.cuda.synchronize(device)
            halo_elapsed = (time.perf_counter() - halo_start) * 1000

            # --- AllReduce phase: global residual norm ---
            allreduce_start = time.perf_counter()

            local_residual = torch.norm(r).unsqueeze(0)
            dist.all_reduce(local_residual, op=dist.ReduceOp.SUM)

            torch.cuda.synchronize(device)
            allreduce_elapsed = (time.perf_counter() - allreduce_start) * 1000

            total_iter_elapsed = (time.perf_counter() - total_iter_start) * 1000

            multi_compute_times.append(compute_elapsed)
            multi_halo_times.append(halo_elapsed)
            multi_allreduce_times.append(allreduce_elapsed)
            multi_total_times.append(total_iter_elapsed)

        multi_elapsed = time.time() - multi_start

        total_multi_flops = len(multi_total_times) * solver_iterations * mesh_size * 14
        multi_tflops = (
            (total_multi_flops / multi_elapsed / 1e12) if multi_elapsed > 0 else 0.0
        )

        results["multi_gpu_tflops"] = multi_tflops
        results["multi_gpu_iterations"] = len(multi_total_times)
        results["compute_stats"] = _timing_stats(multi_compute_times)
        results["halo_stats"] = _timing_stats(multi_halo_times)
        results["allreduce_stats"] = _timing_stats(multi_allreduce_times)
        results["total_iter_stats"] = _timing_stats(multi_total_times)

        # Communication fraction
        avg_compute = (
            sum(multi_compute_times) / len(multi_compute_times)
            if multi_compute_times
            else 0.0
        )
        avg_halo = (
            sum(multi_halo_times) / len(multi_halo_times) if multi_halo_times else 0.0
        )
        avg_allreduce = (
            sum(multi_allreduce_times) / len(multi_allreduce_times)
            if multi_allreduce_times
            else 0.0
        )
        avg_overhead = avg_halo + avg_allreduce
        total_time = avg_compute + avg_overhead
        results["communication_fraction"] = (
            avg_overhead / total_time if total_time > 0 else 0.0
        )
        results["avg_overhead_ms"] = avg_overhead

        # ------------------------------------------------------------------
        # D. Communication benchmarks (10% of duration)
        # ------------------------------------------------------------------
        comm_results: dict[str, Any] = {}
        msg_sizes = [1024, 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024]
        msg_labels = ["1KB", "1MB", "64MB", "256MB"]

        for label, size_bytes in zip(msg_labels, msg_sizes, strict=True):
            num_elements = size_bytes // 4  # float32
            try:
                buf = torch.zeros(num_elements, dtype=torch.float32, device=device)
            except RuntimeError:
                # Skip if buffer doesn't fit
                continue

            # AllReduce benchmark
            ar_times: list[float] = []
            bench_start = time.time()
            time_budget = comm_duration / len(msg_sizes) / 2

            while (time.time() - bench_start) < time_budget:
                t0 = time.perf_counter()
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize(device)
                ar_times.append((time.perf_counter() - t0) * 1000)

            # P2P benchmark (send to right neighbor, recv from left)
            p2p_times: list[float] = []
            recv_buf = torch.zeros_like(buf)
            bench_start = time.time()

            while (time.time() - bench_start) < time_budget:
                t0 = time.perf_counter()
                send_op = dist.isend(buf, dst=right_rank)
                recv_op = dist.irecv(recv_buf, src=left_rank)
                send_op.wait()
                recv_op.wait()
                torch.cuda.synchronize(device)
                p2p_times.append((time.perf_counter() - t0) * 1000)

            # Bandwidth = bytes / time
            size_gb = size_bytes / (1024**3)
            ar_avg_s = (sum(ar_times) / len(ar_times) / 1000) if ar_times else 0
            p2p_avg_s = (sum(p2p_times) / len(p2p_times) / 1000) if p2p_times else 0

            comm_results[label] = {
                "allreduce_avg_ms": ar_avg_s * 1000,
                "allreduce_bandwidth_gbps": size_gb / ar_avg_s if ar_avg_s > 0 else 0,
                "allreduce_iterations": len(ar_times),
                "p2p_avg_ms": p2p_avg_s * 1000,
                "p2p_bandwidth_gbps": size_gb / p2p_avg_s if p2p_avg_s > 0 else 0,
                "p2p_iterations": len(p2p_times),
            }

            del buf, recv_buf

        results["comm_benchmarks"] = comm_results

        # ------------------------------------------------------------------
        # E. Cleanup and report
        # ------------------------------------------------------------------
        del pressure, velocity, residual
        del send_buf_left, send_buf_right, recv_buf_left, recv_buf_right
        torch.cuda.empty_cache()

    except Exception as e:
        results["error"] = str(e)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    # Write results back to shared dict
    shared_dict[rank] = results


# ---------------------------------------------------------------------------
# Main test class
# ---------------------------------------------------------------------------


class GPUMultiScalingTest(StressTest):
    """Multi-GPU scaling stress test.

    Measures how well GPU infrastructure scales across multiple GPUs by
    running CFD workloads with inter-GPU communication (halo exchange,
    residual reductions). Identifies interconnect bottlenecks and computes
    scaling efficiency.

    Phases:
    - Single-GPU baseline on each GPU (30% of duration)
    - Multi-GPU CFD with halo exchange (60% of duration)
    - Communication benchmarks at various message sizes (10% of duration)
    """

    _PARAM_FIELDS = (
        "device_count",
        "mesh_size_per_gpu",
        "burnin_seconds",
        "solver_iterations",
        "halo_depth",
    )

    def __init__(
        self,
        device_count: int | None = None,
        mesh_size_per_gpu: int | None = None,
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
        solver_iterations: int = 100,
        halo_depth: int = 1,
    ):
        """Initialize multi-GPU scaling test.

        Args:
            device_count: Number of GPUs to use. Auto-detects if None.
            mesh_size_per_gpu: Mesh cells per GPU. Auto-sizes from VRAM if None.
            burnin_seconds: Warmup duration before measurement.
            solver_iterations: CG iterations per solver call.
            halo_depth: Boundary cell layers exchanged (1=first-order, 2=second-order).
        """
        self.burnin_seconds = burnin_seconds
        self.solver_iterations = solver_iterations
        self.halo_depth = halo_depth

        # Defer auto-detection to avoid initializing CUDA context
        self._requested_device_count = device_count
        self._requested_mesh_size = mesh_size_per_gpu
        self.device_count = device_count if device_count is not None else 0
        self.mesh_size_per_gpu = (
            mesh_size_per_gpu if mesh_size_per_gpu is not None else 0
        )

        # Set in setup()
        self._master_port: int = 0
        self._topology: str = "unknown"
        self._distributed_backend: str = "nccl"
        self._device_prefix: str = "cuda"
        self._gpu_uuids: list[str] = []
        self._manager: Any = None
        self._shared_dict: Any = None

    def _resolve_device_count(self) -> int:
        """Resolve the actual device count."""
        if self._requested_device_count is not None:
            return self._requested_device_count
        try:
            import torch

            return int(torch.cuda.device_count())
        except ImportError:
            return 0

    def _resolve_mesh_size(self, device_count: int) -> int:
        """Resolve mesh size per GPU based on smallest GPU's VRAM."""
        if self._requested_mesh_size is not None:
            return self._requested_mesh_size
        try:
            import torch

            min_vram_gb = float("inf")
            for i in range(device_count):
                vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                min_vram_gb = min(min_vram_gb, vram)

            # Apply 70% factor for communication buffers
            effective_vram = min_vram_gb * 0.70

            result = 1_500_000  # Default fallback
            for vram_threshold, cells in sorted(_VRAM_MESH_DEFAULTS.items()):
                if effective_vram >= vram_threshold:
                    result = cells
            return result
        except (ImportError, RuntimeError):
            return 1_500_000

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "Multi-GPU Scaling Test"

    def get_description(self) -> str:
        """Return one-line description."""
        return (
            "Measures multi-GPU scaling efficiency via CFD workload with "
            "halo exchange and communication benchmarks"
        )

    def get_category(self) -> TestCategory:
        """Return test category."""
        return TestCategory.ACCELERATOR

    def get_result_type(self) -> type:
        """Return result model type."""
        return GPUSystemResult

    # -------------------------------------------------------------------------
    # Hardware & Availability
    # -------------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check that at least 2 GPUs and NCCL are available."""
        try:
            import torch
            import torch.distributed as dist

            return bool(torch.cuda.device_count() >= 2 and dist.is_nccl_available())
        except ImportError:
            return False

    def validate_configuration(self) -> None:
        """Validate multi-GPU test configuration."""
        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-capable GPUs and PyTorch are required")

        if not dist.is_nccl_available():
            raise RuntimeError("NCCL backend is required for multi-GPU testing")

        # Resolve auto-detected values
        self.device_count = self._resolve_device_count()
        self.mesh_size_per_gpu = self._resolve_mesh_size(self.device_count)

        available = torch.cuda.device_count()
        if self.device_count < 2:
            raise ValueError(
                f"Multi-GPU test requires at least 2 GPUs, found {available}"
            )
        if self.device_count > available:
            raise ValueError(
                f"Requested {self.device_count} GPUs but only {available} available"
            )

        if self.solver_iterations < 10 or self.solver_iterations > 1000:
            raise ValueError("solver_iterations must be between 10 and 1000")

        if self.halo_depth < 1 or self.halo_depth > 4:
            raise ValueError("halo_depth must be between 1 and 4")

        if self.burnin_seconds < 0:
            raise ValueError("burnin_seconds must be >= 0")

        # Check VRAM on each GPU
        bytes_per_cell = 170  # ~56B base + ~114B solver overhead
        required_bytes = self.mesh_size_per_gpu * bytes_per_cell

        gpu_names = set()
        for i in range(self.device_count):
            props = torch.cuda.get_device_properties(i)
            gpu_names.add(props.name)
            if required_bytes > props.total_memory * 0.80:
                vram_gb = props.total_memory / (1024**3)
                need_gb = required_bytes / (1024**3)
                raise ValueError(
                    f"GPU {i} ({props.name}) has "
                    f"{vram_gb:.1f} GB but mesh requires "
                    f"~{need_gb:.1f} GB"
                )

        if len(gpu_names) > 1:
            self.logger.warning(
                f"Heterogeneous GPUs detected: {gpu_names}. "
                "Using smallest VRAM for mesh sizing."
            )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """Detect topology, find free port, create shared dict, get UUIDs."""
        import multiprocessing

        from warpt.backends.factory import get_accelerator_backend

        backend = get_accelerator_backend()

        self._topology = backend.get_topology()
        self._distributed_backend = backend.get_distributed_backend()
        self._device_prefix = backend.get_pytorch_device_string(0).split(":")[0]
        self._master_port = _find_free_port()

        # Extract UUIDs from GPUInfo objects, falling back to model name or index
        devices = backend.list_devices()
        self._gpu_uuids = [
            d.uuid if d.uuid else (d.model if d.model else f"gpu_{d.index}")
            for d in devices[: self.device_count]
        ]

        backend.shutdown()

        self._manager = multiprocessing.Manager()
        self._shared_dict = self._manager.dict()

        self.logger.info(f"Detected topology: {self._topology}")
        self.logger.info(f"Using {self.device_count} GPUs, port {self._master_port}")
        self.logger.info(f"Mesh size per GPU: {self.mesh_size_per_gpu:,} cells")
        self.logger.info(f"Halo depth: {self.halo_depth}")

        for i in range(self.device_count):
            import torch

            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            self.logger.info(f"  GPU {i}: {name} ({vram_gb:.1f} GB)")

    def warmup(self, duration_seconds: int = 0, iterations: int = 3) -> None:
        """No-op: workers handle their own warmup during single-GPU baseline."""
        pass

    def execute_test(self, duration: int, iterations: int) -> GPUSystemResult:
        """Spawn workers and collect multi-GPU scaling results.

        Args:
            duration: Test duration in seconds.
            iterations: Ignored (test runs for duration).

        Returns:
            GPUSystemResult with scaling efficiency and detailed metrics.
        """
        del iterations

        import torch.multiprocessing as mp

        config = {
            "master_port": self._master_port,
            "mesh_size_per_gpu": self.mesh_size_per_gpu,
            "solver_iterations": self.solver_iterations,
            "halo_depth": self.halo_depth,
            "duration": duration,
            "distributed_backend": self._distributed_backend,
            "device_prefix": self._device_prefix,
        }

        self.logger.info(f"Spawning {self.device_count} worker processes...")

        test_start = time.time()

        try:
            mp.spawn(
                _worker_fn,
                args=(self.device_count, self._shared_dict, config),
                nprocs=self.device_count,
                join=True,
            )
        except Exception as e:
            self.logger.error(f"Worker spawn failed: {e}")
            # Try to collect partial results
            if not self._shared_dict:
                raise

        test_elapsed = time.time() - test_start

        # ------------------------------------------------------------------
        # Aggregate results
        # ------------------------------------------------------------------
        worker_results = dict(self._shared_dict)

        # Check for worker errors
        for rank, wr in worker_results.items():
            if "error" in wr:
                self.logger.error(f"Worker {rank} error: {wr['error']}")

        # Collect per-GPU TFLOPS
        single_gpu_tflops = []
        multi_gpu_tflops = []
        per_gpu_metrics: dict[str, Any] = {}

        for rank in range(self.device_count):
            wr = worker_results.get(rank, {})
            sg_tf = wr.get("single_gpu_tflops", 0.0)
            mg_tf = wr.get("multi_gpu_tflops", 0.0)
            single_gpu_tflops.append(sg_tf)
            multi_gpu_tflops.append(mg_tf)

            per_gpu_metrics[f"gpu_{rank}"] = {
                "gpu_name": wr.get("gpu_name", "unknown"),
                "single_gpu_tflops": sg_tf,
                "multi_gpu_tflops": mg_tf,
                "single_gpu_solver_stats": wr.get("single_gpu_solver_stats", {}),
                "compute_stats": wr.get("compute_stats", {}),
                "halo_stats": wr.get("halo_stats", {}),
                "allreduce_stats": wr.get("allreduce_stats", {}),
                "total_iter_stats": wr.get("total_iter_stats", {}),
                "communication_fraction": wr.get("communication_fraction", 0.0),
                "comm_benchmarks": wr.get("comm_benchmarks", {}),
            }

        # Scaling efficiency
        mean_single = (
            sum(single_gpu_tflops) / len(single_gpu_tflops)
            if single_gpu_tflops
            else 0.0
        )
        sum_multi = sum(multi_gpu_tflops)
        ideal_multi = mean_single * self.device_count

        scaling_efficiency = (
            min(sum_multi / ideal_multi, 1.0) if ideal_multi > 0 else 0.0
        )

        # Average overhead across ranks
        overhead_values = [
            worker_results.get(r, {}).get("avg_overhead_ms", 0.0)
            for r in range(self.device_count)
        ]
        avg_overhead = (
            sum(overhead_values) / len(overhead_values) if overhead_values else 0.0
        )

        # Average communication fraction
        comm_fractions = [
            worker_results.get(r, {}).get("communication_fraction", 0.0)
            for r in range(self.device_count)
        ]
        avg_comm_fraction = (
            sum(comm_fractions) / len(comm_fractions) if comm_fractions else 0.0
        )

        # Aggregate comm benchmark from rank 0 (representative)
        rank0_comm = worker_results.get(0, {}).get("comm_benchmarks", {})

        self.logger.info(f"Scaling efficiency: {scaling_efficiency:.3f}")
        self.logger.info(
            f"Single-GPU TFLOPS (mean): {mean_single:.3f}, "
            f"Multi-GPU aggregate: {sum_multi:.3f}"
        )
        self.logger.info(f"Communication overhead: {avg_overhead:.2f}ms avg")
        self.logger.info(f"Communication fraction: {avg_comm_fraction:.3%}")
        self.logger.info(f"Interconnect: {self._topology}")

        return GPUSystemResult(
            devices_used=[f"gpu_{i}" for i in range(self.device_count)],
            gpu_uuids=self._gpu_uuids,
            device_count=self.device_count,
            aggregate_tflops=sum_multi,
            duration=test_elapsed,
            iterations=None,
            scaling_efficiency=scaling_efficiency,
            orchestration_overhead_ms=avg_overhead,
            burnin_seconds=self.burnin_seconds,
            metrics={
                "interconnect_type": self._topology,
                "mesh_size_per_gpu": self.mesh_size_per_gpu,
                "solver_iterations": self.solver_iterations,
                "halo_depth": self.halo_depth,
                "mean_single_gpu_tflops": mean_single,
                "per_gpu_single_tflops": single_gpu_tflops,
                "per_gpu_multi_tflops": multi_gpu_tflops,
                "communication_fraction": avg_comm_fraction,
                "comm_benchmarks": rank0_comm,
                "per_gpu": per_gpu_metrics,
            },
            max_temp_across_gpus=None,
            total_power=None,
        )

    def teardown(self) -> None:
        """Shut down the multiprocessing manager."""
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:
                pass
            self._manager = None
        self._shared_dict = None
