"""GPU CFD simulation stress test.

This test simulates Computational Fluid Dynamics (CFD) workload patterns
to measure realistic performance for fluid simulation applications.
"""

import time
from typing import Any

from warpt.models.constants import DEFAULT_BURNIN_SECONDS
from warpt.stress.base import StressTest, TestCategory


class GPUCFDSimulationTest(StressTest):
    """GPU CFD simulation performance test.

    Simulates realistic CFD workload patterns including:
    - Sparse matrix solver (conjugate gradient)
    - Gradient computations (neighbor cell access)
    - Memory-bound operations (large mesh data)

    This test measures performance for fluid simulation applications.
    """

    _PARAM_FIELDS = ("mesh_size", "device_id", "burnin_seconds", "solver_iterations")

    def __init__(
        self,
        mesh_size: int = 1000000,
        device_id: int = 0,
        burnin_seconds: int = DEFAULT_BURNIN_SECONDS,
        solver_iterations: int = 100,
    ):
        """Initialize GPU CFD simulation test.

        Args:
            mesh_size: Number of mesh cells to simulate.
                Default 1M cells (typical for single GPU CFD).
            device_id: GPU device ID to test. Default 0.
            burnin_seconds: Warmup duration before measurement.
            solver_iterations: Number of solver iterations per solve.
                Default 100 (typical for pressure solver).
        """
        self.mesh_size = mesh_size
        self.device_id = device_id
        self.burnin_seconds = burnin_seconds
        self.solver_iterations = solver_iterations
        self._device = None
        self._gpu_name = None

    # -------------------------------------------------------------------------
    # Identity & Metadata
    # -------------------------------------------------------------------------

    def get_pretty_name(self) -> str:
        """Return human-readable test name."""
        return "GPU CFD Simulation Test"

    def get_description(self) -> str:
        """Return one-line description."""
        return (
            "Simulates CFD workload patterns (sparse solver, gradients, "
            "memory-bound ops) for fluid simulation applications"
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
                f"Available devices: 0-{torch.cuda.device_count()-1}"
            )

        if self.mesh_size < 10000:
            raise ValueError("mesh_size must be >= 10,000 cells")
        if self.mesh_size > 100000000:
            raise ValueError(
                "mesh_size must be <= 100,000,000 cells (avoid excessive memory)"
            )

        if self.solver_iterations < 10:
            raise ValueError("solver_iterations must be >= 10")
        if self.solver_iterations > 1000:
            raise ValueError(
                "solver_iterations must be <= 1000 (avoid excessive runtime)"
            )

        if self.burnin_seconds < 0:
            raise ValueError("burnin_seconds must be >= 0")

        # Estimate memory requirements
        # Each cell needs: pressure (8B), velocity (24B), gradients (24B) = ~56 bytes
        bytes_per_cell = 56
        total_bytes_needed = self.mesh_size * bytes_per_cell
        gpu_memory = torch.cuda.get_device_properties(self.device_id).total_memory

        if total_bytes_needed > gpu_memory * 0.8:  # Leave 20% headroom
            raise ValueError(
                f"mesh_size {self.mesh_size} requires "
                f"{total_bytes_needed / (1024**3):.2f} GB "
                f"but GPU only has {gpu_memory / (1024**3):.2f} GB"
            )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize GPU and get device info."""
        import torch

        self._device = torch.device(f"cuda:{self.device_id}")
        self._gpu_name = torch.cuda.get_device_name(self.device_id)

        # Get GPU properties
        props = torch.cuda.get_device_properties(self.device_id)
        memory_gb = props.total_memory / (1024**3)

        self.logger.info(f"GPU: {self._gpu_name}")
        self.logger.info(f"Device ID: {self.device_id}")
        self.logger.info(f"Memory: {memory_gb:.2f} GB")
        self.logger.info(f"Compute Capability: {props.major}.{props.minor}")
        self.logger.info(f"Mesh size: {self.mesh_size:,} cells")
        self.logger.info(f"Solver iterations: {self.solver_iterations}")

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

        # Create small warmup data (1% of full mesh)
        warmup_size = max(1000, self.mesh_size // 100)

        try:
            # Warmup with small mesh
            pressure = torch.randn(
                warmup_size, dtype=torch.float64, device=self._device
            )
            velocity = torch.randn(
                warmup_size, 3, dtype=torch.float64, device=self._device
            )

            if duration_seconds > 0:
                self.logger.debug(f"Warming up for {duration_seconds}s...")
                start = time.time()
                while (time.time() - start) < duration_seconds:
                    # Simple warmup operation
                    _ = pressure + velocity[:, 0]
                    torch.cuda.synchronize(self._device)
            else:
                self.logger.debug(f"Warming up for {iterations} iterations...")
                for _ in range(iterations):
                    _ = pressure + velocity[:, 0]
                    torch.cuda.synchronize(self._device)

            del pressure, velocity
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.warning("GPU out of memory during warmup.")
                torch.cuda.empty_cache()
            raise

    # -------------------------------------------------------------------------
    # Core Test (TO BE IMPLEMENTED)
    # -------------------------------------------------------------------------

    def execute_test(self, duration: int, iterations: int) -> dict[str, Any]:
        """Execute the CFD simulation test.

        Args:
            duration: Test duration in seconds.
            iterations: Ignored (test runs for duration).

        Returns:
            Dictionary containing CFD performance metrics.
        """
        del iterations  # Unused; test runs for duration

        import torch

        self.logger.info(f"Running CFD simulation test for {duration}s...")
        self.logger.info(f"Simulating {self.mesh_size:,} cell mesh")

        # =====================================================================
        # STEP 1 - Allocate mesh data structures
        # =====================================================================
        # Allocate CFD field data on GPU
        # Each cell stores: pressure (scalar), velocity (3D vector), gradients (3D)
        self.logger.info("Allocating mesh data structures...")

        alloc_start = time.perf_counter()

        try:
            # Pressure field: scalar per cell (Pa or psi)
            pressure = torch.randn(
                self.mesh_size, dtype=torch.float64, device=self._device
            )

            # Velocity field: 3D vector per cell (u, v, w in m/s)
            velocity = torch.randn(
                self.mesh_size, 3, dtype=torch.float64, device=self._device
            )

            # Gradient field: 3D gradient per cell (∂/∂x, ∂/∂y, ∂/∂z)
            gradients = torch.randn(
                self.mesh_size, 3, dtype=torch.float64, device=self._device
            )

            # Residual vector for iterative solver
            residual = torch.zeros(
                self.mesh_size, dtype=torch.float64, device=self._device
            )

            torch.cuda.synchronize(self._device)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"GPU out of memory allocating {self.mesh_size:,} cells. "
                    f"Requires ~{(self.mesh_size * 56) / (1024**3):.2f} GB. "
                    f"Try reducing mesh_size parameter."
                ) from e
            raise

        alloc_time = time.perf_counter() - alloc_start

        # Calculate actual memory used
        bytes_allocated = (
            pressure.element_size() * pressure.nelement()
            + velocity.element_size() * velocity.nelement()
            + gradients.element_size() * gradients.nelement()
            + residual.element_size() * residual.nelement()
        )
        gb_allocated = bytes_allocated / (1024**3)

        self.logger.info(
            f"Allocated {gb_allocated:.2f} GB in {alloc_time*1000:.1f}ms "
            f"({self.mesh_size:,} cells)"
        )

        # =====================================================================
        # STEP 2 - Sparse Matrix Solver (Conjugate Gradient)
        # =====================================================================
        # Simulate CFD pressure solver (70% of CFD runtime)
        # This tests sparse matrix-vector multiply performance
        self.logger.info("Running sparse matrix solver...")

        solver_times = []
        test_start_time = time.time()

        # Run solver multiple times to collect statistics
        while (time.time() - test_start_time) < duration:
            solver_start = time.perf_counter()

            # Simulate conjugate gradient solver
            # In real CFD: solves Ax = b for pressure
            # We simulate the key operations without building full matrix A
            x = pressure.clone()  # Initial guess
            r = residual.clone()  # Residual vector
            p = r.clone()  # Search direction

            for _ in range(self.solver_iterations):
                # Key operation: Sparse matrix-vector multiply (A @ p)
                # Simulates neighbor interactions (adjacency list)
                # In real CFD: this accesses sparse matrix structure
                # We approximate with stencil operations (neighbor access pattern)

                # Simulate 7-point stencil (6 neighbors + self)
                # This mimics sparse matrix access patterns
                ap = (
                    6.0 * p  # Diagonal (self)
                    - torch.roll(p, 1, 0)  # Left neighbor
                    - torch.roll(p, -1, 0)  # Right neighbor
                    - torch.roll(p, 100, 0)  # Up neighbor (approx)
                    - torch.roll(p, -100, 0)  # Down neighbor (approx)
                    - torch.roll(p, 10000, 0)  # Front neighbor (approx)
                    - torch.roll(p, -10000, 0)  # Back neighbor (approx)
                )

                # Conjugate gradient updates (vector operations)
                rtr = torch.dot(r, r)  # Dot product
                alpha = rtr / torch.dot(p, ap)  # Scalar
                x = x + alpha * p  # Update solution (AXPY operation)
                r_new = r - alpha * ap  # Update residual
                beta = torch.dot(r_new, r_new) / rtr  # Scalar
                p = r_new + beta * p  # Update search direction
                r = r_new

                # Early exit if converged (optional, real solvers do this)
                if torch.norm(r) < 1e-6:
                    break

            torch.cuda.synchronize(self._device)
            solver_elapsed = time.perf_counter() - solver_start
            solver_times.append(solver_elapsed * 1000)  # Convert to ms

        # Calculate solver statistics
        if solver_times:
            avg_solver_time = sum(solver_times) / len(solver_times)
            min_solver_time = min(solver_times)
            max_solver_time = max(solver_times)

            sorted_times = sorted(solver_times)
            p50_solver = sorted_times[len(sorted_times) // 2]
            p95_solver = sorted_times[int(len(sorted_times) * 0.95)]
            p99_solver = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            avg_solver_time = 0
            min_solver_time = 0
            max_solver_time = 0
            p50_solver = 0
            p95_solver = 0
            p99_solver = 0

        num_solves = len(solver_times)

        self.logger.info(
            f"Solver: {num_solves} solves, avg={avg_solver_time:.2f}ms, "
            f"p95={p95_solver:.2f}ms"
        )

        # =====================================================================
        # STEP 3 - Gradient Computations
        # =====================================================================
        # Compute spatial gradients using neighbor cell access
        # This tests memory access patterns and bandwidth
        self.logger.info("Computing gradients...")

        gradient_times = []
        gradient_start_time = time.time()

        # Run gradient computations for remaining test duration
        remaining_time = duration - (gradient_start_time - test_start_time)

        while (time.time() - gradient_start_time) < remaining_time / 2:
            grad_start = time.perf_counter()

            # Compute pressure gradients in X, Y, Z directions
            # Uses central difference: ∇P = (P_right - P_left) / (2*dx)
            # This requires accessing neighbor cells (memory-intensive)

            # X-direction gradient (left/right neighbors)
            grad_x = (torch.roll(pressure, -1, 0) - torch.roll(pressure, 1, 0)) / 2.0

            # Y-direction gradient (up/down neighbors, approximate with stride)
            grad_y = (
                torch.roll(pressure, -100, 0) - torch.roll(pressure, 100, 0)
            ) / 2.0

            # Z-direction gradient (front/back neighbors, approximate with stride)
            grad_z = (
                torch.roll(pressure, -10000, 0) - torch.roll(pressure, 10000, 0)
            ) / 2.0

            # Store gradients (memory write)
            gradients[:, 0] = grad_x
            gradients[:, 1] = grad_y
            gradients[:, 2] = grad_z

            # Compute velocity gradients (3 components x 3 directions = 9 gradients)
            # This is more memory-intensive (reading velocity field)
            for component in range(3):
                vel_component = velocity[:, component]
                _ = (
                    torch.roll(vel_component, -1, 0) - torch.roll(vel_component, 1, 0)
                ) / 2.0

            torch.cuda.synchronize(self._device)
            grad_elapsed = time.perf_counter() - grad_start
            gradient_times.append(grad_elapsed * 1000)  # Convert to ms

        # Calculate gradient statistics
        if gradient_times:
            avg_gradient_time = sum(gradient_times) / len(gradient_times)
            min_gradient_time = min(gradient_times)
            max_gradient_time = max(gradient_times)

            sorted_grad_times = sorted(gradient_times)
            p50_gradient = sorted_grad_times[len(sorted_grad_times) // 2]
            p95_gradient = sorted_grad_times[int(len(sorted_grad_times) * 0.95)]
            p99_gradient = sorted_grad_times[int(len(sorted_grad_times) * 0.99)]
        else:
            avg_gradient_time = 0
            min_gradient_time = 0
            max_gradient_time = 0
            p50_gradient = 0
            p95_gradient = 0
            p99_gradient = 0

        num_gradient_ops = len(gradient_times)

        self.logger.info(
            f"Gradients: {num_gradient_ops} ops, avg={avg_gradient_time:.2f}ms, "
            f"p95={p95_gradient:.2f}ms"
        )

        # =====================================================================
        # STEP 4 - Memory-Bound Operations (Sequential Streaming)
        # =====================================================================
        # Simulate flux calculations (mass/momentum/energy transport)
        # This tests sequential memory bandwidth (streaming throughput)
        self.logger.info("Running memory-bound operations...")

        flux_times = []
        flux_start_time = time.time()

        # Run flux calculations for remaining test duration
        remaining_time_flux = duration - (flux_start_time - test_start_time)

        while (time.time() - flux_start_time) < remaining_time_flux / 2:
            flux_iter_start = time.perf_counter()

            # Flux calculations: Sequential read + write operations
            # In CFD: mass_flux = density * velocity
            #         momentum_flux = density * velocity * velocity
            #         energy_flux = (energy + pressure) * velocity

            # Mass flux (3 components)
            # Read: pressure (1M), velocity (3M) = 4M reads
            # Write: 3M values
            mass_flux_x = pressure * velocity[:, 0]
            mass_flux_y = pressure * velocity[:, 1]
            mass_flux_z = pressure * velocity[:, 2]

            # Momentum flux (simplified)
            # Additional reads and writes
            momentum_flux = velocity * velocity[:, 0:1]  # Broadcast multiply

            # Energy flux (combines pressure and velocity)
            energy_flux = (pressure + pressure) * velocity[:, 0]

            # Update fields (memory writes)
            # This simulates updating cell values based on fluxes
            pressure_update = mass_flux_x + mass_flux_y + mass_flux_z
            velocity_update = momentum_flux.sum(dim=1)

            # Apply updates (more memory writes)
            pressure[:] = pressure + 0.01 * pressure_update
            velocity[:, 0] = velocity[:, 0] + 0.01 * velocity_update

            torch.cuda.synchronize(self._device)
            flux_iter_elapsed = time.perf_counter() - flux_iter_start
            flux_times.append(flux_iter_elapsed * 1000)  # Convert to ms

        # Calculate flux operation statistics
        if flux_times:
            avg_flux_time = sum(flux_times) / len(flux_times)
            min_flux_time = min(flux_times)
            max_flux_time = max(flux_times)

            sorted_flux_times = sorted(flux_times)
            p50_flux = sorted_flux_times[len(sorted_flux_times) // 2]
            p95_flux = sorted_flux_times[int(len(sorted_flux_times) * 0.95)]
            p99_flux = sorted_flux_times[int(len(sorted_flux_times) * 0.99)]

            # Estimate memory bandwidth
            # Each flux iteration reads/writes multiple arrays
            # Pressure: 8 bytes x 1M cells = 8 MB (read 3x, write 1x)
            # Velocity: 8 bytes x 3M values = 24 MB (read 5x, write 1x)
            # Total: ~200 MB per iteration
            bytes_per_iteration = (
                pressure.element_size() * pressure.nelement() * 4  # Pressure R/W
                + velocity.element_size() * velocity.nelement() * 6  # Velocity R/W
            )
            gb_per_iteration = bytes_per_iteration / (1024**3)
            memory_bandwidth_gbps = gb_per_iteration / (avg_flux_time / 1000)
        else:
            avg_flux_time = 0
            min_flux_time = 0
            max_flux_time = 0
            p50_flux = 0
            p95_flux = 0
            p99_flux = 0
            memory_bandwidth_gbps = 0.0

        num_flux_ops = len(flux_times)

        self.logger.info(
            f"Flux ops: {num_flux_ops} iterations, avg={avg_flux_time:.2f}ms, "
            f"bandwidth={memory_bandwidth_gbps:.1f} GB/s"
        )

        # Clean up
        del x, r, p, ap, r_new
        del grad_x, grad_y, grad_z
        del mass_flux_x, mass_flux_y, mass_flux_z
        del momentum_flux, energy_flux, pressure_update, velocity_update
        torch.cuda.empty_cache()

        # =====================================================================
        # Calculate Overall Metrics
        # =====================================================================

        test_elapsed = time.time() - test_start_time

        # Calculate cell updates per second
        # Each test iteration processes all cells
        total_iterations = num_solves + num_gradient_ops + num_flux_ops
        total_cell_updates = total_iterations * self.mesh_size
        cell_updates_per_sec = (
            total_cell_updates / test_elapsed if test_elapsed > 0 else 0
        )

        self.logger.info(
            f"\nOverall: {total_cell_updates:,} cell updates in {test_elapsed:.1f}s"
        )
        self.logger.info(f"Performance: {cell_updates_per_sec / 1e6:.1f}M cells/sec")

        return {
            "test_name": self.get_name(),
            "duration": test_elapsed,
            "mesh_size": self.mesh_size,
            "device_id": self.device_id,
            "gpu_name": self._gpu_name,
            "burnin_seconds": self.burnin_seconds,
            "solver_iterations": self.solver_iterations,
            # Overall performance
            "cell_updates_per_sec": cell_updates_per_sec,
            "total_cell_updates": total_cell_updates,
            # Solver metrics
            "num_solves": num_solves,
            "avg_solver_time_ms": avg_solver_time,
            "min_solver_time_ms": min_solver_time,
            "max_solver_time_ms": max_solver_time,
            "p50_solver_time_ms": p50_solver,
            "p95_solver_time_ms": p95_solver,
            "p99_solver_time_ms": p99_solver,
            # Gradient metrics
            "num_gradient_ops": num_gradient_ops,
            "avg_gradient_time_ms": avg_gradient_time,
            "min_gradient_time_ms": min_gradient_time,
            "max_gradient_time_ms": max_gradient_time,
            "p50_gradient_time_ms": p50_gradient,
            "p95_gradient_time_ms": p95_gradient,
            "p99_gradient_time_ms": p99_gradient,
            # Flux/memory metrics
            "num_flux_ops": num_flux_ops,
            "avg_flux_time_ms": avg_flux_time,
            "min_flux_time_ms": min_flux_time,
            "max_flux_time_ms": max_flux_time,
            "p50_flux_time_ms": p50_flux,
            "p95_flux_time_ms": p95_flux,
            "p99_flux_time_ms": p99_flux,
            "memory_bandwidth_gbps": memory_bandwidth_gbps,
        }
