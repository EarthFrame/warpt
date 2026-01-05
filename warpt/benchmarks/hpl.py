"""High Performance Linpack (HPL) benchmark implementation."""

import os
import subprocess
import time
from pathlib import Path

from warpt.benchmarks.base import Benchmark, BenchmarkResult, RunMode


class HPLBenchmark(Benchmark):
    """High Performance Linpack benchmark for measuring floating-point performance.

    HPL solves a dense linear system Ax=b in double precision arithmetic
    using LU factorization with partial pivoting. This is the standard
    benchmark for measuring peak floating-point performance of HPC systems.

    This implementation supports two execution modes:
    1. 'numpy': Pure Python/NumPy implementation (single node).
    2. 'docker': Parallel execution via HPL in a Docker container.
    """

    _PARAM_FIELDS = (
        "problem_size",
        "block_size",
        "p_grid",
        "q_grid",
        "pfact",
        "nbmin",
        "ndiv",
        "rfact",
        "bcast",
        "depth",
        "swap",
        "swapping_threshold",
        "execution_mode",
        "docker_image",
    )

    def __init__(
        self,
        problem_size: int = 1024,
        block_size: int = 128,
        p_grid: int = 1,
        q_grid: int = 1,
        pfact: int = 2,  # 0=left, 1=Crout, 2=Right
        nbmin: int = 4,
        ndiv: int = 2,
        rfact: int = 1,  # 0=left, 1=Crout, 2=Right
        bcast: int = 1,  # 1=1rM
        depth: int = 1,
        swap: int = 2,  # 2=mix
        swapping_threshold: int = 64,
        execution_mode: str = "numpy",
        docker_image: str = "warpt-hpl:arm64",
    ):
        """Initialize HPL benchmark.

        Args:
            problem_size: Size of the coefficient matrix (N x N).
            block_size: Blocking factor (NB).
            p_grid: Number of process rows (P).
            q_grid: Number of process columns (Q).
            pfact: Panel factorization algorithm.
            nbmin: Recursive stopping column.
            ndiv: Panels in recursion.
            rfact: Recursive panel factorization.
            bcast: Broadcast algorithm.
            depth: Lookahead depth.
            swap: Swapping algorithm.
            swapping_threshold: Swapping threshold.
            execution_mode: 'numpy' or 'docker'.
            docker_image: Docker image to use for 'docker' mode.
        """
        self.problem_size = problem_size
        self.block_size = block_size
        self.p_grid = p_grid
        self.q_grid = q_grid
        self.pfact = pfact
        self.nbmin = nbmin
        self.ndiv = ndiv
        self.rfact = rfact
        self.bcast = bcast
        self.depth = depth
        self.swap = swap
        self.swapping_threshold = swapping_threshold
        self.execution_mode = execution_mode
        self.docker_image = docker_image

    def get_pretty_name(self) -> str:
        """Get human-readable benchmark name."""
        return "High Performance Linpack (HPL)"

    def get_description(self) -> str:
        """Get benchmark description."""
        return "Measures floating-point performance via LU factorization"

    def get_run_mode(self) -> RunMode:
        """HPL runs once for a fixed problem size."""
        return RunMode.SINGLE_RUN

    def is_available(self) -> bool:
        """Check if HPL benchmark can run on this system."""
        if self.execution_mode == "numpy":
            try:
                import numpy as np

                # Check for BLAS/LAPACK availability
                np.linalg.solve(
                    np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 1.0])
                )
                return True
            except ImportError:
                return False
        elif self.execution_mode == "docker":
            try:
                subprocess.run(
                    ["docker", "--version"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
        return False

    def validate_configuration(self) -> None:
        """Validate benchmark configuration."""
        if self.problem_size < 64:
            raise ValueError("problem_size must be >= 64")
        if self.block_size < 16:
            raise ValueError("block_size must be >= 16")
        if self.problem_size % self.block_size != 0:
            raise ValueError("problem_size must be divisible by block_size")

        if self.execution_mode not in ("numpy", "docker"):
            raise ValueError(f"Invalid execution_mode: {self.execution_mode}")

        if not self.is_available():
            raise RuntimeError(f"HPL ({self.execution_mode}) not available")

    def setup(self) -> None:
        """Prepare resources for benchmark."""
        # Auto-tune P and Q if needed
        if self.execution_mode == "docker" and self.p_grid == 1 and self.q_grid == 1:
            self._auto_tune_grid()

        if self.execution_mode == "numpy":
            import numpy as np

            # Generate test matrix
            np.random.seed(42)
            n = self.problem_size
            self.A = np.random.rand(n, n).astype(np.float64)
            self.b = np.random.rand(n).astype(np.float64)

            # Make matrix diagonally dominant
            for i in range(n):
                self.A[i, i] += n

            self.logger.info(f"Allocated {n}x{n} matrix ({n*n*8/(1024**3):.2f} GB)")
        elif self.execution_mode == "docker":
            # Generate HPL.dat in the current directory or a temp directory
            self.generate_hpl_dat("HPL.dat")

    def _auto_tune_grid(self) -> None:
        """Automatically determine P and Q based on available CPU cores."""
        try:
            from warpt.backends.system import CPU

            cpu_info = CPU().get_cpu_info()
            cores = cpu_info.total_physical_cores

            # For HPL, P and Q should be as close as possible, with Q >= P
            # Find factors of 'cores'
            import math

            p = int(math.sqrt(cores))
            while p > 0:
                if cores % p == 0:
                    q = cores // p
                    self.p_grid = p
                    self.q_grid = q
                    self.logger.info(
                        f"Auto-tuned HPL grid to {p}x{q} for {cores} cores"
                    )
                    break
                p -= 1
        except Exception as e:
            self.logger.warning(f"Failed to auto-tune HPL grid: {e}")
            # Fallback to 1x1 already set in __init__

    def teardown(self) -> None:
        """Clean up resources."""
        if self.execution_mode == "numpy":
            if hasattr(self, "A"):
                del self.A
            if hasattr(self, "b"):
                del self.b
        elif self.execution_mode == "docker":
            if os.path.exists("HPL.dat"):
                os.remove("HPL.dat")

    def generate_hpl_dat(self, output_path: str) -> None:
        """Generate HPL.dat configuration file.

        Args:
            output_path: Path to save the HPL.dat file.
        """
        content = f"""HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
{self.problem_size}        Ns
1            # of NBs
{self.block_size}          NBs
0            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
{self.p_grid}            Ps
{self.q_grid}            Qs
16.0         threshold
1            # of panel fact
{self.pfact}            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stoppinging column
{self.nbmin}            NBMINs (>= 1)
1            # of panels in recursion
{self.ndiv}            NDIVs
1            # of recursive panel fact.
{self.rfact}            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
{self.bcast}            BCASTs (0=1rg, 1=1rM, 2=2rg, 3=2rM, 4=Lng, 5=LnM)
1            # of lookahead depth
{self.depth}            DEPTHs (>=0)
1            # of swapping
{self.swap}            SWAP (0=bin-exch,1=det,2=mix)
{self.swapping_threshold}           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
"""
        with open(output_path, "w") as f:
            f.write(content)
        self.logger.info(f"Generated HPL.dat at {output_path}")

    def execute_benchmark(
        self, _duration: int | None = None, _iterations: int | None = None
    ) -> BenchmarkResult:
        """Execute HPL benchmark."""
        if self.execution_mode == "numpy":
            return self._execute_numpy()
        else:
            return self._execute_docker()

    def _execute_numpy(self) -> BenchmarkResult:
        """Execute HPL using NumPy."""
        import numpy as np

        start_time = time.time()
        x = np.linalg.solve(self.A, self.b)
        solve_time = time.time() - start_time

        residual = np.linalg.norm(self.A @ x - self.b)
        n = self.problem_size
        flops = self._calculate_flops()
        gflops = flops / solve_time / 1e9

        return BenchmarkResult(
            metrics={"gflops": gflops, "solve_time": solve_time},
            metadata={
                "problem_size": n,
                "block_size": self.block_size,
                "execution_mode": "numpy",
            },
            validation={
                "residual": float(residual),
                "residual_normalized": float(residual / (n * np.linalg.norm(self.b))),
            },
            timing={"solve": solve_time},
        )

    def _execute_docker(self) -> BenchmarkResult:
        """Execute HPL using Docker."""
        # Total processes = P * Q
        num_procs = self.p_grid * self.q_grid

        # Run Docker container
        # Mount the generated HPL.dat into the container
        cwd = os.getcwd()
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{cwd}/HPL.dat:/hpl/HPL.dat",
            self.docker_image,
            "xhpl",
            "-np",
            str(num_procs),
            "xhpl",
        ]

        self.logger.info(f"Running HPL in Docker: {' '.join(cmd)}")
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        total_time = time.time() - start_time

        # Parse output for GFLOPS
        # Example output line:
        # WR11C2R4  10240  128  2  2  54.34  1.312e+01
        gflops = 0.0
        for line in result.stdout.splitlines():
            if "WR" in line and self.problem_size == int(line.split()[1]):
                try:
                    # GFLOPS is usually the second to last column
                    # But it depends on the exact HPL output format.
                    # Standard HPL output: T/V N NB P Q Time Gflops
                    parts = line.split()
                    gflops = float(parts[6])
                except (ValueError, IndexError):
                    pass

        if gflops == 0.0:
            self.logger.warning("Could not parse GFLOPS from HPL output")
            self.logger.debug(f"HPL Output: {result.stdout}")

        return BenchmarkResult(
            metrics={"gflops": gflops, "total_time": total_time},
            metadata={
                "problem_size": self.problem_size,
                "block_size": self.block_size,
                "p_grid": self.p_grid,
                "q_grid": self.q_grid,
                "execution_mode": "docker",
                "docker_image": self.docker_image,
            },
            raw_data=result.stdout,
            timing={"total_execution": total_time},
        )

    def build(self) -> None:
        """Build the HPL benchmark.

        For 'docker' mode, this builds the Docker image if it doesn't exist or
        if forced.
        """
        if self.execution_mode == "docker":
            self.logger.info(f"Building HPL Docker image: {self.docker_image}")

            # Determine the path to the Dockerfile relative to project root
            root = Path(__file__).parents[2]
            dockerfile = root / "docker/benchmarks/hpl/Dockerfile.arm64"
            context = root / "docker/benchmarks/hpl/"

            if not dockerfile.exists():
                self.logger.error(f"Dockerfile not found: {dockerfile}")
                return

            cmd = [
                "docker",
                "build",
                "-t",
                self.docker_image,
                "-f",
                str(dockerfile),
                str(context),
            ]

            self.logger.info(f"Running: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
                self.logger.info("HPL Docker image built successfully")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to build HPL Docker image: {e}")
                raise RuntimeError(f"HPL build failed: {e}") from e
        else:
            self.logger.info("NumPy mode: no build required")

    def _calculate_flops(self) -> float:
        """Calculate total floating-point operations."""
        n = self.problem_size
        lu_flops = (2.0 / 3.0) * n * n * n
        solve_flops = 2.0 * n * n
        return lu_flops + solve_flops
