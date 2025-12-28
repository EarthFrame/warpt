"""Base class for performance benchmarks."""

import logging
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class RunMode(Enum):
    """Benchmark execution mode."""

    TIME_BOUND = "time"  # Run for specified duration
    ITERATION_BOUND = "iteration"  # Run for specified iterations
    SINGLE_RUN = "single"  # Run once (typical for problem-based benchmarks)


class BenchmarkResult:
    """Standard benchmark result container.

    Provides a consistent structure for benchmark results across different
    benchmark types. Benchmarks can extend this or return custom structures.
    """

    def __init__(
        self,
        metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
        timing: dict[str, float] | None = None,
        validation: dict[str, Any] | None = None,
        raw_data: Any | None = None,
    ):
        """Initialize benchmark result.

        Args:
            metrics: Primary performance metrics (e.g., {"gflops": 125.5})
            metadata: Benchmark configuration and system info
            timing: Timing breakdown (setup, warmup, execution, etc.)
            validation: Validation results (residuals, accuracy, etc.)
            raw_data: Raw measurement data for detailed analysis
        """
        self.metrics = metrics
        self.metadata = metadata or {}
        self.timing = timing or {}
        self.validation = validation or {}
        self.raw_data = raw_data
        self.timestamp = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "metrics": self.metrics,
            "metadata": self.metadata,
            "timing": self.timing,
            "validation": self.validation,
            "timestamp": self.timestamp,
        }


class Benchmark(ABC):
    """Abstract base class for all performance benchmarks.

    Designed to support diverse benchmark types including:
    - HPL (High Performance Linpack)
    - HPCG (High Performance Conjugate Gradient)
    - MLPerf (Machine Learning Performance)
    - Custom HPC and AI benchmarks

    Configuration Management:
        Child classes should define _PARAM_FIELDS as a tuple of parameter
        names for automatic serialization and configuration management.

    Run Modes:
        - TIME_BOUND: Run for a specified duration (e.g., MLPerf inference)
        - ITERATION_BOUND: Run for N iterations (e.g., training epochs)
        - SINGLE_RUN: Run once (e.g., HPL with fixed problem size)

    Lifecycle:
        1. is_available() - Check if benchmark can run on current system
        2. validate_configuration() - Validate config and hardware
        3. build() - Optional: Compile/prepare benchmark binaries
        4. setup() - Allocate resources (memory, initialize devices)
        5. warmup() - Warm caches, trigger frequency scaling
        6. execute_benchmark() - Run timed benchmark
        7. teardown() - Clean up resources

    Example:
        >>> class HPLBenchmark(Benchmark):
        ...     _PARAM_FIELDS = ("problem_size", "block_size")
        ...
        ...     def get_run_mode(self) -> RunMode:
        ...         return RunMode.SINGLE_RUN
        ...
        ...     def execute_benchmark(self) -> BenchmarkResult:
        ...         # Solve Ax=b once for given problem size
        ...         gflops = self._run_hpl()
        ...         return BenchmarkResult(
        ...             metrics={"gflops": gflops},
        ...             metadata={"problem_size": self.problem_size},
        ...         )
    """

    # -------------------------------------------------------------------------
    # Identity & Metadata (Override in child classes)
    # -------------------------------------------------------------------------

    def get_name(self) -> str:
        """Get the internal benchmark name (used for programmatic identification).

        Returns:
            Benchmark name, defaults to class name.
        """
        return self.__class__.__name__

    @abstractmethod
    def get_pretty_name(self) -> str:
        """Get the human-readable benchmark name for display.

        Returns:
            Pretty-printed benchmark name (e.g., "High Performance Linpack").
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get a one-line description of what the benchmark measures.

        Returns:
            Brief description (e.g., "Measures floating-point performance
            via LU factorization").
        """
        pass

    # -------------------------------------------------------------------------
    # Configuration & Parameters
    # -------------------------------------------------------------------------

    _PARAM_FIELDS: tuple[str, ...] = ()

    @abstractmethod
    def get_run_mode(self) -> RunMode:
        """Get the execution mode for this benchmark.

        Returns:
            RunMode indicating how the benchmark should be executed.

        Example:
            >>> def get_run_mode(self) -> RunMode:
            ...     return RunMode.SINGLE_RUN  # HPL runs once per problem
        """
        pass

    def supports_custom_duration(self) -> bool:
        """Check if benchmark supports custom duration control.

        Returns:
            True if benchmark can run for arbitrary durations.

        Override this if your benchmark is time-bound and can be stopped
        at any point (e.g., continuous inference workload).
        """
        return self.get_run_mode() == RunMode.TIME_BOUND

    def supports_custom_iterations(self) -> bool:
        """Check if benchmark supports custom iteration counts.

        Returns:
            True if benchmark can run for arbitrary iteration counts.

        Override this if your benchmark is iteration-bound (e.g., training).
        """
        return self.get_run_mode() == RunMode.ITERATION_BOUND

    def get_parameters(self) -> dict[str, Any]:
        """Get the current benchmark parameters for logging and result metadata.

        Default implementation automatically extracts parameter values based on
        the _PARAM_FIELDS tuple defined in child classes.

        Returns:
            Dictionary of parameter names to values.

        Raises:
            AttributeError: If any field in _PARAM_FIELDS is not set as an
                instance attribute.
        """
        # Validate all fields exist before building dict
        missing = [f for f in self._PARAM_FIELDS if not hasattr(self, f)]
        if missing:
            raise AttributeError(
                f"{self.__class__.__name__} missing required parameter fields: "
                f"{', '.join(missing)}. Ensure all fields in _PARAM_FIELDS are "
                "initialized in __init__()."
            )

        return {field: getattr(self, field) for field in self._PARAM_FIELDS}

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set benchmark parameters from a dictionary (e.g., from config file).

        Default implementation sets each field in _PARAM_FIELDS that appears
        in the input dictionary. Override for custom type handling.

        Args:
            params: Dictionary of parameter names to values.
        """
        for field in self._PARAM_FIELDS:
            if field in params:
                # Try to preserve type if possible
                current_val = getattr(self, field, None)
                if current_val is not None and not isinstance(current_val, str):
                    param_type = type(current_val)
                    setattr(self, field, param_type(params[field]))
                else:
                    setattr(self, field, params[field])

    # -------------------------------------------------------------------------
    # Hardware & Availability
    # -------------------------------------------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this benchmark can run on the current system.

        Returns:
            True if benchmark can run, False otherwise.
        """
        pass

    @abstractmethod
    def validate_configuration(self) -> None:
        """Validate benchmark configuration before running.

        Raises:
            ValueError: If configuration is invalid.
            RuntimeError: If hardware requirements aren't met.
        """
        pass

    # -------------------------------------------------------------------------
    # Lifecycle Hooks
    # -------------------------------------------------------------------------

    @abstractmethod
    def build(self) -> None:
        """Build or compile benchmark if needed.

        Override this for benchmarks that require compilation or
        preparation of binary artifacts before execution.

        Examples:
            - Compiling HPL with optimized BLAS
            - Building Docker containers
            - Downloading model weights for MLPerf

        Default implementation is a no-op.
        """
        ...

    @abstractmethod
    def setup(self) -> None:
        """Prepare runtime resources before benchmark execution.

        Allocate memory, initialize devices, load data, etc.
        Called after build() and before warmup().

        Examples:
            - Allocate matrices for linear algebra benchmarks
            - Initialize GPU devices and allocate memory
            - Load model weights into memory
            - Open database connections

        Should be fast and focused on runtime resource allocation.
        """
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources after benchmark execution.

        Free memory, reset devices, close connections, etc.
        Called after execute_benchmark() in a finally block.

        Examples:
            - Free allocated memory
            - Reset GPU state
            - Close file handles
            - Save logs or artifacts

        Must be robust and not raise exceptions.
        """
        ...

    def warmup(self, warmup_iterations: int = 3) -> None:
        """Run warmup phase before measurement.

        Default implementation runs the benchmark workload for a few iterations
        to warm caches, trigger CPU/GPU frequency scaling, and stabilize
        performance.

        Args:
            warmup_iterations: Number of warmup iterations to run.

        Override this to customize warmup behavior. For time-bound benchmarks,
        you might want to run for a fixed duration instead.

        Example override:
            >>> def warmup(self, warmup_iterations: int = 3) -> None:
            ...     # Run actual benchmark workload for warmup
            ...     for _ in range(warmup_iterations):
            ...         self._run_single_iteration()
        """
        # Default: just sleep briefly to allow system to stabilize
        for _ in range(warmup_iterations):
            time.sleep(0.1)

    # -------------------------------------------------------------------------
    # Core Benchmark Execution
    # -------------------------------------------------------------------------

    def run(
        self,
        duration: int | None = None,
        iterations: int | None = None,
        warmup_iterations: int = 3,
    ) -> BenchmarkResult | dict[str, Any]:
        """Run the complete benchmark lifecycle and return results.

        Handles the full lifecycle: validate -> setup -> warmup -> execute -> teardown.

        Args:
            duration: Duration in seconds (for TIME_BOUND mode).
            iterations: Number of iterations (for ITERATION_BOUND mode).
            warmup_iterations: Number of warmup iterations.

        Returns:
            BenchmarkResult or dict with benchmark results.

        Raises:
            ValueError: If required parameters for run mode are missing.
            RuntimeError: If benchmark fails validation or execution.

        Example:
            >>> # SINGLE_RUN benchmark (like HPL)
            >>> result = benchmark.run()
            >>>
            >>> # TIME_BOUND benchmark (like continuous inference)
            >>> result = benchmark.run(duration=60)
            >>>
            >>> # ITERATION_BOUND benchmark (like training)
            >>> result = benchmark.run(iterations=100)
        """
        run_mode = self.get_run_mode()

        # Validate run parameters based on mode
        if run_mode == RunMode.TIME_BOUND and duration is None:
            raise ValueError(f"{self.get_name()} requires 'duration' parameter")
        if run_mode == RunMode.ITERATION_BOUND and iterations is None:
            raise ValueError(f"{self.get_name()} requires 'iterations' parameter")

        # Store run parameters for logging
        self._run_duration = duration
        self._run_iterations = iterations

        # Execute lifecycle
        timing: dict[str, float] = {}

        start_time = time.time()
        self.validate_configuration()
        timing["validation"] = time.time() - start_time

        start_time = time.time()
        self.setup()
        timing["setup"] = time.time() - start_time

        try:
            self.log_warmup_start()
            start_time = time.time()
            self.warmup(warmup_iterations)
            timing["warmup"] = time.time() - start_time

            self.log_benchmark_start()
            start_time = time.time()
            results = self.execute_benchmark(duration, iterations)
            timing["execution"] = time.time() - start_time
            self.log_benchmark_complete()

            # Add timing info if result is BenchmarkResult
            if isinstance(results, BenchmarkResult):
                results.timing.update(timing)
                return results

            return results

        finally:
            start_time = time.time()
            self.teardown()
            timing["teardown"] = time.time() - start_time

    @abstractmethod
    def execute_benchmark(
        self, duration: int | None = None, iterations: int | None = None
    ) -> BenchmarkResult | dict[str, Any]:
        """Execute the core benchmark workload and return results.

        This is the main method to override in benchmark implementations.
        Handle the actual benchmark execution based on your run mode.

        Args:
            duration: Duration in seconds (for TIME_BOUND mode, may be None).
            iterations: Number of iterations (for ITERATION_BOUND mode, may be None).

        Returns:
            BenchmarkResult with metrics, metadata, and validation info,
            or a dict for simpler benchmarks.

        Example for SINGLE_RUN (HPL):
            >>> def execute_benchmark(self, duration, iterations):
            ...     start = time.time()
            ...     gflops = self._solve_linear_system()
            ...     elapsed = time.time() - start
            ...     return BenchmarkResult(
            ...         metrics={"gflops": gflops},
            ...         timing={"solve": elapsed},
            ...         metadata={"problem_size": self.problem_size},
            ...     )

        Example for TIME_BOUND (inference):
            >>> def execute_benchmark(self, duration, iterations):
            ...     start = time.time()
            ...     inferences = 0
            ...     while (time.time() - start) < duration:
            ...         self._run_inference()
            ...         inferences += 1
            ...     elapsed = time.time() - start
            ...     return BenchmarkResult(
            ...         metrics={"throughput": inferences / elapsed},
            ...     )

        Example for ITERATION_BOUND (training):
            >>> def execute_benchmark(self, duration, iterations):
            ...     losses = []
            ...     for i in range(iterations):
            ...         loss = self._train_step()
            ...         losses.append(loss)
            ...     return BenchmarkResult(
            ...         metrics={"final_loss": losses[-1]},
            ...         validation={"losses": losses},
            ...     )
        """
        pass

    # -------------------------------------------------------------------------
    # Result Type & Formatting
    # -------------------------------------------------------------------------

    def get_result_type(self) -> type | None:
        """Get the Pydantic model type for this benchmark's results.

        Returns:
            Pydantic model class, or None if using raw dict.
        """
        return None

    # -------------------------------------------------------------------------
    # Timestamps & Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def generate_timestamp() -> str:
        """Generate an ISO format timestamp (UTC).

        Returns:
            ISO format timestamp string.
        """
        return datetime.now(UTC).isoformat()

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    @property
    def logger(self) -> logging.Logger:
        """Get the logger for this benchmark.

        Returns:
            Logger instance for this benchmark class.
        """
        from warpt.utils.logger import Logger

        return Logger.get(f"benchmark.{self.__class__.__name__}")

    def log_warmup_start(self) -> None:
        """Log that warmup phase is starting."""
        self.logger.info("Starting warmup phase...")

    def log_benchmark_start(self) -> None:
        """Log that benchmark phase is starting."""
        run_mode = self.get_run_mode()
        if run_mode == RunMode.SINGLE_RUN:
            self.logger.info("Running benchmark (single run)...")
        elif run_mode == RunMode.TIME_BOUND and hasattr(self, "_run_duration"):
            self.logger.info(f"Running benchmark for {self._run_duration}s...")
        elif run_mode == RunMode.ITERATION_BOUND and hasattr(self, "_run_iterations"):
            iters = self._run_iterations
            self.logger.info(f"Running benchmark for {iters} iterations...")
        else:
            self.logger.info("Running benchmark...")

    def log_benchmark_complete(self) -> None:
        """Log that benchmark is complete."""
        self.logger.info("Benchmark complete")
