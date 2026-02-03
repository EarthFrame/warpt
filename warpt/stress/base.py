"""Base class for stress tests."""

import logging
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any

if sys.version_info >= (3, 11):  # noqa: UP036
    from datetime import UTC
else:
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017

from warpt.utils.env import get_env


class TestCategory(Enum):
    """Categories of stress tests."""

    __test__ = False  # Tell pytest not to collect this as a test class

    CPU = "cpu"
    ACCELERATOR = "accelerator"
    RAM = "ram"
    STORAGE = "storage"
    NETWORK = "network"


class StressTest(ABC):
    """Abstract base class for all stress tests.

    Provides a standard interface for stress tests with lifecycle hooks,
    hardware validation, and result handling. Child classes must implement
    the abstract methods.

    Configuration Management:
        Child classes should define _PARAM_FIELDS as a tuple of parameter
        names. The base class automatically provides get_parameters() and
        set_parameters() implementations that work with this tuple.

    Attributes (expected in child classes):
        _PARAM_FIELDS: tuple[str, ...] - Names of configuration fields
            (e.g., ("matrix_size", "burnin_seconds", "device_id"))
        burnin_seconds: int - Warmup duration before measurement.
        Additional test-specific parameters (matrix_size, device_id, etc.)

    Lifecycle:
        1. is_available() - Check if test can run on current system
        2. validate_configuration() - Validate config and hardware
        3. setup() - Prepare resources before test
        4. warmup() - Warmup/burnin phase (uses burnin_seconds)
        5. run() - Execute the timed test
        6. teardown() - Clean up resources after test

    Example:
        >>> class GPUMatMulTest(StressTest):
        ...     _PARAM_FIELDS = ("device_id", "matrix_size", "burnin_seconds")
        ...
        ...     def __init__(self, device_id: int = 0, matrix_size: int = 4096,
        ...                  burnin_seconds: int = 5):
        ...         self.device_id = device_id
        ...         self.matrix_size = matrix_size
        ...         self.burnin_seconds = burnin_seconds
        ...         # get_parameters() and set_parameters() are automatic!
        ...
        ...     def run(self, duration: int) -> dict:
        ...         self.validate_configuration()
        ...         self.setup()
        ...         try:
        ...             self.warmup(duration_seconds=self.burnin_seconds)
        ...             # ... timed test logic ...
        ...             return {"tflops": 12.5, "iterations": 100}
        ...         finally:
        ...             self.teardown()
    """

    # -------------------------------------------------------------------------
    # Identity & Metadata (Override in child classes)
    # -------------------------------------------------------------------------

    def get_name(self) -> str:
        """Get the internal test name (used for programmatic identification).

        Returns:
            Test name, defaults to class name.
        """
        return self.__class__.__name__

    @abstractmethod
    def get_pretty_name(self) -> str:
        """Get the human-readable test name for display.

        Returns:
            Pretty-printed test name (e.g., "GPU Matrix Multiplication").
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get a one-line description of what the test measures.

        Returns:
            Brief description (e.g., "Measures GPU compute throughput via FP32 matmul").
        """
        pass

    @abstractmethod
    def get_category(self) -> TestCategory:
        """Get the test category (cpu, gpu, ram, etc.).

        Returns:
            TestCategory enum value.
        """
        pass

    # -------------------------------------------------------------------------
    # Configuration & Parameters
    # -------------------------------------------------------------------------

    _duration_seconds: int | None = None
    _PARAM_FIELDS: tuple[str, ...] = ()

    @property
    def duration_seconds(self) -> int:
        """Test duration in seconds. Defaults to get_default_duration() if not set."""
        if self._duration_seconds is None:
            return self.get_default_duration()
        return self._duration_seconds

    @duration_seconds.setter
    def duration_seconds(self, value: int) -> None:
        """Set test duration in seconds."""
        self._duration_seconds = value

    def get_parameters(self) -> dict[str, Any]:
        """Get the current test parameters for logging and result metadata.

        Default implementation automatically extracts parameter values based on
        the _PARAM_FIELDS tuple defined in child classes. Each field name in
        _PARAM_FIELDS must correspond to an instance attribute.

        Returns:
            Dictionary of parameter names to values.

        Raises:
            AttributeError: If any field in _PARAM_FIELDS is not set as an
                instance attribute.

        Warns:
            UserWarning: If instance attributes exist that are not listed in
                _PARAM_FIELDS (excluding private/protected attributes starting
                with underscore).

        Override this method for custom behavior (e.g., enum conversion,
        computed values, or non-standard types).

        Example in child class:
            >>> class MyTest(StressTest):
            ...     _PARAM_FIELDS = ("matrix_size", "burnin_seconds", "device_id")
            ...
            ...     def __init__(self, matrix_size: int = 4096, burnin_seconds: int = 5,
            ...                  device_id: int = 0):
            ...         self.matrix_size = matrix_size
            ...         self.burnin_seconds = burnin_seconds
            ...         self.device_id = device_id
            ...
            ...     # get_parameters() is inherited and automatic!
            ...
            ... # Calling get_parameters() returns:
            ... # {"matrix_size": 4096, "burnin_seconds": 5, "device_id": 0}

        For custom type conversion (e.g., enums):
            >>> class CustomTest(StressTest):
            ...     _PARAM_FIELDS = ("matrix_size", "burnin_seconds")
            ...     # ... fields ...
            ...     precision: Precision = Precision.FP32
            ...
            ...     def get_parameters(self) -> dict[str, Any]:
            ...         params = super().get_parameters()
            ...         params["precision"] = self.precision.value
            ...         return params
        """
        # Validate all fields exist before building dict
        missing = [f for f in self._PARAM_FIELDS if not hasattr(self, f)]
        if missing:
            raise AttributeError(
                f"{self.__class__.__name__} missing required parameter fields: "
                f"{', '.join(missing)}. Ensure all fields in _PARAM_FIELDS are "
                "initialized in __init__()."
            )

        # Warn about public instance attributes not in _PARAM_FIELDS
        public_attrs = {
            name
            for name in dir(self)
            if not name.startswith("_") and not callable(getattr(self, name))
        }
        param_fields_set = set(self._PARAM_FIELDS)
        # Filter out class-level attributes (from parent classes)
        instance_only = {name for name in public_attrs if name in self.__dict__}
        unlisted = instance_only - param_fields_set
        if unlisted:
            import warnings

            warnings.warn(
                f"{self.__class__.__name__} has instance attributes not listed "
                f"in _PARAM_FIELDS: {', '.join(sorted(unlisted))}. "
                "Add them to _PARAM_FIELDS if they should be included in parameters.",
                UserWarning,
                stacklevel=2,
            )

        return {field: getattr(self, field) for field in self._PARAM_FIELDS}

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set test parameters from a dictionary (e.g., from config file).

        Default implementation sets each field in _PARAM_FIELDS that appears
        in the input dictionary. Values are converted to int by default.

        Unknown keys in params are silently ignored.

        Args:
            params: Dictionary of parameter names to values. Keys should match
                those returned by get_parameters().

        Raises:
            ValueError: If a parameter value is invalid.

        Example in child class:
            >>> class MyTest(StressTest):
            ...     _PARAM_FIELDS = ("matrix_size", "burnin_seconds", "device_id")
            ...     # ... fields ...
            ...
            ...     # set_parameters() is inherited and automatic!
            ...
            ... # Calling set_parameters({"matrix_size": 2048, "burnin_seconds": 10})
            ... # sets self.matrix_size and self.burnin_seconds

        For custom type conversion (e.g., enums):
            >>> class CustomTest(StressTest):
            ...     _PARAM_FIELDS = ("matrix_size", "burnin_seconds")
            ...     precision: Precision = Precision.FP32
            ...
            ...     def set_parameters(self, params: dict[str, Any]) -> None:
            ...         if "precision" in params:
            ...             self.precision = Precision(params["precision"])
            ...         super().set_parameters(params)
        """
        for field in self._PARAM_FIELDS:
            if field in params:
                setattr(self, field, int(params[field]))

    def get_default_duration(self) -> int:
        """Get the default test duration in seconds.

        Override in child classes if a different default makes sense.

        Returns:
            Default duration in seconds.
        """
        return get_env(
            "WARPT_STRESS_DEFAULT_DURATION", as_type=int, default=30, log=False
        )

    def get_minimum_duration(self) -> int:
        """Get the minimum sensible test duration in seconds.

        Tests shorter than this may produce unreliable results.
        Override in child classes if a different minimum makes sense.

        Returns:
            Minimum duration in seconds.
        """
        return get_env(
            "WARPT_STRESS_MINIMUM_DURATION", as_type=int, default=5, log=False
        )

    # -------------------------------------------------------------------------
    # Hardware & Availability
    # -------------------------------------------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this test can run on the current system.

        Check for required hardware, libraries, or features. Should not raise
        exceptions - return False if unavailable.

        Returns:
            True if test can run, False otherwise.

        Example:
            >>> def is_available(self) -> bool:
            ...     try:
            ...         import torch
            ...         return torch.cuda.is_available()
            ...     except ImportError:
            ...         return False
        """
        pass

    @abstractmethod
    def validate_configuration(self) -> None:
        """Validate test configuration before running.

        Called before setup(). Should raise ValueError for invalid config,
        RuntimeError if hardware requirements aren't met.

        Raises:
            ValueError: If configuration is invalid (e.g., matrix_size < 1).
            RuntimeError: If hardware requirements aren't met.

        Example:
            >>> def validate_configuration(self) -> None:
            ...     if not self.is_available():
            ...         raise RuntimeError("CUDA not available")
            ...     if self.matrix_size < 64:
            ...         raise ValueError("matrix_size must be >= 64")
            ...     if self.device_id >= torch.cuda.device_count():
            ...         raise ValueError(f"GPU {self.device_id} not found")
        """
        pass

    # -------------------------------------------------------------------------
    # Lifecycle Hooks
    # -------------------------------------------------------------------------

    @abstractmethod
    def setup(self) -> None:
        """Prepare resources before test execution.

        Called after validate_configuration() but before warmup()/run().
        Allocate memory, initialize devices, configure settings.

        Example:
            >>> def setup(self) -> None:
            ...     import torch
            ...     self.device = torch.device(f"cuda:{self.device_id}")
            ...     torch.cuda.set_device(self.device)
            ...     torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            ...     # Pre-allocate tensors if reusing across iterations
            ...     self.matrix_a = torch.empty(
            ...         self.matrix_size, self.matrix_size,
            ...         dtype=torch.float32, device=self.device
            ...     )
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources after test execution.

        Called in a finally block to ensure cleanup even on errors.
        Free memory, reset device state, stop monitoring.

        Example:
            >>> def teardown(self) -> None:
            ...     # Free pre-allocated tensors
            ...     if hasattr(self, 'matrix_a'):
            ...         del self.matrix_a
            ...     # Clear GPU cache
            ...     import torch
            ...     torch.cuda.empty_cache()
            ...     # Reset TF32 to default
            ...     torch.backends.cuda.matmul.allow_tf32 = True
        """
        pass

    def warmup(self, duration_seconds: int = 0, iterations: int = 3) -> None:
        """Run warmup/burnin phase before measurement.

        Default implementation busy-waits for the specified duration or iterations.
        Override to implement test-specific warmup (e.g., run actual workload to
        warm caches, trigger CPU/GPU frequency scaling, etc.).

        Args:
            duration_seconds: Warmup duration in seconds. If > 0, spin for this long.
            iterations: Number of spin iterations if duration_seconds is 0.
                Each iteration is ~10ms.

        Example override for GPU test:
            >>> def warmup(self, duration_seconds: int = 0, iterations: int = 3):
            ...     if duration_seconds > 0:
            ...         start = time.time()
            ...         while (time.time() - start) < duration_seconds:
            ...             self._run_single_iteration()
            ...     else:
            ...         for _ in range(iterations):
            ...             self._run_single_iteration()
        """
        if duration_seconds > 0:
            # Spin for specified duration
            start = time.time()
            while (time.time() - start) < duration_seconds:
                # Busy-wait with small sleep to avoid pegging CPU
                time.sleep(0.01)
        else:
            # Spin for specified iterations (~10ms each)
            for _ in range(iterations):
                time.sleep(0.01)

    # -------------------------------------------------------------------------
    # Core Test Execution
    # -------------------------------------------------------------------------

    def run(self, duration: int, iterations: int = 1) -> Any:
        """Run the stress test and return results.

        This method implements the Template Method pattern, handling the common
        lifecycle (validate, setup, warmup, run, teardown). Child classes
        override execute_test() instead of run().

        Args:
            duration: Test duration in seconds.
            iterations: Number of iterations (passed to execute_test).

        Returns:
            Test results - typically a dict or Pydantic BaseModel.
        """
        self.duration_seconds = duration
        self.validate_configuration()
        self.setup()

        try:
            self.log_warmup_start()
            self.warmup()

            self.log_test_start()
            results = self.execute_test(duration, iterations)
            self.log_test_complete()

            return results
        finally:
            self.teardown()

    @abstractmethod
    def execute_test(self, duration: int, iterations: int) -> Any:
        """Execute the core test logic and return results.

        This is the method child classes override instead of run().
        The run() method handles all lifecycle management (validation, setup,
        warmup, teardown), so your implementation only needs to focus on
        the actual test measurements.

        Args:
            duration: Test duration in seconds.
            iterations: Number of iterations (if applicable to your test).

        Returns:
            Test results - typically a dict or Pydantic BaseModel.

        Example:
            >>> def execute_test(self, duration: int, iterations: int) -> dict:
            ...     import torch
            ...
            ...     start_time = time.time()
            ...     iter_count = 0
            ...
            ...     while (time.time() - start_time) < duration:
            ...         a = torch.randn(self.matrix_size, self.matrix_size,
            ...                        device=self._device)
            ...         b = torch.randn(self.matrix_size, self.matrix_size,
            ...                        device=self._device)
            ...         _ = torch.matmul(a, b)
            ...         torch.cuda.synchronize()
            ...         iter_count += 1
            ...
            ...     elapsed = time.time() - start_time
            ...     tflops = self._calculate_tflops(iter_count, elapsed)
            ...
            ...     return {"tflops": tflops, "duration": elapsed, ...}
        """
        pass

    # -------------------------------------------------------------------------
    # Result Type & Formatting
    # -------------------------------------------------------------------------

    def get_result_type(self) -> type | None:
        """Get the Pydantic model type for this test's results.

        Override to return the specific result model class. Useful for
        type checking and documentation.

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

    @staticmethod
    def generate_timestamp_start() -> str:
        """Generate a start timestamp for the test.

        Returns:
            ISO format timestamp string (UTC).
        """
        return datetime.now(UTC).isoformat()

    @staticmethod
    def generate_timestamp_end() -> str:
        """Generate an end timestamp for the test.

        Returns:
            ISO format timestamp string (UTC).
        """
        return datetime.now(UTC).isoformat()

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    @property
    def logger(self) -> logging.Logger:
        """Get the logger for this test.

        Returns a child logger under the warpt.stress namespace.
        Raises LoggerNotConfiguredError if Logger.configure() hasn't been called.

        Returns:
            Logger instance for this test class.
        """
        from warpt.utils.logger import Logger

        return Logger.get(f"stress.{self.__class__.__name__}")

    def log_warmup_start(self) -> None:
        """Log that warmup phase is starting.

        Logs the burnin_seconds value if available, otherwise nothing.
        """
        if hasattr(self, "burnin_seconds"):
            self.logger.info(f"Warming up for {self.burnin_seconds}s...")

    def log_test_start(self) -> None:
        """Log that test phase is starting.

        Args:
            duration: Test duration in seconds.
        """
        self.logger.info(f"Running test for {self.duration_seconds}s...")

    def log_test_complete(self) -> None:
        """Log that test is complete."""
        self.logger.info("Test complete")
