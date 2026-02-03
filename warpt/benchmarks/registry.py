"""Registry for automatic discovery and registration of benchmarks.

Usage:
    from warpt.benchmarks.registry import BenchmarkRegistry

    # Discover all benchmarks from default location
    registry = BenchmarkRegistry()

    # Get all registered benchmarks
    all_benchmarks = registry.get_all_benchmarks()

    # Get a specific benchmark by name
    benchmark = registry.get_benchmark("HPLBenchmark")
"""

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from warpt.benchmarks.base import Benchmark

if TYPE_CHECKING:
    pass


class BenchmarkRegistryError(Exception):
    """Base exception for registry errors."""

    pass


class BenchmarkNameCollisionError(BenchmarkRegistryError):
    """Raised when two benchmarks have the same name."""

    def __init__(self, name: str, benchmark1: type, benchmark2: type) -> None:
        self.name = name
        self.benchmark1 = benchmark1
        self.benchmark2 = benchmark2
        super().__init__(
            f"Benchmark name collision: '{name}' is defined in both "
            f"{benchmark1.__module__} and {benchmark2.__module__}"
        )


class BenchmarkNotFoundError(BenchmarkRegistryError):
    """Raised when a requested benchmark is not found."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Benchmark not found: '{name}'")


class BenchmarkRegistry:
    """Registry for automatic discovery and management of benchmarks.

    Discovers all Benchmark subclasses from specified directories and
    provides methods to query and instantiate them.

    Raises BenchmarkNameCollisionError if two benchmarks have the same name.

    Example:
        >>> registry = BenchmarkRegistry()
        >>> available = registry.get_available_benchmarks()
        >>> for benchmark_cls in available:
        ...     benchmark = benchmark_cls()
        ...     if benchmark.is_available():
        ...         result = benchmark.run(duration=60)
    """

    # Default paths to search for benchmarks
    DEFAULT_PATHS: ClassVar[list[Path]] = [
        Path(__file__).parent,  # warpt/benchmarks/
    ]

    def __init__(
        self,
        search_paths: list[str | Path] | None = None,
        include_defaults: bool = True,
        lazy: bool = True,
    ) -> None:
        """Initialize the benchmark registry.

        Args:
            search_paths: Additional directories to search for benchmark modules.
            include_defaults: If True, include the default warpt/benchmarks/ directory.
            lazy: If True, defer benchmark discovery until first access.

        Raises:
            BenchmarkNameCollisionError: If two benchmarks have the same name.
        """
        self._benchmarks: dict[str, type[Benchmark]] = {}
        self._paths: list[Path] = []
        self._discovered: bool = False

        if include_defaults:
            self._paths.extend(self.DEFAULT_PATHS)

        if search_paths:
            self._paths.extend(Path(p) for p in search_paths)

        if not lazy:
            self._discover_benchmarks()

    def _ensure_discovered(self) -> None:
        """Ensure benchmarks have been discovered."""
        if not self._discovered:
            self._discover_benchmarks()
            self._discovered = True

    def _discover_benchmarks(self) -> None:
        """Discover all Benchmark subclasses in registered paths."""
        for path in self._paths:
            if not path.exists():
                continue

            if path.is_file() and path.suffix == ".py":
                self._load_benchmarks_from_file(path)
            elif path.is_dir():
                for py_file in path.glob("*.py"):
                    if py_file.name.startswith("_"):
                        continue
                    self._load_benchmarks_from_file(py_file)

    def _load_benchmarks_from_file(self, filepath: Path) -> None:
        """Load all Benchmark subclasses from a Python file."""
        module_name = f"warpt.benchmarks.{filepath.stem}"

        # Check if already imported
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            try:
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                if spec is None or spec.loader is None:
                    return
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            except Exception:
                # Skip files that can't be imported
                return

        # Find all Benchmark subclasses
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, Benchmark)
                and obj is not Benchmark
                and not inspect.isabstract(obj)
                and obj.__module__ == module_name
            ):
                self._register_benchmark(obj)

    def _register_benchmark(self, benchmark_cls: type[Benchmark]) -> None:
        """Register a benchmark class, checking for name collisions."""
        name = benchmark_cls.__name__

        if name in self._benchmarks:
            existing = self._benchmarks[name]
            if existing is not benchmark_cls:
                raise BenchmarkNameCollisionError(name, existing, benchmark_cls)
        else:
            self._benchmarks[name] = benchmark_cls

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def get_all_benchmarks(self) -> list[type[Benchmark]]:
        """Get all registered benchmark classes.

        Returns:
            List of Benchmark subclasses.
        """
        self._ensure_discovered()
        return list(self._benchmarks.values())

    def get_benchmark(self, name: str) -> type[Benchmark]:
        """Get a specific benchmark class by name.

        Args:
            name: The benchmark class name (e.g., "HPLBenchmark").

        Returns:
            The benchmark class.

        Raises:
            BenchmarkNotFoundError: If benchmark is not found.
        """
        self._ensure_discovered()
        if name not in self._benchmarks:
            raise BenchmarkNotFoundError(name)
        return self._benchmarks[name]

    def get_available_benchmarks(self) -> list[type[Benchmark]]:
        """Get all benchmarks that are available on this system.

        Instantiates each benchmark and checks is_available().

        Returns:
            List of available benchmark classes.
        """
        self._ensure_discovered()
        available = []
        for benchmark_cls in self._benchmarks.values():
            try:
                # Try to instantiate with minimal args
                instance = self._try_instantiate(benchmark_cls)
                if instance and instance.is_available():
                    available.append(benchmark_cls)
            except Exception:
                continue
        return available

    def _try_instantiate(self, benchmark_cls: type[Benchmark]) -> Benchmark | None:
        """Try to instantiate a benchmark with default arguments.

        Args:
            benchmark_cls: The benchmark class to instantiate.

        Returns:
            Instance or None if instantiation fails.
        """
        try:
            # Try with no args first
            return benchmark_cls()
        except TypeError:
            # Some benchmarks require args
            return None

    def list_benchmarks(self) -> list[dict[str, Any]]:
        """Get a summary of all registered benchmarks.

        Returns:
            List of dicts with name, pretty_name, description, available.
        """
        self._ensure_discovered()
        summaries = []
        for name, benchmark_cls in sorted(self._benchmarks.items()):
            try:
                instance = self._try_instantiate(benchmark_cls)
                if instance:
                    summaries.append(
                        {
                            "name": name,
                            "pretty_name": instance.get_pretty_name(),
                            "description": instance.get_description(),
                            "available": instance.is_available(),
                        }
                    )
            except Exception:
                summaries.append(
                    {
                        "name": name,
                        "pretty_name": name,
                        "description": "(error getting info)",
                        "available": False,
                    }
                )
        return summaries

    def __len__(self) -> int:
        """Return number of registered benchmarks."""
        return len(self._benchmarks)

    def __contains__(self, name: str) -> bool:
        """Check if a benchmark is registered."""
        self._ensure_discovered()
        return name in self._benchmarks
