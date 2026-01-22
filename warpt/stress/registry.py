"""Test registry for automatic discovery and registration of stress tests.

Usage:
    from warpt.stress.registry import TestRegistry

    # Discover all tests from default location
    registry = TestRegistry()

    # Get all registered tests
    all_tests = registry.get_all_tests()

    # Get tests for a specific target
    gpu_tests = registry.get_tests_by_category(TestCategory.ACCELERATOR)

    # Get a specific test by name
    test = registry.get_test("GPUMatMulTest")
"""

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from warpt.stress.base import StressTest, TestCategory

if TYPE_CHECKING:
    pass


class TestRegistryError(Exception):
    """Base exception for registry errors."""

    pass


class TestNameCollisionError(TestRegistryError):
    """Raised when two tests have the same name."""

    def __init__(self, name: str, test1: type, test2: type) -> None:
        self.name = name
        self.test1 = test1
        self.test2 = test2
        super().__init__(
            f"Test name collision: '{name}' is defined in both "
            f"{test1.__module__} and {test2.__module__}"
        )


class TestNotFoundError(TestRegistryError):
    """Raised when a requested test is not found."""

    __test__ = False  # Tell pytest not to collect this as a test class

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Test not found: '{name}'")


class TestRegistry:
    """Registry for automatic discovery and management of stress tests.

    Discovers all StressTest subclasses from specified directories and
    provides methods to query and instantiate them.

    Raises TestNameCollisionError if two tests have the same name.

    Example:
        >>> registry = TestRegistry()
        >>> gpu_tests = registry.get_tests_by_category(TestCategory.ACCELERATOR)
        >>> for test_cls in gpu_tests:
        ...     test = test_cls()
        ...     if test.is_available():
        ...         result = test.run(duration=30)
    """

    __test__ = False  # Tell pytest not to collect this as a test class

    # Default paths to search for tests
    DEFAULT_PATHS: ClassVar[list[Path]] = [
        Path(__file__).parent,  # warpt/stress/
    ]

    def __init__(
        self,
        search_paths: list[str | Path] | None = None,
        include_defaults: bool = True,
        lazy: bool = True,
    ) -> None:
        """Initialize the test registry.

        Args:
            search_paths: Additional directories to search for test modules.
            include_defaults: If True, include the default warpt/stress/ directory.
            lazy: If True, defer test discovery until first access.

        Raises:
            TestNameCollisionError: If two tests have the same name.
        """
        self._tests: dict[str, type[StressTest]] = {}
        self._paths: list[Path] = []
        self._discovered: bool = False

        if include_defaults:
            self._paths.extend(self.DEFAULT_PATHS)

        if search_paths:
            self._paths.extend(Path(p) for p in search_paths)

        if not lazy:
            self._discover_tests()

    def _ensure_discovered(self) -> None:
        """Ensure tests have been discovered."""
        if not self._discovered:
            self._discover_tests()
            self._discovered = True

    def _discover_tests(self) -> None:
        """Discover all StressTest subclasses in registered paths."""
        for path in self._paths:
            if not path.exists():
                continue

            if path.is_file() and path.suffix == ".py":
                self._load_tests_from_file(path)
            elif path.is_dir():
                for py_file in path.glob("*.py"):
                    if py_file.name.startswith("_"):
                        continue
                    self._load_tests_from_file(py_file)

    def _load_tests_from_file(self, filepath: Path) -> None:
        """Load all StressTest subclasses from a Python file."""
        module_name = f"warpt.stress.{filepath.stem}"

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

        # Find all StressTest subclasses
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, StressTest)
                and obj is not StressTest
                and not inspect.isabstract(obj)
                and obj.__module__ == module_name
            ):
                self._register_test(obj)

    def _register_test(self, test_cls: type[StressTest]) -> None:
        """Register a test class, checking for name collisions."""
        name = test_cls.__name__

        if name in self._tests:
            existing = self._tests[name]
            if existing is not test_cls:
                raise TestNameCollisionError(name, existing, test_cls)
        else:
            self._tests[name] = test_cls

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def get_all_tests(self) -> list[type[StressTest]]:
        """Get all registered test classes.

        Returns:
            List of StressTest subclasses.
        """
        self._ensure_discovered()
        return list(self._tests.values())

    def get_test(self, name: str) -> type[StressTest]:
        """Get a specific test class by name.

        Args:
            name: The test class name (e.g., "GPUMatMulTest").

        Returns:
            The test class.

        Raises:
            TestNotFoundError: If test is not found.
        """
        self._ensure_discovered()
        if name not in self._tests:
            raise TestNotFoundError(name)
        return self._tests[name]

    def get_tests_by_category(self, category: TestCategory) -> list[type[StressTest]]:
        """Get all tests for a specific category.

        Args:
            category: The TestCategory to filter by.

        Returns:
            List of test classes matching the category.
        """
        self._ensure_discovered()
        result = []
        for test_cls in self._tests.values():
            # Instantiate briefly to check category
            try:
                instance = test_cls.__new__(test_cls)
                # Call get_category without full init
                if hasattr(test_cls, "get_category"):
                    # Check if it's a classmethod or instance method
                    cat = test_cls.get_category(instance)
                    if cat == category:
                        result.append(test_cls)
            except Exception:
                continue
        return result

    def get_available_tests(self) -> list[type[StressTest]]:
        """Get all tests that are available on this system.

        Instantiates each test and checks is_available().

        Returns:
            List of available test classes.
        """
        self._ensure_discovered()
        available = []
        for test_cls in self._tests.values():
            try:
                # Try to instantiate with minimal args
                instance = self._try_instantiate(test_cls)
                if instance and instance.is_available():
                    available.append(test_cls)
            except Exception:
                continue
        return available

    def _try_instantiate(self, test_cls: type[StressTest]) -> StressTest | None:
        """Try to instantiate a test with default arguments.

        Args:
            test_cls: The test class to instantiate.

        Returns:
            Instance or None if instantiation fails.
        """
        try:
            # Try with no args first
            return test_cls()
        except TypeError:
            # Some tests require args (e.g., device_id)
            try:
                return test_cls(device_id=0)  # type: ignore[call-arg]
            except (TypeError, AttributeError):
                return None

    def list_tests(self) -> list[dict[str, str | bool]]:
        """Get a summary of all registered tests.

        Returns:
            List of dicts with name, pretty_name, description, category.
        """
        self._ensure_discovered()
        summaries = []
        for name, test_cls in sorted(self._tests.items()):
            try:
                instance = self._try_instantiate(test_cls)
                if instance:
                    summaries.append(
                        {
                            "name": name,
                            "pretty_name": instance.get_pretty_name(),
                            "description": instance.get_description(),
                            "category": instance.get_category().value,
                            "available": instance.is_available(),
                        }
                    )
            except Exception:
                summaries.append(
                    {
                        "name": name,
                        "pretty_name": name,
                        "description": "(error getting info)",
                        "category": "unknown",
                        "available": False,
                    }
                )
        return summaries

    def __len__(self) -> int:
        """Return number of registered tests."""
        return len(self._tests)

    def __contains__(self, name: str) -> bool:
        """Check if a test is registered."""
        return name in self._tests
