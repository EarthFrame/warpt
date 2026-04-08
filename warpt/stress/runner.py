"""Test runner for executing stress tests and collecting results.

Usage:
    from warpt.stress.runner import TestRunner
    from warpt.stress.registry import TestRegistry

    registry = TestRegistry()
    runner = TestRunner()

    # Add tests to run
    runner.add_test(registry.get_test("GPUMatMulTest"))
    runner.add_tests(registry.get_tests_by_category(TestCategory.CPU))

    # Run all tests
    results = runner.run(duration=30)

    # Emit results
    results.emit_json("results.json")
    results.emit_stdout()
"""

from typing import Any

from warpt.stress.base import StressTest, TestCategory
from warpt.stress.results import TestResults


class TestRunner:
    """Runs stress tests and collects results.

    Manages test execution, handles errors gracefully, and collects
    results into a TestResults object for emission.

    Example:
        >>> runner = TestRunner()
        >>> runner.add_test(GPUMatMulTest)
        >>> results = runner.run(duration=30)
        >>> results.emit_stdout()
    """

    def __init__(self) -> None:
        """Initialize the test runner."""
        self._tests: list[tuple[type[StressTest], str]] = []
        self._test_configs: dict[str, dict[str, Any]] = {}

    def add_test(
        self,
        test_cls: type[StressTest],
        config: dict[str, Any] | None = None,
        run_key: str | None = None,
    ) -> None:
        """Add a test class to run.

        Args:
            test_cls: The StressTest subclass to run.
            config: Optional configuration overrides for this test.
            run_key: Unique key for this run.  Defaults to the class name.
                     Use a custom key to queue the same test class multiple
                     times with different configs (e.g. different device_ids).
        """
        key = run_key or test_cls.__name__
        if key not in {k for _, k in self._tests}:
            self._tests.append((test_cls, key))
        if config:
            self._test_configs[key] = config

    def add_tests(
        self,
        test_classes: list[type[StressTest]],
    ) -> None:
        """Add multiple test classes.

        Args:
            test_classes: List of StressTest subclasses.
        """
        for test_cls in test_classes:
            self.add_test(test_cls)

    def clear(self) -> None:
        """Clear all queued tests."""
        self._tests.clear()
        self._test_configs.clear()

    @property
    def test_count(self) -> int:
        """Return number of queued tests."""
        return len(self._tests)

    def run(
        self,
        duration: int = 30,
        skip_unavailable: bool = True,
        stop_on_error: bool = False,
    ) -> TestResults:
        """Run all queued tests and collect results.

        Args:
            duration: Test duration in seconds for each test.
            skip_unavailable: If True, skip tests that aren't available.
            stop_on_error: If True, stop on first error.

        Returns:
            TestResults object with all results.
        """
        results = TestResults()

        for test_cls, run_key in self._tests:
            try:
                # Instantiate test with config if provided
                config = self._test_configs.get(run_key, {})
                test = self._instantiate_test(test_cls, config)

                if test is None:
                    results.add_error(run_key, "Failed to instantiate test")
                    if stop_on_error:
                        break
                    continue

                # Check availability
                if skip_unavailable and not test.is_available():
                    results.add_error(run_key, "Test not available on this system")
                    continue

                # Run the test
                result = test.run(duration=duration)

                # Convert result to dict if needed
                if hasattr(result, "model_dump"):
                    # Pydantic model
                    result_dict = result.model_dump()
                elif hasattr(result, "dict"):
                    # Pydantic v1
                    result_dict = result.dict()
                elif isinstance(result, dict):
                    result_dict = result
                else:
                    result_dict = {"result": str(result)}

                results.add_result(run_key, result_dict)

            except Exception as e:
                results.add_error(run_key, str(e))
                if stop_on_error:
                    break

        return results

    def _instantiate_test(
        self,
        test_cls: type[StressTest],
        config: dict[str, Any],
    ) -> StressTest | None:
        """Instantiate a test with configuration.

        Args:
            test_cls: The test class.
            config: Configuration dictionary.

        Returns:
            Test instance or None if failed.
        """
        try:
            # Try with no args first
            test = test_cls()
        except TypeError:
            try:
                # Try with device_id for GPU tests
                # Use **kwargs to avoid type checker complaints
                test = test_cls(**{"device_id": 0})
            except (TypeError, AttributeError):
                return None

        # Apply config overrides
        if config:
            test.set_parameters(config)

        return test

    def run_by_category(
        self,
        category: TestCategory,
        duration: int = 30,
    ) -> TestResults:
        """Run all tests of a specific category.

        Filters queued tests by category and runs them.

        Args:
            category: TestCategory to filter by.
            duration: Test duration in seconds.

        Returns:
            TestResults for matching tests.
        """
        results = TestResults()

        for test_cls, run_key in self._tests:
            try:
                config = self._test_configs.get(run_key, {})
                test = self._instantiate_test(test_cls, config)
                if test is None:
                    continue

                if test.get_category() != category:
                    continue

                if not test.is_available():
                    results.add_error(
                        run_key, "Test not available on this system"
                    )
                    continue

                result = test.run(duration=duration)

                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                elif isinstance(result, dict):
                    result_dict = result
                else:
                    result_dict = {"result": str(result)}

                results.add_result(run_key, result_dict)

            except Exception as e:
                results.add_error(run_key, str(e))

        return results


def run_all_available_tests(
    duration: int = 30,
    categories: list[TestCategory] | None = None,
) -> TestResults:
    """Discover and run all available tests.

    Args:
        duration: Test duration in seconds.
        categories: Optional list of categories to filter by.

    Returns:
        TestResults with all results.
    """
    from warpt.stress.registry import TestRegistry

    registry = TestRegistry()
    runner = TestRunner()

    if categories:
        for category in categories:
            runner.add_tests(registry.get_tests_by_category(category))
    else:
        runner.add_tests(registry.get_available_tests())

    return runner.run(duration=duration)
