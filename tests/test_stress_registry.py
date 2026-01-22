"""Tests for the stress test registry."""

from unittest.mock import patch

import pytest

from warpt.stress.base import StressTest, TestCategory
from warpt.stress.registry import TestNotFoundError, TestRegistry


class MockTest(StressTest):
    """A mock stress test for registry testing."""

    def run(self, **kwargs):
        """Mock run."""
        _ = kwargs
        return {}

    def is_available(self):
        """Mock availability."""
        return True

    def get_category(self):
        """Mock category."""
        return TestCategory.ACCELERATOR

    def get_description(self):
        """Mock description."""
        return "Mock"

    def get_pretty_name(self):
        """Mock pretty name."""
        return "Mock Test"

    def validate_configuration(self):
        """Mock validation."""
        pass

    def setup(self):
        """Mock setup."""
        pass

    def teardown(self):
        """Mock teardown."""
        pass

    def execute_test(self, duration, iterations):
        """Mock execute."""
        _ = (duration, iterations)
        return {}


def test_registry_lazy_discovery():
    """Test that registry discovery is lazy by default."""
    with patch.object(TestRegistry, "_discover_tests") as mock_discover:
        registry = TestRegistry(lazy=True)
        assert not registry._discovered
        mock_discover.assert_not_called()

        # Trigger discovery
        _ = registry.get_all_tests()
        assert registry._discovered
        mock_discover.assert_called_once()


def test_registry_get_test():
    """Test getting a specific test by name."""
    registry = TestRegistry(include_defaults=False)
    # Manually register a test
    registry._tests["MockTest"] = MockTest
    registry._discovered = True

    assert registry.get_test("MockTest") == MockTest

    with pytest.raises(TestNotFoundError):
        registry.get_test("NonExistent")


def test_registry_get_by_category():
    """Test filtering tests by category."""
    registry = TestRegistry(include_defaults=False)
    registry._tests["MockTest"] = MockTest
    registry._discovered = True

    accel_tests = registry.get_tests_by_category(TestCategory.ACCELERATOR)
    assert MockTest in accel_tests

    cpu_tests = registry.get_tests_by_category(TestCategory.CPU)
    assert MockTest not in cpu_tests
