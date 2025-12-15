"""Stress testing module for warpt.

This module provides:
- StressTest: Base class for all stress tests
- TestRegistry: Automatic test discovery and registration
- TestRunner: Test execution and result collection
- TestResults: Flexible result emission (JSON/YAML/stdout)

Quick Start:
    from warpt.stress import TestRegistry, TestRunner

    # Discover all tests
    registry = TestRegistry()

    # Run all available tests
    runner = TestRunner()
    runner.add_tests(registry.get_available_tests())
    results = runner.run(duration=30)

    # Emit results
    results.emit_json("results.json")
    results.emit_stdout()
"""

from warpt.stress.base import StressTest, TestCategory
from warpt.stress.registry import (
    TestNameCollisionError,
    TestNotFoundError,
    TestRegistry,
    TestRegistryError,
)
from warpt.stress.results import OutputFormat, TestResults
from warpt.stress.runner import TestRunner, run_all_available_tests

__all__ = [
    "OutputFormat",
    # Base
    "StressTest",
    "TestCategory",
    "TestNameCollisionError",
    "TestNotFoundError",
    # Registry
    "TestRegistry",
    "TestRegistryError",
    # Results
    "TestResults",
    # Runner
    "TestRunner",
    "run_all_available_tests",
]
