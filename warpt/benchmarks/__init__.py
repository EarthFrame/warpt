"""Benchmark subsystem for warpt.

Provides a framework for implementing and running performance benchmarks
with automatic discovery and standardized result formats.
"""

from warpt.benchmarks.base import Benchmark, BenchmarkResult, RunMode
from warpt.benchmarks.registry import (
    BenchmarkNameCollisionError,
    BenchmarkNotFoundError,
    BenchmarkRegistry,
    BenchmarkRegistryError,
)

__all__ = [
    "Benchmark",
    "BenchmarkNameCollisionError",
    "BenchmarkNotFoundError",
    "BenchmarkRegistry",
    "BenchmarkRegistryError",
    "BenchmarkResult",
    "RunMode",
]
