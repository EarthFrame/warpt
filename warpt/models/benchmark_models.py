"""Models for benchmark configuration and results."""

from typing import Any

from pydantic import BaseModel, Field


class BenchmarkConfig(BaseModel):
    """Configuration for a single benchmark run."""

    name: str = Field(..., description="Name of the benchmark (e.g., 'HPLBenchmark')")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Benchmark-specific parameters"
    )
    description: str | None = Field(
        None, description="Optional description of this run"
    )


class BenchmarksConfig(BaseModel):
    """Root configuration containing benchmark definitions."""

    benchmarks: list[BenchmarkConfig] = Field(
        ..., description="List of benchmarks to run"
    )
