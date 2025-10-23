"""Pydantic models for stress test command output."""

from pydantic import BaseModel, ConfigDict, Field


class StressResults(BaseModel):
    """Results from a stress test."""

    max_utilization: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Maximum utilization percentage during the test",
    )
    avg_utilization: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Average utilization percentage during the test",
    )
    peak_temperature: int | None = Field(
        default=None,
        ge=0,
        description="Peak temperature in Celsius during the test",
    )
    power_draw_avg: int | None = Field(
        default=None,
        ge=0,
        description="Average power draw in Watts during the test",
    )
    errors: int = Field(
        default=0,
        ge=0,
        description="Number of errors encountered during the test",
    )
    throttling_events: int = Field(
        default=0,
        ge=0,
        description="Number of thermal or power throttling events",
    )


class StressOutput(BaseModel):
    """Output model for stress test command."""

    model_config = ConfigDict(
        validate_assignment=True,
    )

    test: str = Field(
        ...,
        description="Type of stress test performed (e.g., 'cpu_stress', 'gpu_stress', 'memory_stress')",
    )
    duration: int = Field(
        ...,
        ge=1,
        description="Duration of the stress test in seconds",
    )
    results: StressResults = Field(
        ...,
        description="Detailed results from the stress test",
    )
    status: str = Field(
        ...,
        description="Test status (e.g., 'pass', 'fail')",
    )
