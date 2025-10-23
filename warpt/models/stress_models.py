"""Pydantic models for stress test command output."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from warpt.models.constants import Limits


class StressResults(BaseModel):
    """Results from a stress test."""

    max_utilization: float = Field(
        ...,
        ge=Limits.PERCENT_MIN,
        le=Limits.PERCENT_MAX,
        description="Maximum utilization percentage during the test",
    )
    avg_utilization: float = Field(
        ...,
        ge=Limits.PERCENT_MIN,
        le=Limits.PERCENT_MAX,
        description="Average utilization percentage during the test",
    )
    peak_temperature: int | None = Field(
        default=None,
        ge=Limits.TEMPERATURE_MIN,
        le=Limits.TEMPERATURE_MAX,
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
        description="Number of errors encountered during the test", # need to create categorization of errors
    )
    throttling_events: int = Field(
        default=0,
        ge=0,
        description="Number of thermal or power throttling events", # use pynvml.nvmlDeviceGetCurrentClocksThrottleReasons 
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
    status: Literal["pass", "fail", "warning", "running", "stopped"] = Field(
        ...,
        description="Test status",
    ) # Literal type requires actual value
