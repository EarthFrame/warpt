"""Pydantic models for device behavioral profiles."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class HardwareCategory(str, Enum):
    """Hardware categories for behavioral profiles.

    Separate from ``stress.base.TestCategory`` because profile categories map
    to physical hardware (e.g., GPU) rather than test types (e.g., ACCELERATOR).
    """

    GPU = "gpu"
    CPU = "cpu"
    RAM = "ram"
    STORAGE = "storage"
    NETWORK = "network"


class DeviceFingerprint(BaseModel):
    """Unique identifier for a hardware component."""

    fingerprint_id: str = Field(
        ..., description="Unique device ID (e.g., NVIDIA UUID for GPUs)"
    )
    category: HardwareCategory = Field(
        ..., description="Hardware category this device belongs to"
    )
    source: str = Field(
        ...,
        description="How the fingerprint was generated (e.g., 'nvml_uuid')",
    )


class ObservationEnvironment(BaseModel):
    """Environmental conditions during a test run.

    Captures the state of the hardware while the observation was being
    recorded, enabling correlation between performance and conditions
    (e.g., "this GPU hit 45 TFLOPS but was thermally throttling at 89C").
    """

    temperature_celsius: float | None = Field(
        None, description="Device temperature during test in Celsius"
    )
    power_watts: float | None = Field(
        None, description="Power draw during test in Watts"
    )
    throttle_events: list[dict[str, Any]] = Field(
        default_factory=list, description="Throttling events that occurred during test"
    )
    hostname: str | None = Field(None, description="Machine that ran the test")
    platform: str | None = Field(None, description="OS platform (linux/darwin/windows)")
    driver_version: str | None = Field(
        None, description="Driver version at time of test"
    )


class Observation(BaseModel):
    """A single test run record tied to a device.

    Each time a stress test runs, it can produce one of these.
    """

    id: str = Field(..., description="UUID for this observation")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    test_name: str = Field(
        ...,
        description="Test class name (e.g., 'GPUMatMulTest', 'GPUMemoryBandwidthTest')",
    )
    category: HardwareCategory = Field(
        ..., description="Hardware category of the tested device"
    )
    duration: float = Field(..., ge=0, description="Test duration in seconds")
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Test-specific metrics (tflops, bandwidth_gbps, etc.)",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Test configuration used (matrix_size, block_size, etc.)",
    )
    environment: ObservationEnvironment | None = Field(
        None, description="Environmental conditions during the test"
    )


class DeviceProfile(BaseModel):
    """Behavioral profile for a single hardware component.

    Accumulates observations over time to track performance history,
    detect degradation, and inform workload fitness decisions.
    """

    fingerprint: DeviceFingerprint = Field(..., description="Unique device identifier")
    model: str = Field(..., description="Hardware model name (e.g., 'NVIDIA RTX 4090')")
    vendor: str | None = Field(None, description="Hardware manufacturer")
    specs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Category-specific static specs "
            "(e.g., {'vram_gb': 24, 'compute_capability': '8.9'} for GPU)"
        ),
    )
    first_seen: str = Field(..., description="ISO 8601 timestamp of first observation")
    last_seen: str = Field(
        ..., description="ISO 8601 timestamp of most recent observation"
    )
    observation_count: int = Field(
        0, ge=0, description="Total number of observations recorded"
    )
    observations: list[Observation] = Field(
        default_factory=list, description="Chronological list of test observations"
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="User-defined metadata (e.g., {'rack': 'A3', 'role': 'training'})",
    )
