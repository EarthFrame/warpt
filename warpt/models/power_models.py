"""Pydantic models for power monitoring data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PowerDomain(str, Enum):
    """Power measurement domains."""

    PACKAGE = "package"  # Total CPU package power
    CORE = "core"  # CPU cores only
    UNCORE = "uncore"  # Integrated GPU, memory controller, etc.
    DRAM = "dram"  # Memory subsystem
    GPU = "gpu"  # Discrete GPU
    ANE = "ane"  # Apple Neural Engine
    PSYS = "psys"  # Platform/system (includes everything)


class PowerSource(str, Enum):
    """Source of power measurements."""

    RAPL = "rapl"  # Intel/AMD Running Average Power Limit
    POWERMETRICS = "powermetrics"  # macOS powermetrics
    NVML = "nvml"  # NVIDIA Management Library
    ROCM_SMI = "rocm_smi"  # AMD ROCm SMI
    IOKIT = "iokit"  # macOS IOKit (battery)
    ESTIMATED = "estimated"  # Calculated/estimated from utilization


@dataclass
class DomainPower:
    """Power measurement for a single domain.

    Attributes:
        domain: The power domain being measured.
        power_watts: Current power consumption in watts.
        energy_joules: Cumulative energy in joules (if available).
        source: Source of the measurement.
        metadata: Additional domain-specific info (e.g., GPU index).
    """

    domain: PowerDomain
    power_watts: float
    energy_joules: float | None = None
    source: PowerSource = PowerSource.ESTIMATED
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "domain": self.domain.value,
            "power_watts": self.power_watts,
            "energy_joules": self.energy_joules,
            "source": self.source.value,
            "metadata": self.metadata,
        }


@dataclass
class ProcessPower:
    """Power attribution for a single process.

    Attributes:
        pid: Process ID.
        name: Process name.
        cpu_power_watts: Estimated CPU power consumption.
        gpu_power_watts: GPU power consumption (if using GPU).
        total_power_watts: Total attributed power.
        cpu_percent: CPU utilization percentage.
        gpu_percent: GPU utilization percentage (if applicable).
        memory_mb: Memory usage in MB.
    """

    pid: int
    name: str
    cpu_power_watts: float
    gpu_power_watts: float = 0.0
    total_power_watts: float = 0.0
    cpu_percent: float = 0.0
    gpu_percent: float = 0.0
    memory_mb: float = 0.0

    def __post_init__(self) -> None:
        """Calculate total power if not set."""
        if self.total_power_watts == 0.0:
            self.total_power_watts = self.cpu_power_watts + self.gpu_power_watts

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "pid": self.pid,
            "name": self.name,
            "cpu_power_watts": round(self.cpu_power_watts, 3),
            "gpu_power_watts": round(self.gpu_power_watts, 3),
            "total_power_watts": round(self.total_power_watts, 3),
            "cpu_percent": round(self.cpu_percent, 1),
            "gpu_percent": round(self.gpu_percent, 1),
            "memory_mb": round(self.memory_mb, 1),
        }


@dataclass
class GPUPowerInfo:
    """Power information for a single GPU.

    Attributes:
        index: GPU index.
        name: GPU model name.
        power_watts: Current power draw in watts.
        power_limit_watts: Power limit in watts.
        utilization_percent: GPU utilization percentage.
        memory_utilization_percent: Memory utilization percentage.
        temperature_celsius: GPU temperature.
        processes: List of processes using this GPU.
    """

    index: int
    name: str
    power_watts: float
    power_limit_watts: float | None = None
    utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    temperature_celsius: float | None = None
    processes: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "index": self.index,
            "name": self.name,
            "power_watts": round(self.power_watts, 2),
            "power_limit_watts": self.power_limit_watts,
            "utilization_percent": round(self.utilization_percent, 1),
            "memory_utilization_percent": round(self.memory_utilization_percent, 1),
            "temperature_celsius": self.temperature_celsius,
            "processes": self.processes,
            "metadata": self.metadata,
        }


@dataclass
class PowerSnapshot:
    """Complete power snapshot at a point in time.

    Attributes:
        timestamp: Unix timestamp of the measurement.
        total_power_watts: Total system power (if measurable).
        domains: Power breakdown by domain.
        gpus: Per-GPU power information.
        processes: Per-process power attribution.
        platform: Operating system platform.
        available_sources: Power sources available on this system.
    """

    timestamp: float
    total_power_watts: float | None = None
    domains: list[DomainPower] = field(default_factory=list)
    gpus: list[GPUPowerInfo] = field(default_factory=list)
    processes: list[ProcessPower] = field(default_factory=list)
    platform: str = ""
    available_sources: list[PowerSource] = field(default_factory=list)

    def get_domain_power(self, domain: PowerDomain) -> float | None:
        """Get power for a specific domain."""
        for d in self.domains:
            if d.domain == domain:
                return d.power_watts
        return None

    def get_cpu_power(self) -> float | None:
        """Get total CPU power (package or core)."""
        # Prefer package power, fall back to core
        pkg = self.get_domain_power(PowerDomain.PACKAGE)
        if pkg is not None:
            return pkg
        return self.get_domain_power(PowerDomain.CORE)

    def get_gpu_power(self) -> float:
        """Get total GPU power across all GPUs."""
        return sum(gpu.power_watts for gpu in self.gpus)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        from datetime import datetime

        return {
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "total_power_watts": (
                round(self.total_power_watts, 2) if self.total_power_watts else None
            ),
            "domains": [d.to_dict() for d in self.domains],
            "gpus": [g.to_dict() for g in self.gpus],
            "processes": [p.to_dict() for p in self.processes],
            "platform": self.platform,
            "available_sources": [s.value for s in self.available_sources],
        }


@dataclass
class RAPLDomain:
    """Information about a RAPL power domain.

    Used internally for Linux RAPL reading.

    Attributes:
        name: Domain name (e.g., "package-0", "core", "dram").
        path: Sysfs path to the domain.
        energy_path: Path to energy_uj file.
        max_energy_uj: Maximum energy counter value before wrap.
        last_energy_uj: Last read energy value.
        last_timestamp: Timestamp of last read.
    """

    name: str
    path: str
    energy_path: str
    max_energy_uj: int = 0
    last_energy_uj: int = 0
    last_timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "max_energy_uj": self.max_energy_uj,
        }
