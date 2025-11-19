"""Pydantic models for structured output."""

from warpt.models.list_models import (
    CompilerInfo,
    CPUInfo,
    CUDAInfo,
    FrameworkInfo,
    GPUInfo,
    HardwareInfo,
    ListOutput,
    MemoryInfo,
    PythonInfo,
    SoftwareInfo,
    StorageDevice,
)

from warpt.models.stress_models import (
    CPUSystemResult,
    CPUTestResults,
    CPUSummary,
    GPUDeviceResult,
    GPUSystemResult,
    GPUTestResults,
    GPUSummary,
    RAMTestResults,
    RAMSummary,
    StressTestExport,
)

__all__ = [
    "CPUInfo",
    "CUDAInfo",
    "CompilerInfo",
    "FrameworkInfo",
    "GPUInfo",
    "HardwareInfo",
    "ListOutput",
    "MemoryInfo",
    "PythonInfo",
    "SoftwareInfo",
    "StorageDevice",
    # Stress test models
    "CPUSystemResult",
    "CPUTestResults",
    "CPUSummary",
    "GPUDeviceResult",
    "GPUSystemResult",
    "GPUTestResults",
    "GPUSummary",
    "RAMTestResults",
    "RAMSummary",
    "StressTestExport",
]
