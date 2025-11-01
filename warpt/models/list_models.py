"""Pydantic models for the list command JSON output."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# Hardware Models


class CPUInfo(BaseModel):
    """CPU information."""

    model: str = Field(..., description="CPU model name")
    cores: int = Field(..., description="Number of physical cores", ge=1)
    threads: int = Field(..., description="Number of logical threads", ge=1)
    features: List[str] = Field(
        default_factory=list, description="CPU features (AVX, SSE, etc.)"
    )


class GPUInfo(BaseModel):
    """Individual GPU information."""

    index: int = Field(..., description="GPU index", ge=0)
    model: str = Field(..., description="GPU model name")
    memory_gb: int = Field(..., description="Total GPU memory in GB", ge=0)
    compute_capability: Optional[str] = Field(
        None, description="CUDA compute capability (e.g., '8.9')"
    )
    pcie_gen: Optional[int] = Field(
        None, description="PCIe generation", ge=1, le=5
    )
    extra_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="vendor specific metrics without validation"
    )


class MemoryInfo(BaseModel):
    """System memory information."""

    total_gb: int = Field(..., description="Total system memory in GB", ge=1)
    type: Optional[str] = Field(None, description="Memory type (DDR4, DDR5, etc.)")


class StorageDevice(BaseModel):
    """Storage device information."""

    device_path: str = Field(..., description="Device path (e.g., /dev/nvme0n1)")
    capacity_gb: int = Field(..., description="Storage capacity in GB", ge=1)
    type: str = Field(..., description="Storage type (NVMe SSD, SATA SSD, HDD, etc.)")


class HardwareInfo(BaseModel):
    """Container for all hardware information."""

    cpu: Optional[CPUInfo] = Field(None, description="CPU information")
    gpu: Optional[List[GPUInfo]] = Field(
        None, description="List of GPUs (empty list if no GPUs)"
    )
    memory: Optional[MemoryInfo] = Field(None, description="System memory information")
    storage: Optional[List[StorageDevice]] = Field(
        None, description="List of storage devices"
    )


# Software Models


class PythonInfo(BaseModel):
    """Python installation information."""

    version: str = Field(..., description="Python version (e.g., '3.11.4')")
    path: str = Field(..., description="Path to Python executable")


class CUDAInfo(BaseModel):
    """CUDA toolkit information."""

    version: str = Field(..., description="CUDA version (e.g., '12.1.1')")
    driver: str = Field(..., description="NVIDIA driver version")


class FrameworkInfo(BaseModel):
    """ML framework information."""

    version: str = Field(..., description="Framework version")
    cuda_support: bool = Field(
        default=False, description="Whether CUDA support is enabled"
    )


class CompilerInfo(BaseModel):
    """Compiler information."""

    version: str = Field(..., description="Compiler version")
    path: Optional[str] = Field(None, description="Path to compiler executable")


class SoftwareInfo(BaseModel):
    """Container for all software information."""

    python: Optional[PythonInfo] = Field(None, description="Python installation")
    cuda: Optional[CUDAInfo] = Field(None, description="CUDA toolkit")
    frameworks: Optional[Dict[str, FrameworkInfo]] = Field(
        None,
        description="ML frameworks (pytorch, tensorflow, jax, etc.)",
    )
    compilers: Optional[Dict[str, CompilerInfo]] = Field(
        None,
        description="Compilers (gcc, nvcc, clang, etc.)",
    )


# Top-level Output Model


class ListOutput(BaseModel):
    """Top-level output for the list command."""

    model_config = ConfigDict(
        validate_assignment=True,
    )

    hardware: Optional[HardwareInfo] = Field(
        None, description="Hardware information (populated with --hardware or default)"
    )
    software: Optional[SoftwareInfo] = Field(
        None, description="Software information (populated with --software or default)"
    )
