"""Pydantic models for the list command JSON output."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Hardware Models


class CPUInfo(BaseModel):
    """CPU information."""

    manufacturer: str = Field(
        ..., description="CPU manufacturer (e.g., 'Apple', 'Intel', 'AMD')"
    )
    model: str = Field(..., description="CPU model name")
    architecture: str | None = Field(
        None, description="CPU architecture (e.g., 'amd64', 'arm64')"
    )
    cores: int = Field(..., description="Number of physical cores", ge=1)
    threads: int = Field(..., description="Number of logical threads", ge=1)
    base_frequency_mhz: float | None = Field(
        None, description="Base/minimum frequency in MHz"
    )
    boost_frequency_single_core_mhz: float | None = Field(
        None, description="Maximum single-core boost frequency in MHz"
    )
    boost_frequency_multi_core_mhz: float | None = Field(
        None, description="Multi-core boost frequency in MHz"
    )
    current_frequency_mhz: float | None = Field(
        None, description="Current frequency in MHz"
    )
    instruction_sets: list[str] | None = Field(
        None, description="CPU instruction sets (e.g., AVX, AVX2, SSE4.2)"
    )
    features: list[str] = Field(
        default_factory=list, description="CPU features (AVX, SSE, etc.)"
    )


class GPUInfo(BaseModel):
    """Individual GPU information."""

    index: int = Field(..., description="GPU index", ge=0)
    model: str = Field(..., description="GPU model name")
    memory_gb: int = Field(..., description="Total GPU memory in GB", ge=0)
    compute_capability: str | None = Field(
        None, description="CUDA compute capability (e.g., '8.9')"
    )
    pcie_gen: int | None = Field(None, description="PCIe generation", ge=1, le=5)
    driver_version: str | None = Field(
        None, description="GPU driver version (e.g., '535.104.05')"
    )
    extra_metrics: dict[str, Any] | None = Field(
        default=None, description="vendor specific metrics without validation"
    )


class MemoryInfo(BaseModel):
    """System memory information."""

    total_gb: int = Field(..., description="Total system memory in GB", ge=1)
    free_gb: float = Field(..., description="Free system memory in GB", ge=0)
    type: str | None = Field(None, description="Memory type (DDR4, DDR5, etc.)")
    speed_mhz: int | None = Field(None, description="Memory speed in MHz")
    channels: int | None = Field(None, description="Number of memory channels")


class StorageDevice(BaseModel):
    """Storage device information."""

    device_path: str = Field(..., description="Device path (e.g., /dev/nvme0n1)")
    capacity_gb: int = Field(..., description="Storage capacity in GB", ge=1)
    type: str = Field(..., description="Storage type (NVMe SSD, SATA SSD, HDD, etc.)")
    model: str | None = Field(None, description="Device model string (if available)")
    manufacturer: str | None = Field(
        None, description="Device manufacturer/vendor (if detected)"
    )
    serial: str | None = Field(None, description="Device serial number")
    bus_type: str | None = Field(
        None, description="Bus/interconnect type (PCIe, SATA, USB, etc.)"
    )
    link_speed_gbps: float | None = Field(
        None, description="Reported interface speed in Gbps"
    )


class HardwareInfo(BaseModel):
    """Container for all hardware information."""

    cpu: CPUInfo | None = Field(None, description="CPU information")
    gpu_count: int | None = Field(None, description="Number of GPUs detected", ge=0)
    gpu: list[GPUInfo] | None = Field(
        None, description="List of GPUs (empty list if no GPUs)"
    )
    memory: MemoryInfo | None = Field(None, description="System memory information")
    storage: list[StorageDevice] | None = Field(
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


class NvidiaContainerToolkitInfo(BaseModel):
    """NVIDIA Container Toolkit information."""

    installed: bool = Field(..., description="Whether the toolkit is available")
    cli_version: str | None = Field(None, description="nvidia-container-cli version")
    cli_path: str | None = Field(None, description="Path to nvidia-container-cli")
    runtime_path: str | None = Field(
        None, description="Path to nvidia-container-runtime"
    )
    docker_runtime_ready: bool | None = Field(
        None,
        description="Whether Docker exposes the 'nvidia' runtime",
    )


class DockerInfo(BaseModel):
    """Docker CLI information."""

    installed: bool = Field(..., description="Whether Docker CLI is available")
    version: str | None = Field(None, description="Detected Docker version")
    path: str | None = Field(None, description="Path to docker executable")


class FrameworkInfo(BaseModel):
    """ML framework information."""

    version: str = Field(..., description="Framework version")
    cuda_support: bool = Field(
        default=False, description="Whether CUDA support is enabled"
    )


class CompilerInfo(BaseModel):
    """Compiler information."""

    version: str = Field(..., description="Compiler version")
    path: str | None = Field(None, description="Path to compiler executable")


class SoftwareInfo(BaseModel):
    """Container for all software information."""

    python: PythonInfo | None = Field(None, description="Python installation")
    cuda: CUDAInfo | None = Field(None, description="CUDA toolkit")
    nvidia_container_toolkit: NvidiaContainerToolkitInfo | None = Field(
        None, description="NVIDIA Container Toolkit detection results"
    )
    docker: DockerInfo | None = Field(None, description="Docker CLI information")
    frameworks: dict[str, FrameworkInfo] | None = Field(
        None,
        description="ML frameworks (pytorch, tensorflow, jax, etc.)",
    )
    compilers: dict[str, CompilerInfo] | None = Field(
        None,
        description="Compilers (gcc, nvcc, clang, etc.)",
    )


# Top-level Output Model


class ListOutput(BaseModel):
    """Top-level output for the list command."""

    model_config = ConfigDict(
        validate_assignment=True,
    )

    hardware: HardwareInfo | None = Field(
        None, description="Hardware information (populated with --hardware or default)"
    )
    software: SoftwareInfo | None = Field(
        None, description="Software information (populated with --software or default)"
    )
