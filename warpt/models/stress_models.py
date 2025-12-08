"""Pydantic models for stress test JSON export."""

from typing import Any, Literal

from pydantic import BaseModel, Field

# ============================================================================
# Supporting Types
# ============================================================================


class ThrottleEvent(BaseModel):
    """Represents a single throttling event."""

    timestamp: float = Field(..., description="Unix timestamp when throttling occurred")
    device_id: str = Field(
        ..., description="Fully qualified device ID (e.g., 'cpu_0', 'gpu_0', 'gpu_1')"
    )
    reasons: list[str] = Field(
        ...,
        description=(
            "List of throttle reasons "
            "(e.g., 'thermal', 'power_limit', 'SW Power Cap')"
        ),
    )

    # TODO: add duration tracking
    # - start_timestamp
    # - end_timestamp
    # - duration_ms


# ============================================================================
# Individual Test Result Models
# ============================================================================


class CPUSystemResult(BaseModel):
    """Results from CPU system-level stress test."""

    # Identifiers
    cpu_model: str = Field(..., description="CPU model name")
    cpu_architecture: str = Field(
        ..., description="CPU architecture (e.g., 'x86_64', 'arm64')"
    )

    # Performance metrics
    tflops: float = Field(..., ge=0, description="Computational throughput in TFLOPS")
    duration: float = Field(..., ge=0, description="Actual test duration in seconds")
    iterations: int | None = Field(
        None, ge=0, description="Number of iterations completed (if applicable)"
    )
    total_operations: int = Field(
        ..., ge=0, description="Total floating point operations"
    )
    burnin_seconds: int = Field(..., ge=0, description="Warmup period in seconds")

    # Test-specific metrics (flexible for different test types)
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Test-specific metrics (e.g., matrix_size, vector_length, etc.)",
    )

    # CPU topology
    sockets_used: int = Field(..., ge=1, description="Number of CPU sockets used")
    physical_cores: int = Field(..., ge=1, description="Total physical cores")
    logical_cores: int = Field(..., ge=1, description="Total logical cores")

    # Optional monitoring fields (for future implementation)
    max_temp: float | None = Field(None, description="Maximum temperature in Celsius")
    avg_power: float | None = Field(None, description="Average power draw in Watts")
    throttle_events: list[ThrottleEvent] = Field(
        default_factory=list, description="List of throttling events during test"
    )


class PrecisionResult(BaseModel):
    """Results for a single precision test."""

    supported: bool = Field(..., description="Whether this precision is supported")
    iterations: int | None = Field(None, description="Number of iterations completed")
    avg_time_per_iter: float | None = Field(
        None, description="Average time per iteration in seconds"
    )
    dtype: str = Field(..., description="Data type used (e.g., 'torch.float16')")
    tflops: float | None = Field(None, description="Performance in TFLOPS")
    matrix_size: int = Field(..., description="Matrix size used for test")
    hardware_supported: bool | None = Field(
        None, description="Hardware support (e.g., Tensor Cores)"
    )
    runtime_supported: bool | None = Field(None, description="Runtime/driver support")
    method: str | None = Field(
        None, description="Method used (e.g., 'pytorch_quantization' for INT8)"
    )
    note: str | None = Field(None, description="Additional notes about the test")


class MixedPrecisionResults(BaseModel):
    """Complete mixed precision test results."""

    fp32: PrecisionResult = Field(..., description="FP32 test results")
    fp16: PrecisionResult | None = Field(None, description="FP16 test results")
    bf16: PrecisionResult | None = Field(None, description="BF16 test results")
    int8: PrecisionResult | None = Field(None, description="INT8 test results")

    # Speedup comparisons (relative to FP32 baseline)
    fp16_speedup: float | None = Field(
        None, description="FP16 speedup over FP32 (< 1.0 means slower)", gt=0
    )
    bf16_speedup: float | None = Field(
        None, description="BF16 speedup over FP32 (< 1.0 means slower)", gt=0
    )
    int8_speedup: float | None = Field(
        None, description="INT8 speedup over FP32 (< 1.0 means slower)", gt=0
    )

    mixed_precision_ready: bool = Field(
        ..., description="Whether GPU is ready for mixed precision training"
    )
    tf32_enabled: bool = Field(
        ..., description="Whether TF32 was enabled for FP32 operations"
    )


class GPUMemoryBandwidthResult(BaseModel):
    """Results from GPU memory bandwidth test."""

    # Device identifiers
    device_id: str = Field(..., description="Logical device ID (e.g., 'gpu_0')")
    gpu_uuid: str = Field(
        ..., description="Persistent GPU UUID for tracking across systems"
    )
    gpu_name: str = Field(..., description="GPU model name")

    # Test metadata
    duration: float = Field(..., ge=0, description="Test duration in seconds")
    burnin_seconds: int = Field(..., ge=0, description="Warmup period in seconds")
    data_size_gb: float = Field(
        ..., gt=0, description="Size of data transferred per test in GB"
    )
    used_pinned_memory: bool = Field(
        ..., description="Whether pinned memory was used for H2D/D2H transfers"
    )

    # Device-to-Device (GPU memory bandwidth)
    d2d_bandwidth_gbps: float = Field(
        ..., ge=0, description="Device-to-device memory bandwidth in GB/s"
    )
    d2d_iterations: int = Field(
        ..., ge=0, description="Number of D2D iterations completed"
    )

    # Host-to-Device (PCIe upload bandwidth)
    h2d_bandwidth_gbps: float | None = Field(
        None, ge=0, description="Host-to-device memory bandwidth in GB/s"
    )
    h2d_iterations: int | None = Field(
        None, ge=0, description="Number of H2D iterations completed"
    )

    # Device-to-Host (PCIe download bandwidth)
    d2h_bandwidth_gbps: float | None = Field(
        None, ge=0, description="Device-to-host memory bandwidth in GB/s"
    )
    d2h_iterations: int | None = Field(
        None, ge=0, description="Number of D2H iterations completed"
    )


class GPUDeviceResult(BaseModel):
    """Results from individual GPU stress test."""

    # Identifiers
    device_id: str = Field(..., description="Logical device ID (e.g., 'gpu_0')")
    gpu_uuid: str = Field(
        ..., description="Persistent GPU UUID for tracking across systems"
    )
    gpu_name: str = Field(..., description="GPU model name")

    # Performance metrics (from compute test - optional if only other tests run)
    tflops: float | None = Field(
        None, ge=0, description="GPU computational throughput in TFLOPS"
    )
    duration: float | None = Field(None, ge=0, description="Test duration in seconds")
    iterations: int | None = Field(
        None, ge=0, description="Iterations completed (if applicable)"
    )
    total_operations: int | None = Field(
        None, ge=0, description="Total floating point operations"
    )
    burnin_seconds: int = Field(..., ge=0, description="Warmup period in seconds")

    # Test-specific metrics (flexible for different test types)
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Test-specific metrics " "(e.g., matrix_size, memory_bandwidth_gbps, etc.)"
        ),
    )

    # GPU-specific monitoring
    memory_used_gb: float | None = Field(
        None, ge=0, description="GPU memory used in GB"
    )
    max_temp: float | None = Field(None, description="Peak temperature in Celsius")
    avg_power: float | None = Field(None, description="Average power in Watts")
    throttle_events: list[ThrottleEvent] = Field(
        default_factory=list, description="List of throttling events during test"
    )

    # Mixed precision results
    mixed_precision: MixedPrecisionResults | None = Field(
        None, description="Mixed precision test results"
    )

    # Memory bandwidth results
    memory_bandwidth: GPUMemoryBandwidthResult | None = Field(
        None, description="Memory bandwidth test results"
    )


class GPUSystemResult(BaseModel):
    """Results from GPU system-level stress test (multi-GPU workload)."""

    # System identifiers
    devices_used: list[str] = Field(
        ..., description="GPU device IDs (e.g., ['gpu_0', 'gpu_1'])"
    )
    gpu_uuids: list[str] = Field(
        ..., description="Persistent UUIDs for all GPUs in test"
    )
    device_count: int = Field(..., ge=1, description="Number of GPUs in system test")

    # Performance metrics
    aggregate_tflops: float = Field(
        ..., ge=0, description="Combined TFLOPS across all GPUs"
    )
    duration: float = Field(..., ge=0, description="Test duration in seconds")
    iterations: int | None = Field(
        None, ge=0, description="Iterations completed (if applicable)"
    )
    scaling_efficiency: float = Field(
        ..., ge=0, le=1, description="Multi-GPU scaling efficiency (0-1)"
    )
    orchestration_overhead_ms: float = Field(
        ..., ge=0, description="Overhead from multi-GPU coordination"
    )
    burnin_seconds: int = Field(..., ge=0, description="Warmup period in seconds")

    # Test-specific metrics (flexible for different test types)
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Test-specific metrics (e.g., matrix_size, workload_type, etc.)",
    )

    # Optional monitoring
    max_temp_across_gpus: float | None = Field(
        None, description="Highest temp across all GPUs"
    )
    total_power: float | None = Field(None, description="Combined power draw in Watts")
    throttle_events: list[ThrottleEvent] = Field(
        default_factory=list,
        description="List of throttling events across all GPUs during test",
    )


# ============================================================================
# Target-specific Result Containers
# ============================================================================


class CPUTestResults(BaseModel):
    """CPU test results container."""

    test_mode: Literal["system_level"] = "system_level"
    device_count: int = Field(..., ge=1, description="Number of CPU sockets")
    results: dict[Literal["cpu_system"], CPUSystemResult] = Field(
        ..., description="CPU system-level results"
    )


class GPUTestResults(BaseModel):
    """GPU test results container."""

    test_mode: Literal["per_device", "system_level"] = Field(
        ..., description="Test mode: per_device or system_level"
    )
    device_count: int = Field(..., ge=1, description="Number of GPUs")
    results: dict[str, GPUDeviceResult | GPUSystemResult] = Field(
        ...,
        description=(
            "GPU results - per_device uses 'gpu_0', 'gpu_1', etc.; "
            "system_level uses 'gpu_system'"
        ),
    )


class RAMTestResults(BaseModel):
    """RAM test results container (placeholder for future implementation)."""

    test_mode: Literal["system_level"] = "system_level"
    device_count: int = Field(1, description="System RAM (always 1)")
    results: dict[str, dict] = Field(..., description="RAM test results")


# ============================================================================
# Summary Models
# ============================================================================


class CPUSummary(BaseModel):
    """Summary of CPU test results."""

    status: Literal["pass", "fail", "warning"] = Field(
        ..., description="Overall test status"
    )
    performance: str = Field(..., description="Performance assessment")
    tflops: float = Field(..., ge=0, description="TFLOPS achieved")


class GPUSummary(BaseModel):
    """Summary of GPU test results."""

    total_devices_tested: int = Field(..., ge=0, description="Number of GPUs tested")
    avg_tflops: float = Field(..., ge=0, description="Average TFLOPS across all GPUs")
    healthy_devices: int = Field(..., ge=0, description="Number of healthy GPUs")
    warnings: list[str] = Field(default_factory=list, description="Warning messages")


class RAMSummary(BaseModel):
    """Summary of RAM test results (placeholder for future implementation)."""

    status: Literal["pass", "fail", "warning"] = Field(
        ..., description="Overall test status"
    )
    bandwidth_gbps: float | None = Field(None, description="Memory bandwidth in GB/s")


# ============================================================================
# Top-level Export Model
# ============================================================================


class StressTestExport(BaseModel):
    """Top-level stress test export model for JSON output."""

    targets_tested: list[str] = Field(
        ..., description="List of targets tested (e.g., ['cpu', 'gpu', 'ram'])"
    )

    results: dict[str, CPUTestResults | GPUTestResults | RAMTestResults] = Field(
        ..., description="Test results organized by target type"
    )

    summary: dict[str, CPUSummary | GPUSummary | RAMSummary] = Field(
        ..., description="Test summaries organized by target type"
    )

    # Optional metadata
    timestamp_start: str | None = Field(
        None, description="Test start timestamp (ISO format)"
    )
    timestamp_end: str | None = Field(
        None, description="Test end timestamp (ISO format)"
    )
    warpt_version: str | None = Field(None, description="Warpt version used")
