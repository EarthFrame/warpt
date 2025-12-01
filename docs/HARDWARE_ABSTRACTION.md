# Hardware Abstraction Architecture

This document outlines the design for warpt's hardware abstraction layer, enabling
vendor-agnostic hardware detection, listing, and benchmarking.

## Design Goals

1. **Vendor Extensibility**: Hardware vendors can implement interfaces to add support
1. **Auto-Discovery**: Automatically detect available hardware via bus scanning
1. **Unified API**: Common interface across storage, compute, and accelerators
1. **Graceful Degradation**: Work with partial information when drivers unavailable
1. **Simple Partnership Model**: Easy integration for accelerator vendors

## Overview

Warpt aims to support eight primary accelerator vendors through a clean, flat backend architecture:

- NVIDIA (GPUs)
- AMD (GPUs)
- Intel (Arc/Data Center GPUs)
- Apple (M-series GPU + Neural Engine)
- Qualcomm (Hexagon DSP/NPU)
- Google (TPU)
- Groq (LPU)
- Cerebras (WSE)

## Recommended Directory Structure

```text
warpt/backends/
├── base.py                    # AcceleratorBackend ABC
├── factory.py                 # Auto-detection and registration
│
├── accelerators/              # All accelerator vendors
│   ├── __init__.py
│   ├── nvidia.py             # NVIDIA GPUs
│   ├── amd.py                # AMD GPUs (ROCm)
│   ├── intel.py              # Intel Arc/Data Center GPUs
│   ├── apple.py              # Apple Neural Engine + M-series GPU
│   ├── qualcomm.py           # Qualcomm Hexagon DSP/NPU
│   ├── google.py             # Google TPU
│   ├── groq.py               # Groq LPU
│   └── cerebras.py           # Cerebras WSE
│
├── cpu/                       # CPU backends
│   ├── __init__.py
│   ├── base.py
│   └── generic.py            # Cross-platform CPU detection
│
├── memory/                    # Memory backends
│   ├── __init__.py
│   └── ram.py
│
└── storage/                   # Storage backends
    ├── __init__.py
    ├── base.py
    ├── factory.py
    └── local.py
```

## Core Concepts

### Hardware Categories

```text
Hardware
├── Accelerators (Primary focus for vendor partnerships)
│   ├── GPUs
│   │   ├── NVIDIA (CUDA)
│   │   ├── AMD (ROCm)
│   │   ├── Intel (oneAPI)
│   │   └── Apple (Metal)
│   ├── NPUs/TPUs
│   │   ├── Google TPU
│   │   ├── Apple Neural Engine
│   │   └── Qualcomm Hexagon
│   └── Custom Accelerators
│       ├── Groq LPU
│       └── Cerebras WSE
│
├── CPU (x86, ARM, RISC-V)
│
├── Memory
│   ├── System RAM
│   ├── GPU Memory (HBM, GDDR)
│   └── Persistent Memory (Optane)
│
└── Storage
    ├── Local Block Devices (NVMe, SATA, HDD)
    ├── Network Storage (NFS, iSCSI)
    ├── Distributed/Parallel (Lustre, GPFS, BeeGFS)
    └── Object Storage (S3, Azure Blob, GCS)
```

### Hardware Discovery Methods

Accelerator discovery uses vendor-specific libraries and platform APIs:

**Vendor Libraries**:

- **NVIDIA**: nvidia-ml-py (NVML wrapper)
- **AMD**: amdsmi (ROCm System Management Interface)
- **Intel**: Level Zero, oneAPI libraries
- **Apple**: IOKit (Metal devices, Neural Engine)
- **Qualcomm**: Hexagon SDK
- **Google**: TPU detection via cloud APIs

**Bus/Interconnect Detection**:

- **PCIe**: GPUs, FPGAs, custom accelerators
- **SXM/NVLink**: High-bandwidth GPU interconnects (NVIDIA)
- **Infinity Fabric**: AMD GPU interconnects
- **Platform-specific**: Apple M-series (integrated), ARM SoC accelerators

**Platform APIs**:

- Linux: `/sys/class/` entries, vendor device drivers
- macOS: IOKit framework for device enumeration
- Windows: WMIC, vendor-specific management interfaces

## Implementation Roadmap

### Phase 1: Accelerator Reorganization (Current Priority)

**Goal**: Refactor existing GPU backends to new unified architecture

- Create `warpt/backends/accelerators/` directory
- Move `nvidia.py`, `amd.py`, `intel.py` to `accelerators/`
- Refactor to implement `AcceleratorBackend` base class
- Update `factory.py` with new functions:
  - `get_all_accelerator_backends()`
  - `get_accelerator_backend(vendor=None)`
  - `get_accelerator_backend_by_index(index)`
  - `get_backend_count()`
  - `list_available_vendors()`
- Update commands to use new factory functions
- Update data models (`AcceleratorInfo` replaces vendor-specific models)

**Deliverables**:

- Clean migration of existing NVIDIA support
- Framework for vendor partnerships
- Backward compatibility maintained

### Phase 2: Initial Vendor Partnerships (Next 3-6 months)

**Goal**: Add 2-3 partner vendor backends

Priority partners:

- **Apple**: M-series GPU + Neural Engine (broad user base)
- **AMD**: ROCm ecosystem (enterprise customers)
- **Qualcomm**: Edge AI/mobile (emerging market)

Per-vendor implementation:

- Implement `AcceleratorBackend` interface
- Add vendor detection via platform APIs
- Integrate with vendor monitoring tools
- Add stress test support
- Documentation and examples

### Phase 3: Expand Vendor Ecosystem (6-12 months)

**Goal**: Add remaining vendor backends

Additional vendors:

- **Google TPU**: Cloud AI workloads
- **Intel**: Data center GPU market
- **Groq**: LPU for inference
- **Cerebras**: Large-scale training

### Phase 4: Enhanced Features

**Goal**: Advanced monitoring and benchmarking

Features:

- Multi-device coordination testing
- Cross-vendor performance comparison
- Power efficiency metrics
- Thermal profiling over time
- Framework-specific benchmarks (PyTorch, TensorFlow, JAX)

### Phase 5: Storage Expansion (Parallel track)

**Goal**: Build out storage backend ecosystem

- Enhanced local storage (NVMe, SATA with SMART)
- Network storage (NFS, iSCSI)
- Distributed storage (Lustre, GPFS)
- Object storage (S3, Azure, GCS)

## Unified Accelerator Interface

### AcceleratorBackend Base Class

All accelerator vendors implement this interface:

```python
# warpt/backends/base.py

from abc import ABC, abstractmethod

class AcceleratorBackend(ABC):
    """Base class for all accelerator vendors.

    This interface supports GPUs, NPUs, TPUs, and custom accelerators
    from any vendor. Each vendor implements this interface to provide
    hardware detection, monitoring, and framework integration.
    """

    # Vendor identification (class attributes)
    VENDOR_NAME: str = "Unknown"  # e.g., "NVIDIA", "AMD", "Apple"
    DEVICE_TYPE: str = "unknown"  # e.g., "gpu", "npu", "tpu", "lpu"

    # Hardware Detection
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this vendor's hardware is present on the system.

        Returns:
            bool: True if at least one device is detected
        """
        pass

    @abstractmethod
    def get_device_count(self) -> int:
        """Get the number of devices from this vendor.

        Returns:
            int: Number of devices detected
        """
        pass

    @abstractmethod
    def list_devices(self) -> list[AcceleratorInfo]:
        """List all devices with their specifications.

        Returns:
            List of AcceleratorInfo objects with device details
        """
        pass

    # Monitoring (for stress tests and real-time monitoring)
    @abstractmethod
    def get_temperature(self, index: int) -> float | None:
        """Get device temperature in degrees Celsius.

        Args:
            index: Device index (0-based)

        Returns:
            Temperature in Celsius, or None if unavailable
        """
        pass

    @abstractmethod
    def get_memory_usage(self, index: int) -> dict | None:
        """Get current device memory usage.

        Args:
            index: Device index (0-based)

        Returns:
            dict with keys:
                - total (int): Total memory in bytes
                - used (int): Used memory in bytes
                - free (int): Free memory in bytes
            Or None if unavailable
        """
        pass

    @abstractmethod
    def get_utilization(self, index: int) -> dict | None:
        """Get device utilization percentages.

        Args:
            index: Device index (0-based)

        Returns:
            dict with keys:
                - compute (float): Compute utilization (0-100)
                - memory (float): Memory bandwidth utilization (0-100)
            Or None if unavailable
        """
        pass

    @abstractmethod
    def get_power_usage(self, index: int) -> float | None:
        """Get current device power usage in Watts.

        Args:
            index: Device index (0-based)

        Returns:
            Power usage in Watts, or None if unavailable
        """
        pass

    @abstractmethod
    def get_throttle_reasons(self, index: int) -> list[str]:
        """Get current device throttling reasons.

        Common throttle reasons:
        - 'thermal' - Temperature limit reached
        - 'power_limit' - Power limit reached
        - 'sw_power_cap' - Software-imposed power cap

        Args:
            index: Device index (0-based)

        Returns:
            List of active throttle reasons, empty list if not throttling
        """
        pass

    # Framework Integration
    @abstractmethod
    def get_framework_device_string(
        self, device_id: int, framework: str = "pytorch"
    ) -> str:
        """Get framework-specific device string for this vendor.

        Examples:
            NVIDIA: "cuda:0"
            AMD: "cuda:0" (with ROCm PyTorch)
            Intel: "xpu:0"
            Apple: "mps"

        Args:
            device_id: Device index (0-based)
            framework: Framework name (default: "pytorch")

        Returns:
            Framework-specific device string
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Cleanup and shutdown vendor-specific libraries.

        Should handle vendor-specific cleanup (e.g., nvmlShutdown()).
        """
        pass


# Example vendor implementations:

class NvidiaBackend(AcceleratorBackend):
    """NVIDIA GPU backend using nvidia-ml-py."""

    VENDOR_NAME = "NVIDIA"
    DEVICE_TYPE = "gpu"

    def __init__(self):
        import pynvml
        pynvml.nvmlInit()

    def get_framework_device_string(self, device_id: int, framework: str = "pytorch") -> str:
        return f"cuda:{device_id}"

    # ... implement other methods


class AppleBackend(AcceleratorBackend):
    """Apple Neural Engine + M-series GPU backend."""

    VENDOR_NAME = "Apple"
    DEVICE_TYPE = "npu"  # Can support both 'npu' and 'gpu'

    def get_framework_device_string(self, device_id: int, framework: str = "pytorch") -> str:
        # Apple uses 'mps' for Metal Performance Shaders
        return "mps"

    # ... implement other methods


class GoogleTPUBackend(AcceleratorBackend):
    """Google TPU backend."""

    VENDOR_NAME = "Google"
    DEVICE_TYPE = "tpu"

    def get_framework_device_string(self, device_id: int, framework: str = "pytorch") -> str:
        # JAX/TensorFlow format
        return f"tpu:{device_id}"

    # ... implement other methods
```

## Auto-Discovery Factory

### Backend Registry and Factory Functions

```python
# warpt/backends/factory.py

from typing import Type
import importlib

from warpt.backends.base import AcceleratorBackend

# Registry of all supported backend classes
# Add new vendors here to enable auto-detection
ACCELERATOR_BACKENDS = [
    "warpt.backends.accelerators.nvidia.NvidiaBackend",
    "warpt.backends.accelerators.amd.AMDBackend",
    "warpt.backends.accelerators.intel.IntelBackend",
    "warpt.backends.accelerators.apple.AppleBackend",
    "warpt.backends.accelerators.qualcomm.QualcommBackend",
    "warpt.backends.accelerators.google.GoogleTPUBackend",
    "warpt.backends.accelerators.groq.GroqBackend",
    "warpt.backends.accelerators.cerebras.CerebrasBackend",
]


def get_all_accelerator_backends() -> list[AcceleratorBackend]:
    """Get all available accelerator backends on this system.

    Attempts to initialize all registered backends and returns those
    that successfully detect hardware. Backends without available hardware
    or missing vendor libraries are silently skipped.

    Returns:
        List of initialized backend instances for detected hardware.
        Empty list if no accelerators are detected.

    Example:
        backends = get_all_accelerator_backends()
        for backend in backends:
            print(f"{backend.VENDOR_NAME}: {backend.get_device_count()} devices")
    """
    available_backends = []

    for backend_path in ACCELERATOR_BACKENDS:
        try:
            # Dynamically import the backend class
            module_path, class_name = backend_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            backend_class: Type[AcceleratorBackend] = getattr(module, class_name)

            # Try to initialize the backend
            backend = backend_class()

            # Only include if hardware is available
            if backend.is_available():
                available_backends.append(backend)

        except (ImportError, AttributeError, NotImplementedError, Exception):
            # Skip backends that:
            # - Have missing vendor libraries (ImportError)
            # - Are not yet implemented (NotImplementedError)
            # - Fail initialization for any reason
            continue

    return available_backends


def get_accelerator_backend(vendor: str | None = None) -> AcceleratorBackend:
    """Get a specific vendor backend or auto-detect the first available.

    Args:
        vendor: Vendor name (case-insensitive) or None for auto-detect.
                Examples: "nvidia", "amd", "apple", "google"

    Returns:
        Initialized AcceleratorBackend instance

    Raises:
        RuntimeError: If no accelerators found or specified vendor not available

    Examples:
        # Auto-detect first available accelerator
        backend = get_accelerator_backend()

        # Get specific vendor
        nvidia_backend = get_accelerator_backend(vendor="nvidia")
        apple_backend = get_accelerator_backend(vendor="apple")
    """
    available_backends = get_all_accelerator_backends()

    if not available_backends:
        raise RuntimeError(
            "No accelerators detected on this system. "
            "Please ensure accelerator drivers are installed and "
            "hardware is properly configured."
        )

    # If vendor specified, find matching backend
    if vendor:
        vendor_lower = vendor.lower()
        for backend in available_backends:
            if backend.VENDOR_NAME.lower() == vendor_lower:
                return backend

        # Vendor requested but not found
        available_vendors = [b.VENDOR_NAME for b in available_backends]
        raise RuntimeError(
            f"No {vendor} accelerators detected. "
            f"Available vendors: {', '.join(available_vendors)}"
        )

    # Return first available backend
    return available_backends[0]


def get_accelerator_backend_by_index(index: int) -> AcceleratorBackend:
    """Get accelerator backend by index from available backends.

    Backends are ordered by detection priority (typically NVIDIA → AMD → Intel
    → Apple → others). Use this when you want to select a specific backend
    from the available set.

    Args:
        index: Backend index (0-based) from the list of available backends

    Returns:
        Initialized AcceleratorBackend instance at the specified index

    Raises:
        RuntimeError: If no accelerators found
        IndexError: If index is out of range

    Examples:
        # Get second available backend
        backend = get_accelerator_backend_by_index(1)

        # Iterate through all backends
        backends = get_all_accelerator_backends()
        for i in range(len(backends)):
            backend = get_accelerator_backend_by_index(i)
            print(f"{backend.VENDOR_NAME}: {backend.get_device_count()} devices")
    """
    available_backends = get_all_accelerator_backends()

    if not available_backends:
        raise RuntimeError(
            "No accelerators detected on this system. "
            "Please ensure accelerator drivers are installed and "
            "hardware is properly configured."
        )

    if index < 0 or index >= len(available_backends):
        raise IndexError(
            f"Backend index {index} out of range. "
            f"Available backends: {len(available_backends)} "
            f"(indices 0-{len(available_backends) - 1})"
        )

    return available_backends[index]


def get_backend_count() -> int:
    """Get the number of available accelerator backends.

    Returns:
        Number of detected backends (vendor types, not device count)

    Example:
        count = get_backend_count()
        print(f"Detected {count} different accelerator vendor(s)")
    """
    return len(get_all_accelerator_backends())


def list_available_vendors() -> list[str]:
    """Get list of vendor names for all available backends.

    Returns:
        List of vendor names (e.g., ["NVIDIA", "Apple"])

    Example:
        vendors = list_available_vendors()
        print(f"Available accelerators: {', '.join(vendors)}")
    """
    backends = get_all_accelerator_backends()
    return [backend.VENDOR_NAME for backend in backends]
```

## Data Models

### AcceleratorInfo Model

```python
# warpt/models/list_models.py

from pydantic import BaseModel

class AcceleratorInfo(BaseModel):
    """Universal accelerator device information.

    This model works for GPUs, NPUs, TPUs, and custom accelerators
    from any vendor.
    """

    # Vendor Identification
    vendor: str                    # "NVIDIA", "AMD", "Intel", "Apple", etc.
    device_type: str               # "gpu", "npu", "tpu", "lpu", "wse"
    index: int                     # Device index (0-based)
    model: str                     # "RTX 4090", "MI300X", "H100", "M2 Ultra"

    # Compute Capability
    compute_capability: str | None  # Vendor-specific (e.g., "8.9" for CUDA)
    memory_gb: int | None           # Device memory in GB

    # Interconnect
    bus_type: str | None           # "pcie", "sxm", "nvlink", "infinity_fabric"
    pcie_gen: int | None           # PCIe generation (3, 4, 5)
    bandwidth_gbps: float | None   # Interconnect bandwidth

    # Software
    driver_version: str | None
    firmware_version: str | None

    # Vendor-specific extras
    # Use this for vendor-specific metrics that don't fit the common schema
    extra_info: dict | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "vendor": "NVIDIA",
                "device_type": "gpu",
                "index": 0,
                "model": "NVIDIA H100 PCIe",
                "compute_capability": "9.0",
                "memory_gb": 80,
                "bus_type": "pcie",
                "pcie_gen": 5,
                "bandwidth_gbps": 128.0,
                "driver_version": "535.154.05",
                "extra_info": {
                    "cuda_cores": 14592,
                    "sm_count": 114,
                    "tensor_cores": 456
                }
            }
        }


class HardwareInfo(BaseModel):
    """Complete hardware inventory."""

    cpu: CPUInfo | None = None
    memory: MemoryInfo | None = None

    # Accelerators (replaces gpu-only model)
    accelerators: list[AcceleratorInfo] | None = None
    accelerator_count: int | None = None

    storage: list[StorageDeviceInfo] | None = None
```

### StorageDeviceInfo Model

```python
class StorageDeviceInfo(BaseModel):
    """Comprehensive storage device information."""

    # Identification
    device_path: str          # /dev/nvme0n1, /dev/sda
    model: str                # "Samsung 990 Pro 2TB"
    serial: str | None        # Serial number
    firmware: str | None      # Firmware version

    # Classification
    device_type: StorageType  # nvme_ssd, sata_ssd, hdd, network, object
    bus_type: str             # pcie, sata, sas, usb, network

    # Capacity
    capacity_bytes: int
    capacity_gb: int          # Convenience field

    # Performance (if available)
    max_read_mbps: int | None
    max_write_mbps: int | None
    max_iops: int | None

    # Health (if SMART available)
    health_status: str | None        # "healthy", "warning", "critical"
    temperature_celsius: int | None
    wear_level_percent: int | None   # For SSDs
    power_on_hours: int | None

    # Mount information
    mount_points: list[str] | None   # ["/", "/home"]
    filesystem: str | None           # ext4, xfs, apfs, ntfs
```

## Command Integration Examples

### Updated List Command

```python
# warpt/commands/list_cmd.py

def run_list(export_format=None, export_filename=None) -> None:
    """Display comprehensive hardware information."""

    # ... existing CPU/RAM code ...

    # Accelerator Detection (replaces GPU-only detection)
    print("\nAccelerators:")
    try:
        from warpt.backends.factory import (
            get_all_accelerator_backends,
            list_available_vendors
        )

        backends = get_all_accelerator_backends()

        if not backends:
            print("  No accelerators detected")
        else:
            all_accelerators = []

            for backend in backends:
                devices = backend.list_devices()
                print(f"\n  {backend.VENDOR_NAME} {backend.DEVICE_TYPE.upper()}s:")

                for device in devices:
                    print(f"    [{device.index}] {device.model}")
                    if device.memory_gb:
                        print(f"        Memory:         {device.memory_gb} GB")
                    if device.compute_capability:
                        print(f"        Compute Cap:    {device.compute_capability}")
                    if device.driver_version:
                        print(f"        Driver:         {device.driver_version}")

                    all_accelerators.append(device)

            # Build output model with all accelerators
            hardware = HardwareInfo(
                cpu=cpu_model,
                memory=memory_info,
                accelerators=all_accelerators,
                accelerator_count=len(all_accelerators),
                storage=None,
            )

    except Exception as e:
        print(f"  Accelerator detection failed: {e}")
```

### Updated Stress Command

```python
# warpt/commands/stress_cmd.py

# Valid targets now include vendor names
VALID_STRESS_TARGETS = (
    "cpu", "ram",
    "nvidia", "amd", "intel", "apple", "qualcomm", "google", "groq", "cerebras",
    "all"
)

def run_stress(...):
    """Run stress tests based on user specifications."""

    # ... existing validation ...

    # For accelerator targets
    accelerator_vendors = ["nvidia", "amd", "intel", "apple", "qualcomm",
                          "google", "groq", "cerebras"]

    for target in parsed_targets:
        if target in accelerator_vendors:
            from warpt.backends.factory import get_accelerator_backend
            from warpt.stress.accelerator_compute import AcceleratorMatMulTest

            try:
                backend = get_accelerator_backend(vendor=target)
            except RuntimeError as e:
                print(f"⚠️  Skipping {target}: {e}")
                continue

            # Test each device from this vendor
            for device_id in device_ids or range(backend.get_device_count()):
                print(f"=== {backend.VENDOR_NAME} {target.upper()} {device_id} Stress Test ===\n")

                test = AcceleratorMatMulTest(
                    backend=backend,
                    device_id=device_id,
                    burnin_seconds=burnin_seconds
                )
                results = test.run(duration=test_duration)

                # Display results
                print(f"\nResults for {backend.VENDOR_NAME} {device_id}:")
                print(f"  Device:             {results.get('device_name', 'Unknown')}")
                print(f"  Performance:        {results['tflops']:.2f} TFLOPS")
                print(f"  Duration:           {results['duration']:.2f}s")
                # ... more results ...
```

## Implementation Best Practices

### Error Handling

Hardware detection should never crash the application:

```python
def get_all_accelerator_backends() -> list[AcceleratorBackend]:
    """Robust error handling for vendor libraries."""
    available_backends = []

    for backend_path in ACCELERATOR_BACKENDS:
        try:
            backend = load_and_initialize_backend(backend_path)
            if backend.is_available():
                available_backends.append(backend)
        except ImportError:
            # Vendor library not installed - skip silently
            continue
        except NotImplementedError:
            # Backend stubbed but not implemented - skip silently
            continue
        except Exception as e:
            # Unexpected error - log but don't crash
            logger.debug(f"Failed to load {backend_path}: {e}")
            continue

    return available_backends
```

**Error Scenarios**:

- **Missing vendor library** → Skip silently (ImportError)
- **Hardware not present** → `is_available()` returns False
- **Permission denied** → Skip device with warning
- **Driver not loaded** → Return partial info where possible

### Platform-Specific Considerations

**Linux**:

- Most vendor libraries available via pip
- PCIe device enumeration via `/sys/bus/pci/devices/`
- May require elevated permissions for some metrics

**macOS**:

- Apple hardware via IOKit (native)
- External GPUs via Thunderbolt detection
- Metal framework for Apple GPU/Neural Engine

**Windows**:

- NVIDIA, AMD, Intel via Windows binaries
- WMIC for general device enumeration
- PowerShell for detailed metrics

### Monitoring and Stress Testing

Accelerator stress tests focus on:

| Metric | Description | Use Case |
|--------|-------------|----------|
| TFLOPS | Floating-point operations/sec | Compute performance |
| Memory Bandwidth | GB/s sustained transfer | Data-intensive workloads |
| Temperature | Celsius under load | Thermal design validation |
| Power Usage | Watts consumed | Efficiency metrics |
| Throttling | Thermal/power limits | Stability testing |

**Stress Test Pattern**:

1. **Warmup**: 30s burnin period to reach steady state
1. **Sustained Load**: User-defined duration (default: 60s)
1. **Monitoring**: Real-time metrics collection
1. **Results**: Aggregate performance and stability metrics

### Thread Safety

Vendor libraries have varying thread safety:

- **NVIDIA NVML**: Thread-safe after initialization
- **AMD ROCm**: Check vendor documentation
- **Intel Level Zero**: Generally thread-safe

**Recommendation**: Initialize backends in main thread, use locks for metric collection if running parallel stress tests.

## Vendor Partnership Model

### Benefits for Hardware Vendors

Integrating with warpt provides accelerator vendors with:

1. **Easy Detection**: Automatic hardware discovery in user environments
1. **Standardized Listing**: Consistent device information display
1. **Built-in Stress Testing**: Ready-to-use benchmarking for validation
1. **Framework Integration**: PyTorch/TensorFlow device string generation
1. **Community Visibility**: Exposure to warpt user base

### Integration Requirements

To add vendor support, implement:

**Minimum Requirements** (for listing):

- `is_available()` - Hardware detection
- `get_device_count()` - Count devices
- `list_devices()` - Device specifications

**Full Support** (for stress testing):

- All monitoring methods (temperature, memory, utilization, power)
- Framework device string generation
- Throttle detection

### Vendor Implementation Template

```python
# warpt/backends/accelerators/example_vendor.py

from warpt.backends.base import AcceleratorBackend
from warpt.models.list_models import AcceleratorInfo

class ExampleVendorBackend(AcceleratorBackend):
    """Example Vendor accelerator backend."""

    VENDOR_NAME = "ExampleVendor"
    DEVICE_TYPE = "gpu"  # or "npu", "tpu", "lpu", etc.

    def __init__(self):
        """Initialize vendor SDK/library."""
        # Import and initialize vendor library
        # Raise ImportError if library not available
        # Raise NotImplementedError if not yet implemented
        pass

    def is_available(self) -> bool:
        """Check if vendor hardware is present."""
        # Query vendor API for device count
        # Return True if devices found
        return False

    def get_device_count(self) -> int:
        """Get number of vendor devices."""
        # Query vendor API
        return 0

    def list_devices(self) -> list[AcceleratorInfo]:
        """List all vendor devices."""
        devices = []
        for i in range(self.get_device_count()):
            # Query device properties
            device = AcceleratorInfo(
                vendor=self.VENDOR_NAME,
                device_type=self.DEVICE_TYPE,
                index=i,
                model="Device Model Name",
                compute_capability="1.0",
                memory_gb=16,
                bus_type="pcie",
                pcie_gen=4,
                driver_version="1.0.0",
                extra_info={
                    "vendor_specific_metric": 12345,
                }
            )
            devices.append(device)
        return devices

    # ... implement monitoring methods ...

    def get_framework_device_string(
        self, device_id: int, framework: str = "pytorch"
    ) -> str:
        """Get framework device string."""
        # Return vendor-specific device string
        # e.g., "xpu:0", "vendor:0", etc.
        return f"vendor:{device_id}"

    def shutdown(self) -> None:
        """Cleanup vendor library."""
        pass
```

## Benefits of This Architecture

### For Users

1. **Unified Interface**: Same commands work across all vendors
1. **Auto-Discovery**: No manual configuration needed
1. **Comprehensive Monitoring**: Consistent metrics across vendors
1. **Easy Comparison**: Compare accelerators from different vendors

### For Developers

1. **Simple Extension**: Add vendors by implementing one interface
1. **Clear Separation**: Each vendor in isolated module
1. **No Breaking Changes**: New vendors don't affect existing code
1. **Type Safety**: Pydantic models ensure data consistency

### For Vendors

1. **Easy Integration**: ~200 lines of code for full support
1. **Standalone Testing**: Each backend testable independently
1. **Graceful Degradation**: Missing libraries don't break the tool
1. **Marketing Opportunity**: Visibility to warpt users

## Design Principles

### 1. Flat Over Hierarchical

- Each vendor gets one backend file
- No complex inheritance hierarchies
- Easy to understand and maintain

### 2. Graceful Degradation

- Missing vendor libraries → skip silently
- Unavailable hardware → clear error messages
- Partial information → display what's available

### 3. Vendor Independence

- Each backend is self-contained
- No cross-vendor dependencies
- Easy to add/remove vendors

### 4. Factory Pattern

- Single point for backend registration
- Automatic detection and initialization
- Extensible without code changes

## Practical Usage Examples

### Basic Device Listing

```python
# List all available accelerators
from warpt.backends.factory import get_all_accelerator_backends

backends = get_all_accelerator_backends()

for backend in backends:
    print(f"\n{backend.VENDOR_NAME} {backend.DEVICE_TYPE}s:")
    devices = backend.list_devices()

    for device in devices:
        print(f"  [{device.index}] {device.model}")
        print(f"      Memory: {device.memory_gb} GB")
        print(f"      Driver: {device.driver_version}")
```

### Vendor-Specific Backend

```python
# Get NVIDIA backend specifically
from warpt.backends.factory import get_accelerator_backend

try:
    nvidia = get_accelerator_backend(vendor="nvidia")
    print(f"Found {nvidia.get_device_count()} NVIDIA GPUs")

    # Monitor GPU 0
    temp = nvidia.get_temperature(0)
    mem = nvidia.get_memory_usage(0)
    util = nvidia.get_utilization(0)

    print(f"GPU 0: {temp}°C, {util['compute']}% utilized")

except RuntimeError as e:
    print(f"NVIDIA hardware not available: {e}")
```

### Multi-Backend Iteration

```python
# Iterate through backends by index
from warpt.backends.factory import get_backend_count, get_accelerator_backend_by_index

for i in range(get_backend_count()):
    backend = get_accelerator_backend_by_index(i)

    print(f"\nBackend {i}: {backend.VENDOR_NAME}")
    print(f"  Device type: {backend.DEVICE_TYPE}")
    print(f"  Device count: {backend.get_device_count()}")
```

### Framework Integration

```python
# Get PyTorch device strings for stress testing
import torch
from warpt.backends.factory import get_all_accelerator_backends

for backend in get_all_accelerator_backends():
    for i in range(backend.get_device_count()):
        device_str = backend.get_framework_device_string(i, framework="pytorch")

        try:
            device = torch.device(device_str)
            tensor = torch.randn(1000, 1000, device=device)
            print(f"✓ {backend.VENDOR_NAME} device {i} ready: {device_str}")
        except Exception as e:
            print(f"✗ {backend.VENDOR_NAME} device {i} unavailable: {e}")
```

### Comprehensive Monitoring

```python
# Monitor all accelerators in real-time
import time
from warpt.backends.factory import get_all_accelerator_backends

backends = get_all_accelerator_backends()

while True:
    print("\033[2J\033[H")  # Clear screen
    print("Accelerator Monitor\n")

    for backend in backends:
        print(f"\n{backend.VENDOR_NAME}:")

        for i in range(backend.get_device_count()):
            temp = backend.get_temperature(i) or 0.0
            util = backend.get_utilization(i) or {}
            power = backend.get_power_usage(i) or 0.0
            throttle = backend.get_throttle_reasons(i)

            status = "⚠️  THROTTLED" if throttle else "✓ OK"

            print(f"  [{i}] {temp:.1f}°C | "
                  f"{util.get('compute', 0):.0f}% util | "
                  f"{power:.1f}W | {status}")

    time.sleep(1.0)
```

## Summary

This architecture provides:

1. **Clean Vendor Abstraction**: Unified interface for 8+ accelerator vendors
1. **Simple Partnership Model**: Easy integration for new vendors
1. **Robust Auto-Discovery**: Automatic hardware detection across platforms
1. **Incremental Implementation**: Phase-based rollout plan
1. **Future-Proof Design**: Extensible for emerging accelerator types

### Key Factory Functions

```python
# Get all available backends
backends = get_all_accelerator_backends()

# Get specific vendor
nvidia = get_accelerator_backend(vendor="nvidia")

# Get backend by index
first_backend = get_accelerator_backend_by_index(0)

# Count available backend types
count = get_backend_count()

# List vendor names
vendors = list_available_vendors()
```

### Next Steps

1. **Phase 1**: Reorganize existing GPU backends into new structure
1. **Phase 2**: Partner with 2-3 initial accelerator vendors
1. **Phase 3**: Expand to all 8 target vendors
1. **Phase 4**: Add advanced monitoring and benchmarking features

See implementation roadmap above for detailed timeline and deliverables.
