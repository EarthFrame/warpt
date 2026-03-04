<!-- DO NOT EDIT — This file is maintained by the warpt core team. -->

---
title: Adding a Hardware Accelerator Backend
description: Learn how to implement new hardware accelerator backends for warpt.
order: 1
---

# Adding a Hardware Accelerator Backend to warpt

----------

## Overview

`warpt` is meant to be a vendor-agnostic package for power measurement, hardware configuration detection, and stress testing. New hardware support can be added by implementing three classes in `warpt`:

1.  **Accelerator Backend** — The accelerator backend exposes device discovery, memory, utilization, identity, and basic telemetry. This is the core requirement for integration with warpt.
2.  **Power Backend** — Integrates power draw, temperature, and energy telemetry per device. Enables warpt's power profiling capabilities.
3.  **Stress Backend (Optional but Recommended)** — Allows the implementation of stress test workloads so warpt can apply standardized load to your hardware.

An "accelerator" may be a GPU, TPU, NPU, or any other compute accelerator supported by your platform. The sections below walk through each backend type, the required methods, and concrete examples.

## Integration Checklist

| Step | File(s) | Done |
|------|---------|------|
| 1. Implement AcceleratorBackend subclass | warpt/backends/yourvendor.py | ☐ |
| 2. Add factory registration | warpt/backends/factory.py | ☐ |
| 3. Add package to setuptools | pyproject.toml | ☐ |
| 4. Add optional dependency group | pyproject.toml | ☐ |
| 5. Write factory fallthrough test | tests/test_backends_factory.py | ☐ |
| 6. Write backend unit tests (mocked) | tests/test_yourvendor_backend.py | ☐ |
| 7. Verify list_devices() returns valid GPUInfo | Manual test with hardware | ☐ |
| 8. Verify stress tests run on your device | warpt stress -c accelerator | ☐ |
| 9. Verify power readings work | warpt list (check power column) | ☐ |
| 10. (Optional) Implement PowerBackend | warpt/backends/power/yourvendor_power.py | ☐ |
| 11. (Optional) Add custom stress tests | warpt/stress/yourvendor_compute.py | ☐ |
| 12. Submit PR with tests passing | All files | ☐ |

----------

## Architecture Overview

The following diagram shows how warpt's layered architecture separates hardware-specific code from the core logic. Your backend sits at the bottom of this stack:

```
  CLI Commands (warpt stress, warpt list, warpt profile)
           │
  Stress Tests / Profiling Engine / Report Generation
           │
  Backend Abstraction Layer (AcceleratorBackend ABC)
           │
  ┌───────────────┬───────────────┬──────────────────┐
  │ NvidiaBackend │    (empty)    │  YourBackend     │
  │  (pynvml)     │               │  (your SDK)      │
  └───────────────┴───────────────┴──────────────────┘

```

The core never imports vendor-specific libraries directly. All hardware interaction is mediated through the backend abstraction, enabling warpt to support heterogeneous multi-vendor environments.

> **Note:** Only the NVIDIA backend is currently shipped as a complete reference implementation. We use the NvidiaBackend as an example throughout this document. We encourage users to refer to its implementation for guidance.

### Source Code Organization

```
warpt/
├── backends/
│   ├── base.py                 # AcceleratorBackend Abstract Base Class (ABC)
│   ├── factory.py              # Add your try/except block here
│   ├── nvidia.py               # Reference implementation
│   ├── yourvendor.py           # Add this: Implements the AcceleratorBackend ABC
│   └── power/
│       ├── base.py             # PowerBackend ABC
│       ├── factory.py          # Add your power init here
│       ├── nvidia_power.py     # Reference
│       └── yourvendor_power.py # Add this for power readings (optional, but recommended)
├── stress/
│   ├── base.py                 # StressTest ABC
│   ├── gpu_compute.py          
│   └── yourvendor_compute.py   # Add this to enable stress testing on your hardware
├── models/
│   ├── list_models.py          # GPUInfo (Pydantic) — use extra_metrics
│   ├── power_models.py         # Power data contracts (@dataclass)
│   └── profile_models.py       # Profile schemas (Pydantic)

```

----------

## 1. AcceleratorBackend

The accelerator backend is the only required abstract class that must be implemented to enable basic functionality with `warpt`. Implementing this class enables the collection of hardware information and basic telemetry from your hardware.

### 1.1 Responsibilities

- Device discovery and enumeration
- Device identity (model, UUID, PCIe generation, driver version)
- Memory reporting (total, used, free)
- Utilization reporting (compute %, memory %)
- Temperature reading
- Basic power reading (point-in-time wattage)
- Throttle reason reporting
- PyTorch device string mapping for stress test workloads

### 1.2 The AcceleratorBackend Interface

Every accelerator backend must implement the `AcceleratorBackend` Abstract Base Class defined in `warpt/backends/base.py`. This abstract base class defines the complete contract between warpt's core and your hardware SDK.

> **Why "Accelerator Backend"?** warpt uses the term "accelerator" rather than "GPU" because the backend interface is designed to support a variety of compute accelerators — GPUs, TPUs, NPUs, or custom ASICs.

### 1.3 Full Method Reference

The following table lists every abstract method you must implement. All methods should handle vendor SDK errors gracefully and return `None` where indicated rather than raising exceptions.

| Method | Returns | Purpose |
|--------|---------|---------|
| `__init__(self)` | `None` | Initialize your vendor SDK (e.g., `nvmlInit`). Called once when the backend is instantiated. |
| `is_available(self)` | `bool` | Return `True` if at least one of your devices is present and accessible on this system. |
| `get_device_count(self)` | `int` | Return the number of your accelerator devices detected on this system. |
| `list_devices(self)` | `list[GPUInfo]` | Return full device info for every detected device. See GPUInfo schema below. |
| `get_temperature(self, index)` | `float` or `None` | Return the current temperature in Celsius for the device at the given index. |
| `get_memory_usage(self, index)` | `dict` or `None` | Return `{total, used, free}` in bytes for device memory. |
| `get_utilization(self, index)` | `dict` or `None` | Return `{gpu: 0–100, memory: 0–100}` utilization percentages. |
| `get_pytorch_device_string(self, device_id)` | `str` | Return the PyTorch device string (e.g., `"cuda:0"`, `"xpu:0"`, `"tt:0"`). |
| `get_power_usage(self, index)` | `float` or `None` | Return current power draw in Watts for the device at the given index. |
| `get_throttle_reasons(self, index)` | `list[str]` | Return active throttle reasons: `["thermal", "power_limit", ...]`. Empty list if none. |
| `get_driver_version(self)` | `str` or `None` | Return the vendor driver version string. |
| `shutdown(self)` | `None` | Clean up vendor SDK resources. Called when warpt exits. |

### 1.4 GPUInfo Data Contract

Defined in `warpt/models/list_models.py`. This is the Pydantic model returned by `list_devices()`:

```python
class GPUInfo(BaseModel):
    index: int                            # 0-based device index
    model: str                            # e.g., "NVIDIA RTX 4090"
    memory_gb: int                        # Total device memory in GB
    uuid: str | None                      # Persistent unique device ID
    compute_capability: str | None        # e.g., "8.9" (None if N/A)
    pcie_gen: int | None                  # PCIe generation (3, 4, 5)
    driver_version: str | None            # Vendor driver version
    extra_metrics: dict[str, Any] | None  # Vendor-specific extras

```

The `extra_metrics` field is your extension point. Use it for vendor-specific data that doesn't map to the standard fields (e.g., board type, firmware version, ethernet port count, ASIC temperature breakdown). The profiling engine stores this data alongside the standard fields.

> **Note on UUIDs (Highly Recommended):** If your platform exposes a stable, unique device identifier, we recommend populating the `uuid` field. A persistent hardware ID allows warpt to more accurately track the number and identity of accelerators in a system and enables future features such as per-device telemetry. If your platform does not provide a stable identifier, return `None`.

### 1.5 Minimal Implementation Skeleton

Replace the placeholders with your vendor SDK calls:

```python
# warpt/backends/yourvendor.py

from warpt.backends.base import AcceleratorBackend
from warpt.models.list_models import GPUInfo

class YourVendorBackend(AcceleratorBackend):
    """Backend for YourVendor accelerators."""

    def __init__(self):
        # Initialize your vendor SDK here.
        # This is called once when the backend is first created.
        # Equivalent to pynvml.nvmlInit() in the NVIDIA backend.
        your_sdk.init()

    def is_available(self) -> bool:
        # Return True if at least one device is accessible.
        # The factory calls this to decide whether to use your backend.
        return self.get_device_count() > 0

    def get_device_count(self) -> int:
        # Query your SDK for the number of detected devices.
        return your_sdk.device_count()

    def list_devices(self) -> list[GPUInfo]:
        # Build a GPUInfo object for each device.
        # Use extra_metrics for any vendor-specific data that
        # doesn't map to the standard GPUInfo fields.
        devices = []
        for i in range(self.get_device_count()):
            devices.append(GPUInfo(
                index=i,
                model="YourVendor Accelerator",
                memory_gb=your_sdk.get_memory(i) // (1024**3),
                uuid=your_sdk.get_uuid(i),
                compute_capability=None,
                pcie_gen=your_sdk.get_pcie_gen(i),
                driver_version=self.get_driver_version(),
                extra_metrics={
                    "board_type": your_sdk.get_board_type(i),
                    "firmware_version": your_sdk.get_fw_ver(i),
                },
            ))
        return devices

    def get_temperature(self, index: int) -> float | None:
        return your_sdk.get_temp(index)

    def get_memory_usage(self, index: int) -> dict | None:
        return {"total": ..., "used": ..., "free": ...}

    def get_utilization(self, index: int) -> dict | None:
        return {"gpu": ..., "memory": ...}

    def get_pytorch_device_string(self, device_id: int) -> str:
        # Must match what PyTorch uses for your hardware.
        return f"your_device:{device_id}"

    def get_power_usage(self, index: int) -> float | None:
        # Return power in Watts. Convert if your SDK
        # reports in milliwatts (like NVIDIA does).
        return your_sdk.get_power(index)

    def get_throttle_reasons(self, index: int) -> list[str]:
        return []  # Implement if your SDK exposes this

    def get_driver_version(self) -> str | None:
        return your_sdk.get_driver_version()

    def shutdown(self):
        your_sdk.cleanup()

```

### 1.6 Defining a Custom Device Info Model

For most backends, `GPUInfo` with `extra_metrics` is sufficient. You can store any vendor-specific data in that dict field and it will be persisted alongside the standard fields. No new model required.

However, if your hardware has fundamental properties that don't map to any existing `GPUInfo` field — for example, a TPU with `tpu_version` and `mesh_topology`, or a quantum backend with `qubit_count` and `coherence_time_us` — you should define a custom device info model. Typed fields give you validation, autocompletion, and self-documenting schemas, which are better than opaque dicts for anything that's core to your hardware's identity.

To define a custom model:

1. Subclass `GPUInfo` (or a future `DeviceInfo` base) in `warpt/models/list_models.py`:
```python
    class TPUInfo(GPUInfo):
        tpu_version: int
        mesh_topology: str
        hbm_bandwidth_gbps: float | None = None
```

2. Register it in `warpt/models/__init__.py` and add it to the `__all__` export.

3. Update `HardwareInfo` in `warpt/models/list_models.py` to accept your new model type.

4. In your backend subclass, you can type `list_devices()` as returning `list[TPUInfo]` instead of `list[GPUInfo]`. This works because `TPUInfo` is a subclass of `GPUInfo`, so it has all the same fields plus your extras. The rest of warpt treats them as normal `GPUInfo` objects and continues to work without modification.

Only define a custom model when `extra_metrics` genuinely falls short. If you find yourself putting the same keys into `extra_metrics` on every single device and validating them manually, that's a signal to promote them to typed fields on a proper subclass.

----------

## 2. Power Backend

The power backend provides advanced continuous power monitoring beyond the simple `get_power_usage()` method on the accelerator backend. It is separate from the accelerator backend and optional, but required for vendors who want full warpt power profiling support.

### 2.1 Responsibilities

- Continuous power telemetry (not just point-in-time)
- Domain-level power breakdown (GPU core, DRAM, voltage regulator, etc.)
- Energy accumulation (joules over time)
- Correlated thermal data (temperature alongside power readings)

### 2.2 How It Differs from get_power_usage()

| | get_power_usage() (Accelerator Backend) | Power Backend |
|---|---|---|
| **Scope** | Single wattage reading | Full domain breakdown |
| **Frequency** | On-demand, point-in-time | Continuous sampling |
| **Data** | float (watts) | PowerSnapshot with domains, processes, metadata |
| **Required?** | Yes (part of AcceleratorBackend) | Optional |

### 2.3 Power Data Contracts

Defined in `warpt/models/power_models.py`. Note: these are plain Python `@dataclass` classes, **not** Pydantic models — they do not have `.model_dump()` or `.model_validate()` methods.

```python
@dataclass
class GPUPowerInfo:
    index: int                                # Device index
    name: str                                 # Device model name
    power_watts: float                        # Current power draw
    power_limit_watts: float | None           # TDP / power cap
    utilization_percent: float                # Compute utilization
    memory_utilization_percent: float          # Memory utilization
    temperature_celsius: float | None         # Current temp
    processes: list[dict[str, Any]]           # Running processes
    metadata: dict[str, Any]                  # Vendor extras

@dataclass
class PowerSnapshot:
    timestamp: float                          # Unix timestamp
    total_power_watts: float | None           # System total
    domains: list[DomainPower]                # Per-domain breakdown
    gpus: list[GPUPowerInfo]                  # Per-device readings
    processes: list[ProcessPower]             # Per-process power
    platform: str                             # OS identifier
    available_sources: list[PowerSource]      # Active sources

```

### 2.4 Implementation

Your backend should subclass `PowerBackend` defined in `warpt/backends/power/base.py`. See `warpt/backends/power/nvidia_power.py` as the reference implementation. Your file goes in:

```
warpt/backends/power/yourvendor_power.py

```

Register it in `warpt/backends/power/factory.py` by adding your power source to the `PowerMonitor.initialize()` method.

----------

## 3. Stress Backend

warpt's stress test system is backend-aware but backend-agnostic. The core orchestrates all workloads; your backend provides the hardware access layer. Implementing custom stress tests is optional but recommended for vendors whose hardware may not have a standard PyTorch device integration.

### 3.1 How Stress Tests Use Backends

Stress tests subclass `StressTest` (`warpt/stress/base.py`). They receive a `AcceleratorBackend` instance via constructor injection and use it primarily for one thing: getting the correct PyTorch device string via `get_pytorch_device_string()`.

```python
class StressTest(ABC):
    @abstractmethod def get_pretty_name(self) -> str
    @abstractmethod def get_description(self) -> str
    @abstractmethod def get_category(self) -> TestCategory
    @abstractmethod def is_available(self) -> bool
    @abstractmethod def validate_configuration(self) -> None
    @abstractmethod def setup(self) -> None
    @abstractmethod def teardown(self) -> None
    @abstractmethod def execute_test(self, duration, iterations) -> Any

    def run(self, duration, iterations=1) -> Any:
        self.validate_configuration()
        self.setup()
        try:
            self.warmup()
            return self.execute_test(duration, iterations)
        finally:
            self.teardown()

```

### 3.2 Two Paths for Stress Test Support

**Path A: Your hardware has a PyTorch integration.** If PyTorch recognizes your device string (e.g., `"tt:0"` via `torch.compile(backend='tt')`), then existing warpt stress tests (matrix multiply, memory bandwidth, etc.) work automatically. Your `get_pytorch_device_string()` just returns the right string.

**Path B: Your hardware does not have a PyTorch integration.** You write custom stress test classes in `warpt/stress/` that exercise your hardware through its native SDK. Same `StressTest` ABC interface, just different internals in `execute_test()`.

### 3.3 Test Discovery

The `TestRegistry` (`warpt/stress/registry.py`) auto-scans `warpt/stress/*.py` for non-abstract `StressTest` subclasses. Any new test file you add is automatically discovered — no registration required.

### 3.4 Custom Result Models

If you take Path B specified in sub-section 3.2 and write custom stress tests using your native SDK, your `execute_test()` method can return any object — the return type is `Any`. However, for structured results that integrate cleanly with warpt's reporting and profiling pipeline, you should define a custom result model.

Follow the same Pydantic `BaseModel` pattern used by `GPUDeviceResult` in `warpt/models/stress_models.py`:
```python
# warpt/models/stress_models.py

class YourVendorStressResult(BaseModel):
    device_index: int
    test_name: str
    duration_seconds: float
    operations_per_second: float
    peak_power_watts: float | None = None
    peak_temperature_celsius: float | None = None
    vendor_metrics: dict[str, Any] | None = None  # SDK-specific extras
```

The results collector serializes whatever `execute_test()` returns, so as long as your model is JSON-serializable (which Pydantic models are by default), it will flow through warpt's reporting layer correctly.

Add your model to `warpt/models/stress_models.py` and export it from `warpt/models/__init__.py`.

----------

## 4. Registration: Wiring Your Backend In

warpt currently uses a hardcoded priority chain for backend discovery. There is no plugin registry or entry-point system. To add your backend, you modify two files: `warpt/backends/factory.py` (backend registration) and `pyproject.toml` (package registration).

### 4.1 Backend Factory

File: `warpt/backends/factory.py`

The factory tries each vendor backend in priority order. The first backend where `is_available()` returns `True` wins. Add a new `try/except` block for your backend:

```python
def get_gpu_backend() -> AcceleratorBackend:
    # Priority 1: NVIDIA (reference implementation)
    try:
        from warpt.backends.nvidia import NvidiaBackend
        backend = NvidiaBackend()
        if backend.is_available():
            return backend
    except Exception:
        pass

    # ← ADD YOUR BACKEND HERE
    # Priority N: YourVendor
    try:
        from warpt.backends.yourvendor import YourVendorBackend
        backend = YourVendorBackend()
        if backend.is_available():
            return backend
    except Exception:
        pass

    raise RuntimeError("No accelerators detected on this system. ...")

```

The lazy import pattern (importing inside the `try` block) is intentional. It means your vendor SDK is only imported if earlier backends fail, so users without your SDK installed won't see `ImportError`s.

### 4.2 Package Registration

Add your backend's package path to the `[tool.setuptools] packages` list in `pyproject.toml` so it gets included in builds.

----------

## 5. Dependencies and Packaging

Currently, `nvidia-ml-py` is a hard dependency in `pyproject.toml`. For multi-vendor support, vendor SDKs may be declared as optional extras so that users only install the dependencies they need.

### 5.1 Current Structure

```toml
[project]
dependencies = [
    "click>=8.0.0",
    "psutil>=5.9.0",
    "pydantic>=2.0.0",
    "nvidia-ml-py>=12.0.0",   # Currently hard dep
]

[project.optional-dependencies]
dev = ["pytest>=7.0.0", "black>=22.0.0", "ruff>=0.1.0"]
stress = ["torch>=2.0.0", "numpy>=1.24.0"]

```

### 5.2 Recommended Pattern for New Backends

When submitting your backend, declare your vendor SDK as an optional dependency group:

```toml
[project.optional-dependencies]
nvidia = ["nvidia-ml-py>=12.0.0"]
tenstorrent = ["pyluwen>=0.7.11", "tt-tools-common>=1.4.28"]
stress = ["torch>=2.0.0", "numpy>=1.24.0"]

```

This lets users install only what they need: `pip install warpt[tenstorrent]`

----------

## 6. Testing Without Hardware

warpt's test suite uses mock-based testing so that backend tests can run in CI without physical hardware. Follow this pattern for your backend tests.

### 6.1 Factory Fallthrough Test Pattern

From `tests/test_backends_factory.py` — the key pattern is using `patch.dict(sys.modules, ...)` to mock vendor imports and `MagicMock()` for backend instances:

```python
def test_get_gpu_backend_yourvendor():
    """Mock YourVendorBackend, verify factory returns it."""
    # Mock all higher-priority backends as unavailable
    mock_nvidia = MagicMock()
    mock_nvidia.NvidiaBackend.side_effect = RuntimeError("No NVML")

    # Mock your backend as available
    mock_yours = MagicMock()
    mock_instance = MagicMock()
    mock_instance.is_available.return_value = True
    mock_yours.YourVendorBackend.return_value = mock_instance

    with patch.dict(sys.modules, {
        "warpt.backends.nvidia": mock_nvidia,
        "warpt.backends.yourvendor": mock_yours,
    }):
        backend = get_gpu_backend()
        assert backend == mock_instance

```

### 6.2 Key Test Files


| Test File | What It Tests |
|-----------|---------------|
| test_backends_factory.py | Factory priority chain and fallthrough logic |
| test_nvidia_toolkit_detection.py | Mocking CLI tools (shutil.which, subprocess.run) |
| test_stress_registry.py | Test auto-discovery |
| test_stress_results.py | Result collection and emission |

----------

## 7. Runtime Backend Selection

Currently, warpt uses auto-detection only. There are no CLI flags, config files, or environment variables for selecting a backend. The factory tries backends in priority order and uses the first one where `is_available()` returns `True`.

This means your backend will be used automatically on systems where your hardware is present and higher-priority backends are not available. On systems with multiple vendor accelerators, the factory priority order determines which backend wins.

> **Future improvement:** warpt may add explicit backend selection via a `--backend` CLI flag or `WARPT_BACKEND` environment variable. If you need to test your backend on a system that also has NVIDIA GPUs, you can temporarily reorder the priority chain in `factory.py` during development.

----------

## 8. Glossary

A quick-reference for warpt-specific terms used throughout this document.

### Core Abstractions

**AcceleratorBackend** — The primary abstract base class every hardware backend must subclass. Defines the full contract for device discovery, telemetry, and power reading. (`warpt/backends/base.py`)

**PowerBackend** — The abstract base class for continuous power monitoring. Distinct from the point-in-time `get_power_usage()` on AcceleratorBackend. (`warpt/backends/power/base.py`)

**StressTest** — The abstract base class for workload tests. Defines the lifecycle: `validate_configuration()` → `setup()` → `warmup()` → `execute_test()` → `teardown()`. (`warpt/stress/base.py`)

**TestCategory** — Enum that classifies stress tests: `CPU`, `ACCELERATOR`, `RAM`, `STORAGE`, `NETWORK`. Note: warpt uses "accelerator" rather than "gpu". (`warpt/stress/base.py`)

### Data Contracts

**GPUInfo** — Pydantic model returned by `list_devices()`. Contains device identity, memory, PCIe, and driver info. (`warpt/models/list_models.py`)

**extra_metrics** — The `dict[str, Any]` field on GPUInfo for vendor-specific data that doesn't map to standard fields (e.g., board type, firmware version).

**PowerSnapshot** — Top-level dataclass for a complete power reading. Contains timestamp, total power, per-domain breakdown, per-device readings, and per-process power. (`warpt/models/power_models.py`)

**GPUPowerInfo** — Per-device power dataclass with power draw, utilization, temperature, and running processes. (`warpt/models/power_models.py`)

**DomainPower** — Power reading for a single measurement domain (package, core, GPU, DRAM, etc.). (`warpt/models/power_models.py`)

**PowerDomain** — Enum for power measurement domains: `PACKAGE`, `CORE`, `UNCORE`, `DRAM`, `GPU`, `ANE`, `PSYS`. (`warpt/models/power_models.py`)

**PowerSource** — Enum for measurement source: `RAPL`, `POWERMETRICS`, `NVML`, `ROCM_SMI`, etc. (`warpt/models/power_models.py`)

### Architecture & Patterns

**Backend factory** — The `get_accelerator_backend()` function that tries vendor backends in priority order. The first backend where `is_available()` returns `True` wins. (`warpt/backends/factory.py`)

**Lazy import** — The `try: from ... import; except: pass` pattern used in the factory so vendor SDKs are only imported when needed. Users without a given SDK installed won't see `ImportError`s.

**TestRegistry** — Auto-discovery system that scans `warpt/stress/*.py` for non-abstract `StressTest` subclasses. New test files are discovered automatically with no manual registration. (`warpt/stress/registry.py`)

**Device string** — The PyTorch device identifier (e.g., `"cuda:0"`, `"xpu:0"`) returned by `get_pytorch_device_string()`. Used by stress tests to place workloads on the correct device.

### Concepts

**Accelerator** — warpt's vendor-neutral term for any compute accelerator: GPU, TPU, NPU, or custom ASIC. This is why the base class is named `AcceleratorBackend` rather than `GPUBackend`.

**Burnin / Warmup** — The pre-measurement phase in stress tests (`burnin_seconds`). Runs the workload briefly before collecting results to ensure the device reaches steady state.

**Throttle reasons** — The list of strings returned by `get_throttle_reasons()` indicating performance-limiting conditions (e.g., `"thermal"`, `"power_limit"`). Empty list if the device is running unrestricted.

----------

_EarthFrame Inc. | earthframe.com | Questions? Open an issue on the warpt repository._
