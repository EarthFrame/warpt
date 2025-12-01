# warpt: Hardware Performance Made Simple

**W**orkload **A**nalysis & **R**esource **P**rofiling **T**ool

## What is warpt?

warpt is a unified command-line tool for understanding, testing, and validating computational hardware. It answers the fundamental questions that everyone working with high-performance computing asks daily:

- **"What hardware do I actually have?"** - Hardware discovery and enumeration
- **"Is it working correctly?"** - Health checks and diagnostics
- **"How fast is it?"** - Benchmarking and performance measurement
- **"Can it handle the load?"** - Stress testing and reliability validation
- **"What software can use it?"** - Framework and driver detection

Whether you're provisioning a cloud instance, debugging a training run, validating a new workstation, or investigating performance issues, warpt provides the answers you need.

## The Problem We're Solving

Modern computing hardware is increasingly heterogeneous and complex:

- **Multiple vendors**: NVIDIA, AMD, Intel, Apple, Qualcomm, Google TPU, Groq, Cerebras
- **Multiple accelerator types**: GPUs, TPUs, NPUs, custom AI accelerators
- **Fragmented tooling**: Each vendor has different tools, APIs, and diagnostic utilities
- **Software compatibility maze**: PyTorch vs TensorFlow, CUDA vs ROCm vs oneAPI
- **Opaque cloud instances**: You ordered "GPU compute" but what did you actually get?

Current solutions are vendor-specific, require deep expertise, or provide incomplete information. Engineers waste time stitching together outputs from `nvidia-smi`, `rocm-smi`, `lscpu`, framework imports, and vendor-specific tools.

**warpt provides a single, unified interface to understand and validate your entire computational environment.**

## Who Uses warpt?

### Machine Learning Engineers

*"I need to know if my training environment is set up correctly."*

- Verify GPU availability and driver compatibility before long training runs
- Check that PyTorch/TensorFlow can actually see the GPUs
- Quickly diagnose why distributed training isn't using all GPUs
- Validate cloud instances match what you paid for

### Research Scientists

*"I'm running experiments across different systems and need consistent measurements."*

- Compare hardware performance across lab machines, clusters, and cloud
- Document exact hardware specifications for reproducibility
- Detect thermal throttling or performance degradation during long runs
- Export hardware specs in standard formats for paper methods sections

### DevOps & Platform Engineers

*"I maintain infrastructure and need to verify system health at scale."*

- Automate hardware validation in CI/CD pipelines
- Generate machine-readable reports for monitoring systems
- Stress test new hardware before production deployment
- Detect failing GPUs or degraded performance across fleets

### HPC System Administrators

*"I manage clusters and need to diagnose hardware issues quickly."*

- Rapidly identify problematic nodes in large clusters
- Validate storage performance (local NVMe, Lustre, GPFS, BeeGFS)
- Test system stability after driver updates
- Generate health reports across heterogeneous hardware

### Hardware Vendors & Partners

*"We want our accelerators to work seamlessly with the ML ecosystem."*

- Implement the `AcceleratorBackend` interface to add hardware support
- Provide users with consistent diagnostics across all hardware
- Join a neutral, vendor-agnostic ecosystem
- Benefit from community-maintained tooling

## Core Philosophy

### Unified Interface

One tool, one syntax, all hardware. Whether you're checking an NVIDIA GPU, AMD Radeon, Apple Neural Engine, or Google TPU, the commands are the same:

```bash
warpt list          # What do I have?
warpt check         # Is it working?
warpt benchmark     # How fast is it?
warpt stress        # Can it handle load?
```

### Vendor Extensibility

Hardware evolves faster than tools. warpt's backend architecture lets vendors add support for new accelerators without changing the core tool. Implement the `AcceleratorBackend` interface and you're in.

### Graceful Degradation

Not all information is available in all environments. warpt works with what it can find:

- No vendor drivers? Fall back to PCIe detection
- Missing permissions? Report what's accessible
- Cloud with limited visibility? Show what the hypervisor exposes

### Truth Over Marketing

You ordered an "A100 instance" but got a 40GB variant when you needed 80GB. warpt shows you what you *actually have*, not what the sales page promised.

### Developer-Friendly

- **JSON/YAML/TOML output** for scripting and automation
- **Python API** for integration into workflows
- **Exit codes** that make sense in CI/CD
- **Comprehensive error messages** that help you fix problems

## What Makes warpt Different?

### vs. nvidia-smi / rocm-smi / intel-gpu-top

**Them**: Vendor-specific, different syntax, different output formats\
**warpt**: Unified interface across all vendors

### vs. lshw / lspci / dmidecode

**Them**: Raw hardware enumeration, no performance testing\
**warpt**: Discovery + diagnostics + testing + framework compatibility

### vs. MLPerf

**Them**: Standardized benchmarks, complex setup, research focus\
**warpt**: Quick diagnostics and stress tests, production focus

### vs. Cloud provider dashboards

**Them**: Shows what you're *supposed* to have\
**warpt**: Shows what you *actually* have (and if it works)

## Example Workflows

### Scenario: New Cloud GPU Instance

```bash
# What did I get?
warpt list --format json > hardware.json

# Can PyTorch see it?
warpt check --framework pytorch

# Is it actually an A100?
warpt benchmark gpu --quick

# Can it handle a training run?
warpt stress gpu --duration 5m --monitor
```

**Value**: 2 minutes to full confidence vs. 30 minutes of manual checking

### Scenario: Distributed Training Not Using All GPUs

```bash
# Are all GPUs visible?
warpt list gpu

# Are they all healthy?
warpt check gpu

# Any thermal throttling?
warpt stress gpu --devices 0,1,2,3 --monitor

# Framework compatibility?
warpt check --framework pytorch --verbose
```

**Value**: Quickly isolate hardware vs. software issues

### Scenario: CI/CD Hardware Validation

```yaml
- name: Validate GPU Environment
  run: |
    warpt check gpu --format json | jq '.status == "pass"'
    warpt check --framework pytorch --cuda-required
```

**Value**: Catch configuration drift before expensive training runs

### Scenario: Research Reproducibility

```bash
# Document exact hardware
warpt list --all --format yaml > system_config.yaml

# Run benchmarks
warpt benchmark --all --export results.json

# Include in paper supplementary materials
```

**Value**: Reviewers can see exactly what hardware experiments used

## Current Status

warpt is in **active development** as an internal tool. Current capabilities:

✅ **Implemented**:

- CPU detection and monitoring
- NVIDIA GPU support (via `nvidia-ml-py`)
- PyTorch framework detection
- Storage abstraction (local, preparing for network/distributed)
- CLI with `list`, `version`, `stress` commands
- JSON export for automation
- Three-layer architecture (CLI → Commands → Backends)

🚧 **In Progress**:

- Hardware abstraction layer for multi-vendor support
- Storage backend expansion (NFS, Lustre, GPFS, BeeGFS, S3, Azure Blob, GCS)
- Comprehensive stress testing framework
- Health monitoring and reporting

📋 **Planned**:

- AMD GPU support (ROCm)
- Intel GPU support (oneAPI)
- Apple Metal/Neural Engine support
- Benchmark suite
- Advanced framework compatibility checking
- Real-time monitoring mode
- Comparative performance reporting

## Design Principles

### Clean Architecture

```
CLI Layer (cli.py)          ← User interface
    ↓
Commands Layer (commands/)  ← Business logic & formatting
    ↓
Backends Layer (backends/)  ← Pure hardware interaction
```

Each layer has a single responsibility. Backends have zero CLI dependencies and return structured data. Commands handle formatting and user interaction. CLI handles argument parsing.

### Type Safety

Extensive use of Pydantic models ensures:

- Validated data structures
- Clear contracts between layers
- Self-documenting APIs
- Easy serialization to JSON/YAML/TOML

### Testability

Backend separation makes it easy to:

- Mock hardware for testing
- Test commands without real hardware
- Validate output formats
- Test error handling

### Extensibility

New hardware support requires implementing one interface:

```python
from warpt.backends.base import AcceleratorBackend

class MyAcceleratorBackend(AcceleratorBackend):
    def detect(self) -> list[AcceleratorInfo]:
        """Detect available accelerators."""

    def get_info(self, device_id: int) -> AcceleratorInfo:
        """Get detailed info for one device."""

    def run_health_check(self, device_id: int) -> HealthStatus:
        """Verify device is working."""
```

Register it and it works with all existing commands.

## The Vision

warpt aims to become **the standard tool for hardware diagnostics in ML and HPC**:

- **Universal**: Works everywhere - laptop to supercomputer
- **Trusted**: Neutral, open, community-driven
- **Essential**: The first command you run on any new system
- **Extensible**: Vendors can add support easily
- **Comprehensive**: Hardware + software + performance in one tool

We're building the `git status` for hardware - a tool so useful that it becomes muscle memory.

## Technical Stack

- **Language**: Python 3.11+
- **CLI**: Click for argument parsing
- **Models**: Pydantic for data validation
- **Hardware**: `psutil`, `nvidia-ml-py`, vendor SDKs
- **Frameworks**: `torch`, `tensorflow` detection
- **Testing**: pytest
- **Code Quality**: ruff, black, pre-commit hooks
- **Distribution**: pip, conda, poetry, uv

## Try It

```bash
# Installation (development)
git clone https://github.com/your-org/warpt.git
cd warpt
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Basic commands
warpt list              # See what you have
warpt version          # Check warpt version
warpt stress gpu       # Stress test GPU
```

## Contributing

We're actively developing warpt and welcome contributions:

- **Hardware vendors**: Implement backends for your accelerators
- **ML engineers**: Request features, report bugs, share use cases
- **HPC admins**: Help us understand enterprise needs
- **Developers**: Contribute code, tests, documentation

See `CONTRIBUTING.md` for guidelines.

## License

Internal development build. Not for public release.

______________________________________________________________________

**warpt**: Because you deserve to know what hardware you're actually running on.
