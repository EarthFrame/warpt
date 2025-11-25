# WARPT - System Diagnostics & Performance Testing CLI

## Package Name: `warpt`

**W**orkload **A**nalysis & **R**esource **P**rofiling **T**ool

## Installation

### pip (Recommended)

```bash
pip install warpt
```

### conda

```bash
conda install -c conda-forge warpt
```

### poetry

```bash
poetry add warpt
```

### uv

```bash
uv pip install warpt
```

### From Source

```bash
git clone https://github.com/your-org/warpt.git
cd warpt
pip install -e .
```

______________________________________________________________________

## Core Command Structure

```bash
warpt <command> [options] [targets...]
```

## Core Commands

### 1. `check` - Quick Health Checks

Fast, non-destructive health checks for system components.

```bash
# Basic usage
warpt check                    # Check all components
warpt check cpu gpu memory     # Check specific components
warpt check --all              # Comprehensive check

# Options
warpt check --quick            # Fast check only
warpt check --verbose          # Detailed output
warpt check --format json      # Machine-readable output
```

**Output (text format):**

```
✓ CPU: 16 cores, all functional
✓ GPU: NVIDIA RTX 4090 (24GB), CUDA 12.1
✓ Memory: 64GB, no errors detected
✗ CUDA: Driver version mismatch (found 11.8, expected 12.1)

Summary: 3 passed, 1 failed
```

**Output (JSON format):**

```json
{
  "timestamp": "2025-10-14T10:30:00Z",
  "results": {
    "cpu": {"status": "pass", "cores": 16, "threads": 32},
    "gpu": {"status": "pass", "model": "NVIDIA RTX 4090", "memory": "24GB"},
    "memory": {"status": "pass", "total": "64GB", "available": "48GB"},
    "cuda": {"status": "fail", "error": "Driver version mismatch"}
  },
  "summary": {"total": 4, "passed": 3, "failed": 1}
}
```

### 2. `stress` - Stress Testing

Intensive testing for reliability and stability.

```bash
# Basic usage
warpt stress cpu               # Stress CPU
warpt stress gpu               # Stress GPU
warpt stress memory            # Stress memory

# Options
warpt stress cpu --duration 5m --threads 16
warpt stress gpu --intensity high --duration 60s
warpt stress memory --size 32GB --pattern random
warpt stress --all --duration 10m
```

**Output:**

```
[████████░░] 80% CPU Stress Test (1m 12s remaining)
Current: 16 threads at 100% utilization
Peak Temperature: 78°C
Power Draw: 125W
```

**JSON Output:**

```json
{
  "test": "cpu_stress",
  "duration": 300,
  "results": {
    "max_utilization": 100.0,
    "avg_utilization": 98.5,
    "peak_temperature": 78,
    "power_draw_avg": 125,
    "errors": 0,
    "throttling_events": 0
  },
  "status": "pass"
}
```

### 3. `benchmark` - Performance Benchmarking

Measure and compare performance.

```bash
# Basic usage
warpt benchmark cpu            # CPU benchmark
warpt benchmark gpu            # GPU benchmark
warpt benchmark --all          # Full system benchmark

# Options
warpt benchmark gpu --mlperf             # MLPerf benchmark
warpt benchmark cpu --threads 8          # Specify thread count
warpt benchmark --compare baseline.json  # Compare to baseline
warpt benchmark --output results.json    # Save results
```

**Output:**

```
GPU Benchmark Results:
  FP32 Performance: 82.3 TFLOPS
  FP16 Performance: 165.2 TFLOPS
  Memory Bandwidth: 1008 GB/s
  MLPerf Score: 1250

Comparison to baseline:
  FP32: +15.3% (baseline: 71.4 TFLOPS)
  Memory: +2.1% (baseline: 988 GB/s)
```

**JSON Output:**

```json
{
  "benchmark": "gpu",
  "timestamp": "2025-10-14T10:30:00Z",
  "hardware": {
    "model": "NVIDIA RTX 4090",
    "memory": "24GB",
    "driver": "530.30.02"
  },
  "results": {
    "fp32_tflops": 82.3,
    "fp16_tflops": 165.2,
    "memory_bandwidth_gbps": 1008,
    "mlperf_score": 1250
  },
  "comparison": {
    "baseline_file": "baseline.json",
    "fp32_delta_percent": 15.3,
    "memory_delta_percent": 2.1
  }
}
```

### 4. `run` - Profile Command Execution

Profile resource usage of any command.

```bash
# Basic usage
warpt run python train.py
warpt run ./my_application
warpt run "npm test"

# Options
warpt run --gpu python train.py              # Include GPU profiling
warpt run --memory python script.py          # Memory profiling
warpt run --interval 100ms python train.py   # Sampling interval
warpt run --output profile.json python train.py
```

**Output:**

```
Command: python train.py
Duration: 5m 32s
Exit Code: 0

Resource Usage:
  CPU Peak: 89.3% (avg: 67.2%)
  Memory Peak: 12.3GB (avg: 8.7GB)
  GPU Utilization: 95.2% (avg: 87.4%)
  GPU Memory: 18.2GB peak

Timeline:
  [0-1m]   CPU: 45%, GPU: 78%, Mem: 4.2GB
  [1-3m]   CPU: 82%, GPU: 95%, Mem: 12.1GB  (training)
  [3-5m]   CPU: 52%, GPU: 89%, Mem: 9.8GB
```

**JSON Output:**

```json
{
  "command": "python train.py",
  "start_time": "2025-10-14T10:30:00Z",
  "end_time": "2025-10-14T10:35:32Z",
  "duration": 332.5,
  "exit_code": 0,
  "resources": {
    "cpu": {
      "peak_percent": 89.3,
      "avg_percent": 67.2,
      "timeline": [...]
    },
    "memory": {
      "peak_gb": 12.3,
      "avg_gb": 8.7,
      "timeline": [...]
    },
    "gpu": {
      "utilization_peak": 95.2,
      "utilization_avg": 87.4,
      "memory_peak_gb": 18.2,
      "timeline": [...]
    }
  }
}
```

### 5. `monitor` - Real-time Monitoring

Real-time system monitoring.

```bash
# Basic usage
warpt monitor                  # Monitor all resources
warpt monitor gpu              # Monitor GPU only
warpt monitor --live           # Live dashboard

# Options
warpt monitor --interval 1s    # Update interval
warpt monitor --dashboard      # TUI dashboard
warpt monitor --duration 5m    # Monitor for duration
warpt monitor --alert cpu:80   # Alert when CPU > 80%
```

**Output (Live Dashboard):**

```
╔══════════════════════════════════════════════════════════╗
║  WARPT Monitor - ml-workstation-01  [10:30:45]          ║
╠══════════════════════════════════════════════════════════╣
║  CPU     [████████░░] 78%   16 cores @ 2.8 GHz          ║
║  Memory  [██████░░░░] 62%   39.7GB / 64GB                ║
║  GPU 0   [█████████░] 89%   RTX 4090 @ 1920 MHz         ║
║    Mem   [███████░░░] 72%   17.3GB / 24GB                ║
║    Temp  [██████░░░░] 68°C  Power: 320W / 450W           ║
║  GPU 1   [████████░░] 82%   RTX 4090 @ 1890 MHz         ║
║    Mem   [██████░░░░] 65%   15.6GB / 24GB                ║
║    Temp  [██████░░░░] 65°C  Power: 305W / 450W           ║
╚══════════════════════════════════════════════════════════╝
```

### 6. `report` - Generate Reports

Generate comprehensive reports from logs or live data.

```bash
# Basic usage
warpt report                          # Generate current system report
warpt report --output report.html     # HTML report
warpt report --format json            # JSON report

# From logs
warpt report --log benchmark.json     # Report from specific log
warpt report --logs ~/.warpt/logs/    # Aggregate all logs
warpt report --since 7d               # Logs from last 7 days

# Comparisons
warpt report --compare log1.json log2.json
warpt report --trend --since 30d      # Trend analysis
```

**Output (Text):**

```
System Report - ml-workstation-01
Generated: 2025-10-14 10:30:00

Hardware Summary:
  CPU: Intel Xeon E5-2686 v4 (16 cores)
  GPU: 2x NVIDIA RTX 4090 (24GB each)
  Memory: 64GB DDR4
  Storage: 2TB NVMe SSD

Recent Activity (7 days):
  Total Runs: 145
  Stress Tests: 23 (100% passed)
  Benchmarks: 12 (avg GPU: 82.1 TFLOPS)
  Failed Tests: 2 (CUDA version issues)

Performance Trends:
  GPU Performance: Stable (82.1 ± 1.2 TFLOPS)
  CPU Performance: Stable (95% of theoretical)
  No degradation detected
```

### 7. `list` - Discover Available Resources

List available hardware and software.

```bash
# Basic usage
warpt list                     # List all resources
warpt list hardware            # Hardware only
warpt list software            # Software only

# Specific queries
warpt list --gpu               # GPUs only
warpt list --frameworks        # ML frameworks
warpt list --detailed          # Verbose output
```

**Output:**

```
Hardware:
  CPU: Intel Xeon E5-2686 v4
    Cores: 16, Threads: 32
    Features: AVX, AVX2, SSE4.2, FMA

  GPU:
    [0] NVIDIA RTX 4090 (24GB, CUDA 8.9)
    [1] NVIDIA RTX 4090 (24GB, CUDA 8.9)

  Memory: 64GB DDR4

  Storage:
    /dev/nvme0n1: 2TB NVMe SSD

Software:
  Python: 3.11.4 (/usr/bin/python3.11)
  CUDA: 12.1.1 (driver 530.30.02)

  Frameworks:
    PyTorch: 2.0.1 (CUDA 12.1)
    TensorFlow: 2.13.0 (CUDA 12.1)

  Compilers:
    GCC: 11.4.0
    NVCC: 12.1.105
```

**JSON Output:**

```json
{
  "hardware": {
    "cpu": {
      "model": "Intel Xeon E5-2686 v4",
      "cores": 16,
      "threads": 32,
      "features": ["AVX", "AVX2", "SSE4.2", "FMA"]
    },
    "gpu": [
      {
        "index": 0,
        "model": "NVIDIA RTX 4090",
        "memory_gb": 24,
        "compute_capability": "8.9",
        "pcie_gen": 4
      }
    ],
    "memory": {
      "total_gb": 64,
      "type": "DDR4"
    }
  },
  "software": {
    "python": {
      "version": "3.11.4",
      "path": "/usr/bin/python3.11"
    },
    "cuda": {
      "version": "12.1.1",
      "driver": "530.30.02"
    },
    "frameworks": {
      "pytorch": {
        "version": "2.0.1",
        "cuda_support": true
      }
    }
  }
}
```

______________________________________________________________________

## Target Components

### Hardware Targets

- `cpu` - CPU diagnostics and testing
- `gpu` - GPU diagnostics and compute tests
- `memory` / `ram` - Memory testing
- `storage` / `disk` - Disk I/O and health
- `network` - Network interface testing

### Software Targets

- `pytorch` - PyTorch installation and functionality
- `tensorflow` - TensorFlow checks
- `cuda` - CUDA toolkit and drivers
- `drivers` - Driver health checks

______________________________________________________________________

## Global Options

```bash
--verbose, -v          Verbose output
--quiet, -q            Suppress output (errors only)
--output, -o FILE      Save results to file
--format FORMAT        Output format: text, json, yaml, html
--config FILE          Use config file
--timeout DURATION     Timeout (e.g., 5m, 300s)
--no-color             Disable colored output
--help, -h             Show help
--version              Show version
```

______________________________________________________________________

## Logging System

### Automatic Logging

All commands automatically log to `~/.warpt/logs/` unless `--no-log` is specified.

**Log filename format:**

```
{timestamp}-{command}-{target}-{id}.json
Example: 20251014-103045-benchmark-gpu-a1b2c3.json
```

**Log contents:**

```json
{
  "metadata": {
    "timestamp": "2025-10-14T10:30:45Z",
    "command": "benchmark",
    "target": "gpu",
    "hostname": "ml-workstation-01",
    "warpt_version": "1.0.0"
  },
  "system": {
    "os": "Ubuntu 22.04",
    "cpu": "Intel Xeon E5-2686 v4",
    "gpu": ["NVIDIA RTX 4090"],
    "memory_gb": 64
  },
  "execution": {
    "start": "2025-10-14T10:30:45Z",
    "end": "2025-10-14T10:32:15Z",
    "duration_seconds": 90,
    "exit_code": 0
  },
  "results": {
    "fp32_tflops": 82.3,
    "fp16_tflops": 165.2,
    "memory_bandwidth_gbps": 1008
  }
}
```

### Log Management

```bash
# List logs
warpt logs list
warpt logs list --since 7d
warpt logs list --command benchmark

# View log
warpt logs show <log-id>
warpt logs show --latest

# Clean old logs
warpt logs clean --older-than 30d
```

______________________________________________________________________

## Configuration

### Config File Locations

1. `~/.config/warpt/config.yaml` (user config)
1. `./warpt.yaml` (project config)
1. Environment variables: `WARPT_*`

### Sample Configuration

```yaml
# ~/.config/warpt/config.yaml

# Default settings
defaults:
  output_format: text
  timeout: 300
  log_enabled: true

# Logging
logging:
  directory: ~/.warpt/logs
  retention_days: 30
  max_files: 1000

# Target-specific settings
targets:
  gpu:
    cuda_checks: true
    memory_test: true

  pytorch:
    version_check: true
    gpu_support: true

# Benchmark baselines
baselines:
  gpu_fp32_tflops: 82.0
  cpu_gflops: 450
```

______________________________________________________________________

## Usage Examples

### Quick Start

```bash
# Check if system is healthy
warpt check

# Run a quick benchmark
warpt benchmark --all

# Monitor GPU during training
warpt monitor gpu --live
```

### Benchmarking Workflow

```bash
# 1. Create baseline
warpt benchmark --all --output baseline.json

# 2. Make changes (update drivers, etc.)

# 3. Re-benchmark and compare
warpt benchmark --all --compare baseline.json

# 4. Generate report
warpt report --compare baseline.json current.json --output report.html
```

### Profiling Workflow

```bash
# Profile a training script
warpt run --gpu --memory python train.py --output training-profile.json

# View resource usage
cat training-profile.json | jq '.resources.gpu.utilization_avg'

# Monitor live during execution
warpt monitor gpu &
python train.py
```

### CI/CD Integration

```bash
# In CI pipeline
warpt check --all --format json > system-check.json

# Parse results
FAILED=$(jq '.summary.failed' system-check.json)
if [ "$FAILED" -gt 0 ]; then
  echo "System check failed"
  exit 1
fi

# Run benchmarks
warpt benchmark --quick --output ci-benchmark.json
```

______________________________________________________________________

## Environment Management (Extended Features)

This section covers advanced environment discovery, export, and reproducibility features.

### Commands

#### `warpt inspect` - Detailed Environment Inspection

```bash
# Full inspection
warpt inspect

# Container-specific
warpt inspect --container      # Detect container runtime
warpt inspect --devices        # Show GPU passthrough
warpt inspect --mounts         # Show mounted volumes
```

**Output:**

```json
{
  "environment": {
    "runtime": "docker",
    "container_id": "abc123",
    "base_image": "nvidia/cuda:12.1-runtime-ubuntu22.04"
  },
  "devices": {
    "nvidia": ["/dev/nvidia0", "/dev/nvidia1", "/dev/nvidiactl"],
    "passthrough": true
  },
  "mounts": [
    {"source": "/data", "target": "/workspace/data", "type": "bind"}
  ],
  "environment_vars": {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "NVIDIA_VISIBLE_DEVICES": "all"
  }
}
```

#### `warpt export` - Export Environment Specification

```bash
# Export current environment
warpt export env.yaml

# Export with installation instructions
warpt export --with-install env.yaml

# Generate Dockerfile
warpt export --dockerfile Dockerfile.warpt

# Minimal export (essentials only)
warpt export --minimal minimal-env.yaml
```

**Environment Spec Format (env.yaml):**

```yaml
metadata:
  exported: "2025-10-14T10:30:00Z"
  hostname: "ml-workstation-01"
  warpt_version: "1.0.0"

hardware:
  cpu:
    model: "Intel Xeon E5-2686 v4"
    cores: 16
  gpu:
    - model: "NVIDIA RTX 4090"
      memory_gb: 24
      compute_capability: "8.9"
  memory_gb: 64

software:
  python:
    version: "3.11.4"
  cuda:
    version: "12.1.1"
    driver: "530.30.02"
  frameworks:
    - name: pytorch
      version: "2.0.1"
    - name: tensorflow
      version: "2.13.0"

# Installation commands (with --with-install)
install:
  system:
    - apt-get install build-essential cmake
  python:
    - pip install torch==2.0.1
    - pip install tensorflow==2.13.0
```

**Generated Dockerfile:**

```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.0.1 \
    tensorflow==2.13.0

ENV CUDA_VISIBLE_DEVICES=0,1
WORKDIR /workspace
```

#### `warpt recreate` - Recreate Environment

```bash
# Recreate from spec
warpt recreate env.yaml

# Dry run (show what would be installed)
warpt recreate env.yaml --dry-run

# Build Docker container
warpt recreate env.yaml --build-docker --tag ml-env:v1
```

#### `warpt diff` - Compare Environments

```bash
# Compare two specs
warpt diff env1.yaml env2.yaml

# Compare current with spec
warpt diff --current env.yaml

# Show summary only
warpt diff env1.yaml env2.yaml --summary
```

**Output:**

```
Environment Comparison: baseline.yaml vs current.yaml

Hardware:
  GPU: NVIDIA RTX 3090 → RTX 4090 (upgraded)

Software:
  PyTorch: 2.0.1 → 2.1.0 (upgraded)
  CUDA: 12.1 → 12.2 (upgraded)
  + Added: JAX 0.4.13
  - Removed: TensorFlow 2.13.0

Summary: 5 changes (2 hardware, 3 software)
Breaking changes: 1 (CUDA upgrade may require rebuilds)
```

### Container Workflows

#### Verify GPU in Container

```bash
docker run --gpus all my-image warpt check gpu
docker run --gpus all my-image warpt list --gpu --detailed
```

#### Export and Recreate

```bash
# On dev machine
warpt export dev-env.yaml --with-install

# Build container
warpt export --dockerfile Dockerfile.dev
docker build -f Dockerfile.dev -t ml-dev:v1 .

# Verify
docker run ml-dev:v1 warpt check --all
```

#### Environment Versioning

```bash
# Save baseline
warpt export baseline.yaml

# After changes
warpt export current.yaml

# Compare
warpt diff baseline.yaml current.yaml --output changes.txt
```

______________________________________________________________________

## Exit Codes

```
0  - Success
1  - Failure (tests failed, errors encountered)
2  - Warnings (tests passed with warnings)
3  - Invalid arguments
```

______________________________________________________________________

## Implementation Notes

### Technology Stack

- **CLI Framework**: Click or Typer (Python)
- **Hardware Detection**: psutil, GPUtil, pynvml
- **Container Detection**: Check `/proc/1/cgroup`, `/.dockerenv`
- **Output Formats**: rich (terminal), JSON, YAML, HTML templates

### Key Features

1. **Progressive Disclosure**: Simple commands for beginners, advanced options for power users
1. **Machine-Readable Output**: All commands support JSON/YAML for automation
1. **Automatic Logging**: Every run logged for historical analysis
1. **Container-Aware**: Detects and adapts to containerized environments
1. **Extensible**: Plugin system for custom targets and tests

### Container Detection

```python
def is_container():
    """Detect if running in a container"""
    if os.path.exists('/.dockerenv'):
        return True
    try:
        with open('/proc/1/cgroup') as f:
            return 'docker' in f.read() or 'lxc' in f.read()
    except:
        return False
```

### GPU Detection

```python
def detect_gpus():
    """Detect available GPUs"""
    gpus = []
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({
                'index': i,
                'name': name,
                'memory_gb': mem.total / 1024**3
            })
    except:
        pass
    return gpus
```
