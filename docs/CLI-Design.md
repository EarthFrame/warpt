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



### From Source
```bash
git clone https://github.com/your-org/warpt.git
cd warpt
pip install -e .
```

### poetry
```bash
# Add to existing project
poetry add warpt

# Install from source
git clone https://github.com/your-org/warpt.git
cd warpt
poetry install
```

### uv
```bash
# Install package
uv pip install warpt

# Install from source
git clone https://github.com/your-org/warpt.git
cd warpt
uv pip install -e .
```

### Verify Installation
```bash
warpt --version
warpt check --quick
```

## Core Command Structure

```bash
warpt <command> [options] [targets...]
```

## Main Commands

### 1. `check` - System Health Checks
```bash
# Quick health check
warpt check

# Check specific components
warpt check cpu gpu memory
warpt check pytorch cuda
warpt check --all

# Quick system overview
warpt check --summary
```

### 2. `stress` - Stress Testing
```bash
# Run stress tests
warpt stress

# Test specific components
warpt stress cpu --duration 60s
warpt stress gpu --intensity high
warpt stress memory --size 8GB

# Test suites
warpt stress --suite ml-training
warpt stress --suite networking
```

### 3. `benchmark` - Performance Benchmarking
```bash
# Run benchmarks
warpt benchmark
warpt benchmark gpu --mlperf
warpt benchmark cpu --threads 8

# Compare against theoretical performance
warpt benchmark gpu --theoretical --api-key $WARPT_API_KEY
warpt benchmark --all --theoretical --output benchmark-report.html
```

### 4. `run` - Profile Command Execution
```bash
# Profile any command
warpt run python train.py
warpt run ./my_application --config app.yaml
warpt run "npm test"

# Detailed profiling
warpt run --profile-gpu python train.py
warpt run --profile-memory --profile-io ./data_processor
warpt run --interval 100ms python long_running_task.py

# Save profile results
warpt run --output profile.json python script.py
warpt run --format html python script.py
```

### 5. `monitor` - Real-time Monitoring
```bash
# Monitor system in real-time
warpt monitor
warpt monitor gpu --live
warpt monitor --dashboard
```

### 6. `report` - Generate Reports
```bash
# Generate comprehensive report
warpt report --output report.html
warpt report --format json

# Generate reports from logs
warpt report --from-logs ~/.warpt/logs/
warpt report --from-logs ~/.warpt/logs/20250716-*
warpt report --combine-logs --since 7d
warpt report --trend --logs ~/.warpt/logs/benchmark-*

# Theoretical performance analysis
warpt report --theoretical --api-key $WARPT_API_KEY
warpt report --log benchmark.json --theoretical --output analysis.html

# Specific log analysis
warpt report --log-file ~/.warpt/logs/20250716-143022-test-gpu.json
```

### 7. `list` - Discover Available Hardware & Software
```bash
# List all available hardware
warpt list hardware
warpt list hw              # Short alias

# List specific hardware types
warpt list hw --cpu
warpt list hw --gpu
warpt list hw --memory
warpt list hw --storage
warpt list hw --network

# List all available software/frameworks
warpt list software
warpt list sw              # Short alias

# List specific software categories
warpt list sw --frameworks      # ML frameworks (PyTorch, TensorFlow, etc.)
warpt list sw --cuda            # CUDA toolkit and drivers
warpt list sw --python          # Python and installed packages
warpt list sw --compilers       # Available compilers
warpt list sw --libraries       # System libraries

# List everything
warpt list all
warpt list --detailed           # Include versions, paths, capabilities

# Container-aware listing
warpt list --container-info     # Show container runtime details
warpt list --host-passthrough   # Show host devices available in container

# Output formats
warpt list hw --format json
warpt list sw --format yaml
warpt list all --format table
```

### 8. `inspect` - Comprehensive Environment Inspection
```bash
# Full environment inspection
warpt inspect

# Inspect specific components
warpt inspect hardware
warpt inspect software
warpt inspect network

# Detailed inspection with capabilities
warpt inspect --capabilities    # Include what each component can do
warpt inspect --versions        # Show all version information
warpt inspect --paths           # Show installation paths

# Container-specific inspection
warpt inspect --container       # Detect and show container environment
warpt inspect --runtime         # Show runtime environment (Docker, containerd, etc.)
warpt inspect --mounts          # Show mounted volumes and devices
warpt inspect --devices         # Show available devices (/dev/nvidia*, etc.)
```

### 9. `export` - Export Environment Specification
```bash
# Export complete environment
warpt export env.yaml
warpt export env.json
warpt export env.toml

# Export specific components
warpt export --hardware-only hw.yaml
warpt export --software-only sw.yaml
warpt export --minimal minimal-env.yaml    # Only essential components

# Include installation instructions
warpt export env.yaml --with-install       # Add installation commands
warpt export env.yaml --dockerfile         # Generate Dockerfile
warpt export env.yaml --requirements       # Generate requirements.txt

# Container-optimized export
warpt export --container-optimized container-env.yaml
warpt export --docker-compose docker-compose.yml
warpt export --dockerfile Dockerfile.warpt
```

### 10. `recreate` - Recreate Environment from Specification
```bash
# Recreate from exported file
warpt recreate env.yaml
warpt recreate env.json

# Dry-run to see what would be installed
warpt recreate env.yaml --dry-run

# Selective recreation
warpt recreate env.yaml --hardware-only
warpt recreate env.yaml --software-only
warpt recreate env.yaml --only pytorch,cuda

# Container recreation
warpt recreate env.yaml --build-dockerfile
warpt recreate env.yaml --build-docker --tag my-env:latest
```

## Target Components (Extensible)

### Hardware Targets
- `cpu` - CPU diagnostics and stress testing
- `gpu` - GPU diagnostics and compute tests
- `ram` - Memory testing and stress
- `storage` - Disk I/O and health
- `network` - Network interface testing
- `interconnect` - High-speed interconnects
- `nvlink` - NVIDIA NVLink testing

### Software/Framework Targets
- `pytorch` - PyTorch installation and functionality
- `cuda` - CUDA toolkit and drivers
- `framework` - ML framework testing
- `mlperf` - MLPerf benchmarking
- `drivers` - Driver health checks
- `libraries` - Essential library checks

## Global Options

```bash
--verbose, -v          Verbose output
--quiet, -q            Suppress non-error output
--output, -o FILE      Output results to file
--format FORMAT        Output format (json, yaml, html, text)
--config CONFIG        Use custom configuration file
--parallel, -p N       Run N tests in parallel
--timeout DURATION     Set operation timeout
--no-color             Disable colored output
--dry-run             Show what would be done
--log-dir DIR          Override default log directory
--no-log              Disable automatic logging
--log-id ID           Use custom log identifier
--overwrite-log       Allow overwriting existing log files
--help, -h            Show help
--version             Show version
```

## Usage Examples

### Beginner Examples
```bash
# "Is my system healthy?"
warpt check

# "Test my GPU for ML workloads"
warpt test gpu --suite ml

# "Generate a system report"
warpt report
```

### Intermediate Examples
```bash
# Test specific components with custom settings
warpt test cpu --duration 5m --threads 16
warpt test ram --size 50% --pattern random

# Check ML stack is working
warpt check pytorch cuda --verbose

# Monitor GPU during stress test
warpt test gpu --monitor
```

### Advanced Examples
```bash
# Custom test configuration
warpt test --config custom-tests.yaml

# Comprehensive system validation
warpt check --all --output system-health.json
warpt test --suite enterprise --parallel 4
warpt benchmark --mlperf --compare baseline.json

# Pipeline integration
warpt check --format json --quiet | jq '.failed | length'

# Log-based workflows
warpt test gpu --log-id baseline-gpu
warpt test gpu --log-id updated-gpu
warpt report --compare-logs baseline-gpu updated-gpu

# Automated reporting
warpt report --comprehensive --since 7d --output weekly-report.html
```

## Help System Design

### Main Help
```bash
warpt --help
```
Shows overview, common commands, and getting started guide.

### Command-Specific Help
```bash
warpt check --help
warpt test --help
warpt benchmark --help
```

### Target-Specific Help
```bash
warpt check gpu --help
warpt test pytorch --help
```

### Discovery Commands
```bash
# List all available targets
warpt list targets

# Show what's installed/available
warpt list available

# Show test suites
warpt list suites

# Show recent logs
warpt list logs
warpt list logs --since 7d
```

## Logging System Design

### Default Logging Behavior
All commands automatically log to `~/.warpt/logs/` by default:
- **Always enabled**: Logging happens automatically unless `--no-log` is specified
- **Unique filenames**: Each run gets a unique log file to prevent overwrites
- **Structured format**: Logs are in JSON format for easy parsing and reporting

### Log File Naming Convention
```
~/.warpt/logs/{timestamp}-{command}-{targets}-{run_id}.json
```

Examples:
```
20250716-143022-check-all-a1b2c3d4.json
20250716-143157-test-gpu-cuda-f5e6d7c8.json
20250716-144301-benchmark-mlperf-9a8b7c6d.json
20250716-145445-monitor-system-2e3f4g5h.json
```

### Log File Structure
```json
{
  "meta": {
    "version": "1.2.3",
    "timestamp": "2025-07-16T14:30:22Z",
    "run_id": "a1b2c3d4",
    "command": "test",
    "targets": ["gpu", "cuda"],
    "user": "username",
    "hostname": "workstation-01",
    "working_dir": "/home/user/project",
    "cli_args": ["test", "gpu", "cuda", "--duration", "60s"],
    "config_file": "~/.config/warpt/config.yaml",
    "environment": {
      "CUDA_VISIBLE_DEVICES": "0,1",
      "PATH": "/usr/local/cuda/bin:...",
      "python_version": "3.11.4",
      "warpt_version": "1.2.3"
    },
    "system_info": {
      "os": "Ubuntu 22.04.3 LTS",
      "kernel": "5.15.0-78-generic",
      "cpu": "Intel Xeon E5-2686 v4",
      "memory": "64GB",
      "gpu": ["NVIDIA RTX 4090", "NVIDIA RTX 4090"]
    }
  },
  "execution": {
    "start_time": "2025-07-16T14:30:22Z",
    "end_time": "2025-07-16T14:31:22Z",
    "duration": 60.234,
    "status": "completed",
    "exit_code": 0,
    "interrupted": false
  },
  "results": {
    "summary": {
      "total_tests": 8,
      "passed": 7,
      "failed": 1,
      "warnings": 0,
      "skipped": 0
    },
    "targets": {
      "gpu": {
        "status": "pass",
        "duration": 45.123,
        "tests": [
          {
            "name": "gpu_memory_test",
            "status": "pass",
            "duration": 15.456,
            "result": {"allocated": "8GB", "tested": "8GB", "errors": 0}
          },
          {
            "name": "gpu_compute_test",
            "status": "pass",
            "duration": 29.667,
            "result": {"ops_per_sec": 1250000, "utilization": 98.5}
          }
        ]
      },
      "cuda": {
        "status": "fail",
        "duration": 5.234,
        "error": "CUDA driver version mismatch",
        "tests": [
          {
            "name": "cuda_version_check",
            "status": "fail",
            "duration": 0.123,
            "error": "Expected 12.1, found 11.8"
          }
        ]
      }
    }
  },
  "metrics": {
    "resource_usage": {
      "peak_memory": "2.3GB",
      "peak_cpu": 45.6,
      "peak_gpu": 87.3,
      "disk_io": {"read": "1.2GB", "write": "0.8GB"}
    }
  },
  "warnings": [],
  "errors": [
    {
      "timestamp": "2025-07-16T14:30:45Z",
      "target": "cuda",
      "message": "CUDA driver version mismatch",
      "details": "Expected 12.1, found 11.8"
    }
  ]
}
```

### Log Management Commands
```bash
# List recent logs
warpt logs list
warpt logs list --since 7d
warpt logs list --command test
warpt logs list --target gpu

# Show log details
warpt logs show 20250716-143022-test-gpu-a1b2c3d4
warpt logs show --latest
warpt logs show --latest --command benchmark

# Clean up old logs
warpt logs clean --older-than 30d
warpt logs clean --keep 100
warpt logs clean --failed-only

# Export logs
warpt logs export --since 7d --output logs-export.tar.gz
```

### Report Generation from Logs
```bash
# Single log report
warpt report --log 20250716-143022-test-gpu-a1b2c3d4.json

# Multiple log reports
warpt report --logs "20250716-*-test-gpu-*"
warpt report --logs ~/.warpt/logs/ --since 7d

# Trend analysis
warpt report --trend --target gpu --since 30d
warpt report --trend --command benchmark --output gpu-trends.html

# Comparison reports
warpt report --compare before.json after.json
warpt report --compare-logs --baseline "20250701-*" --current "20250716-*"

# Comprehensive reports
warpt report --comprehensive --since 7d --output system-health-report.html
```

## Configuration System

### Default Config Locations
- `~/.config/warpt/config.yaml`
- `./warpt.yaml`
- Environment variables: `SYSDIAG_*`

### Sample Configuration
```yaml
# warpt.yaml
defaults:
  timeout: 300s
  parallel: 2
  output_format: text

# Logging configuration
logging:
  enabled: true
  directory: ~/.warpt/logs
  format: json
  retention_days: 30
  max_files: 1000
  compress_old: true
  include_system_info: true
  include_environment: true

targets:
  gpu:
    cuda_checks: true
    memory_test: true
    compute_test: true
  
  pytorch:
    version_check: true
    gpu_support: true
    basic_ops: true

suites:
  ml-training:
    - pytorch
    - cuda
    - gpu
    - ram
  
  networking:
    - network
    - interconnect
    - nvlink
```

## Output Formats

### Text Output (Default)
```
✓ CPU: 16 cores, all functional
✓ GPU: NVIDIA RTX 4090, CUDA 12.1
✗ RAM: 2GB failed sector detected
⚠ PyTorch: Installed but no GPU support
```

### JSON Output
```json
{
  "timestamp": "2025-07-16T10:30:00Z",
  "summary": {
    "total": 8,
    "passed": 6,
    "failed": 1,
    "warnings": 1
  },
  "results": {
    "cpu": {"status": "pass", "details": {...}},
    "gpu": {"status": "pass", "details": {...}},
    "ram": {"status": "fail", "error": "Memory sector failure"}
  }
}
```

## Error Handling & User Experience

### Clear Error Messages
```bash
# Bad command
$ warpt invalid-command
Error: Unknown command 'invalid-command'
Did you mean: check, test, benchmark?

Run 'warpt --help' for usage information.

# Missing dependency
$ warpt check cuda
Error: CUDA not found
Suggestion: Install CUDA toolkit or use --skip-missing
```

### Progress Indication
```bash
$ warpt test --all
Running system diagnostics...
[●●●●●●●●●○] 90% CPU stress test (45s remaining)
```

### Extensible Architecture

The CLI should be designed with a plugin system:

```bash
# Plugin discovery
warpt plugins list
warpt plugins install networking-extended
warpt plugins enable custom-gpu-tests
```

## Implementation Notes

### Argument Parsing
- Use `argparse` or `click` for robust argument handling
- Support both short and long options
- Validate arguments early with clear error messages

### Extensibility
- Plugin architecture for adding new targets
- Configuration-driven test definitions
- Modular result reporting

### Performance
- Parallel execution where safe
- Progress reporting for long operations
- Graceful handling of timeouts and interrupts

### Integration
- Exit codes for CI/CD (0=success, 1=failure, 2=warnings)
- Machine-readable output formats
- Logging integration

This design provides a clean, intuitive interface that grows with user expertise while maintaining consistency and discoverability.

---

## CLI Design Critique & Improvement Suggestions

### Strengths of Current Design

1. **Clear Command Structure**: The verb-based command structure (`check`, `stress`, `benchmark`, `run`, `monitor`, `report`) is intuitive and follows Unix conventions.

2. **Progressive Disclosure**: The design accommodates beginners through advanced users with increasing complexity of options.

3. **Comprehensive Logging**: Automatic logging with unique filenames prevents data loss and enables historical analysis.

4. **Machine-Readable Output**: Multiple output formats (JSON, YAML, HTML) support automation and integration.

### Critical Issues & Improvements

#### 1. **Command Overlap & Confusion**

**Issue**: `stress` and `benchmark` commands have significant overlap. Users may not understand when to use which.

**Suggestion**:
```bash
# Consider consolidating or clarifying:
warpt test <target> --mode [stress|benchmark|quick]
# OR maintain separation but make it clearer:
warpt stress <target>  # For reliability/stability testing
warpt perf <target>    # For performance measurement (shorter than benchmark)
```

#### 2. **Inconsistent Command Naming**

**Issue**: Mix of `check`, `test`, and `stress` creates ambiguity. What's the difference between checking and testing?

**Suggestion**: Standardize terminology:
```bash
warpt check <target>     # Health/status check (fast, non-destructive)
warpt test <target>      # Functional testing (moderate, validates capabilities)
warpt stress <target>    # Stress/endurance testing (intensive, long-running)
warpt bench <target>     # Performance benchmarking (measurement-focused)
```

#### 3. **The `run` Command Ambiguity**

**Issue**: `warpt run` is potentially confusing - it doesn't "run tests" but profiles external commands.

**Suggestion**: Rename to be more explicit:
```bash
warpt profile <command>  # More descriptive
warpt trace <command>    # Alternative
warpt exec <command> --profile  # Makes relationship clearer
```

#### 4. **Over-Complex Logging System**

**Issue**: The logging filename format `{timestamp}-{command}-{targets}-{run_id}.json` could become unwieldy.

**Suggestions**:
- Add `warpt logs` as a top-level command for better discoverability
- Simplify to: `{timestamp}-{run_id}.json` with searchable metadata inside
- Add tags/labels for filtering: `warpt test gpu --tags baseline,nightly`
```bash
warpt logs               # Interactive log browser
warpt logs list --tag baseline
warpt logs show <id>     # Auto-complete from run_id prefix
warpt logs diff <id1> <id2>  # Compare two runs
```

#### 5. **Missing Quick Feedback Mechanisms**

**Issue**: No obvious way to get instant, minimal output for quick checks.

**Suggestion**: Add express modes:
```bash
warpt status             # One-line system status
warpt quick              # Alias for common quick check
warpt check --minimal    # Minimal output mode
warpt check --summary    # Summary only (already exists but promote it)
```

#### 6. **Target Specification Inconsistency**

**Issue**: Mixing hardware (`cpu`, `gpu`, `ram`) and software (`pytorch`, `cuda`) targets without clear categorization.

**Suggestion**: Use namespacing or clearer grouping:
```bash
warpt check hw:gpu           # Hardware
warpt check sw:pytorch       # Software
warpt check stack:ml         # Pre-defined stacks
warpt check all:hw           # All hardware
warpt check all:sw           # All software

# Or use subcommands:
warpt hw check gpu
warpt sw check pytorch
```

#### 7. **Duration/Size Format Inconsistency**

**Issue**: Multiple formats (`5m`, `60s`, `50%`, `8GB`) without clear documentation.

**Suggestion**: Document supported formats prominently and be flexible:
```bash
# Support common formats with clear documentation:
--duration 5m|5min|300s|300
--size 8GB|8G|8192M|8192
--timeout 1h|60m|3600s|3600
```

#### 8. **Missing Comparison Features**

**Issue**: Comparison features are buried in reports. Should be more prominent.

**Suggestion**: Add dedicated comparison command:
```bash
warpt compare <run1> <run2>
warpt compare --baseline <id> --current <id>
warpt diff <id1> <id2>  # Alias for comparison
warpt regression check --since 7d  # Check for performance regressions
```

#### 9. **Theoretical Performance API Integration**

**Issue**: `--theoretical --api-key` pattern is clunky and exposes keys in command history.

**Suggestion**: Better API key management:
```bash
# Use environment variable or config file:
export WARPT_API_KEY=<key>
# Or:
warpt config set api_key <key>  # Stores securely

# Then simply:
warpt bench gpu --theoretical
```

#### 10. **No Dry-Run Feedback**

**Issue**: `--dry-run` exists but unclear what output it provides.

**Suggestion**: Make dry-run informative:
```bash
warpt test gpu --dry-run
# Output:
# Would run the following tests:
#   - gpu_memory_test (estimated 15s)
#   - gpu_compute_test (estimated 30s)
#   - cuda_version_check (estimated 1s)
# Estimated total duration: 46s
# Log file: ~/.warpt/logs/20250716-143022-test-gpu-a1b2c3d4.json
```

#### 11. **Missing Watch/Continuous Mode**

**Issue**: No way to continuously monitor or re-run checks.

**Suggestion**: Add watch functionality:
```bash
warpt monitor --watch <target>  # Already exists as 'monitor'
warpt check gpu --watch --interval 30s  # Continuous checking
warpt test --watch --on-failure alert   # Re-run on change
```

#### 12. **Unclear Suite Management**

**Issue**: Suites are defined in config but no clear way to manage them via CLI.

**Suggestion**: Add suite management:
```bash
warpt suite list
warpt suite show ml-training
warpt suite create my-suite --targets cpu,gpu,pytorch
warpt suite run ml-training
```

### Additional Enhancement Suggestions

#### 13. **Add Context-Aware Suggestions**

When commands fail or targets are unavailable:
```bash
$ warpt check cuda
✗ CUDA not found

Suggestions:
  • Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
  • Skip CUDA checks: warpt check --skip cuda
  • Check GPU without CUDA: warpt check gpu --no-cuda
```

#### 14. **Interactive Mode**

For exploratory workflows:
```bash
warpt interactive
# or
warpt
# Launches interactive prompt:
warpt> check gpu
warpt> stress cpu --duration 1m
warpt> report --last
```

#### 15. **Better Progress Indicators**

Enhance progress indication with context:
```bash
[●●●●●●●●●○] 90% GPU Memory Test (45s remaining)
Current: Testing 8GB allocation
Last: Verified 7.2GB successfully
```

#### 16. **Integrate with System Alerts**

```bash
warpt test gpu --notify-on-failure
warpt monitor --alert-threshold 90% --alert-email user@example.com
```

#### 17. **Add Explain Mode**

For understanding what's happening:
```bash
warpt check gpu --explain
# Output:
# Running GPU health check...
# 
# This check will:
# 1. Detect installed GPUs via PCIe enumeration
# 2. Verify driver installation and version
# 3. Check GPU memory accessibility
# 4. Run basic compute validation
# 
# This is non-destructive and typically takes 5-10 seconds.
```

### Recommended Implementation Priority

1. **High Priority** (Core usability):
   - Fix command naming inconsistencies (#2)
   - Improve `run` command clarity (#3)
   - Better API key management (#9)
   - Add suite management (#12)

2. **Medium Priority** (Enhanced UX):
   - Simplify logging interface (#4)
   - Add quick feedback mechanisms (#5)
   - Improve comparison features (#8)
   - Better dry-run output (#10)

3. **Low Priority** (Nice to have):
   - Interactive mode (#14)
   - Explain mode (#17)
   - System alerts (#16)
   - Target namespacing (#6)

### Summary

The CLI design is solid and comprehensive, but suffers from:
- **Too many overlapping commands** without clear differentiation
- **Inconsistent terminology** that could confuse users
- **Hidden power features** that should be more discoverable
- **Complex patterns** that could be simplified

The core philosophy is right: progressive disclosure, machine-readable output, comprehensive logging. The improvements suggested focus on **consistency**, **clarity**, and **discoverability** while maintaining the tool's power and flexibility.