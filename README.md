# warpt

A unified command-line tool for hardware discovery, stress testing, and performance monitoring.

warpt provides a vendor-agnostic interface for understanding and validating computational resources—answering questions like *"What hardware do I have?"*, *"Is it working correctly?"*, and *"How fast is it?"*

## Installation

```bash
pip install warpt
```

For stress testing capabilities:

```bash
pip install warpt[stress]
```

**Requirements:** Python 3.8+ (3.11+ recommended) | Linux, macOS, or Windows

## Quick Start

```bash
# Discover your hardware
warpt list

# Run CPU stress tests
warpt stress -c cpu

# Monitor system in real-time
warpt monitor

# Check power consumption (Linux/macOS)
warpt power
```

## Features

| Command | Description |
|---------|-------------|
| `warpt list` | Detect CPU, GPU, memory, storage, and installed ML frameworks |
| `warpt stress` | Run stress tests across CPU, GPU, RAM, storage, and network |
| `warpt monitor` | Real-time system monitoring with TUI dashboard |
| `warpt power` | Power consumption monitoring and per-process attribution |
| `warpt benchmark` | Performance benchmarking suite |

## Documentation

- [Getting Started](https://docs.earthframe.com/getting_started) — Installation and first steps
- [CLI Reference](https://docs.earthframe.com/cli_reference) — Complete command and option reference
- [Support Matrix](https://docs.earthframe.com/support_matrix) — System requirements and platform compatibility

## Platform Support

| Platform | Status |
|----------|--------|
| Linux | Full support |
| macOS | Full support (power monitoring requires sudo) |
| Windows | Limited support (see [Known Limitations](https://docs.earthframe.com/support_matrix#known-limitations)) |

**GPU Support:** NVIDIA GPUs supported. AMD, Intel, and Apple Silicon GPU support coming soon.

## Example Output

```
$ warpt list

CPU Information:
  Make:               Intel
  Model:              Xeon W-2295
  Architecture:       x86_64

Topology:
  Total Sockets:      1
  Total Phys Cores:   18
  Total Logic Cores:  36

Memory Information:
  Total:              128.0 GB
  Type:               DDR4

GPU Information:
  GPU 0: NVIDIA RTX 4090
    Memory: 24576 MB
    Driver: 545.23.08
```

## Alpha Release

This is **v0.1.0-alpha**. Some features are still in development:

- AMD GPU support (ROCm) — in progress
- Intel GPU support (oneAPI) — in progress
- Apple Neural Engine — in progress
- Additional benchmarks — expanding

See the [Support Matrix](https://docs.earthframe.com/support_matrix) for full details.

## Feedback

We'd love to hear from you:

- **Report bugs:** [GitHub Issues](https://github.com/EarthFrame/warpt/issues)
- **Feature requests:** [GitHub Issues](https://github.com/EarthFrame/warpt/issues)

## License

MIT License — see [LICENSE](LICENSE) for details.
