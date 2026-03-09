# warpt: EarthFrame's Workload and Resource Performance and Transparency Toolkit

`warpt` is a unified command-line tool for hardware discovery, stress testing, and performance monitoring.

warpt provides a vendor-agnostic reporting interface for software and hardware resources. `warpt` brings transparency to system configuration, benchmarking, and stress testing, answering questions such as:

- *"What hardware do I have?"*,
- *"How much power is it actively using?"*,
- *"Is it working correctly?"*, and
- *"How fast is it?"*

## Installation

`warpt` requires Python 3.8 or newer, though we recommend 3.10+. `warpt` has full support on Mac OS X and Linux; Windows support is currently limited.

**Note:** We recommend using `warpt` in a virtualenv. Before installing warpt, you can create and activate a virtualenv by running:

```bash
python -m venv warpt-env
source warpt-env/bin/activate
```

You can install the basic `warpt` command set like so:

```bash
pip install warpt
```

Stress testing has some vendor-specific code at the moment; you can install this to enable all compatible stress tests by running:

```bash
pip install warpt[stress]
```

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
| `warpt carbon` | Track energy consumption, CO2 emissions, and estimated cost |
| `warpt benchmark` | Performance benchmarking suite |
| `warpt integrate` | AI-assisted hardware backend integration |

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

## Carbon Tracking

warpt automatically tracks energy usage and CO2 emissions during stress tests and power monitoring. You can also track any workload manually:

```bash
# Automatic — built into stress tests
warpt stress -c cpu -d 30
# [carbon] 30.2s | 23.8W avg | 199.7 mWh | 0.08g CO2 | $0.0000 | less than breathing for a minute

# Manual — track any workload
warpt carbon start
# ... run your workload ...
warpt carbon stop

# View history and totals
warpt carbon history
warpt carbon summary

# Check available grid regions and carbon intensities
warpt carbon regions
```

Carbon calculations use regional grid intensity data to estimate CO2 emissions from energy consumption. Configure your region with `--region` (defaults to US).

## Backend Integration

`warpt integrate` uses an AI agent (Claude Code SDK) to generate new hardware backend implementations from vendor SDK documentation. It creates the backend, power backend, factory registration, and tests automatically.

### Prerequisites

`warpt integrate` requires [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (Anthropic's CLI) and the Claude Code Python SDK.

1. **Install Claude Code CLI:**

   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Authenticate** — choose one:

   - **Anthropic API key** (recommended for CI/automation):
     ```bash
     export ANTHROPIC_API_KEY=sk-ant-...
     ```
   - **Interactive login** (opens browser, uses your Claude account):
     ```bash
     claude login
     ```

3. **Install warpt with integration dependencies:**

   ```bash
   pip install warpt[integrate]
   ```

4. **Verify setup:**

   ```bash
   claude --version          # confirm CLI is installed
   warpt integrate --vendor test --sdk-docs ./some-docs/ --dry-run
   ```

   The `--dry-run` validates everything without starting the agent.

### Quick Start

```bash
# Start a new backend integration
warpt integrate --vendor amd --sdk-docs ./amd-sdk/python/

# Check integration status
warpt integrate status

# Process answered questions
warpt integrate

# Validate generated code (ruff + pytest)
warpt integrate validate

# Reset and start over
warpt integrate reset
```

### Dry Run

Use `--dry-run` to validate your setup before spending tokens on an agent session:

```bash
warpt integrate --vendor amd --sdk-docs ./amd-sdk/python/ --dry-run
```

This runs all validation (vendor name, existing backend check, doc loading, token limit) and prints a summary:

```
==================================================
DRY RUN SUMMARY
==================================================
  SDK docs:      ~18,500 tokens from 42 file(s)
  System prompt: ~12,000 tokens
  Total context: ~30,500 tokens (limit ~200k)

  Files to generate:
    warpt/backends/amd.py
    warpt/backends/power/amd_power.py
    tests/test_amd_backend.py
    questions.yaml

  Branch: backend/amd
==================================================

Press Enter to start the agent, or Ctrl+C to abort:
```

Press Enter to proceed into the full agent run, or Ctrl+C to abort without creating anything.

### Post-Agent File Audit

After the agent session completes, `warpt integrate` automatically audits the generated files:

```
Generated files:
  questions.yaml: created (8 question(s))
  tests/test_amd_backend.py: created (320 lines)
  warpt/backends/amd.py: created (245 lines)
  warpt/backends/power/amd_power.py: created (180 lines)
```

If the agent modified files outside the expected set (`factory.py`, `pyproject.toml`, and the generated files), a warning is printed:

```
Warning: Unexpected file modifications:
  warpt/utils/helpers.py
```

### Workflow

1. **Init** (`--sdk-docs`): Agent reads SDK docs, generates backend + tests, logs questions to `questions.yaml`
2. **Review**: Engineer reviews generated code and answers questions (set `status: answered`)
3. **Iterate** (no `--sdk-docs`): Agent reads answers, updates code, runs tests
4. **Validate**: Run `warpt integrate validate` to confirm ruff + pytest pass

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

This is an **alpha release**. Some features are still in development:

- Carbon tracking — new in v0.2.0
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

`warpt` is licensed under the MIT License — see [LICENSE](LICENSE) for details.
