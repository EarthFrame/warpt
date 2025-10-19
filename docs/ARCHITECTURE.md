# Warpt Architecture

## Three-Layer Design

```
CLI Layer (cli.py)
    ↓
Commands Layer (commands/)
    ↓
Backends Layer (backends/)
```

### CLI Layer
- Click command definitions
- Argument parsing
- Calls command handlers

### Commands Layer
- Handles user interaction and output formatting
- Calls backend(s) to get data

### Backends Layer
- Pure data collection
- No CLI dependencies
- Returns raw data structures

## Backend Split

**`system.py`** - Generic system info using `psutil`
- CPU cores, usage
- Memory, disk (future)
- Platform-agnostic

**`nvidia.py`** - NVIDIA-specific using `pynvml`
- GPU detection
- Memory, temperature
- CUDA info

**`future additions`** - additional vendor logic
- amd.py
- intel.py
- interface to implement shared methods

## Current Structure

```
warpt/
├── __init__.py
├── cli.py                    # Entry point
├── backends/
│   ├── __init__.py
│   ├── system.py            # System class - psutil
│   └── nvidia.py            # NvidiaBackend - pynvml
├── commands/
│   ├── __init__.py
│   └── list_cmd.py          # run_list()
└── utils/
    └── __init__.py

docs/
└── CLI-Design.md            # External interface spec
```

## Example Flow

```
$ warpt list
  ↓
cli.py: list() command
  ↓
commands/list_cmd.py: run_list()
  ↓
backends/system.py: System.list_devices()
  ↓
Returns CPU data → formats → prints
```

## Future Additions

Commands need to be created for:
- `monitor` - Real-time monitoring
- `benchmark` - Performance tests
- `check` - Health checks
- `stress` - Stress testing
- `run` - Profile execution (might not need this?)
- `report` - Generate reports (rename to export?)
- `health` - do we need seperate health report command? we can include health check as part of every 
  stress test and benchmark run

Each will follow same pattern: `cli.py` → `commands/{name}_cmd.py` → `backends/*`
