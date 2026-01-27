## Warpt Benchmark – Design Overview

### 1. Goals and Scope

- **Primary goal**: Provide a unified `warpt benchmark` framework that can:
  - Run **named benchmark suites** (e.g. MLPerf, Linpack, `pepbench`, `biobench`).
  - Run **component-targeted benchmarks** (CPU, GPU, RAM, storage, network, system).
  - Produce **structured, exportable results** that integrate with existing `warpt` models.
- **Design constraints**:
  - Reuse existing patterns (`warpt/backends`, `warpt/models`, `warpt/commands`).
  - Use Pydantic models with type hints and `X | None` for optional fields.
  - Keep the design extensible so new suites (e.g. future `Xbench`) can be added cleanly.

______________________________________________________________________

### 2. Core Concepts

- **Component**: A hardware or system area to benchmark.
  - Examples: `cpu`, `gpu`, `ram`, `storage`, `network`, `system`.
- **Benchmark**: A single runnable workload.
  - Examples: “Linpack FP64 CPU”, “MLPerf ResNet50 infer”, “pepbench LLM infer”.
- **Suite**: A named collection of benchmarks with shared semantics.
  - Examples: `mlperf`, `linpack`, `pepbench`, `biobench`, `micro`.
- **Run**: One execution of a benchmark (or a suite) with specific parameters.
- **Metric**: A single numeric result (e.g. GFLOP/s, samples/s, GB/s, latency).

These concepts are separated so that:

- Suites can be added or removed without touching the core runner.
- Benchmarks can be targeted either by **suite name** or by **component**.

______________________________________________________________________

### 3. High-Level Architecture

- **New package**: `warpt/benchmark/`
  - `base.py`: core interfaces (benchmark, suite, runner utilities).
  - `suites/`: built-in suite implementations:
    - `mlperf`, `linpack`, `pepbench`, `biobench`, `micro` (microbenchmarks).
  - `components/`: mapping from components → default benchmarks.
- **Models**: `warpt/models/benchmark_models.py`
  - Pydantic models for configuration, run metadata, and results.
- **CLI**: extend `warpt/cli.py` and add a command module:
  - `warpt/commands/benchmark_cmd.py`
    - Implements `warpt benchmark` commands using the core interfaces.

The goal is to make `warpt benchmark` feel similar to `warpt list` and `warpt stress`:
structured models underneath, a thin CLI layer, and JSON export by default.

______________________________________________________________________

### 4. Data Model (Pydantic)

All models live in `warpt/models/benchmark_models.py` and follow existing style:
docstrings with Args/Returns, type hints everywhere, and JSON-serializable fields.

#### 4.1 Enums

- **BenchmarkComponent**
  - Represents the hardware or system area a benchmark exercises.

```python
class BenchmarkComponent(str, Enum):
    cpu = "cpu"
    gpu = "gpu"
    ram = "ram"
    storage = "storage"
    network = "network"
    system = "system"
```

- **BenchmarkKind**
  - High-level category of a benchmark.

```python
class BenchmarkKind(str, Enum):
    synthetic = "synthetic"      # microbench / stress-like workloads
    application = "application"  # end-to-end ML or app-level workloads
    domain = "domain"            # domain-specific (e.g. bio/omics)
```

#### 4.2 Benchmark identity and spec

- **BenchmarkId**
  - Unique identifier for a benchmark within a suite.

```python
class BenchmarkId(BaseModel):
    suite: str  # "mlperf", "linpack", "pepbench", "biobench", "micro"
    name: str   # "resnet50_infer", "fp64_linpack", etc.
```

- **BenchmarkSpec**
  - Describes an available benchmark and how it should be run.

```python
class BenchmarkSpec(BaseModel):
    id: BenchmarkId
    components: list[BenchmarkComponent]
    kind: BenchmarkKind
    description: str | None = None
    default_duration_seconds: int | None = None
    external_tool: str | None = None  # e.g. "mlperf_runner", "linpack_binary"
    params: dict[str, str | int | float | bool] | None = None
```

#### 4.3 Metrics and results

- **BenchmarkMetric**
  - A single scalar result from a benchmark run.

```python
class BenchmarkMetric(BaseModel):
    name: str              # "gflops", "throughput", "latency_p50"
    value: float
    unit: str              # "GFLOP/s", "samples/s", "ms"
    higher_is_better: bool
    tags: dict[str, str] | None = None  # e.g. {"component": "cpu", "phase": "train"}
```

- **BenchmarkRunResult**
  - Result of executing one benchmark.

```python
class BenchmarkRunResult(BaseModel):
    spec: BenchmarkSpec
    metrics: list[BenchmarkMetric]
    success: bool
    error_message: str | None = None
    start_time: datetime
    end_time: datetime
    system: ListOutput | None = None  # optional snapshot from `warpt list`
```

- **BenchmarkSuiteRun**
  - Aggregated result for a suite or for a component-driven run.

```python
class BenchmarkSuiteRun(BaseModel):
    suite: str
    components: list[BenchmarkComponent]
    benchmarks: list[BenchmarkRunResult]
    metadata: dict[str, str] | None = None
```

`BenchmarkSuiteRun` is the main export shape for JSON results.

______________________________________________________________________

### 5. Core Interfaces (`warpt/benchmark/base.py`)

#### 5.1 BaseBenchmark

Defines the protocol for any runnable benchmark.

```python
class BaseBenchmark(ABC):
    spec: BenchmarkSpec

    @abstractmethod
    def prepare(self) -> None:
        """Prepare environment and validate prerequisites."""

    @abstractmethod
    def run(self) -> BenchmarkRunResult:
        """Execute the benchmark and return structured results."""
```

Implementations are responsible for:

- Checking dependencies (binaries, containers, datasets).
- Running the actual workload.
- Converting raw measurements into `BenchmarkMetric` instances.

#### 5.2 BenchmarkSuite

Represents a collection of related benchmarks (e.g. MLPerf, Linpack).

```python
class BenchmarkSuite(ABC):
    name: str  # "mlperf", "linpack", "pepbench", "biobench", "micro"

    @abstractmethod
    def list_benchmarks(self) -> list[BenchmarkSpec]:
        """Return the benchmarks available in this suite."""

    @abstractmethod
    def get_benchmark(self, name: str) -> BaseBenchmark:
        """Return a runnable benchmark instance by name."""
```

Each suite implementation can carry any internal configuration it needs
(e.g. paths to binaries, dataset locations, container configuration).

#### 5.3 Suite registry and factory

A central registry maps suite names to suite classes:

```python
SUITE_REGISTRY: dict[str, type[BenchmarkSuite]] = {
    "mlperf": MlperfSuite,
    "linpack": LinpackSuite,
    "pepbench": PepbenchSuite,
    "biobench": BiobenchSuite,
    "micro": MicrobenchSuite,
}
```

A simple helper function creates suite instances:

```python
def get_suite(name: str) -> BenchmarkSuite:
    """Instantiate a benchmark suite by name."""
    ...
```

This keeps CLI code thin and makes adding a new suite a one-line change
to the registry plus a new suite implementation.

______________________________________________________________________

### 6. Component-Targeted Design

The benchmark system must support running benchmarks “by component” in addition
to running full suites.

#### 6.1 Component-to-benchmark mapping

Define a declarative mapping from `BenchmarkComponent` to default
benchmarks (by `BenchmarkId`):

```python
COMPONENT_DEFAULTS: dict[BenchmarkComponent, list[BenchmarkId]] = {
    BenchmarkComponent.cpu: [BenchmarkId(suite="micro", name="cpu_flops")],
    BenchmarkComponent.gpu: [BenchmarkId(suite="micro", name="gpu_flops")],
    BenchmarkComponent.ram: [BenchmarkId(suite="micro", name="mem_bandwidth")],
    BenchmarkComponent.storage: [BenchmarkId(suite="micro", name="disk_io")],
    BenchmarkComponent.network: [BenchmarkId(suite="micro", name="net_bandwidth")],
    BenchmarkComponent.system: [
        BenchmarkId(suite="pepbench", name="end_to_end_infer")
    ],
}
```

When a user requests `--component cpu`, the runner expands this mapping
into a concrete list of benchmarks to run.

#### 6.2 System snapshot integration

For context, the runner can optionally:

- Capture a `ListOutput` snapshot via the existing `list` backends.
- Attach it to each `BenchmarkRunResult.system`.

This turns each benchmark JSON into a portable record of both:

- The workload that was run.
- The system it was run on.

______________________________________________________________________

### 7. CLI Design (`warpt benchmark`)

The existing `benchmark` stub in `warpt/cli.py` is replaced with a real
command that mirrors the style of `list` and `stress`.

#### 7.1 Command structure

- Convert `benchmark` into a group with subcommands:
  - `warpt benchmark list`
  - `warpt benchmark describe`
  - `warpt benchmark run`

Alternatively, the initial implementation can expose only
`warpt benchmark run` and add the others later, but the design
assumes a group structure for clarity.

#### 7.2 `warpt benchmark list`

- **Purpose**: enumerate available suites and benchmarks.
- **Key options**:
  - `--suite mlperf` (filter by suite).
  - `--component cpu,gpu` (filter by component).
  - `--json` (output machine-readable listing).

This is wired to `BenchmarkSuite.list_benchmarks()` for each
registered suite.

#### 7.3 `warpt benchmark describe`

- **Purpose**: show detailed information about a single benchmark.
- **Key options**:
  - `--suite mlperf`.
  - `--benchmark resnet50_infer`.

Output fields come from `BenchmarkSpec`:

- Description, components, kind, default duration, external tool, etc.

#### 7.4 `warpt benchmark run`

- **Purpose**: execute benchmarks and export results.

- **Example usage**:

  - Suite-centric:
    - `warpt benchmark run --suite mlperf`
    - `warpt benchmark run --suite linpack --benchmark linpack_fp64_cpu`
  - Component-centric:
    - `warpt benchmark run --component cpu`
    - `warpt benchmark run --component cpu --component gpu`
  - Hybrid:
    - `warpt benchmark run --suite pepbench --component gpu`

- **Key options (initial set)**:

  - `--suite TEXT`:
    - Name of suite to run (e.g. `mlperf`, `linpack`, `pepbench`, `biobench`, `micro`).
  - `--benchmark TEXT` (repeatable):
    - Specific benchmark name within the suite. If omitted, runs suite defaults.
  - `--component TEXT` (repeatable, supports comma-separated values):
    - Components to target; used when no suite is provided or to filter suite benchmarks.
  - `--duration SECONDS`:
    - Default duration for synthetic/micro benchmarks (where applicable).
  - `--burnin-seconds SECONDS`:
    - Warmup period before measurements (reuse `DEFAULT_BURNIN_SECONDS` semantics).
  - `--export` / `--export-file PATH`:
    - Structured JSON export, consistent with `list` and `stress`.
  - `--dry-run`:
    - Print which benchmarks would be run and exit.

- **Export behavior**:

  - Default filename pattern:
    - `warpt_benchmark_<SUITE_OR_COMPONENT>_<TIMESTAMP>_<RANDOM>.json`
  - Contents:
    - A single `BenchmarkSuiteRun` instance serialized via
      `model_dump_json(indent=2)`.

### 7.5 Background Monitoring

Resource telemetry can be collected while benchmarks or stress tests run by
reusing `warpt.monitoring.SystemMonitorDaemon` in the background. The daemon
captures CPU usage, wired/available RAM, and (when `pynvml` is installed)
NVIDIA GPU utilization, power, and GUIDs.

- `--monitor` (stress only, benchmark run will eventually share the same flag)
  keeps the daemon alive while work is executing.
- `--monitor-interval <seconds>` sets the sampling cadence (default `1.0s`). Lower
  values increase fidelity; higher values reduce monitoring overhead.
- `--monitor-output <path>` writes the collected timeline to JSON for
  correlation with benchmark results. Without a path the monitor still runs but
  nothing is persisted.

Example:

```
warpt stress gpu --monitor --monitor-interval 0.5 \
    --monitor-output monitor-timeline.json
```

`monitor-timeline.json` contains records such as:

```json
[
  {
    "timestamp": "2025-12-05T14:00:01",
    "cpu_utilization_percent": 12.3,
    "total_memory_bytes": 68719476736,
    "memory_utilization_percent": 32.4,
    "gpu_usage": [
      {
        "index": 0,
        "model": "NVIDIA RTX 4090",
        "utilization_percent": 84.2,
        "memory_utilization_percent": 65.0,
        "power_watts": 310.5,
        "guid": "GPU-abcdef12-3456-7890-abcd-ef1234567890"
      }
    ]
  }
]
```

Basic tooling can load this JSON to augment benchmark/ stress results with a
timeline of GPU power and CPU/memory utilization. When `pynvml` is unavailable,
the daemon still records CPU and RAM metrics.

______________________________________________________________________

### 8. Suite Implementations

Each suite lives in `warpt/benchmark/suites/` and implements the
`BenchmarkSuite` interface.

#### 8.1 Microbench suite (`MicrobenchSuite`)

- Focused on fast, internal microbenchmarks:
  - CPU compute throughput.
  - GPU compute throughput (if available).
  - Memory bandwidth and latency.
  - Storage and network I/O.
- Likely reuses or wraps the existing `warpt.stress` compute helpers.
- Provides sensible defaults for component-driven runs.

This suite is the easiest starting point and should be implemented first.

#### 8.2 Linpack suite (`LinpackSuite`)

- Wraps external Linpack binaries:
  - CPU Linpack (`linpack_fp64_cpu`).
  - GPU Linpack (`linpack_fp64_gpu`), where applicable.
- Implementation details:
  - Call external binaries via `subprocess`.
  - Parse stdout into `BenchmarkMetric` instances (e.g. GFLOP/s).
  - Map failures or missing binaries to `success=False` with an error message.

#### 8.3 Pepbench suite (`PepbenchSuite`)

- New benchmark focused on ML workloads defined within Warpt:
  - Could include:
    - Small LLM inference benchmarks.
    - Data pipeline throughput tests.
    - Tokenization or embedding throughput.
- Likely a mix of `application` and `domain` kind benchmarks.
- Should be designed to:
  - Be runnable without huge datasets.
  - Provide clear, interpretable metrics (throughput, latency, memory usage).

#### 8.4 Biobench suite (`BiobenchSuite`)

- Domain-specific suite for bio/omics workloads:
  - Examples:
    - Sequence alignment.
    - Variant calling.
    - Genome assembly microcases.
- Emphasis on:
  - Realistic, domain-relevant workloads.
  - Metrics such as reads/s, variants/s, or similar domain metrics.

#### 8.5 MLPerf suite (`MlperfSuite`)

- Integration with MLPerf, potentially via:
  - Official MLPerf reference containers.
  - Helper scripts for invoking specific benchmarks.
- Benchmarks might include:
  - `resnet50_train`, `resnet50_infer`.
  - `bert_infer`, etc.
- This suite is more complex:
  - Requires careful environment checks.
  - May need configuration files for dataset paths and container runtimes.

Implementation of `MlperfSuite` can be staged after the micro and linpack
work is complete.

______________________________________________________________________

### 9. Interaction with Existing Warpt Functionality

#### 9.1 Reusing `warpt list` data

- Benchmarks can call into the same backends used by `warpt list` to
  capture a `ListOutput` snapshot before or after running.
- This snapshot:
  - Provides hardware and software context (CPU, GPU, RAM, storage, Docker, etc).
  - Is attached as `BenchmarkRunResult.system`.

#### 9.2 Reusing `warpt stress` components

- Existing stress code (`warpt/stress`) already:
  - Drives CPU and GPU with synthetic workloads.
  - Handles durations and monitoring.
- Microbenchmarks can wrap these primitives:
  - For example, a CPU FLOPS microbenchmark that calls into `cpu_compute`
    with a fixed problem size and records GFLOP/s.

This reduces duplication and keeps stress and benchmark behavior aligned.

______________________________________________________________________

### 10. Implementation Phases

- **Phase 1 – Skeleton and microbench**

  - Add `benchmark_models.py` with core models and enums.
  - Add `warpt/benchmark/base.py` and a minimal registry.
  - Implement `MicrobenchSuite` for CPU/GPU/RAM (and optionally storage/network).
  - Replace the current `benchmark` stub with:
    - `warpt benchmark run --component cpu` (and friends).
  - Support JSON export of `BenchmarkSuiteRun`.

- **Phase 2 – Linpack and Pepbench**

  - Implement `LinpackSuite` and wire to external binaries.
  - Implement an initial `PepbenchSuite` with a small number of ML workloads.
  - Extend CLI to allow:
    - `warpt benchmark run --suite linpack`.
    - `warpt benchmark run --suite pepbench`.

- **Phase 3 – MLPerf and Biobench**

  - Add `MlperfSuite` integration and configuration handling.
  - Add `BiobenchSuite` with realistic bio workloads.
  - Enhance CLI ergonomics:
    - `warpt benchmark list` and `warpt benchmark describe`.
  - Iterate on component-to-benchmark defaults based on real usage.

______________________________________________________________________

### 11. Open Questions and Future Enhancements

- **Configuration story**:
  - How should users specify dataset locations, container runtime options,
    and suite-specific parameters (e.g. via config file, env vars, or CLI)?
- **Concurrency**:
  - Should multi-benchmark runs execute sequentially or in parallel,
    especially for microbenchmarks on multi-GPU systems?
- **Result aggregation**:
  - Do we want optional summary metrics across a suite run
    (e.g. “overall GPU score”), or keep results strictly per benchmark?
- **History and comparison**:
  - Future: commands like `warpt benchmark compare` to diff two JSON results.

These can be addressed incrementally once the core benchmark system is in place.
