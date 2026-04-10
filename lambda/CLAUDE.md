# Lambda Multi-GPU Test Runbook

You are on a Lambda Labs instance with multiple NVIDIA GPUs. This directory contains everything needed to run warpt multi-GPU stress tests and daemon monitoring, capture detailed results, and push them back to the repo.

## CRITICAL: Issue Tracking

**When anything goes wrong — a script fails, a test errors, an import breaks, a command hangs, anything unexpected — you MUST document it immediately in `lambda/results/ISSUES.md`.**

For every issue, record:
1. **What failed** — the exact command or step
2. **Error output** — full traceback or error message (copy it verbatim)
3. **Environment context** — GPU count, Python version, any relevant nvidia-smi output
4. **What you tried** — any workarounds or debugging steps attempted
5. **Root cause** (if identified) — why it failed
6. **Fix needed** — what code changes are required, with file paths and line numbers

Format each issue like this:
```markdown
### Issue N: <short title>
**Status:** open | worked-around | blocked
**Step:** setup / stress / daemon / push
**Command:** `the exact command that failed`
**Error:**
\```
paste full error output here
\```
**Context:** <GPU count, relevant versions, etc>
**Tried:** <what you attempted>
**Root cause:** <if known>
**Fix needed:** <specific code changes with file:line references>
```

Do NOT silently skip failures. Do NOT just retry and move on without documenting. The whole point is to capture what breaks so we can fix it off-machine. Even if you work around an issue, still document it.

After documenting, commit and push `ISSUES.md` along with any partial results so nothing is lost.

## Quick Start

```bash
# 1. Setup (run once after cloning)
bash lambda/setup.sh

# 2. Run all tests (stress + daemon)
bash lambda/run.sh

# 3. Push results back (always include ISSUES.md)
git add lambda/results/
git commit -m "lambda multi-gpu results $(date +%Y%m%d_%H%M%S)"
git push origin lambda-multigpu
```

## What the Scripts Do

### `lambda/setup.sh`
- Verifies NVIDIA drivers, CUDA, Python, PyTorch, NCCL
- Installs warpt with `pip install -e ".[stress,daemon]"`
- Prints GPU topology and NVLink status
- Shows Claude Code install instructions

### Claude Code on this machine
```bash
# Install
npm install -g @anthropic-ai/claude-code

# Login (opens browser, no API key needed)
claude

# If no browser available (headless), it prints a URL to open elsewhere
```
On first launch Claude opens an OAuth login in the browser. If the machine is headless it prints a URL you can open on your laptop. No `ANTHROPIC_API_KEY` needed.

### `lambda/run.sh [stress|daemon|all]`
Runs tests and captures everything into `lambda/results/<timestamp>/`.

**Stress mode** (`bash lambda/run.sh stress`):
- `GPUMultiScalingTest` — multi-GPU CFD with NCCL halo exchange, measures scaling efficiency
- Per-GPU baselines: MatMul, Compute, Memory, FP64, Precision, CFD on every GPU individually
- Continuous nvidia-smi dmon (1s interval) captures power, temp, utilization, clocks, throttle throughout

**Daemon mode** (`bash lambda/run.sh daemon`):
- Starts warpt daemon (threshold-based GPU monitoring → DuckDB)
- Runs 90s GPU workload to trigger threshold breaches
- Captures daemon status snapshots every 15s
- Stops daemon, copies DuckDB, dumps all tables to CSV + JSON:
  - `gpu_profiles` — GPU identity registry
  - `vitals` — system snapshots (heartbeats + threshold events)
  - `vitals_per_gpu` — expanded view (one row per GPU per timestamp)
  - `events` — threshold breach events with metadata
  - `cases` — diagnostic cases opened by breaches

**Default** (`bash lambda/run.sh`) runs both stress then daemon.

## Results Directory Structure

After a run, `lambda/results/<timestamp>/` contains:

```
├── environment.json          # Python, PyTorch, CUDA, NCCL, per-GPU specs
├── nvidia_smi.txt            # nvidia-smi at start
├── nvidia_smi_full_query.txt # nvidia-smi -q (all details)
├── nvidia_smi_final.txt      # nvidia-smi at end
├── gpu_details.csv           # per-GPU: clocks, power limits, ECC, PCIe, temp
├── topology.txt              # nvidia-smi topo -m (NVLink/PCIe matrix)
├── nvlink.txt                # NVLink status
├── system_info.txt           # CPU, RAM, kernel
├── nccl_env.txt              # NCCL environment variables
├── warpt_list.txt            # warpt device listing
│
├── multi_gpu_scaling.json    # GPUMultiScalingTest results (scaling efficiency,
│                             #   per-GPU TFLOPS, comm benchmarks, halo stats)
├── multi_gpu_scaling.yaml    # same in YAML
├── per_gpu_tests.json        # MatMul/Compute/Memory/FP64/Precision per GPU
├── per_gpu_tests.yaml
├── per_gpu_cfd.json          # CFD simulation per GPU (baseline for scaling)
│
├── daemon_workload_results.json  # stress test output during daemon monitoring
├── daemon_cases.log              # warpt daemon inspect --list
├── daemon_status_start.log
├── daemon_status_end.log
├── daemon_status_periodic.log    # snapshots every 15s during workload
│
├── warpt.db                  # full DuckDB database (copy for offline analysis)
├── warpt.db.wal
│
├── smi_logs/
│   ├── dmon.csv              # nvidia-smi dmon at 1s (power, temp, util, clocks)
│   └── periodic_smi.log     # full nvidia-smi every 5s
│
├── db_dumps/
│   ├── db_summary.json       # row counts per table
│   ├── gpu_profiles.csv
│   ├── gpu_profiles.json
│   ├── vitals.csv
│   ├── vitals.json
│   ├── vitals_per_gpu.csv    # one row per GPU per timestamp (best for analysis)
│   ├── vitals_per_gpu.json
│   ├── events.csv
│   ├── events.json
│   ├── events_detail.json    # events with full metadata
│   ├── cases.csv
│   ├── cases.json
│   └── cases_detail.json     # cases with all diagnostic fields
│
├── stress_multi_gpu.log      # stdout from multi-GPU test
├── stress_per_gpu.log        # stdout from per-GPU tests
├── stress_per_gpu_cfd.log
├── daemon_workload.log
│
└── manifest.json             # file inventory with sizes
```

## Config-Driven Alternative

Instead of CLI flags, you can run via config:
```bash
warpt stress --config lambda/multigpu.yaml -o lambda/results/config_run.json
```

## Key Things to Check in Results

1. **Scaling efficiency** — `multi_gpu_scaling.json` → `scaling_efficiency` (0-1, closer to 1 = better)
2. **Communication overhead** — `communication_fraction` and `comm_benchmarks` (allreduce/P2P bandwidth)
3. **Interconnect** — `topology.txt` shows NVLink vs PCIe between GPUs
4. **Thermal/power** — `smi_logs/dmon.csv` for continuous power draw and temperature
5. **Daemon threshold detection** — `db_dumps/events_detail.json` for breach events, `cases_detail.json` for opened cases
6. **Per-GPU consistency** — compare TFLOPS across GPUs in `per_gpu_tests.json` (should be similar)

## Offline Analysis with DuckDB

After pulling results locally, query the database directly:
```python
import duckdb
db = duckdb.connect("lambda/results/<timestamp>/warpt.db", read_only=True)

# Vitals timeline per GPU
db.sql("""
    SELECT v.ts, g.gpu_index, g.utilization_pct, g.power_w, g.temperature_c
    FROM vitals v, UNNEST(v.gpus) AS g
    ORDER BY v.ts, g.gpu_index
""").show()

# Threshold breach events
db.sql("SELECT ts, severity, gpu_guid, summary FROM events ORDER BY ts").show()

# Cases opened
db.sql("SELECT case_id, status, title, opened_at FROM cases").show()
```

## Pushing Results

After tests complete (or if blocked mid-run):
```bash
git add lambda/results/
git commit -m "lambda multi-gpu results $(date +%Y%m%d_%H%M%S)"
git push origin lambda-multigpu
```

**Always push even if tests failed partway through.** Partial results + ISSUES.md are more valuable than nothing. The `.gitignore` excludes temp PID files and the daemon's working `warpt_home/` directory. The DuckDB copy, all CSVs, JSONs, and logs are tracked.
