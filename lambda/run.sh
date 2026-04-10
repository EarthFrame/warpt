#!/usr/bin/env bash
# Lambda Labs multi-GPU test runner for warpt
# All results write to lambda/results/ inside the repo for easy git push.
#
# Usage:
#   bash lambda/run.sh              # full suite (stress + daemon)
#   bash lambda/run.sh stress       # stress test only
#   bash lambda/run.sh daemon       # daemon test only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
RESULTS_DIR="${SCRIPT_DIR}/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_DIR}/${TIMESTAMP}"
MODE="${1:-all}"

cd "${REPO_DIR}"

mkdir -p "${RUN_DIR}/smi_logs"
mkdir -p "${RUN_DIR}/db_dumps"

echo "============================================"
echo "  warpt Lambda Multi-GPU Test Run"
echo "  Mode: ${MODE}"
echo "  Results: ${RUN_DIR}"
echo "  Started: $(date -Iseconds)"
echo "  Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "  Commit: $(git rev-parse --short HEAD)"
echo "============================================"

# ── SYSTEM SNAPSHOT ───────────────────────────────────────────────
capture_system_info() {
    echo ""
    echo "Capturing system info..."

    nvidia-smi > "${RUN_DIR}/nvidia_smi.txt" 2>&1
    nvidia-smi -q > "${RUN_DIR}/nvidia_smi_full_query.txt" 2>&1
    nvidia-smi topo -m > "${RUN_DIR}/topology.txt" 2>&1 || true
    nvidia-smi nvlink --status > "${RUN_DIR}/nvlink.txt" 2>&1 || true

    # Per-GPU details: clocks, power limits, ECC, PCIe, temps
    nvidia-smi --query-gpu=index,gpu_name,gpu_uuid,memory.total,memory.free,\
power.limit,power.default_limit,clocks.max.sm,clocks.max.mem,\
pcie.link.gen.current,pcie.link.width.current,temperature.gpu,\
ecc.mode.current,compute_mode \
        --format=csv > "${RUN_DIR}/gpu_details.csv" 2>&1 || true

    # System info
    uname -a > "${RUN_DIR}/system_info.txt" 2>&1
    cat /proc/cpuinfo | head -50 >> "${RUN_DIR}/system_info.txt" 2>&1 || true
    free -h >> "${RUN_DIR}/system_info.txt" 2>&1 || true
    lscpu >> "${RUN_DIR}/system_info.txt" 2>&1 || true

    # Python / PyTorch / CUDA / NCCL versions
    python3 -c "
import torch, sys, json
info = {
    'python': sys.version,
    'pytorch': torch.__version__,
    'cuda': torch.version.cuda,
    'cudnn': str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
    'nccl': torch.cuda.nccl.version() if hasattr(torch.cuda, 'nccl') else None,
    'device_count': torch.cuda.device_count(),
    'devices': [
        {
            'index': i,
            'name': torch.cuda.get_device_name(i),
            'vram_gb': round(torch.cuda.get_device_properties(i).total_mem / (1024**3), 2),
            'compute_capability': f'{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}',
            'multi_processor_count': torch.cuda.get_device_properties(i).multi_processor_count,
        }
        for i in range(torch.cuda.device_count())
    ],
}
print(json.dumps(info, indent=2))
" > "${RUN_DIR}/environment.json" 2>&1

    # NCCL env vars (Lambda may set these)
    env | grep -i nccl > "${RUN_DIR}/nccl_env.txt" 2>&1 || echo "(no NCCL env vars)" > "${RUN_DIR}/nccl_env.txt"

    # warpt device list
    warpt list > "${RUN_DIR}/warpt_list.txt" 2>&1 || true
}

# ── CONTINUOUS nvidia-smi MONITORING ──────────────────────────────
start_gpu_monitor() {
    echo "  Starting continuous GPU monitor (nvidia-smi dmon, 1s interval)..."

    # dmon: per-second telemetry (power, utilization, clocks, violations, memory, ECC, temp)
    nvidia-smi dmon -s pucvmet -d 1 \
        > "${RUN_DIR}/smi_logs/dmon.csv" 2>&1 &
    echo "$!" > "${RUN_DIR}/.dmon_pid"

    # Full nvidia-smi snapshot every 5s
    (while true; do
        echo "=== $(date -Iseconds) ===" >> "${RUN_DIR}/smi_logs/periodic_smi.log"
        nvidia-smi >> "${RUN_DIR}/smi_logs/periodic_smi.log" 2>&1
        sleep 5
    done) &
    echo "$!" > "${RUN_DIR}/.smi_periodic_pid"
}

stop_gpu_monitor() {
    echo "  Stopping GPU monitors..."
    for pidfile in "${RUN_DIR}/.dmon_pid" "${RUN_DIR}/.smi_periodic_pid"; do
        if [ -f "${pidfile}" ]; then
            kill "$(cat "${pidfile}")" 2>/dev/null || true
            rm -f "${pidfile}"
        fi
    done
    echo "  GPU monitors stopped."
}

# Cleanup on exit
trap 'stop_gpu_monitor; warpt daemon stop 2>/dev/null || true' EXIT

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "GPUs: ${GPU_COUNT}"

capture_system_info

# ── STRESS TEST ───────────────────────────────────────────────────
run_stress() {
    echo ""
    echo "============================================"
    echo "  [1a] Multi-GPU Scaling Stress Test"
    echo "       GPUs: ${GPU_COUNT}"
    echo "       Start: $(date -Iseconds)"
    echo "============================================"
    echo ""

    start_gpu_monitor

    # Multi-GPU scaling test (auto-detects all GPUs, NCCL)
    warpt stress -t GPUMultiScalingTest \
        -o "${RUN_DIR}/multi_gpu_scaling.json" \
        -o "${RUN_DIR}/multi_gpu_scaling.yaml" \
        2>&1 | tee "${RUN_DIR}/stress_multi_gpu.log"

    echo ""
    echo "  Multi-GPU scaling test done."

    # Per-GPU individual tests
    echo ""
    echo "============================================"
    echo "  [1b] Per-GPU Individual Tests"
    echo "       Start: $(date -Iseconds)"
    echo "============================================"

    GPU_IDS=$(seq -s, 0 $((GPU_COUNT - 1)))

    warpt stress \
        -t GPUMatMulTest \
        -t GPUComputeTest \
        -t GPUMemoryTest \
        -t GPUFP64ComputeTest \
        -t GPUPrecisionTest \
        --device-id "${GPU_IDS}" \
        -o "${RUN_DIR}/per_gpu_tests.json" \
        -o "${RUN_DIR}/per_gpu_tests.yaml" \
        2>&1 | tee "${RUN_DIR}/stress_per_gpu.log"

    # Per-GPU CFD (single-GPU baseline vs multi-GPU scaling)
    echo ""
    echo "============================================"
    echo "  [1c] Per-GPU CFD Baseline"
    echo "       Start: $(date -Iseconds)"
    echo "============================================"

    warpt stress \
        -t GPUCFDSimulationTest \
        --device-id "${GPU_IDS}" \
        -o "${RUN_DIR}/per_gpu_cfd.json" \
        2>&1 | tee "${RUN_DIR}/stress_per_gpu_cfd.log"

    stop_gpu_monitor

    echo ""
    echo "  All stress tests complete. $(date -Iseconds)"
}

# ── DAEMON TEST ───────────────────────────────────────────────────
run_daemon() {
    echo ""
    echo "============================================"
    echo "  [2] Daemon Multi-GPU Monitoring Test"
    echo "      Start: $(date -Iseconds)"
    echo "============================================"
    echo ""

    # Ensure clean state
    warpt daemon stop 2>/dev/null || true
    sleep 1

    # Daemon home lives inside results dir
    export WARPT_DIR="${RUN_DIR}/warpt_home"
    mkdir -p "${WARPT_DIR}"

    cat > "${WARPT_DIR}/config.yaml" << 'YAMLEOF'
intelligence_enabled: false
YAMLEOF

    echo "  Starting daemon (WARPT_DIR=${WARPT_DIR})..."
    warpt daemon start
    sleep 3

    warpt daemon status | tee "${RUN_DIR}/daemon_status_start.log"
    echo ""

    start_gpu_monitor

    # Run GPU workload to generate vitals + trigger thresholds
    echo "  Running GPU workload (GPUComputeTest + GPUMatMulTest, 90s)..."
    GPU_IDS=$(seq -s, 0 $((GPU_COUNT - 1)))
    warpt stress \
        -t GPUComputeTest \
        -t GPUMatMulTest \
        --device-id "${GPU_IDS}" \
        --duration 90 \
        -o "${RUN_DIR}/daemon_workload_results.json" \
        2>&1 | tee "${RUN_DIR}/daemon_workload.log" &
    WORKLOAD_PID=$!

    # Periodic daemon status while workload runs
    (
        i=0
        while kill -0 ${WORKLOAD_PID} 2>/dev/null; do
            echo "--- snapshot ${i} $(date -Iseconds) ---" >> "${RUN_DIR}/daemon_status_periodic.log"
            warpt daemon status >> "${RUN_DIR}/daemon_status_periodic.log" 2>&1
            echo "" >> "${RUN_DIR}/daemon_status_periodic.log"
            sleep 15
            i=$((i + 1))
        done
    ) &
    STATUS_PID=$!

    wait ${WORKLOAD_PID} 2>/dev/null || true
    kill ${STATUS_PID} 2>/dev/null || true
    echo ""

    # Final heartbeats
    sleep 15

    warpt daemon status | tee "${RUN_DIR}/daemon_status_end.log"
    echo ""

    # Inspect cases
    warpt daemon inspect --list 2>&1 | tee "${RUN_DIR}/daemon_cases.log" || true
    echo ""

    stop_gpu_monitor

    # Stop daemon before DB copy
    warpt daemon stop
    sleep 2
    echo "  Daemon stopped."

    # Copy DuckDB (daemon is stopped, no WAL contention)
    cp "${WARPT_DIR}/warpt.db" "${RUN_DIR}/warpt.db" 2>/dev/null || true
    cp "${WARPT_DIR}/warpt.db.wal" "${RUN_DIR}/warpt.db.wal" 2>/dev/null || true

    # ── Full DuckDB table dumps ──────────────────────────────────
    echo ""
    echo "  Dumping DuckDB tables to CSV + JSON..."

    RUN_DIR="${RUN_DIR}" python3 << 'PYEOF'
import json, os, sys

try:
    import duckdb
except ImportError:
    print("  WARNING: duckdb not available, skipping DB dump")
    sys.exit(0)

run_dir = os.environ["RUN_DIR"]
db_path = os.path.join(run_dir, "warpt.db")
dump_dir = os.path.join(run_dir, "db_dumps")

if not os.path.exists(db_path):
    print("  WARNING: warpt.db not found, skipping DB dump")
    sys.exit(0)

db = duckdb.connect(db_path, read_only=True)

tables = ["gpu_profiles", "vitals", "events", "cases", "schema_migrations"]
summary = {}

for table in tables:
    try:
        count = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        summary[table] = count
        print(f"    {table}: {count} rows")

        if count == 0:
            continue

        # CSV
        db.execute(
            f"COPY {table} TO '{dump_dir}/{table}.csv' (HEADER, DELIMITER ',')"
        )

        # JSON via pandas
        rows = db.execute(f"SELECT * FROM {table}").fetchdf()
        rows.to_json(
            os.path.join(dump_dir, f"{table}.json"),
            orient="records", indent=2, date_format="iso", default_handler=str,
        )
    except Exception as e:
        print(f"    {table}: ERROR - {e}")
        summary[table] = f"error: {e}"

# Expanded vitals: one row per GPU per timestamp
try:
    gpu_vitals = db.execute("""
        SELECT
            v.ts,
            v.cpu_utilization_pct,
            v.mem_utilization_pct,
            v.total_power_w,
            v.collection_type,
            g.gpu_guid,
            g.gpu_index,
            g.utilization_pct AS gpu_util_pct,
            g.mem_utilization_pct AS gpu_mem_util_pct,
            g.power_w AS gpu_power_w,
            g.temperature_c AS gpu_temp_c,
            g.mem_used_bytes AS gpu_mem_used,
            g.mem_total_bytes AS gpu_mem_total
        FROM vitals v, UNNEST(v.gpus) AS g
        ORDER BY v.ts, g.gpu_index
    """).fetchdf()

    gpu_vitals.to_csv(os.path.join(dump_dir, "vitals_per_gpu.csv"), index=False)
    gpu_vitals.to_json(
        os.path.join(dump_dir, "vitals_per_gpu.json"),
        orient="records", indent=2, date_format="iso", default_handler=str,
    )
    print(f"    vitals_per_gpu: {len(gpu_vitals)} rows (expanded)")
except Exception as e:
    print(f"    vitals_per_gpu: ERROR - {e}")

# Events detail
try:
    events_detail = db.execute("""
        SELECT event_id, ts, kind, severity, gpu_guid, summary,
               case_id, triggered_by, metadata
        FROM events ORDER BY ts
    """).fetchdf()
    events_detail.to_json(
        os.path.join(dump_dir, "events_detail.json"),
        orient="records", indent=2, date_format="iso", default_handler=str,
    )
except Exception as e:
    print(f"    events_detail: ERROR - {e}")

# Cases detail
try:
    cases_detail = db.execute("SELECT * FROM cases ORDER BY opened_at").fetchdf()
    cases_detail.to_json(
        os.path.join(dump_dir, "cases_detail.json"),
        orient="records", indent=2, date_format="iso", default_handler=str,
    )
except Exception as e:
    print(f"    cases_detail: ERROR - {e}")

with open(os.path.join(dump_dir, "db_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

db.close()
print("    DB dump complete.")
PYEOF

    echo ""
    echo "  Daemon test complete. $(date -Iseconds)"
}

# ── DISPATCH ──────────────────────────────────────────────────────
case "${MODE}" in
    stress)
        run_stress
        ;;
    daemon)
        run_daemon
        ;;
    all)
        run_stress
        run_daemon
        ;;
    *)
        echo "Usage: $0 [stress|daemon|all]"
        exit 1
        ;;
esac

# ── PACKAGE RESULTS ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Packaging Results"
echo "============================================"

nvidia-smi > "${RUN_DIR}/nvidia_smi_final.txt" 2>&1

# Manifest
python3 -c "
import os, json, datetime
run_dir = '${RUN_DIR}'
files = {}
for root, dirs, fnames in os.walk(run_dir):
    for f in fnames:
        path = os.path.join(root, f)
        rel = os.path.relpath(path, run_dir)
        files[rel] = {
            'size_bytes': os.path.getsize(path),
            'size_human': f'{os.path.getsize(path) / 1024:.1f} KB',
        }
manifest = {
    'timestamp': datetime.datetime.now().isoformat(),
    'run_dir': run_dir,
    'file_count': len(files),
    'total_size_bytes': sum(f['size_bytes'] for f in files.values()),
    'files': files,
}
with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2))
"

echo ""
echo "============================================"
echo "  Run Complete!"
echo "  Finished: $(date -Iseconds)"
echo "============================================"
echo ""
echo "Results in: ${RUN_DIR}/"
ls -lhR "${RUN_DIR}/"
echo ""
echo "To push results:"
echo "  cd ${REPO_DIR}"
echo "  git add lambda/results/"
echo "  git commit -m 'lambda multi-gpu test results ${TIMESTAMP}'"
echo "  git push origin lambda-multigpu"
