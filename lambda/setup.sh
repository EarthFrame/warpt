#!/usr/bin/env bash
# Lambda Labs multi-GPU setup for warpt
#
# Usage:
#   git clone --branch lambda-multigpu https://github.com/EarthFrame/warpt.git
#   cd warpt
#   bash lambda/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

cd "${REPO_DIR}"

echo "============================================"
echo "  warpt Lambda Labs Setup"
echo "  repo: ${REPO_DIR}"
echo "  branch: $(git rev-parse --abbrev-ref HEAD)"
echo "  commit: $(git rev-parse --short HEAD)"
echo "============================================"

# ── 1. System checks ──────────────────────────────────────────────
echo ""
echo "[1/5] System checks..."

if ! command -v nvidia-smi &>/dev/null; then
    echo "FATAL: nvidia-smi not found. Need NVIDIA drivers."
    exit 1
fi
echo "  nvidia-smi: OK"

GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "  GPUs detected: ${GPU_COUNT}"
nvidia-smi -L

if ! command -v nvcc &>/dev/null; then
    echo "  WARNING: nvcc not found (PyTorch may still work with bundled CUDA)"
else
    echo "  CUDA: $(nvcc --version | grep release | awk '{print $6}')"
fi

PYTHON="${PYTHON:-python3}"
PY_VERSION=$($PYTHON --version 2>&1)
echo "  Python: ${PY_VERSION}"

# ── 2. Install warpt with stress + daemon extras ──────────────────
echo ""
echo "[2/5] Installing warpt[stress,daemon]..."

$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -e ".[stress,daemon]"

# ── 3. Verify PyTorch + CUDA ──────────────────────────────────────
echo ""
echo "[3/5] Verifying PyTorch CUDA..."

$PYTHON -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version: {torch.version.cuda}')
print(f'  Device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
    print(f'         VRAM: {mem:.1f} GB')
"

# ── 4. Verify NCCL (required for multi-GPU) ──────────────────────
echo ""
echo "[4/5] Verifying NCCL..."

$PYTHON -c "
import torch.distributed as dist
nccl = dist.is_nccl_available()
print(f'  NCCL available: {nccl}')
if not nccl:
    print('  FATAL: NCCL not available — multi-GPU tests will fail')
    exit(1)
print('  NCCL: OK')
"

# ── 5. Verify warpt CLI ──────────────────────────────────────────
echo ""
echo "[5/5] Verifying warpt CLI..."

warpt --version || $PYTHON -m warpt --version
echo ""
warpt stress --list -c accelerator

# ── GPU Topology ──────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  GPU Topology"
echo "============================================"
nvidia-smi topo -m 2>/dev/null || echo "(topology query not supported)"

echo ""
echo "============================================"
echo "  NVLink Status"
echo "============================================"
nvidia-smi nvlink --status 2>/dev/null || echo "(NVLink query not supported)"

# ── Install Claude Code (optional) ────────────────────────────────
echo ""
echo "============================================"
echo "  Claude Code (optional)"
echo "============================================"
if command -v claude &>/dev/null; then
    echo "  Claude Code already installed: $(claude --version 2>/dev/null || echo 'unknown')"
elif command -v npm &>/dev/null; then
    echo "  npm found. Install Claude Code with:"
    echo "    npm install -g @anthropic-ai/claude-code"
    echo "    claude    # opens browser login, no API key needed"
else
    echo "  npm not found. Install Node.js first, then:"
    echo "    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -"
    echo "    sudo apt-get install -y nodejs"
    echo "    npm install -g @anthropic-ai/claude-code"
    echo "    claude    # opens browser login, no API key needed"
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Run:  bash lambda/run.sh"
echo "============================================"
