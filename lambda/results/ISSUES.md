# Lambda Multi-GPU Test Issues

Issues encountered during testing on Lambda Labs. Each issue documents what failed, the full error, what was tried, and what code changes are needed to fix off-machine.

---

### Issue 1: NumPy 2.x incompatible with system PyTorch
**Status:** worked-around
**Step:** setup
**Command:** `bash lambda/setup.sh`
**Error:**
```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.
```
**Context:** Lambda instance has PyTorch 2.7.0 compiled against NumPy 1.x. `pip install warpt[stress,daemon]` pulls numpy 2.2.6.
**Tried:** `pip install 'numpy<2'` — installs numpy 1.26.4
**Root cause:** warpt's pyproject.toml specifies `numpy>=1.24.0` with no upper bound; pip resolves to 2.x which is ABI-incompatible with the system PyTorch.
**Fix needed:** Pin `numpy>=1.24.0,<2` in pyproject.toml, or detect the system PyTorch's numpy requirement during setup.

---

### Issue 2: `total_mem` attribute error in setup.sh and run.sh
**Status:** worked-around
**Step:** setup
**Command:** `bash lambda/setup.sh` (also in `lambda/run.sh` line 70)
**Error:**
```
AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'. Did you mean: 'total_memory'?
```
**Context:** PyTorch 2.7.0 on Lambda, 8x V100-SXM2-16GB
**Tried:** Changed `total_mem` to `total_memory` in run.sh line 70
**Root cause:** The attribute is `torch.cuda.get_device_properties(i).total_memory`, not `total_mem`. Likely a rename between PyTorch versions.
**Fix needed:** `lambda/run.sh:70` — change `total_mem` to `total_memory`. Also check `lambda/setup.sh` for the same issue.

---

### Issue 3: NCCL hangs on init_process_group — InfiniBand misconfiguration
**Status:** worked-around (code fix applied)
**Step:** stress (GPUMultiScalingTest)
**Command:** `warpt stress -t GPUMultiScalingTest`
**Error:**
```
[rank7]:[W410 21:36:52.176066491 socket.cpp:460] [c10d] waitForInput: poll for socket
SocketImpl(fd=5, addr=[localhost]:39284, remote=[localhost]:47115) returned 0, likely a timeout
[rank7]:[W410 21:36:52.177160212 socket.cpp:485] [c10d] waitForInput: socket timed out after 300000ms
```
**Context:** 8x V100-SXM2-16GB. Machine has Mellanox ConnectX mlx5_0 (MT4120) with IB hardware active in Ethernet/RoCE mode (`ibstat` shows State: Active, Link layer: Ethernet). NCCL detects IB hardware and tries to use it for bootstrap/communication but it's not configured for IB transport.
**Tried:**
1. `NCCL_SOCKET_IFNAME=lo` alone — GPUs hit 100% util but still hung (env var only affects NCCL, not PyTorch TCPStore)
2. `NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo` as env vars — still hung because env vars were set AFTER `import torch` in worker
3. `NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1` — still hung (same import ordering issue)
4. Setting env vars BEFORE `import torch` in `_worker_fn` — **this worked**
5. Verified with `torchrun --nproc_per_node=8` minimal test — NCCL works fine
6. Verified with `mp.spawn` + Manager dict minimal test — also works fine

**Root cause:** Two issues combined:
- Lambda V100 instances have Mellanox IB hardware in RoCE mode. NCCL tries to use IB for bootstrap and hangs.
- The `_worker_fn` in `gpu_multi_scaling.py` imported `torch` (line 90) BEFORE setting `NCCL_IB_DISABLE` and `NCCL_SOCKET_IFNAME` (lines 106-107). Since `mp.spawn` uses the "spawn" start method, child processes re-import everything from scratch. `libnccl.so` reads env vars at load time during `import torch`, so env vars set after import are ignored.

**Fix needed:** In `warpt/stress/gpu_multi_scaling.py`, in `_worker_fn()`:
1. Move all `os.environ` NCCL settings BEFORE `import torch` (currently they're after)
2. Add `NCCL_IB_DISABLE` and `NCCL_SOCKET_IFNAME` defaults for robustness:
```python
# BEFORE import torch:
os.environ["NCCL_IB_DISABLE"] = os.environ.get("NCCL_IB_DISABLE", "1")
os.environ["NCCL_SOCKET_IFNAME"] = os.environ.get("NCCL_SOCKET_IFNAME", "lo")
# THEN:
import torch
import torch.distributed as dist
```

---

### Issue 4: NCCL P2P isend/irecv deadlocks in halo exchange
**Status:** worked-around (code fix applied)
**Step:** stress (GPUMultiScalingTest multi-GPU phase)
**Command:** `warpt stress -t GPUMultiScalingTest` (after Issue 3 fix)
**Error:** All 8 workers complete single-GPU baseline then hang silently in the multi-GPU halo exchange phase. No error output — GPU utilization drops to 0%, workers spin at 100% CPU.
**Context:** 8x V100-SXM2-16GB, NCCL_IB_DISABLE=1, NCCL_SOCKET_IFNAME=lo
**Tried:**
1. `NCCL_P2P_DISABLE=1` — still hangs
2. `batch_isend_irecv()` instead of individual `isend`/`irecv` — still hangs
3. Replaced isend/irecv with `dist.all_gather()` — **this worked**

**Root cause:** NCCL's `isend`/`irecv` implementation has reliability issues on certain configurations (IB disabled, socket-only). Even `batch_isend_irecv` doesn't help. The point-to-point operations are implemented as group calls internally and deadlock when IB is disabled and only socket transport is available.

**Fix needed:** In `warpt/stress/gpu_multi_scaling.py`, replace the halo exchange `isend`/`irecv` pattern (lines ~269-275) with `dist.all_gather()`:
```python
# Replace:
ops.append(dist.isend(send_buf_right, dst=right_rank))
ops.append(dist.irecv(recv_buf_left, src=left_rank))
...
# With:
all_left = [torch.zeros_like(send_buf_left) for _ in range(world_size)]
all_right = [torch.zeros_like(send_buf_right) for _ in range(world_size)]
dist.all_gather(all_left, send_buf_left)
dist.all_gather(all_right, send_buf_right)
recv_buf_left[:] = all_right[left_rank]
recv_buf_right[:] = all_left[right_rank]
```
Same fix needed for the P2P comm benchmark (~line 366-369).

---

### Issue 5: Collective ordering mismatch — missing barriers between phases
**Status:** worked-around (code fix applied)
**Step:** stress (GPUMultiScalingTest multi-GPU phase)
**Command:** `warpt stress -t GPUMultiScalingTest` (after Issues 3+4 fixes)
**Error:** Multi-GPU loop completes ~160 iterations successfully, then hangs. Workers spin at 100% CPU, 0% GPU util.
**Context:** 8x V100-SXM2-16GB, all_gather working correctly for individual iterations
**Tried:** Added per-iteration debug prints — confirmed all_gather and all_reduce work for hundreds of iterations. Hang occurs when some workers exit the while loop and proceed to the comm benchmark phase while others are still in the multi-GPU loop.

**Root cause:** The multi-GPU while loop (`while (time.time() - multi_start) < multi_duration`) runs different numbers of iterations on different ranks because workers enter the loop at slightly different times. When the fastest rank exits the loop and calls its first comm benchmark `all_reduce`, while the slowest rank calls its last multi-GPU `all_gather`, NCCL sees a collective ordering mismatch and deadlocks.

Same issue exists in the comm benchmark section — the allreduce benchmark while loop and the P2P benchmark while loop can produce different iteration counts across ranks.

**Fix needed:** In `warpt/stress/gpu_multi_scaling.py`:
1. Add `dist.barrier()` after the multi-GPU while loop (after `multi_elapsed = time.time() - multi_start`)
2. Add `dist.barrier()` between each comm benchmark while loop (between allreduce benchmark and P2P benchmark, and after P2P benchmark before next message size)

---

### Issue 6: NCCL watchdog timeout on large comm benchmarks
**Status:** open
**Step:** stress (GPUMultiScalingTest comm benchmarks)
**Command:** `warpt stress -t GPUMultiScalingTest --duration 30`
**Error:**
```
Exception raised from ncclCommWatchdog at ./torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1902
process 4 terminated with signal SIGABRT
```
**Context:** 8x V100-SXM2-16GB, NCCL_IB_DISABLE=1. Test ran at 100% GPU for ~2.5 minutes then crashed.
**Tried:** Reduced max comm benchmark message size from 256MB to 16MB. The 256MB all_gather on 8 GPUs = 2GB total data over socket transport — too slow, triggers NCCL's 5-minute watchdog.

**Root cause:** With IB disabled and NVLink P2P disabled (socket-only fallback), large collective operations (256MB all_gather across 8 GPUs) exceed NCCL's default watchdog timeout. The `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` and `NCCL_TIMEOUT` defaults are too short for socket-only transport at scale.

**Fix needed:** In `warpt/stress/gpu_multi_scaling.py`:
1. Reduce default comm benchmark sizes from `[1KB, 1MB, 64MB, 256MB]` to `[1KB, 1MB, 16MB]` — or make them configurable
2. Increase `dist.init_process_group` timeout from 5 minutes to 10 minutes
3. Consider setting `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600` in the worker env

---

### Issue 7: Orphaned worker processes from killed tests
**Status:** worked-around
**Step:** stress (GPUMultiScalingTest)
**Command:** `timeout N warpt stress -t GPUMultiScalingTest` followed by kill
**Error:** After killing a stuck multi-GPU test, orphaned `multiprocessing.spawn` worker processes remain running at 100%+ CPU indefinitely, holding CUDA contexts on all GPUs.
**Context:** `timeout` sends SIGTERM to the parent process, which calls `mp.spawn(..., join=True)`. If the parent dies before workers, the workers become orphans. Up to 40+ orphaned python processes accumulated from repeated test attempts.
**Tried:** `pkill -9 -f "multiprocessing.spawn"` to clean up orphans. Must be done before retrying tests, otherwise new workers can't initialize CUDA on GPUs held by orphans.

**Root cause:** `mp.spawn` doesn't set up a process group or signal handler to clean up workers when the parent is killed. The `trap` in `run.sh` only handles the shell-level cleanup, not the internal mp.spawn workers.

**Fix needed:** Consider using `torchrun` instead of `mp.spawn` for multi-GPU tests, as torchrun handles process lifecycle and cleanup properly. Alternatively, add a signal handler in the parent that kills the worker process group.

---

### Issue 8: Daemon only detects threshold breaches on 2 of 8 GPUs during sequential stress test
**Status:** open
**Step:** daemon
**Command:** `warpt stress -t GPUMatMulTest --device-id 0,1,2,3,4,5,6,7 --duration 15` (run while daemon monitors)
**Error:** Not an error — a test design problem. Only 2 of 8 GPUs (GPU 0 and GPU 1) triggered threshold breach events. GPUs 2-7 were never detected.
**Context:** 8x V100-SXM2-16GB. Daemon monitoring with default thresholds (utilization >80% sustained for ~15s). The stress test runs GPUs **sequentially** — GPU 0 for 20s (5s warmup + 15s test), then GPU 1, then GPU 2, etc. Total wall time ~160s.

**What the DB captured:**
```
gpu_profiles: 8 rows (all 8 GPUs registered correctly)
vitals:       12 rows (heartbeat snapshots)
events:       3 rows (threshold breaches)
cases:        2 rows (diagnostic cases)
```

Event detail:
- Event 1: GPU 0 (eb2cb3) utilization 100% for 15s → Case 1 (ts: 23:44:58)
- Event 2: GPU 0 (eb2cb3) utilization 100% for 15s → Case 1 (ts: 23:48:23) — second breach on same GPU
- Event 3: GPU 1 (76d489) utilization 100% for 15s → Case 2 (ts: 23:48:42)

GPUs 2-7: zero events. They each ran a 15s burst, but the daemon polls on an interval (~15s heartbeat). By the time the daemon's next poll fired, the GPU had already finished its burst and returned to idle. The daemon never observed sustained >80% utilization on those GPUs.

**Root cause:** The `warpt stress` CLI runs tests sequentially per GPU (GPU 0 first, then GPU 1, etc.). Each GPU is only hot for ~20s (warmup + test). The daemon's heartbeat interval is ~15s, so there's only a 1-2 heartbeat window to catch each GPU. If the burst starts right after a poll, the daemon won't see it until the next poll — by which time the GPU may already be done.

GPU 0 got caught twice because it was tested first (before the daemon's initial poll delay) and happened to align with two polling windows. GPU 1 was caught once. GPUs 2-7 all fell between polls.

**Fix needed:** The daemon test workload should run all GPUs **simultaneously** (in parallel) rather than sequentially. This ensures all 8 GPUs are at 100% utilization at the same time, giving the daemon multiple heartbeat windows to observe them all. Two approaches:

1. **Parallel shell approach:** Launch 8 separate `warpt stress -t GPUMatMulTest --device-id N --duration 30` processes in the background, then `wait` for all:
   ```bash
   for gpu in $(seq 0 7); do
       warpt stress -t GPUMatMulTest --device-id $gpu --duration 30 &
   done
   wait
   ```

2. **Single-process approach:** Use a test that exercises all GPUs simultaneously (like `GPUMultiScalingTest`, once its NCCL issues are fixed), or add a `--parallel` flag to the stress CLI that runs per-GPU tests concurrently instead of sequentially.

Also consider: the `run.sh` daemon section specifies `--duration 90` which is excessive. With parallel execution, `--duration 30` gives the daemon plenty of heartbeat windows (at least 2 polls at 15s interval) to detect all 8 GPUs.

---

### Issue 9: `run.sh` references non-existent test names
**Status:** worked-around (code fix applied)
**Step:** daemon / stress
**Command:** `bash lambda/run.sh daemon`
**Error:**
```
Error: Test 'GPUComputeTest' not found.
Use 'warpt stress --list' to see available tests.
```
**Context:** `lambda/run.sh` references `GPUComputeTest` and `GPUMemoryTest` in multiple places. The actual test names are `GPUMemoryBandwidthTest` (no `GPUComputeTest` exists — it may have been renamed or removed).
**Tried:** Replaced all occurrences:
- `GPUComputeTest` → `GPUMemoryBandwidthTest` (lines 154, 218)
- `GPUMemoryTest` → `GPUMemoryBandwidthTest` (line 156)

**Root cause:** Test class was renamed but `run.sh` wasn't updated to match.

**Fix needed:** Already applied in `lambda/run.sh`. Verify test names match `warpt stress --list` output. Available GPU tests are: `GPUCFDSimulationTest`, `GPUFP64ComputeTest`, `GPUMatMulTest`, `GPUMemoryBandwidthTest`, `GPUMultiScalingTest`, `GPUPrecisionTest`.
