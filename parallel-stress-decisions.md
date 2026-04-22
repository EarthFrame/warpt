# Parallel Stress Test Execution — Design Decisions

Decisions from grill session on moving `warpt stress` from sequential to parallel GPU execution.

---

## Context

`TestRunner.run()` executes tests in a flat `for` loop. When `--device-id 0,1,2,3` is passed, `stress_cmd.py` expands it into separate entries (e.g. `GPUMatMulTest (GPU 0)`, `GPUMatMulTest (GPU 1)`, ...) and they all run one after another. Total time = N GPUs × test duration.

## Decisions

### 1. Where does parallelism live?
**Decision: `TestRunner` (option B)**

Fix it in `runner.py` so every multi-device invocation benefits automatically. No shell workarounds, no new CLI flags.

### 2. What gets parallelized?
**Decision: Same test across GPUs only, sequential between test types**

`warpt stress -t MatMul -t MemBW --device-id 0,1,2,3` runs as:
- Round 1: MatMul on GPUs 0,1,2,3 simultaneously
- Round 2: MemBW on GPUs 0,1,2,3 simultaneously

Never two different tests on the same GPU at the same time — that would contaminate results (compute contention, thermal interference).

### 3. Threading or multiprocessing?
**Decision: Threads**

- GPU compute releases the GIL, so threads achieve real parallelism where it matters
- Result collection is simple (shared memory, just needs a lock)
- No orphan process risk (threads die with the process, unlike `mp.spawn` which caused Issue 7 on Lambda)
- Each thread calls `torch.cuda.set_device(N)` which creates a separate CUDA context per GPU automatically

### 4. CLI interface change?
**Decision: No new flag, parallel by default**

When multiple device IDs are passed, the runner automatically parallelizes same-test groups. No `--parallel` flag needed. Sequential was never intentional — it was just the simplest initial implementation.

### 5. GPUMultiScalingTest handling
**Decision: Excluded from parallel grouping, runs alone**

Already special-cased in `stress_cmd.py:454` — it manages its own multi-GPU coordination via NCCL and is never expanded per-device. It runs sequentially after all parallel groups finish.

### 6. Error handling
**Decision: Let other GPUs continue**

If GPU 2's test crashes, catch the exception in that thread, record it as an error in results, let GPUs 0,1,3 finish normally. Same pattern as today's sequential runner (`runner.py:141` catches per-test exceptions).

### 7. Result collection thread safety
**Decision: `threading.Lock` around `TestResults` writes**

Each thread writes to a unique key (`"GPUMatMulTest (GPU 0)"` etc.) so no data contention — just dict mutation safety. Lock is held for microseconds during the dict write, not during GPU compute. Thread-local variables hold result data safely while waiting for the lock.

### 8. How does the runner know what to parallelize?
**Decision: `stress_cmd.py` passes groups explicitly (option B)**

The CLI already knows the grouping because it does the device expansion. Pass structured groups to the runner rather than a flat list, so the runner doesn't have to reverse-engineer groups from string parsing.

---

## Open / Deferred

### Q8: What API shape does the runner expose for parallel groups?

Options to decide:
- **A) New method:** `runner.add_parallel_group(test_cls, device_ids, config)` — explicitly tells the runner "run this test on these GPUs in parallel"
- **B) New `run()` input:** `runner.run()` accepts a list of groups instead of iterating the flat `self._tests` list
- **C) Keep `add_test` as-is, add grouping logic:** runner inspects queued tests and auto-groups by test class name + different device_ids

Need to decide: does `stress_cmd.py` build the groups and pass them in, or does the runner figure it out from the existing flat list?

### Other deferred items

- **`poll_interval` bug in VitalsNurse** — stored and logged but never used; not wired to the monitor subprocess's `--interval` flag. Separate fix.
- **Case closure / cooldown** — no mechanism to close cases or enforce cooldown between events. Phase 2 (Attending agent).
