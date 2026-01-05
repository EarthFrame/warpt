# Tuning HPL Performance on Apple Silicon

This guide explains how to get the best performance and efficiency results when running HPL on macOS using `warpt`.

## 1. Docker Resource Allocation

By default, Docker Desktop on Mac often limits the virtual machine to 2 CPUs and 2GB of RAM. Even if `warpt` detects 8 cores and attempts to run 8 processes, they will be throttled by the VM limits.

**Recommended Settings:**

- **CPUs**: Set to the number of Performance cores (e.g., 4 or 6) or all cores (8+).
- **Memory**: Set to at least 8GB (higher if running large problem sizes).

To change these:

1. Open **Docker Desktop Settings**.
1. Navigate to **Resources**.
1. Adjust the sliders and click **Apply & Restart**.

## 2. Choosing the Right Problem Size (N)

The problem size $N$ is the most critical factor for reaching peak GFLOPS. If $N$ is too small (e.g., 5120), the overhead of process communication and setup dominates the actual computation time.

**Memory Calculation:**
HPL stores the matrix in double precision (8 bytes per element).
Total RAM usage $\\approx N^2 \\times 8$ bytes.

| N | Memory Usage | Target Hardware |
|---|---|---|
| 10240 | ~0.8 GB | Smoke Test |
| 20480 | ~3.2 GB | 8GB RAM Macs |
| 30000 | ~7.2 GB | 16GB RAM Macs |
| 40960 | ~13.4 GB | 24GB+ RAM Macs |

**Tip:** Aim for a run that takes at least 1–2 minutes to allow the CPU to reach a stable thermal/power state.

## 3. Understanding Power Results

When running HPL via Docker, `warpt` uses `powermetrics` to measure energy consumption.

- **Idle Power**: A MacBook Air typically idles at ~1.5W to 3.5W.
- **Load Power**: Under a full HPL load (using all cores), you should see **15W to 30W+**.
- **Efficiency (GFLOPS/W)**: If your average power is very low (e.g., < 5W) while running a benchmark, it usually indicates that the hardware is bottlenecked by Docker resource limits rather than the computation itself.

## 4. Software Limitations (AMX)

The current `warpt` Docker image uses **OpenBLAS** on Ubuntu ARM64. While fast, it does not currently leverage Apple's proprietary **AMX (Apple Matrix Extension)** instructions, which are required to reach the theoretical peak performance of M-series chips.

For comparison:

- **Docker (OpenBLAS)**: ~100–150 Gflops.
- **Native (Apple Accelerate)**: ~250–400+ Gflops.

## 5. Quick Start Tuned Run

Use the provided optimized configuration to see real performance:

```bash
warpt benchmark run --benchmark-config examples/hpl_config.yaml
```
