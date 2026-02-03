# Setting up and Running HPL with warpt

High Performance Linpack (HPL) is the gold standard for measuring peak floating-point performance. This guide covers how to set up and run HPL using `warpt` and Docker.

## Prerequisites

- **Docker**: Installed and running (use `warpt list software` to verify).
- **warpt**: Installed in your Python environment.

## 1. Environment Validation

Before running HPL, use `warpt` to ensure your environment is compatible. You can use the `ListParser` utility to programmatically check for Docker and hardware capabilities.

```python
from warpt.utils import ListParser

# Run 'warpt list --all --format json > system.json' first
output = ListParser.parse_file("system.json")

if not ListParser.get_container_tool(output):
    print("Error: Docker not detected. Please install Docker.")
    exit(1)

print(f"Detected {ListParser.get_gpu_count(output)} GPUs")
print(f"Architecture: {ListParser.get_cpu_arch(output)}")
```

## 2. Using the HPL Container

### Custom Optimized Build (Recommended)

`warpt` provides a custom Dockerfile optimized for Apple Silicon (ARM64). This build uses OpenBLAS and OpenMPI.

To build it using the provided Makefile:

```bash
cd docker
make hpl-arm64
```

Or build it manually:

```bash
docker build -t warpt-hpl:arm64 -f docker/benchmarks/hpl/Dockerfile.arm64 docker/benchmarks/hpl/
```

### Pulling a Pre-built Container

Alternatively, you can pull a general-purpose HPL image:

```bash
# General purpose HPL image (supports x86_64 and arm64)
docker pull uvarshney/hpl:latest
```

## 3. Running HPL

You can run HPL directly via Docker or use the integrated `warpt benchmark run` command.

### Using warpt benchmark run (Recommended)

`warpt` can automate the generation of `HPL.dat` and the execution of the Docker container. Create a configuration file (e.g., `hpl_config.yaml`):

```yaml
# hpl_config.yaml
benchmarks:
  - name: hpl
    parameters:
      problem_size: 10240
      block_size: 128
      p_grid: 2
      q_grid: 2
      execution_mode: "docker"
      docker_image: "warpt-hpl:arm64"
```

Then run the benchmark:

```bash
warpt benchmark run --benchmark-config hpl_config.yaml
```

### Manual Docker Run

To run a basic HPL benchmark in the container manually:

```bash
docker run --rm -it -v $(pwd)/HPL.dat:/hpl/HPL.dat warpt-hpl:arm64 xhpl -np 4 xhpl
```

## 4. Configuring HPL for MacBook Air (M2/M3)

When running on a MacBook Air, keep in mind:

- **Efficiency Cores vs. Performance Cores**: HPL performance will vary depending on how threads are pinned.
- **Memory**: HPL is memory-intensive. Ensure you don't over-allocate `N` beyond your available RAM (check `warpt list hardware`).

For an 8-core M2 (4P + 4E), a good starting `P x Q` is `2 x 4` or `2 x 2` depending on how many cores you want to stress.
