"""Stress test command - runs comprehensive hardware stress tests."""

import sys

import click

from warpt.models.constants import (
    DEFAULT_STRESS_SECONDS,
    VALID_STRESS_TARGETS,
)


def parse_targets(
    targets: tuple, valid_targets: tuple = VALID_STRESS_TARGETS
) -> list[str]:
    """Parse, validate, and deduplicate target specifications.

    Supports both comma-separated and repeated flags:
    - ('cpu', 'gpu') -> ['cpu', 'gpu']
    - ('cpu,gpu', 'ram') -> ['cpu', 'gpu', 'ram']
    - ('cpu', 'cpu', 'gpu') -> ['cpu', 'gpu'] (deduplicated)

    Args:
        targets: Tuple of target strings from Click
        valid_targets: Tuple of valid target strings

    Returns
    -------
        List of deduplicated and validated target strings

    Raises
    ------
        ValueError: If any target is invalid
    """
    parsed_targets = []

    for target_spec in targets:
        # Split by comma and strip whitespace
        for target in target_spec.split(","):
            target = target.strip().lower()
            if target:
                # Validate target
                if target not in valid_targets:
                    raise ValueError(
                        f"Invalid target '{target}'. Valid targets: "
                        f"{', '.join(sorted(valid_targets))}"
                    )
                # Add to list if not duplicate
                if target not in parsed_targets:
                    parsed_targets.append(target)

    return parsed_targets


def parse_device_ids(device_id_str: str | None) -> list[int] | None:
    """Parse comma-separated device IDs into list of integers.

    Args:
        device_id_str: Comma-separated device IDs (e.g., '0' or '0,1,2')

    Returns
    -------
        List of device IDs or None if not provided

    Raises
    ------
        SystemExit: If device IDs are invalid
    """
    if not device_id_str:
        return None

    try:
        # Split and filter out empty strings (handles cases like "0,," or "0,,2")
        id_strings = [s.strip() for s in device_id_str.split(",") if s.strip()]

        if not id_strings:
            print("Error: No valid device IDs provided.")
            print("Run 'warpt list' to see available devices.")
            sys.exit(1)

        device_ids = [int(id_str) for id_str in id_strings]
        return device_ids
    except ValueError:
        print(
            f"Error: Invalid device ID format '{device_id_str}'. "
            f"Must be comma-separated integers (e.g., '0' or '0,1')."
        )
        print("Run 'warpt list' to see available devices.")
        sys.exit(1)


def get_available_gpus() -> list[int]:
    """Get list of available GPU IDs.

    Returns
    -------
        List of GPU IDs or empty list if no GPUs available
    """
    try:
        from warpt.backends.nvidia import NvidiaBackend

        backend = NvidiaBackend()
        gpus = backend.list_devices()
        return [gpu["index"] for gpu in gpus]
    except Exception:
        return []


def get_available_cpus() -> list[int]:
    """Get list of available CPU IDs (socket IDs).

    Returns
    -------
        List of CPU socket IDs
    """
    try:
        from warpt.backends.system import CPU

        cpu = CPU()
        info = cpu.get_cpu_info()
        # Return list of socket IDs (0 to total_sockets-1)
        return list(range(info.total_sockets))
    except Exception:
        # Fallback: assume at least 1 CPU socket
        return [0]


def validate_device_availability(
    parsed_targets: list[str],
    gpu_ids: list[int] | None,
    cpu_ids: list[int] | None,
    explicit_gpu_request: bool = False,
) -> list[str]:
    """Validate that requested devices are available.
    Unavailable targets are removed with warnings.

    Args:
        parsed_targets: List of target strings
        gpu_ids: List of requested GPU IDs or None
        cpu_ids: List of requested CPU IDs or None
        explicit_gpu_request: True if user explicitly specified --target gpu
            (not via 'all')

    Returns
    -------
        List of valid targets (unavailable targets removed)

    Raises
    ------
        SystemExit: If specific device IDs not found or no valid targets remain
    """
    valid_targets = parsed_targets.copy()

    # Validate GPU IDs if GPU is a target
    if "gpu" in parsed_targets:
        available_gpus = get_available_gpus()

        if not available_gpus:
            # No GPUs available - error if explicitly requested, warn otherwise
            if explicit_gpu_request:
                raise click.ClickException(
                    "No GPUs detected but --target gpu was specified. "
                    "Run 'warpt list' to see available hardware."
                )
            else:
                print("⚠️  Warning: No GPUs detected. Skipping GPU tests.")
                print("    Run 'warpt list' to see available hardware.")
                valid_targets.remove("gpu")
        elif gpu_ids:
            # User specified GPU IDs - validate they exist
            for gpu_id in gpu_ids:
                if gpu_id not in available_gpus:
                    print(
                        f"Error: GPU ID {gpu_id} not found. Available GPUs: "
                        f"{', '.join(map(str, available_gpus))}"
                    )
                    print("Run 'warpt list' for detailed GPU information.")
                    sys.exit(1)

    # Validate CPU IDs if CPU is a target
    if "cpu" in parsed_targets:
        available_cpus = get_available_cpus()

        if cpu_ids:
            for cpu_id in cpu_ids:
                if cpu_id not in available_cpus:
                    print(
                        f"Error: CPU ID {cpu_id} not found. Available CPUs: "
                        f"{', '.join(map(str, available_cpus))}"
                    )
                    print("Run 'warpt list' for detailed CPU information.")
                    sys.exit(1)

    # Check if any valid targets remain
    if not valid_targets:
        print("Error: No valid targets available to test.")
        print("Run 'warpt list' to see available hardware.")
        sys.exit(1)

    return valid_targets


def run_stress(
    targets: tuple,
    gpu_id: str | None,
    cpu_id: str | None,
    duration_seconds: int | None,
    burnin_seconds: int,
    export_format: str | None,
    export_filename: str | None,
    log_file: str | None,
) -> None:
    """Run stress tests based on user specifications.

    Args:
        targets: Tuple of target specifications from CLI
        gpu_id: Comma-separated GPU IDs or None
        cpu_id: Comma-separated CPU IDs or None
        duration_seconds: Test duration in seconds or None (use defaults)
        burnin_seconds: Warmup period in seconds
        export_format: Export format ('json' or None)
        export_filename: Custom export filename or None
        log_file: Log file path or None
    """
    # Parse and validate targets
    try:
        parsed_targets = parse_targets(targets)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Parse device IDs early to infer targets if needed
    gpu_ids = parse_device_ids(gpu_id)
    cpu_ids = parse_device_ids(cpu_id)

    # Infer targets from device IDs if no targets specified
    if not parsed_targets:
        if gpu_ids and cpu_ids:
            parsed_targets = ["cpu", "gpu"]
        elif gpu_ids:
            parsed_targets = ["gpu"]
        elif cpu_ids:
            parsed_targets = ["cpu"]
        else:
            parsed_targets = ["all"]

    # Track if user explicitly requested GPU (vs. GPU being part of "all")
    explicit_gpu_request = "gpu" in parsed_targets and "all" not in parsed_targets

    # Expand 'all' target to individual targets
    if "all" in parsed_targets:
        parsed_targets = ["cpu", "gpu", "ram"]

    # Validate device IDs match specified targets
    if cpu_ids and "cpu" not in parsed_targets:
        print("Error: --cpu-id can only be used with --target cpu")
        print(
            f"You specified --target {','.join(parsed_targets)} but provided --cpu-id"
        )
        sys.exit(1)

    if gpu_ids and "gpu" not in parsed_targets:
        print("Error: --gpu-id can only be used with --target gpu")
        print(
            f"You specified --target {','.join(parsed_targets)} but provided --gpu-id"
        )
        sys.exit(1)

    # Determine test duration - user-specified always takes priority
    test_duration = (
        duration_seconds if duration_seconds is not None else DEFAULT_STRESS_SECONDS
    )

    # Validate device availability (returns filtered list of valid targets)
    parsed_targets = validate_device_availability(
        parsed_targets, gpu_ids, cpu_ids, explicit_gpu_request
    )

    # Handle defaulting to all devices when target specified but no device IDs
    default_to_all_gpus = False
    default_to_all_cpus = False

    if "gpu" in parsed_targets and not gpu_ids:
        default_to_all_gpus = True
        gpu_ids = get_available_gpus()

    if "cpu" in parsed_targets and not cpu_ids:
        default_to_all_cpus = True
        cpu_ids = get_available_cpus()

    # Display configuration
    print("Stress Test Configuration:")
    print(f"  Targets:        {', '.join(parsed_targets)}")
    print(f"  Duration:       {test_duration}s per test")
    print(f"  Burnin:         {burnin_seconds}s")

    if "gpu" in parsed_targets:
        if default_to_all_gpus:
            print(
                f"  GPU IDs:        all (defaulting to all available GPUs: "
                f"{', '.join(map(str, gpu_ids or []))})"
            )
            print(
                "                  Use 'warpt list' to view device IDs and "
                "--gpu-id to specify"
            )
        else:
            print(f"  GPU IDs:        {', '.join(map(str, gpu_ids or []))}")

    if "cpu" in parsed_targets:
        if default_to_all_cpus:
            print("  CPU IDs:        all (defaulting to all available CPUs)")
            print(
                "                  Use 'warpt list' to view device IDs and "
                "--cpu-id to specify"
            )
        else:
            print(f"  CPU IDs:        {', '.join(map(str, cpu_ids or []))}")

    if log_file:
        print(f"  Log file:       {log_file}")
    if export_format:
        if export_filename:
            print(f"  Export to:      {export_filename}")
        else:
            print("  Export to:      warpt_stress_<timestamp>.json")

    print("\n" + "=" * 60)
    print("Running Stress Tests...")
    print("=" * 60 + "\n")

    # TODO: Add timestamp tracking for JSON export: start and end times from
    # base class helper functions

    # Run stress tests
    for target in parsed_targets:
        if target == "cpu":
            print("=== CPU Compute Stress Test ===\n")
            from warpt.stress.cpu_compute import CPUMatMulTest

            test = CPUMatMulTest(burnin_seconds=burnin_seconds)
            results = test.run(duration=test_duration)

            # Display results
            print("\nResults:")
            print(f"  Performance:        {results['tflops']:.2f} TFLOPS")
            print(f"  Duration:           {results['duration']:.2f}s")
            print(f"  Iterations:         {results['iterations']}")
            print(
                f"  Matrix Size:        "
                f"{results['matrix_size']}x{results['matrix_size']}"
            )
            print(f"  Total Operations:   {results['total_operations']:,}")
            print(
                f"  CPU Cores:          {results['cpu_physical_cores']} "
                f"physical, {results['cpu_logical_cores']} logical"
            )
            print()

        elif target == "gpu":
            from warpt.stress.gpu_compute import GPUMatMulTest

            if not gpu_ids:
                print("No GPUs available for testing.")
                continue

            # Test each GPU individually
            for gpu_index in gpu_ids:
                print(f"=== GPU {gpu_index} Compute Stress Test ===\n")

                gpu_test = GPUMatMulTest(
                    device_id=gpu_index, burnin_seconds=burnin_seconds
                )
                results = gpu_test.run(duration=test_duration)

                # Display results
                print(
                    f"\nResults for GPU {gpu_index} "
                    f"({results.get('gpu_name', 'Unknown')}):"
                )
                print(f"  Performance:        {results['tflops']:.2f} TFLOPS")
                print(f"  Duration:           {results['duration']:.2f}s")
                print(f"  Iterations:         {results['iterations']}")
                print(
                    "  Matrix Size:        "
                    f"{results['matrix_size']}x{results['matrix_size']}"
                )
                print(f"  Total Operations:   {results['total_operations']:,}")
                print(f"  Precision:          {results['precision'].upper()}")
                print(
                    "  Memory Used:        "
                    f"{results['memory_used_gb']:.2f} GB / "
                    f"{results['memory_total_gb']:.2f} GB"
                )
                print()

        elif target == "ram":
            print("[TODO] RAM stress tests not yet implemented\n")

    print("✓ Stress tests completed")
