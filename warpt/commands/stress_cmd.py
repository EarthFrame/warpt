"""Stress test command - runs comprehensive hardware stress tests."""

import sys
from typing import List, Optional

from warpt.models.constants import (
    DEFAULT_STRESS_DURATION,
    DEFAULT_BURNIN_DURATION,
)


def parse_targets(targets: tuple) -> List[str]:
    """
    Parse and deduplicate target specifications.

    Supports both comma-separated and repeated flags:
    - ('cpu', 'gpu') -> ['cpu', 'gpu']
    - ('cpu,gpu', 'ram') -> ['cpu', 'gpu', 'ram']
    - ('cpu', 'cpu', 'gpu') -> ['cpu', 'gpu'] (deduplicated)

    Args:
        targets: Tuple of target strings from Click

    Returns:
        List of deduplicated target strings
    """
    parsed_targets = []

    for target_spec in targets:
        # Split by comma and strip whitespace
        for target in target_spec.split(','):
            target = target.strip().lower()
            if target and target not in parsed_targets:
                parsed_targets.append(target)

    return parsed_targets


def validate_targets(targets: List[str]) -> None:
    """
    Validate that all targets are recognized.

    Args:
        targets: List of target strings

    Raises:
        SystemExit: If any target is invalid
    """
    valid_targets = {'cpu', 'gpu', 'ram', 'all'}

    for target in targets:
        if target not in valid_targets:
            print(f"Error: Invalid target '{target}'. Valid targets: {', '.join(sorted(valid_targets))}")
            sys.exit(1)


def parse_device_ids(device_id_str: Optional[str]) -> Optional[List[int]]:
    """
    Parse comma-separated device IDs into list of integers.

    Args:
        device_id_str: Comma-separated device IDs (e.g., '0' or '0,1,2')

    Returns:
        List of device IDs or None if not provided

    Raises:
        SystemExit: If device IDs are invalid
    """
    if not device_id_str:
        return None

    try:
        device_ids = [int(id_str.strip()) for id_str in device_id_str.split(',')]
        return device_ids
    except ValueError:
        print(f"Error: Invalid device ID format '{device_id_str}'. Must be comma-separated integers (e.g., '0' or '0,1').")
        print("Run 'warpt list' to see available devices.")
        sys.exit(1)


def get_available_gpus() -> List[int]:
    """
    Get list of available GPU IDs.

    Returns:
        List of GPU IDs or empty list if no GPUs available
    """
    try:
        from warpt.backends.nvidia import NvidiaBackend
        backend = NvidiaBackend()
        gpus = backend.list_devices()
        return [gpu['index'] for gpu in gpus]
    except Exception:
        return []


def validate_device_availability(
    parsed_targets: List[str],
    gpu_ids: Optional[List[int]],
    cpu_ids: Optional[List[int]]
) -> None:
    """
    Validate that requested devices are available.

    Args:
        parsed_targets: List of target strings
        gpu_ids: List of requested GPU IDs or None
        cpu_ids: List of requested CPU IDs or None

    Raises:
        SystemExit: If requested devices are not available
    """
    # Validate GPU IDs if GPU is a target
    if 'gpu' in parsed_targets:
        available_gpus = get_available_gpus()

        if not available_gpus:
            print("Error: No GPUs detected on this system.")
            print("Run 'warpt list' to see available hardware.")
            sys.exit(1)

        if gpu_ids:
            for gpu_id in gpu_ids:
                if gpu_id not in available_gpus:
                    print(f"Error: GPU ID {gpu_id} not found. Available GPUs: {', '.join(map(str, available_gpus))}")
                    print("Run 'warpt list' for detailed GPU information.")
                    sys.exit(1)


def run_stress(
    targets: tuple,
    gpu_id: Optional[str],
    cpu_id: Optional[str],
    duration: Optional[int],
    burnin_seconds: int,
    export_format: Optional[str],
    export_filename: Optional[str],
    log_file: Optional[str],
) -> None:
    """
    Run stress tests based on user specifications.

    Args:
        targets: Tuple of target specifications from CLI
        gpu_id: Comma-separated GPU IDs or None
        cpu_id: Comma-separated CPU IDs or None
        duration: Test duration in seconds or None (use defaults)
        burnin_seconds: Warmup period in seconds
        export_format: Export format ('json' or None)
        export_filename: Custom export filename or None
        log_file: Log file path or None
    """
    # Parse and validate targets
    parsed_targets = parse_targets(targets)

    # Default to all targets if none specified
    if not parsed_targets:
        parsed_targets = ['all']

    # Validate targets
    validate_targets(parsed_targets)

    # Expand 'all' target to individual targets
    if 'all' in parsed_targets:
        parsed_targets = ['cpu', 'gpu', 'ram']

    # Determine test duration - user-specified always takes priority
    test_duration = duration if duration is not None else DEFAULT_STRESS_DURATION

    # Parse device IDs
    gpu_ids = parse_device_ids(gpu_id)
    cpu_ids = parse_device_ids(cpu_id)

    # Validate device availability
    validate_device_availability(parsed_targets, gpu_ids, cpu_ids)

    # Handle defaulting to all devices when target specified but no device IDs
    default_to_all_gpus = False
    default_to_all_cpus = False

    if 'gpu' in parsed_targets and not gpu_ids:
        default_to_all_gpus = True
        available_gpus = get_available_gpus()
        gpu_ids = available_gpus

    if 'cpu' in parsed_targets and not cpu_ids:
        default_to_all_cpus = True
        # TODO: Get actual CPU count when CPU backend is ready
        cpu_ids = [0]  # Placeholder

    # Display configuration
    print("Stress Test Configuration:")
    print(f"  Targets:        {', '.join(parsed_targets)}")
    print(f"  Duration:       {test_duration}s per test")
    print(f"  Burnin:         {burnin_seconds}s")

    if 'gpu' in parsed_targets:
        if default_to_all_gpus:
            print(f"  GPU IDs:        all (defaulting to all available GPUs: {', '.join(map(str, gpu_ids))})")
            print(f"                  Use 'warpt list' to view device IDs and --gpu-id to specify")
        else:
            print(f"  GPU IDs:        {', '.join(map(str, gpu_ids))}")

    if 'cpu' in parsed_targets:
        if default_to_all_cpus:
            print(f"  CPU IDs:        all (defaulting to all available CPUs)")
            print(f"                  Use 'warpt list' to view device IDs and --cpu-id to specify")
        else:
            print(f"  CPU IDs:        {', '.join(map(str, cpu_ids))}")

    if log_file:
        print(f"  Log file:       {log_file}")
    if export_format:
        if export_filename:
            print(f"  Export to:      {export_filename}")
        else:
            print(f"  Export to:      warpt_stress_<timestamp>.json")

    print("\n" + "="*60)
    print("Running Stress Tests...")
    print("="*60 + "\n")

    # TODO: Implement actual stress tests
    # For now, placeholder output
    for target in parsed_targets:
        print(f"[PLACEHOLDER] Running {target.upper()} stress tests...")
        if target == 'gpu' and gpu_ids:
            for gpu_id_val in gpu_ids:
                print(f"  - Testing GPU {gpu_id_val}")
        if target == 'cpu' and cpu_ids:
            for cpu_id_val in cpu_ids:
                print(f"  - Testing CPU {cpu_id_val}")

    print("\n✓ Stress tests completed (placeholder)")
