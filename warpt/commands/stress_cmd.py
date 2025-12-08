"""Stress test command - runs comprehensive hardware stress tests."""

import sys

import click

from warpt.models.constants import (
    DEFAULT_STRESS_SECONDS,
    VALID_STRESS_TARGETS,
    Precision,
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


def parse_precisions(precision_str: str) -> list[Precision]:
    """Parse comma-separated precision string into list of Precision enums.

    Args:
        precision_str: Comma-separated precisions (e.g., 'fp16,bf16' or 'fp32')

    Returns
    -------
        List of Precision enums

    Raises
    ------
        SystemExit: If precision is invalid
    """
    valid_precisions = {
        "fp32": Precision.FP32,
        "fp16": Precision.FP16,
        "bf16": Precision.BF16,
        "int8": Precision.INT8,
    }

    try:
        # Split and filter out empty strings
        precision_strings = [
            s.strip().lower() for s in precision_str.split(",") if s.strip()
        ]

        if not precision_strings:
            print("Error: No valid precisions provided.")
            print(f"Valid precisions: {', '.join(sorted(valid_precisions.keys()))}")
            sys.exit(1)

        parsed = []
        for p in precision_strings:
            if p not in valid_precisions:
                print(
                    f"Error: Invalid precision '{p}'. "
                    f"Valid: {', '.join(sorted(valid_precisions.keys()))}"
                )
                sys.exit(1)
            parsed.append(valid_precisions[p])

        return parsed
    except Exception as e:
        print(f"Error parsing precisions: {e}")
        sys.exit(1)


def get_available_gpus() -> list[int]:
    """Get list of available GPU IDs.

    Returns
    -------
        List of GPU IDs or empty list if no GPUs available
    """
    try:
        from warpt.backends.factory import get_gpu_backend

        backend = get_gpu_backend()
        gpus = backend.list_devices()
        return [gpu.index for gpu in gpus]
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
    compute: bool,
    precision_type: str | None,
    memory: bool,
    allow_tf32: bool,
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
        compute: Run compute stress test (GPU only)
        precision: Run mixed precision profiling test (GPU only)
        memory: Run memory bandwidth test (GPU only)
        allow_tf32: Enable TF32 for GPU tests (default True)
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
        # Track that these came from "all" expansion, not explicit request
        explicitly_requested_targets = set()
    else:
        # User explicitly requested these targets
        explicitly_requested_targets = set(parsed_targets)

    # Parse precision types if provided
    precision_list = None
    if precision_type is not None:
        # User provided --precision flag (with or without value)
        if precision_type == "":
            # --precision with no value -> use defaults
            precision_list = None  # Will use GPUPrecisionTest defaults
        else:
            # --precision fp16,bf16 -> parse the list
            precision_list = parse_precisions(precision_type)

    # Determine which GPU tests to run
    # Default: if no test type flags specified and GPU is in targets, run all tests
    gpu_tests_to_run = []
    if "gpu" in parsed_targets:
        if not compute and precision_type is None and not memory:
            # No flags specified - run all tests by default
            gpu_tests_to_run = ["compute", "precision", "memory"]
        else:
            # User specified specific tests
            if compute:
                gpu_tests_to_run.append("compute")
            if precision_type is not None:
                gpu_tests_to_run.append("precision")
            if memory:
                gpu_tests_to_run.append("memory")

    # Determine which CPU tests to run
    # CPU only has compute test currently
    run_cpu_tests = False
    if "cpu" in parsed_targets:
        if not compute and precision_type is None and not memory:
            # No flags specified - run CPU tests by default
            run_cpu_tests = True
        elif compute:
            # Compute flag specified - run CPU compute
            run_cpu_tests = True
        # else: precision or memory specified - skip CPU (not applicable)

    # Determine which RAM tests to run
    # RAM only has memory stress test TODO
    run_ram_tests = False
    if "ram" in parsed_targets:
        if not compute and precision_type is None and not memory:
            # No flags specified - run RAM tests by default
            run_ram_tests = True
        elif memory:
            # Memory flag specified - run RAM memory test
            run_ram_tests = True
        # else: compute or precision specified - skip RAM (not applicable)

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

    # Remove targets that won't run any tests (clean up before config display)
    if "cpu" in parsed_targets and not run_cpu_tests:
        parsed_targets.remove("cpu")
    if "gpu" in parsed_targets and not gpu_tests_to_run:
        parsed_targets.remove("gpu")
    if "ram" in parsed_targets and not run_ram_tests:
        parsed_targets.remove("ram")

    # Check if any targets remain after filtering
    if not parsed_targets:
        print("No applicable tests will run for the specified configuration.")
        if compute or precision_type is not None or memory:
            print(
                "The specified test flags don't apply to any available "
                "hardware targets."
            )
        print("\nRun 'warpt stress --help' to see available options.")
        print("Run 'warpt list' to see available hardware.")
        sys.exit(0)

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
        print(f"  GPU Tests:      {', '.join(gpu_tests_to_run)}")
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

    # Track timestamps and collect results for JSON export
    from datetime import UTC, datetime

    timestamp_start = datetime.now(UTC).isoformat()

    # Result collectors (will build Pydantic models as we run tests)
    cpu_test_results = None
    gpu_test_results = None
    # ram_test_results = None  # TODO: Implement RAM tests
    targets_tested: list[str] = []

    # Run stress tests
    for target in parsed_targets:
        if target == "cpu":
            if not run_cpu_tests:
                # Only show skip message if explicitly requested
                if "cpu" in explicitly_requested_targets:
                    print("⊘ CPU tests skipped (test flags don't apply)\n")
                continue

            print("=== CPU Compute Stress Test ===\n")
            try:
                from warpt.stress.cpu_compute import CPUMatMulTest
            except ImportError:
                print(
                    "Error: numpy is required for CPU stress tests.\n"
                    "Install with: pip install warpt[stress]"
                )
                sys.exit(1)

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

            # Collect results for JSON export
            from warpt.backends.system import CPU
            from warpt.models.stress_models import CPUSystemResult, CPUTestResults

            # Get CPU info for model/architecture/sockets
            cpu_backend = CPU()
            cpu_info = cpu_backend.get_cpu_info()

            cpu_system_result = CPUSystemResult(
                cpu_model=cpu_info.model,
                cpu_architecture=cpu_info.architecture,
                tflops=results["tflops"],
                duration=results["duration"],
                iterations=results["iterations"],
                total_operations=results["total_operations"],
                burnin_seconds=results["burnin_seconds"],
                metrics={"matrix_size": results["matrix_size"]},
                sockets_used=cpu_info.total_sockets,
                physical_cores=results["cpu_physical_cores"],
                logical_cores=results["cpu_logical_cores"],
                max_temp=None,  # TODO: Implement temperature monitoring
                avg_power=None,  # TODO: Implement power monitoring
            )

            cpu_test_results = CPUTestResults(
                test_mode="system_level",
                device_count=cpu_info.total_sockets,
                results={"cpu_system": cpu_system_result},
            )
            targets_tested.append("cpu")

        elif target == "gpu":
            from warpt.backends.factory import get_gpu_backend
            from warpt.backends.nvidia import NvidiaBackend

            if not gpu_ids:
                print("No GPUs available for testing.")
                continue

            # TODO MVP limitation: Only NVIDIA GPUs supported for stress tests
            try:
                backend = get_gpu_backend()
                if not isinstance(backend, NvidiaBackend):
                    raise click.ClickException(
                        f"GPU stress tests currently only support NVIDIA GPUs.\n"
                        f"Detected: {backend.__class__.__name__}\n"
                        f"AMD and Intel GPU stress test support coming soon."
                    )
            except RuntimeError as e:
                raise click.ClickException(str(e)) from e

            # Import test classes based on which tests we're running
            test_classes: dict[str, type] = {}
            try:
                if "compute" in gpu_tests_to_run:
                    from warpt.stress.gpu_compute import GPUMatMulTest

                    test_classes["compute"] = GPUMatMulTest
                if "precision" in gpu_tests_to_run:
                    from warpt.stress.gpu_precision import GPUPrecisionTest

                    test_classes["precision"] = GPUPrecisionTest
                if "memory" in gpu_tests_to_run:
                    from warpt.stress.gpu_memory import GPUMemoryBandwidthTest

                    test_classes["memory"] = GPUMemoryBandwidthTest
            except ImportError:
                print(
                    "Error: torch is required for GPU stress tests.\n"
                    "Install with: pip install warpt[stress]"
                )
                sys.exit(1)

            # Collect GPU results for export (per-device mode)
            from warpt.models.stress_models import GPUDeviceResult, GPUTestResults

            gpu_device_results: dict[str, dict] = {}  # Temp storage per GPU

            # Test each GPU individually
            for gpu_index in gpu_ids:
                gpu_key = f"gpu_{gpu_index}"
                if gpu_key not in gpu_device_results:
                    gpu_device_results[gpu_key] = {
                        "device_id": gpu_key,
                        "compute_result": None,
                        "precision_result": None,
                        "memory_result": None,
                    }
                # Run each test type for this GPU
                for test_type in gpu_tests_to_run:
                    if test_type == "compute":
                        print(f"=== GPU {gpu_index} Compute Stress Test ===\n")

                        gpu_test = test_classes["compute"](
                            device_id=gpu_index,
                            burnin_seconds=burnin_seconds,
                            allow_tf32=allow_tf32,
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

                        # Store compute results
                        gpu_device_results[gpu_key]["compute_result"] = results

                    if test_type == "precision":
                        print(f"=== GPU {gpu_index} Mixed Precision Profile ===\n")

                        precision_test = test_classes["precision"](
                            device_id=gpu_index,
                            burnin_seconds=burnin_seconds,
                            allow_tf32=allow_tf32,
                            precisions=precision_list,
                        )
                        results = precision_test.run(duration=test_duration)

                        # Display results (summary already printed by test)
                        print()

                        # Store precision results
                        gpu_device_results[gpu_key]["precision_result"] = results

                    if test_type == "memory":
                        from warpt.models.stress_models import (
                            GPUMemoryBandwidthResult,
                        )

                        print(f"=== GPU {gpu_index} Memory Bandwidth Test ===\n")

                        memory_test = test_classes["memory"](
                            device_id=gpu_index,
                            burnin_seconds=burnin_seconds,
                        )
                        mem_results: GPUMemoryBandwidthResult = memory_test.run(
                            duration=test_duration
                        )

                        # Display results
                        print(
                            f"\nResults for GPU {gpu_index} "
                            f"({mem_results.gpu_name}):"
                        )
                        print(
                            f"  D2D Bandwidth:      "
                            f"{mem_results.d2d_bandwidth_gbps:.1f} GB/s"
                        )
                        if mem_results.h2d_bandwidth_gbps is not None:
                            print(
                                f"  H2D Bandwidth:      "
                                f"{mem_results.h2d_bandwidth_gbps:.1f} GB/s"
                            )
                        if mem_results.d2h_bandwidth_gbps is not None:
                            print(
                                f"  D2H Bandwidth:      "
                                f"{mem_results.d2h_bandwidth_gbps:.1f} GB/s"
                            )
                        print(
                            f"  Data Size:          "
                            f"{mem_results.data_size_gb} GB per test"
                        )
                        print(f"  Duration:           " f"{mem_results.duration:.1f}s")
                        print(
                            f"  Pinned Memory:      "
                            f"{'Yes' if mem_results.used_pinned_memory else 'No'}"
                        )
                        print()

                        # Store memory results
                        gpu_device_results[gpu_key]["memory_result"] = mem_results

            # Build GPUDeviceResult models from collected data
            from warpt.models.stress_models import GPUSystemResult

            gpu_results_dict: dict[str, GPUDeviceResult | GPUSystemResult] = {}

            for gpu_key, gpu_data in gpu_device_results.items():
                compute_res = gpu_data["compute_result"]
                precision_res = gpu_data["precision_result"]
                memory_res = gpu_data["memory_result"]

                # We need at least one test result to build the model
                if compute_res or memory_res:
                    # Get GPU UUID and name
                    import pynvml

                    gpu_index_int = int(gpu_key.split("_")[1])
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index_int)
                        gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
                    except Exception:
                        gpu_uuid = f"gpu_{gpu_index_int}_unknown"

                    # Get GPU name from whichever test result is available
                    if compute_res:
                        gpu_name = compute_res["gpu_name"]
                    elif memory_res:
                        gpu_name = memory_res.gpu_name
                    else:
                        gpu_name = "Unknown"

                    # Determine burnin_seconds from any available test
                    if compute_res:
                        burnin = compute_res["burnin_seconds"]
                    elif memory_res:
                        burnin = memory_res.burnin_seconds
                    else:
                        burnin = burnin_seconds  # Fallback to CLI param

                    gpu_device_result = GPUDeviceResult(
                        device_id=gpu_key,
                        gpu_uuid=gpu_uuid,
                        gpu_name=gpu_name,
                        # Compute metrics (optional)
                        tflops=compute_res["tflops"] if compute_res else None,
                        duration=compute_res["duration"] if compute_res else None,
                        iterations=compute_res["iterations"] if compute_res else None,
                        total_operations=(
                            compute_res["total_operations"] if compute_res else None
                        ),
                        burnin_seconds=burnin,
                        metrics=(
                            {
                                "matrix_size": compute_res["matrix_size"],
                                "precision": compute_res["precision"],
                                "tf32_enabled": compute_res.get("tf32_enabled", False),
                            }
                            if compute_res
                            else {}
                        ),
                        memory_used_gb=(
                            compute_res["memory_used_gb"] if compute_res else None
                        ),
                        max_temp=None,  # TODO: Implement temperature monitoring
                        avg_power=None,  # TODO: Implement power monitoring
                        mixed_precision=precision_res if precision_res else None,
                        memory_bandwidth=memory_res if memory_res else None,
                    )
                    gpu_results_dict[gpu_key] = gpu_device_result

            # Build GPUTestResults container
            if gpu_results_dict:
                gpu_test_results = GPUTestResults(
                    test_mode="per_device",
                    device_count=len(gpu_results_dict),
                    results=gpu_results_dict,
                )
                targets_tested.append("gpu")

        elif target == "ram":
            if not run_ram_tests:
                # Only show skip message if explicitly requested
                if "ram" in explicitly_requested_targets:
                    print("⊘ RAM tests skipped (test flags don't apply)\n")
                continue

            print("[TODO] RAM stress tests not yet implemented\n")

    print("✓ Stress tests completed")

    # Capture end timestamp
    timestamp_end = datetime.now(UTC).isoformat()

    # Export to JSON if requested
    if export_format == "json":
        from pathlib import Path

        from warpt.commands.list_cmd import random_string
        from warpt.models.stress_models import (
            CPUSummary,
            GPUSummary,
            StressTestExport,
        )

        # Build results dict
        results_dict: dict = {}
        if cpu_test_results:
            results_dict["cpu"] = cpu_test_results
        if gpu_test_results:
            results_dict["gpu"] = gpu_test_results
        # RAM tests not yet implemented, ram_test_results will always be None

        # Build summary dict (simplified for now)
        summary_dict: dict = {}
        if cpu_test_results:
            cpu_summary = CPUSummary(
                status="pass",
                performance="completed",
                tflops=cpu_test_results.results["cpu_system"].tflops,
            )
            summary_dict["cpu"] = cpu_summary

        if gpu_test_results:
            gpu_tflops_list = [
                gpu_res.tflops
                for gpu_res in gpu_test_results.results.values()
                if isinstance(gpu_res, GPUDeviceResult) and gpu_res.tflops is not None
            ]
            avg_tflops = (
                sum(gpu_tflops_list) / len(gpu_tflops_list) if gpu_tflops_list else 0.0
            )
            gpu_summary = GPUSummary(
                total_devices_tested=gpu_test_results.device_count,
                avg_tflops=avg_tflops,
                healthy_devices=gpu_test_results.device_count,
                warnings=[],
            )
            summary_dict["gpu"] = gpu_summary

        # Build StressTestExport model
        export_model = StressTestExport(
            targets_tested=targets_tested,
            results=results_dict,
            summary=summary_dict,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            warpt_version="0.1.0",  # TODO: Get from package
        )

        # Generate filename if not provided
        if not export_filename:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_tag = random_string(6)
            export_filename = f"warpt_stress_{timestamp_str}_{random_tag}.json"

        # Write JSON file using Pydantic's built-in serialization
        export_path = Path(export_filename)
        export_path.write_text(export_model.model_dump_json(indent=2))

        print(f"\n✓ JSON exported to: {export_filename}")
