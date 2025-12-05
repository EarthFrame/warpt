#!/usr/bin/env python3
"""Warpt CLI - Command-line interface for Warpt."""

import click

from warpt.commands.list_cmd import run_list
from warpt.models.constants import DEFAULT_BURNIN_SECONDS


@click.group()
def warpt():
    """Warpt command-line tool for system monitoring and utilities."""
    pass


@warpt.command()
@click.option(
    "--export",
    is_flag=True,
    default=False,
    help=(
        "Export results to JSON file with default filename "
        "(warpt_list_TIMESTAMP.json)"
    ),
)
@click.option(
    "--export-file",
    default=None,
    help="Export results to JSON file with custom filename",
)
def list(export, export_file):
    """List CPU and GPU information."""
    # Determine export format and filename
    if export_file:
        # --export-file provided (takes precedence)
        export_format = "json"
        export_filename = export_file
    elif export:
        # --export flag used
        export_format = "json"
        export_filename = None  # Will use default timestamp
    else:
        # No export
        export_format = None
        export_filename = None

    run_list(export_format=export_format, export_filename=export_filename)


@warpt.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed version information")
def version(verbose):
    """Display warpt version information."""
    from warpt.commands.version_cmd import run_version

    run_version(verbose=verbose)


@warpt.command()
def monitor():
    """Monitor system performance in real-time."""
    print("Live monitoring!")


@warpt.command()
def benchmark():
    """Run system benchmarks."""
    print("Benchmarking!")


@warpt.command()
def system_info():
    """Display detailed system information."""
    print("System info!")


@warpt.command()
def check():
    """Run system health checks."""
    print("Health check!")


@warpt.command()
@click.option(
    "--target",
    multiple=True,
    help=(
        "Targets to stress test (cpu, gpu, ram, all). Can be comma-separated "
        "or repeated multiple times."
    ),
)
@click.option(
    "--gpu-id",
    default=None,
    help=(
        "Specific GPU ID(s) to test (comma-separated, e.g., '0' or '0,1'). "
        "Use with --target gpu."
    ),
)
@click.option(
    "--cpu-id",
    default=None,
    help=(
        "Specific CPU socket/core ID(s) to test (comma-separated). "
        "Use with --target cpu."
    ),
)
@click.option(
    "--duration",
    type=int,
    default=None,
    help="Duration in seconds for each stress test (default: 30s)",
)
@click.option(
    "--burnin-seconds",
    type=int,
    default=DEFAULT_BURNIN_SECONDS,
    help=(
        f"Warmup period in seconds before measurements "
        f"(default: {DEFAULT_BURNIN_SECONDS}s)"
    ),
)
@click.option(
    "--export",
    is_flag=True,
    default=False,
    help=(
        "Export results to JSON file with default filename "
        "(warpt_stress_<TIMESTAMP>_<RANDOM>.json)"
    ),
)
@click.option(
    "--export-file",
    default=None,
    help="Export results to JSON file with custom filename",
)
@click.option(
    "--log-file", default=None, help="Write detailed execution logs to specified file"
)
@click.option(
    "--compute",
    is_flag=True,
    default=False,
    help="Run compute stress test (sustained workload for GPU health)",
)
@click.option(
    "--precision",
    "precision_type",
    default=None,
    help=(
        "Run mixed precision profiling test. "
        "Optionally specify precisions as comma-separated list (e.g., fp16,bf16). "
        "If no list provided, tests fp32,fp16,bf16 by default."
    ),
)
@click.option(
    "--memory",
    is_flag=True,
    default=False,
    help="Run memory bandwidth test (GPU memory performance)",
)
@click.option(
    "--no-tf32",
    is_flag=True,
    default=False,
    help="Disable TF32 (TensorFloat-32) for GPU tests. By default, TF32 is enabled.",
)
# TODO - add --nic-id
def stress(
    target,
    gpu_id,
    cpu_id,
    duration,
    burnin_seconds,
    export,
    export_file,
    log_file,
    compute,
    precision_type,
    memory,
    no_tf32,
):
    """Run system stress tests."""
    from warpt.commands.stress_cmd import run_stress

    # Determine export format and filename (matches list command pattern)
    if export_file:
        export_format = "json"
        export_filename = export_file
    elif export:
        export_format = "json"
        export_filename = None  # Will use default timestamp
    else:
        export_format = None
        export_filename = None

    run_stress(
        targets=target,
        gpu_id=gpu_id,
        cpu_id=cpu_id,
        duration_seconds=duration,
        burnin_seconds=burnin_seconds,
        export_format=export_format,
        export_filename=export_filename,
        log_file=log_file,
        compute=compute,
        precision_type=precision_type,
        memory=memory,
        allow_tf32=not no_tf32,  # Invert the flag: --no-tf32 -> allow_tf32=False
    )


if __name__ == "__main__":
    warpt()
