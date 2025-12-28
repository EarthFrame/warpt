#!/usr/bin/env python3
"""Warpt CLI - Command-line interface for Warpt."""

import click

from warpt.commands.list_cmd import run_list
from warpt.utils.env import get_env
from warpt.utils.logger import Logger


@click.group()
def warpt():
    """Warpt command-line tool for system monitoring and utilities."""
    # Configure logger at startup if not already configured
    if not Logger.is_configured():
        # Default to INFO level; subcommands can adjust via set_level()
        Logger.configure(
            level=get_env("WARPT_LOG_LEVEL", default="INFO"), timestamps=True
        )


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

    if verbose:
        Logger.set_level("DEBUG")

    run_version(verbose=verbose)


@warpt.command()
def monitor():
    """Monitor system performance in real-time."""
    print("Live monitoring!")


@warpt.command()
@click.option(
    "--benchmark",
    "-b",
    help="Benchmark name (e.g. hpl)",
)
@click.option(
    "--system",
    type=click.Path(exists=True),
    help="System configuration file (JSON, YAML, or HCL)",
)
@click.option(
    "--benchmark-config",
    type=click.Path(exists=True),
    help="Benchmark configuration file (JSON, YAML, or HCL)",
)
@click.option(
    "--cluster",
    type=click.Path(exists=True),
    help="Cluster configuration file (JSON, YAML, or HCL)",
)
@click.option(
    "--set",
    "overrides",
    multiple=True,
    help="Override configuration value (KEY=VALUE format, repeatable)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate and print actions without executing",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug output and stack traces",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Write structured benchmark results to file",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force rebuild even if cached image exists",
)
@click.argument(
    "subcommand",
    type=click.Choice(["list", "validate", "build", "run"]),
    required=False,
)
def benchmark(
    subcommand,
    benchmark,
    system,
    benchmark_config,
    cluster,
    overrides,
    dry_run,
    debug,
    output,
    force,
):
    r"""Run and manage performance benchmarks.

    \b
    Examples:
      warpt benchmark                          # List available benchmarks
      warpt benchmark list                     # List supported benchmarks
      warpt benchmark validate                 # Validate configuration files
      warpt benchmark build -b hpl             # Build benchmark container
      warpt benchmark run -b hpl               # Run HPL benchmark
      warpt benchmark run -b hpl --output results.json  # Run with output file
    """
    from warpt.benchmarks.registry import BenchmarkRegistry
    from warpt.commands.benchmark_cmd import (
        build_benchmark,
        list_benchmarks,
        run_benchmark,
        validate_configs,
    )

    if debug:
        Logger.set_level("DEBUG")

    # Create registry for benchmark discovery
    registry = BenchmarkRegistry()

    # If no subcommand provided, show available benchmarks
    if subcommand is None:
        list_benchmarks(registry)
        return

    # Dispatch to appropriate function based on subcommand
    if subcommand == "list":
        list_benchmarks(registry)
    elif subcommand == "validate":
        validate_configs(
            benchmark, system, benchmark_config, cluster, overrides, registry
        )
    elif subcommand == "build":
        build_benchmark(
            benchmark,
            system,
            benchmark_config,
            cluster,
            overrides,
            force,
            dry_run,
            debug,
            registry,
        )
    elif subcommand == "run":
        run_benchmark(
            benchmark,
            system,
            benchmark_config,
            cluster,
            overrides,
            output,
            dry_run,
            debug,
            registry,
        )


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
    "--list",
    "-l",
    "list_only",
    is_flag=True,
    help="List available stress tests",
)
@click.option(
    "--category",
    "-c",
    multiple=True,
    help="Filter by category: cpu, accelerator, ram, storage, network, all",
)
@click.option(
    "--test",
    "-t",
    "tests",
    multiple=True,
    help="Run specific test(s) by name (e.g., GPUMatMulTest)",
)
@click.option(
    "--duration",
    "-d",
    type=int,
    default=30,
    help="Duration in seconds per test (default: 30)",
)
@click.option(
    "--warmup",
    "-w",
    type=int,
    default=5,
    help="Warmup period in seconds (default: 5)",
)
@click.option(
    "--device-id",
    default=None,
    help="Device ID(s) for accelerator tests (comma-separated, e.g., '0,1')",
)
@click.option(
    "--output",
    "-o",
    "outputs",
    multiple=True,
    help="Output file(s) - format auto-detected (.json/.yaml). Repeatable.",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "yaml", "text"], case_sensitive=False),
    default=None,
    help="Stdout format when no --output specified",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="YAML config file with per-test settings",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output",
)
def stress(
    list_only,
    category,
    tests,
    duration,
    warmup,
    device_id,
    outputs,
    fmt,
    config,
    verbose,
):
    r"""Run hardware stress tests.

    \b
    Examples:
      warpt stress --list                  # List available tests
      warpt stress --list -c cpu           # List CPU tests only
      warpt stress -c all                  # Run all available tests
      warpt stress -c cpu                  # Run CPU category
      warpt stress -c accelerator          # Run accelerator tests
      warpt stress -t GPUMatMulTest        # Run specific test
      warpt stress -o results.json         # Save to JSON
      warpt stress -o a.json -o b.yaml     # Multiple outputs
      warpt stress --config tests.yaml     # Use config file
    """
    from warpt.commands.stress_cmd import run_stress

    if verbose:
        Logger.set_level("DEBUG")

    run_stress(
        categories=category,
        tests=tests,
        duration=duration,
        warmup=warmup,
        device_id=device_id,
        outputs=outputs,
        fmt=fmt,
        config=config,
        list_only=list_only,
        verbose=verbose,
    )


if __name__ == "__main__":
    warpt()
