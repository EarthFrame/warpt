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
        "Export results to JSON file with default filename (warpt_list_TIMESTAMP.json)"
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
@click.option(
    "--interval",
    "-i",
    type=float,
    default=1.0,
    show_default=True,
    help="Sampling interval in seconds for live monitoring",
)
@click.option(
    "--duration",
    "-d",
    type=float,
    default=None,
    help="Stop monitoring after this many seconds (default: run until interrupted)",
)
@click.option(
    "--no-tui",
    is_flag=True,
    default=False,
    help="Run the CLI monitor output instead of the curses dashboard",
)
def monitor(interval, duration, no_tui):
    """Monitor system performance in real-time."""
    if not no_tui:
        try:
            from warpt.commands.monitor_tui import run_monitor_tui

            run_monitor_tui(interval_seconds=interval)
        except Exception as exc:
            raise click.ClickException(f"Monitor TUI failed: {exc}") from exc
        return

    from warpt.commands.monitor_cmd import run_monitor

    try:
        run_monitor(interval_seconds=interval, duration_seconds=duration)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


@warpt.command(hidden=not get_env("WARPT_ENABLE_POWER", default=False, as_type=bool))
@click.option(
    "--interval",
    "-i",
    type=float,
    default=1.0,
    show_default=True,
    help="Sampling interval in seconds",
)
@click.option(
    "--duration",
    "-d",
    type=float,
    default=None,
    help="Stop after this many seconds (default: run until interrupted)",
)
@click.option(
    "--no-processes",
    is_flag=True,
    default=False,
    help="Don't show per-process power attribution",
)
@click.option(
    "--top",
    "-n",
    type=int,
    default=10,
    show_default=True,
    help="Number of top processes to display",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output in JSON format",
)
@click.option(
    "--output",
    "-o",
    type=str,
    default=None,
    help="Write results to JSON file",
)
@click.option(
    "--continuous",
    "-c",
    is_flag=True,
    default=False,
    help="Run continuously (vs single snapshot)",
)
@click.option(
    "--sources",
    is_flag=True,
    default=False,
    help="Show available power sources and exit",
)
def power(
    interval, duration, no_processes, top, output_json, output, continuous, sources
):
    r"""Monitor system power consumption.

    Displays power usage in watts for CPU, GPU, and other components.
    Can attribute power consumption to individual processes.

    \b
    Examples:
      warpt power                    # Single snapshot
      warpt power -c                 # Continuous monitoring
      warpt power -c -d 60           # Monitor for 60 seconds
      warpt power --json             # JSON output
      warpt power -o power.json      # Save to file
      warpt power --sources          # Show available power sources

    \b
    Platform support:
      Linux: Intel/AMD RAPL via /sys/class/powercap/
      macOS: powermetrics (requires sudo)
      All: NVIDIA GPUs via NVML
    """
    from warpt.commands.power_cmd import run_power, show_power_sources

    if sources:
        show_power_sources()
        return

    try:
        run_power(
            interval_seconds=interval,
            duration_seconds=duration,
            show_processes=not no_processes,
            top_n_processes=top,
            output_format="json" if output_json else "text",
            output_file=output,
            continuous=continuous,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


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
@click.option(
    "--no-monitor",
    is_flag=True,
    default=False,
    help="Disable power and resource monitoring during benchmark",
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
    no_monitor,
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
            no_monitor,
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
    type=click.IntRange(min=1),
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
@click.option(
    "--target",
    default=None,
    help=(
        "Target IP(s) for network tests "
        "(comma-separated, e.g., '192.168.1.11,192.168.1.76')"
    ),
)
@click.option(
    "--payload",
    type=int,
    default=None,
    help=(
        "Payload size in bytes for network tests "
        "(e.g., 4096 for latency, 1048576 for bandwidth)"
    ),
)
@click.option(
    "--network-mode",
    type=click.Choice(["latency", "bandwidth", "both"], case_sensitive=False),
    default=None,
    help="Network test mode: latency, bandwidth, or both",
)
@click.option(
    "--read-ratio",
    type=float,
    default=None,
    help=(
        "Read ratio for StorageMixedTest (0.0-1.0). "
        "E.g., 0.7 = 70%% reads, 30%% writes"
    ),
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
    target,
    payload,
    network_mode,
    read_ratio,
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
        target=target,
        payload=payload,
        network_mode=network_mode,
        read_ratio=read_ratio,
    )


if __name__ == "__main__":
    warpt()
