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
    )


if __name__ == "__main__":
    warpt()
