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
@click.option(
    "--results-file",
    default=None,
    help="Path to a JSON results file from 'warpt list' command",
)
def recommend(results_file):
    """Recommend models and configurations based on system specs."""
    from warpt.commands.recommend_cmd import run_recommend

    run_recommend(results_file=results_file)


@warpt.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed version information")
def version(verbose):
    """Display warpt version information."""
    from warpt.commands.version_cmd import run_version

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
# TODO - add --nic-id
@click.option(
    "--monitor",
    is_flag=True,
    default=False,
    help="Run the background resource monitor during stress tests",
)
@click.option(
    "--monitor-interval",
    type=float,
    default=1.0,
    show_default=True,
    help="Sampling interval in seconds for the background monitor",
)
@click.option(
    "--monitor-output",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Path to write the monitor's collected samples as JSON",
)
def stress(
    target,
    gpu_id,
    cpu_id,
    duration,
    burnin_seconds,
    export,
    export_file,
    log_file,
    monitor,
    monitor_interval,
    monitor_output,
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
        monitor=monitor,
        monitor_interval=monitor_interval,
        monitor_output=monitor_output,
    )


if __name__ == "__main__":
    warpt()
