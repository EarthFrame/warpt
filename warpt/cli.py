#!/usr/bin/env python3
"""Warpt CLI - Command-line interface for Warpt."""

import click
from warpt.commands.list_cmd import run_list


@click.group()
def warpt():
    """Warpt command-line tool for system monitoring and utilities."""
    pass


@warpt.command()
@click.option(
    "--export",
    is_flag=True,
    default=False,
    help="Export results to JSON file with default filename (warpt_list_TIMESTAMP.json)"
)
@click.option(
    "--export-file",
    default=None,
    help="Export results to JSON file with custom filename"
)
def list(export, export_file):
    """List CPU and GPU information."""

    # Determine export format and filename
    if export_file:
        # --export-file provided (takes precedence)
        export_format = 'json'
        export_filename = export_file
    elif export:
        # --export flag used
        export_format = 'json'
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
def stress():
    """Run system stress tests."""
    print("Stress test!")


if __name__ == "__main__":
    warpt()
