#!/usr/bin/env python3
"""Warpt CLI - Command-line interface for Warpt."""

import click


@click.group()
def warpt():
    """Warpt command-line tool for system monitoring and utilities."""
    pass


@warpt.command()
def list():
    """List CPU information."""
    from warpt.commands.list_cmd import run_list

    run_list()


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
