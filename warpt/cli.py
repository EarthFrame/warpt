#!/usr/bin/env python3
"""
Warpt CLI - Command-line interface for Warpt
"""

import click

@click.group()
def warpt():
    pass

@warpt.command()
def list():
    """List CPU information"""
    from warpt.commands.list_cmd import run_list
    run_list()

@warpt.command()
@click.option('--verbose', '-v', is_flag=True, help='Show detailed version information')
def version(verbose):
    """Display warpt version information"""
    from warpt.commands.version_cmd import run_version
    run_version(verbose=verbose)
    
@warpt.command()
def monitor():
    print("Live monitoring!")
    
@warpt.command()
def benchmark():
    print("Benchmarking!")
    
@warpt.command()
def system_info():
    print("System info!")
    
@warpt.command()
def check():
    print("Health check!")

@warpt.command()
def stress():
    print("Stress test!")


if __name__ == "__main__":
    warpt()
