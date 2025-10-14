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
    print("Hello, Warpt!")
    
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
