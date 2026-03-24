"""CLI commands for warpt daemon start/stop/status."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import click


def check_duckdb() -> str | None:
    """Check if duckdb is available.

    Returns
    -------
        Error message if duckdb is missing, None if available.
    """
    try:
        importlib.import_module("duckdb")
        return None
    except ImportError:
        return (
            "The daemon requires DuckDB. " "Install it with: pip install warpt[daemon]"
        )


def _get_warpt_dir() -> str:
    """Resolve the warpt directory from env or default."""
    return os.environ.get("WARPT_DIR", os.path.expanduser("~/.warpt"))


@click.group()
def daemon():
    """Manage the warpt background daemon."""


@daemon.command()
def start():
    """Start the warpt daemon in the background."""
    err = check_duckdb()
    if err:
        raise click.ClickException(err)

    from warpt.daemon.daemon_process import DaemonProcess

    warpt_dir = _get_warpt_dir()
    dp = DaemonProcess(warpt_dir=warpt_dir)

    if dp.is_running():
        click.echo("Daemon is already running.")
        return

    # Launch daemon as a detached subprocess
    cmd = [sys.executable, "-m", "warpt.daemon.daemon_process", warpt_dir]
    kwargs: dict = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "start_new_session": True,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )

    subprocess.Popen(cmd, **kwargs)
    click.echo("Daemon started.")


@daemon.command()
def stop():
    """Stop the running warpt daemon."""
    err = check_duckdb()
    if err:
        raise click.ClickException(err)

    from warpt.daemon.daemon_process import send_stop

    warpt_dir = _get_warpt_dir()
    result = send_stop(warpt_dir=warpt_dir)
    click.echo(result)


@daemon.command()
def status():
    """Show daemon status and key stats."""
    err = check_duckdb()
    if err:
        raise click.ClickException(err)

    from warpt.daemon.daemon_process import DaemonProcess

    warpt_dir = _get_warpt_dir()
    dp = DaemonProcess(warpt_dir=warpt_dir)
    info = dp.get_status()

    if info["running"]:
        click.echo(f"Daemon running (PID {info['pid']})")
        if "vitals_count" in info:
            click.echo(f"  Vitals rows:  {info['vitals_count']}")
            click.echo(f"  Events:       {info['events_count']}")
            click.echo(f"  Open cases:   {info['open_cases']}")
            if info.get("last_heartbeat"):
                click.echo(f"  Last heartbeat: {info['last_heartbeat']}")
    else:
        click.echo("Daemon not running.")
