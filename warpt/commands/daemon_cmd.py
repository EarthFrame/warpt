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

    # First-run: check for intelligence config
    from pathlib import Path

    config_path = Path(warpt_dir) / "config.yaml"
    if not config_path.exists():
        click.echo("No intelligence config found.")
        if click.confirm(
            "Would you like to set up AI diagnostics? (requires Ollama)",
            default=False,
        ):
            from warpt.commands.er_cmd import er_wizard

            er_wizard(warpt_dir)
        else:
            click.echo(
                "Tip: run 'warpt daemon er' later to enable intelligence."
            )

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
def er():
    """Run the ER intelligence setup wizard."""
    err = check_duckdb()
    if err:
        raise click.ClickException(err)

    from warpt.commands.er_cmd import er_wizard

    er_wizard(_get_warpt_dir())


@daemon.command()
@click.option(
    "--case", "case_id", type=int, default=None,
    help="Show a specific case by ID.",
)
@click.option("--list", "list_all", is_flag=True, help="List all cases.")
def inspect(case_id, list_all):
    """Inspect cases from the warpt database."""
    err = check_duckdb()
    if err:
        raise click.ClickException(err)

    import logging
    import shutil
    import tempfile

    import duckdb

    from warpt.commands.inspect_cmd import list_cases, show_case, show_latest
    from warpt.daemon.casefile import CaseFile

    logging.getLogger("warpt").setLevel(logging.WARNING)

    warpt_dir = _get_warpt_dir()
    db_path = os.path.join(warpt_dir, "warpt.db")

    # Try read-only first; if the daemon holds the lock, copy the DB
    # to a temp file and query that instead.
    try:
        cf = CaseFile(db_path, read_only=True)
        tmp_path = None
    except duckdb.IOException:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        shutil.copy2(db_path, tmp_path)
        wal_path = db_path + ".wal"
        if os.path.exists(wal_path):
            shutil.copy2(wal_path, tmp_path + ".wal")
        cf = CaseFile(tmp_path, read_only=True)

    try:
        if list_all:
            list_cases(cf)
        elif case_id is not None:
            show_case(cf, case_id)
        else:
            show_latest(cf)
    finally:
        cf.close()
        if tmp_path:
            os.unlink(tmp_path)
            wal_tmp = tmp_path + ".wal"
            if os.path.exists(wal_tmp):
                os.unlink(wal_tmp)


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
