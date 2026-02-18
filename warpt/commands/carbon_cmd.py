"""Carbon tracking CLI command implementation."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime

_SECTION_SEP = "=" * 40
_HEADING_UNDERLINE = "---"


def run_carbon(
    subcommand: str | None,
    label: str | None,
    region: str,
    interval: float,
    limit: int,
    days: int,
    output_json: bool,
) -> None:
    """Dispatch to the appropriate carbon subcommand.

    Parameters
    ----------
    subcommand : str | None
        One of: start, stop, status, history, summary, regions.
    label : str | None
        Session label for start command.
    region : str
        Grid region code.
    interval : float
        Sampling interval for daemon.
    limit : int
        Max sessions for history.
    days : int
        Time window for summary.
    output_json : bool
        Whether to output JSON.
    """
    if subcommand is None or subcommand == "status":
        _show_status(output_json)
    elif subcommand == "start":
        _start_tracking(label, interval, region)
    elif subcommand == "stop":
        _stop_tracking(output_json)
    elif subcommand == "history":
        _show_history(limit, output_json)
    elif subcommand == "summary":
        _show_summary(days, output_json)
    elif subcommand == "regions":
        _show_regions()


def _start_tracking(label: str | None, interval: float, region: str) -> None:
    """Start the carbon tracking daemon."""
    from warpt.carbon.daemon import start_daemon

    effective_label = label or "manual"
    try:
        session_id = start_daemon(
            label=effective_label, interval=interval, region=region
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(_SECTION_SEP)
    print("  carbon tracking started")
    print(_SECTION_SEP)
    print(f"\n  Session:  {session_id[:8]}...")
    print(f"  Label:    {effective_label}")
    print(f"  Region:   {region}")
    print(f"  Interval: {interval}s")
    print("\n  Stop with: warpt carbon stop")


def _stop_tracking(output_json: bool) -> None:
    """Stop the carbon tracking daemon and display results."""
    from warpt.carbon.daemon import stop_daemon

    session = stop_daemon()
    if session is None:
        print("No carbon tracking daemon is running.")
        return

    if output_json:
        print(json.dumps(session.to_dict(), indent=2))
        return

    from warpt.carbon.calculator import CarbonCalculator

    calc = CarbonCalculator(region=session.region)
    humanized = calc.humanize(session.co2_grams or 0.0)

    print(_SECTION_SEP)
    print("  carbon tracking stopped")
    print(_SECTION_SEP)
    print(f"\n  Session:  {session.id[:8]}...")
    print(f"  Label:    {session.label}")
    print(f"  Duration: {_format_duration(session.duration_s or 0)}")
    print(f"  Region:   {session.region}")
    print("\n  [Energy]")
    print(f"  {_HEADING_UNDERLINE}")
    energy_mwh = (session.energy_kwh or 0) * 1_000_000
    print(f"  Energy:   {energy_mwh:.1f} mWh ({session.energy_kwh or 0:.8f} kWh)")
    avg_w = session.metadata.get("avg_power_w", 0)
    peak_w = session.metadata.get("peak_power_w", 0)
    print(f"  Avg Power: {avg_w:.1f} W")
    print(f"  Peak Power: {peak_w:.1f} W")
    print(f"  Samples:  {session.metadata.get('sample_count', 0)}")
    print("\n  [Impact]")
    print(f"  {_HEADING_UNDERLINE}")
    print(f"  CO2:      {session.co2_grams or 0:.4f} g")
    print(f"  Cost:     ${session.cost_usd or 0:.6f}")
    print(f"  ~ {humanized}")


def _show_status(output_json: bool) -> None:
    """Show the current daemon status."""
    from warpt.carbon.daemon import daemon_status

    status = daemon_status()

    if output_json:
        print(json.dumps(status or {"running": False}, indent=2))
        return

    if status is None:
        print("No carbon tracking daemon is running.")
        print("\n  Start with: warpt carbon start")
        return

    print(_SECTION_SEP)
    print("  carbon tracking active")
    print(_SECTION_SEP)
    print(f"\n  PID:      {status['pid']}")
    print(f"  Session:  {status.get('session_id', 'unknown')[:8]}...")
    print(f"  Label:    {status.get('label', 'manual')}")
    print(f"  Region:   {status.get('region', 'US')}")
    if "elapsed_s" in status:
        print(f"  Elapsed:  {_format_duration(status['elapsed_s'])}")
    if "sample_count" in status:
        print(f"  Samples:  {status['sample_count']}")

    avg_w = status.get("avg_power_w", 0)
    peak_w = status.get("peak_power_w", 0)
    if avg_w > 0:
        elapsed = status.get("elapsed_s", 0)
        energy_mwh = (avg_w * elapsed / 3600) * 1000  # W * s -> mWh
        print("\n  [Power]")
        print(f"  {_HEADING_UNDERLINE}")
        print(f"  Avg:      {avg_w:.1f} W")
        print(f"  Peak:     {peak_w:.1f} W")
        print(f"  Energy:   {energy_mwh:.1f} mWh")

    print("\n  Stop with: warpt carbon stop")


def _show_history(limit: int, output_json: bool) -> None:
    """Display recent tracking sessions."""
    from warpt.carbon.store import EnergyStore

    store = EnergyStore()
    sessions = store.get_sessions(limit=limit)

    if output_json:
        print(json.dumps([s.to_dict() for s in sessions], indent=2))
        return

    if not sessions:
        print("No carbon tracking sessions found.")
        print("\n  Start tracking with: warpt carbon start")
        return

    print(_SECTION_SEP)
    print("  carbon tracking history")
    print(_SECTION_SEP)

    # Table header
    print(
        f"\n  {'Date':<12} {'Label':<16} {'Duration':<10} "
        f"{'Avg W':>7} {'mWh':>8} {'gCO2':>8} {'Cost':>8}"
    )
    print(f"  {_HEADING_UNDERLINE * 3}")

    for s in sessions:
        dt = datetime.fromtimestamp(s.start_time).strftime("%Y-%m-%d")
        label = s.label[:15] if len(s.label) > 15 else s.label
        dur = _format_duration(s.duration_s or 0)
        avg_w = s.metadata.get("avg_power_w", 0)
        energy_mwh = (s.energy_kwh or 0) * 1_000_000
        co2 = s.co2_grams or 0
        cost = s.cost_usd or 0

        print(
            f"  {dt:<12} {label:<16} {dur:<10} "
            f"{avg_w:>7.1f} {energy_mwh:>8.1f} {co2:>8.4f} ${cost:>7.4f}"
        )

    print(f"\n  Showing {len(sessions)} session(s)")


def _show_summary(days: int, output_json: bool) -> None:
    """Show aggregated summary across sessions."""
    from warpt.carbon.store import EnergyStore

    since = time.time() - (days * 86400)
    store = EnergyStore()
    summary = store.get_totals(since=since)

    if output_json:
        print(json.dumps(summary.to_dict(), indent=2))
        return

    print(_SECTION_SEP)
    print(f"  carbon summary (last {days} days)")
    print(_SECTION_SEP)

    if summary.total_sessions == 0:
        print("\n  No sessions recorded in this period.")
        return

    print(f"\n  Sessions:   {summary.total_sessions}")
    energy_mwh = summary.total_energy_kwh * 1_000_000
    print(f"  Energy:     {energy_mwh:.1f} mWh ({summary.total_energy_kwh:.8f} kWh)")
    print(f"  Avg Power:  {summary.avg_power_watts:.1f} W")

    print("\n  [Impact]")
    print(f"  {_HEADING_UNDERLINE}")
    print(f"  CO2:        {summary.total_co2_grams:.4f} g")
    print(f"  Cost:       ${summary.total_cost_usd:.6f}")
    print(f"  ~ {summary.humanized}")


def _show_regions() -> None:
    """Display all available grid regions and their carbon intensities."""
    from warpt.carbon.grid_intensity import list_regions

    regions = list_regions()

    print(_SECTION_SEP)
    print("  grid carbon intensity by region")
    print(_SECTION_SEP)
    print(f"\n  {'Region':<10} {'gCO2/kWh':>10}")
    print(f"  {_HEADING_UNDERLINE * 3}")

    for region, intensity in sorted(regions.items()):
        print(f"  {region:<10} {intensity:>10.0f}")

    print("\n  Use --region <code> to set your region.")
    world = regions.get("WORLD", 440)
    print(f"  Unknown regions fall back to WORLD ({world} gCO2/kWh).")


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m{s:.0f}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m}m"
