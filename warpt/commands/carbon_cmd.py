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
    limit: int,
    days: int,
    output_json: bool,
    value: str | None = None,
) -> None:
    """Dispatch to the appropriate carbon subcommand.

    Parameters
    ----------
    subcommand : str | None
        One of: start, stop, status, history, summary, regions,
        set-region, intensity.
    label : str | None
        Session label for start command.
    limit : int
        Max sessions for history.
    days : int
        Time window for summary.
    output_json : bool
        Whether to output JSON.
    value : str | None
        Value for set-region or intensity subcommands.
    """
    if subcommand is None or subcommand == "status":
        _show_status(output_json)
    elif subcommand == "start":
        _start_tracking(label)
    elif subcommand == "stop":
        _stop_tracking(output_json)
    elif subcommand == "history":
        _show_history(limit, output_json)
    elif subcommand == "summary":
        _show_summary(days, output_json)
    elif subcommand == "regions":
        _show_regions()
    elif subcommand == "set-region":
        _set_region(value)
    elif subcommand == "intensity":
        _set_intensity(value)
    elif subcommand == "kwh-price":
        _set_kwh_price(value)


def _start_tracking(label: str | None) -> None:
    """Open a carbon tracking session bookmarked against the power-daemon."""
    from warpt.carbon.config import (
        DEFAULT_KWH_PRICE,
        get_effective_kwh_price,
        get_effective_region_and_intensity,
        load_carbon_config,
    )
    from warpt.carbon.session import start_session

    # Resolve region / intensity / rate from config
    region, intensity = get_effective_region_and_intensity()
    kwh_price = get_effective_kwh_price()
    cfg = load_carbon_config()
    has_config = bool(cfg.get("region") or cfg.get("intensity"))

    # start_session bookmarks the daemon counter and errors if it is unreachable.
    effective_label = label or "manual"
    try:
        session_id = start_session(
            label=effective_label,
            region=region,
            intensity=intensity if region == "CUSTOM" else None,
            kwh_price=kwh_price,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(_SECTION_SEP)
    print("  carbon tracking started")
    print(_SECTION_SEP)
    print(f"\n  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Session:  {session_id[:8]}...")
    print(f"  Label:    {effective_label}")

    # Show region/intensity source
    if has_config and cfg.get("intensity"):
        print(f"  Intensity: {intensity} gCO2/kWh (custom)")
    elif has_config:
        print(f"  Region:   {region} ({intensity} gCO2/kWh)")
    else:
        print()
        print(
            "  \u26a0 No region or intensity set"
            " \u2014 using US (385 gCO2/kWh) as default."
        )
        print(
            "    \u2192 Run `warpt carbon stop` to stop,"
            " then set your region or intensity."
        )
        print("    \u2192 Run `warpt carbon regions` to view available regions")
        print("    \u2192 Run `warpt carbon set-region --value <CODE>` to set a region")
        print(
            "    \u2192 Run `warpt carbon intensity"
            " --value <NUMBER>` to set a custom value"
        )
        print("    \u2192 See docs.earthframe.com/warpt/carbon for full details")
        print()

    if "kwh_price" in cfg:
        print(f"  Rate:     ${kwh_price}/kWh")
    else:
        print(f"  Rate:     ${DEFAULT_KWH_PRICE}/kWh (default)")
        print(
            "    \u2192 Run `warpt carbon kwh-price --value <NUMBER>` to set your rate"
        )

    print("  Source:   daemon")
    print("\n  Stop with: warpt carbon stop")


def _stop_tracking(output_json: bool) -> None:
    """Close the carbon tracking session and display results."""
    from warpt.carbon.session import stop_session

    session = stop_session()
    if session is None:
        print("No carbon tracking session is active.")
        return

    if output_json:
        print(json.dumps(session.to_dict(), indent=2))
        return

    print(_SECTION_SEP)
    print("  carbon tracking stopped")
    print(_SECTION_SEP)
    print(f"\n  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Session:  {session.id[:8]}...")
    print(f"  Label:    {session.label}")
    print(f"  Duration: {_format_duration(session.duration_s or 0)}")
    print(f"  Region:   {session.region}")

    status = session.metadata.get("status", "completed")
    if status != "completed":
        # Daemon connection lost or counter reset mid-session.
        print(f"\n  ⚠ {status}")
        return

    from warpt.carbon.calculator import CarbonCalculator

    calc = CarbonCalculator(region=session.region)
    humanized = calc.humanize(session.co2_grams or 0.0)

    print("\n  [Energy]")
    print(f"  {_HEADING_UNDERLINE}")
    energy_mwh = (session.energy_kwh or 0) * 1_000_000
    print(f"  Energy:   {energy_mwh:.1f} mWh ({session.energy_kwh or 0:.8f} kWh)")
    print(f"  Avg Power: {session.metadata.get('avg_power_w', 0):.1f} W")
    print("\n  [Impact]")
    print(f"  {_HEADING_UNDERLINE}")
    print(f"  CO2:      {session.co2_grams or 0:.4f} g")
    print(f"  Cost:     ${session.cost_usd or 0:.6f}")
    print(f"  ~ {humanized}")


def _show_status(output_json: bool) -> None:
    """Show the open session's live status (polled from the daemon)."""
    from warpt.carbon.session import session_status

    status = session_status()

    if output_json:
        print(json.dumps(status or {"running": False}, indent=2))
        return

    if status is None:
        print("No carbon tracking session is active.")
        print("\n  Start with: warpt carbon start")
        return

    print(_SECTION_SEP)
    print("  carbon tracking active")
    print(_SECTION_SEP)
    print(f"\n  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Session:  {status.get('session_id', 'unknown')[:8]}...")
    print(f"  Label:    {status.get('label', 'manual')}")
    print(f"  Region:   {status.get('region', 'US')}")
    if "elapsed_s" in status:
        print(f"  Elapsed:  {_format_duration(status['elapsed_s'])}")

    if status.get("daemon_unreachable"):
        print("\n  ⚠ power-daemon not reachable — cannot read live power")
    else:
        print("\n  [Power]")
        print(f"  {_HEADING_UNDERLINE}")
        if "current_power_w" in status:
            print(f"  Current:  {status['current_power_w']:.1f} W")
        if "energy_kwh_so_far" in status:
            energy_mwh = status["energy_kwh_so_far"] * 1_000_000
            print(f"  Energy:   {energy_mwh:.1f} mWh (so far)")

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
        f"\n  {'Date':<20} {'Label':<16} {'Duration':<10} "
        f"{'Avg W':>7} {'mWh':>8} {'gCO2':>8} {'Cost':>8}"
    )
    print(f"  {_HEADING_UNDERLINE * 3}")

    for s in sessions:
        dt = datetime.fromtimestamp(s.start_time).strftime("%Y-%m-%d %H:%M:%S")
        label = s.label[:15] if len(s.label) > 15 else s.label
        dur = _format_duration(s.duration_s or 0)
        avg_w = s.metadata.get("avg_power_w", 0)
        energy_mwh = (s.energy_kwh or 0) * 1_000_000
        co2 = s.co2_grams or 0
        cost = s.cost_usd or 0

        print(
            f"  {dt:<20} {label:<16} {dur:<10} "
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
    print(f"\n  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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


def _set_region(value: str | None) -> None:
    """Set the grid region for carbon tracking."""
    from warpt.carbon.config import (
        load_carbon_config,
        save_carbon_config,
        validate_region,
    )
    from warpt.carbon.grid_intensity import get_grid_intensity

    if value is None:
        print("Usage: warpt carbon set-region --value <CODE>", file=sys.stderr)
        sys.exit(1)

    code = value.upper()
    if not validate_region(code):
        print(f"Unknown region '{value}'.", file=sys.stderr)
        print(
            "Run 'warpt carbon regions' to see available regions, or use "
            "'warpt carbon intensity --value <N>' for a custom value.",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = {"region": code}
    # Preserve kwh_price if previously set
    existing = load_carbon_config()
    if "kwh_price" in existing:
        cfg["kwh_price"] = existing["kwh_price"]
    save_carbon_config(cfg)
    intensity = get_grid_intensity(code)
    print(f"Region set to {code} ({intensity} gCO2/kWh)")


def _set_intensity(value: str | None) -> None:
    """Set a custom carbon intensity value."""
    from warpt.carbon.config import load_carbon_config, save_carbon_config

    if value is None:
        print("Usage: warpt carbon intensity --value <NUMBER>", file=sys.stderr)
        sys.exit(1)

    try:
        float_value = float(value)
    except ValueError:
        print("Intensity must be a positive number (gCO2/kWh)", file=sys.stderr)
        sys.exit(1)

    if float_value <= 0:
        print("Intensity must be a positive number (gCO2/kWh)", file=sys.stderr)
        sys.exit(1)

    cfg = {"intensity": float_value}
    # Preserve kwh_price if previously set
    existing = load_carbon_config()
    if "kwh_price" in existing:
        cfg["kwh_price"] = existing["kwh_price"]
    save_carbon_config(cfg)
    print(f"Custom intensity set to {float_value} gCO2/kWh")


def _set_kwh_price(value: str | None) -> None:
    """Set the electricity price per kWh."""
    from warpt.carbon.config import load_carbon_config, save_carbon_config

    if value is None:
        print("Usage: warpt carbon kwh-price --value <NUMBER>", file=sys.stderr)
        sys.exit(1)

    try:
        float_value = float(value)
    except ValueError:
        print("Price must be a positive number ($/kWh)", file=sys.stderr)
        sys.exit(1)

    if float_value <= 0:
        print("Price must be a positive number ($/kWh)", file=sys.stderr)
        sys.exit(1)

    cfg = load_carbon_config()
    cfg["kwh_price"] = float_value
    save_carbon_config(cfg)
    print(f"Electricity price set to ${float_value}/kWh")


def _show_regions() -> None:
    """Display all available grid regions and their carbon intensities."""
    from warpt.carbon.grid_intensity import list_regions

    regions = list_regions()

    print(_SECTION_SEP)
    print("  grid carbon intensity by region")
    print(_SECTION_SEP)
    print()
    print("  Set your region:    warpt carbon set-region --value <CODE>")
    print("  Set custom value:   warpt carbon intensity --value <NUMBER>")
    print(f"\n  {'Region':<10} {'gCO2/kWh':>10}")
    print(f"  {_HEADING_UNDERLINE * 3}")

    for region, intensity in sorted(regions.items()):
        print(f"  {region:<10} {intensity:>10.0f}")

    print("\n  Set your region: warpt carbon set-region --value <CODE>")
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
