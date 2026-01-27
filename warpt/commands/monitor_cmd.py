"""Monitor command helper that drives the live daemon."""

from __future__ import annotations

import time
from datetime import datetime

from warpt.monitoring import ResourceSnapshot, SystemMonitorDaemon
from warpt.utils.env import get_env


def run_monitor(interval_seconds: float, duration_seconds: float | None) -> None:
    """Run the live system monitor until interrupted or duration elapses.

    Args:
        interval_seconds: Sampling interval in seconds. Must be positive.
        duration_seconds: Optional time limit in seconds. If None, runs until
            interrupted by the user.
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero")

    daemon = SystemMonitorDaemon(interval_seconds=interval_seconds)
    daemon.start()
    start_time = time.monotonic()
    last_timestamp: float | None = None

    try:
        while True:
            snapshot = daemon.get_latest_snapshot()
            if snapshot and snapshot.timestamp != last_timestamp:
                print(_format_snapshot(snapshot), flush=True)
                last_timestamp = snapshot.timestamp

            if duration_seconds is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= duration_seconds:
                    break

            time.sleep(min(interval_seconds, 0.25))

    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        daemon.stop()


def _format_snapshot(snapshot: ResourceSnapshot) -> str:
    """Render a snapshot as a multi-line string for CLI output.

    Args:
        snapshot: Snapshot to render.

    Returns
    -------
        Multi-line string describing the snapshot.
    """
    timestamp = datetime.fromtimestamp(snapshot.timestamp).strftime("%Y-%m-%d %H:%M:%S")

    used_memory = max(snapshot.total_memory_bytes - snapshot.available_memory_bytes, 0)
    total_str = _format_bytes(snapshot.total_memory_bytes)
    used_str = _format_bytes(used_memory)

    memory_line = (
        f"Memory: {_format_percent(snapshot.memory_utilization_percent)} "
        f"({used_str} used / {total_str})"
    )
    wired = snapshot.wired_memory_bytes
    if wired is not None:
        memory_line += f" wired: {_format_bytes(wired)}"

    enable_power = get_env("WARPT_ENABLE_POWER", default=False, as_type=bool)

    cpu_line = (
        f"[{timestamp}] CPU: {_format_percent(snapshot.cpu_utilization_percent)} "
    )
    if enable_power:
        cpu_line += f"power: {_format_power(snapshot.cpu_power_watts)}"

    lines = [
        cpu_line,
        memory_line,
    ]

    for gpu in snapshot.gpu_usage:
        model_suffix = f" ({gpu.model})" if gpu.model else ""
        gpu_line = (
            f"GPU {gpu.index}{model_suffix}: "
            f"util: {_format_percent(gpu.utilization_percent)} "
            f"mem: {_format_percent(gpu.memory_utilization_percent)} "
        )
        if enable_power:
            gpu_line += f"power: {_format_power(gpu.power_watts)}"

        lines.append(gpu_line)
        if gpu.guid:
            lines.append(f"    GUID: {gpu.guid}")

    return "\n".join(lines)


def _format_percent(value: float | None) -> str:
    """Format a percentage value for display.

    Args:
        value: Percentage value between 0 and 100 or None.

    Returns
    -------
        Formatted percentage string or 'N/A'.
    """
    return f"{value:.1f}%" if value is not None else "N/A"


def _format_power(value: float | None) -> str:
    """Format a power value in watts.

    Args:
        value: Power draw in watts or None.

    Returns
    -------
        Formatted power string or 'N/A'.
    """
    return f"{value:.1f} W" if value is not None else "N/A"


def _format_bytes(value: int) -> str:
    """Return a human-friendly representation of bytes.

    Args:
        value: Byte count to format.

    Returns
    -------
        Human-readable string (e.g., '3.2 GiB').
    """
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    current = float(value)
    for unit in units:
        if current < 1024.0 or unit == units[-1]:
            return f"{current:.1f} {unit}"
        current /= 1024.0
    return f"{current:.1f} {units[-1]}"
