"""Text-based UI for the live monitor."""

from __future__ import annotations

import curses
import time
from typing import TypedDict

import psutil

from warpt.monitoring import ResourceSnapshot, SystemMonitorDaemon


class StorageUsage(TypedDict):
    """Typed storage usage record for the monitor dashboard."""

    mount: str
    percent: float
    used_gb: float
    total_gb: float
    available_gb: float


def run_monitor_tui(interval_seconds: float = 1.0) -> None:
    """Run the curses-based monitor UI until interrupted.

    Args:
        interval_seconds: Sampling interval in seconds.
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero")

    daemon = SystemMonitorDaemon(interval_seconds=interval_seconds)
    daemon.start()

    try:
        curses.wrapper(_curses_main, daemon, interval_seconds)
    finally:
        daemon.stop()


def _curses_main(
    stdscr: curses._CursesWindow, daemon: SystemMonitorDaemon, interval: float
) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)
    last_snapshot: ResourceSnapshot | None = None

    while True:
        ch = stdscr.getch()
        if ch == ord("q"):
            break

        snapshot = daemon.get_latest_snapshot()
        if snapshot:
            last_snapshot = snapshot

        stdscr.erase()
        _safe_addstr(stdscr, 0, 0, "Warpt Monitor (press q to quit)", curses.A_BOLD)
        _safe_addstr(stdscr, 1, 0, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        _safe_hline(stdscr, 2, 0, "-", 80)

        if last_snapshot:
            _render_snapshot(stdscr, last_snapshot)
        else:
            _safe_addstr(stdscr, 4, 0, "Waiting for first snapshot…")

        stdscr.refresh()
        time.sleep(interval)


def _render_snapshot(stdscr: curses._CursesWindow, snapshot: ResourceSnapshot) -> None:
    row = 4
    row = _render_cpu_section(stdscr, row, snapshot)
    row = _render_memory_section(stdscr, row, snapshot)
    row = _render_gpu_section(stdscr, row, snapshot)
    _render_storage_section(stdscr, row)


def _safe_addstr(
    stdscr: curses._CursesWindow,
    row: int,
    col: int,
    text: str | None = None,
    attr: int = 0,
) -> None:
    try:
        if text is None:
            text = ""
        stdscr.addstr(row, col, text, attr)
    except curses.error:
        pass


def _safe_hline(
    stdscr: curses._CursesWindow, row: int, col: int, ch: str, width: int
) -> None:
    try:
        stdscr.hline(row, col, ch, width)
    except curses.error:
        pass


def _draw_bar(
    stdscr: curses._CursesWindow,
    row: int,
    col: int,
    label: str,
    percent: float,
    suffix: str,
) -> None:
    width = 40
    filled = int(width * min(max(percent, 0.0), 100.0) / 100.0)
    bar = "█" * filled + "░" * (width - filled)
    text = f"{label:<18} [{bar}] {suffix}"
    _safe_addstr(stdscr, row, col, text)


def _render_cpu_section(
    stdscr: curses._CursesWindow, row: int, snapshot: ResourceSnapshot
) -> int:
    cpu_percent = snapshot.cpu_utilization_percent or 0.0
    total_cores = psutil.cpu_count(logical=True) or 0
    _safe_addstr(stdscr, row, 0, "CPU Overview:")
    _draw_bar(
        stdscr,
        row + 1,
        2,
        f"Aggregate ({total_cores} cores)",
        cpu_percent,
        f"{cpu_percent:.1f}%",
    )
    core_percents = psutil.cpu_percent(percpu=True)
    for idx, percent in enumerate(core_percents[:8], start=1):
        _draw_bar(
            stdscr,
            row + 1 + idx,
            4,
            f"Core {idx}",
            percent,
            f"{percent:.1f}%",
        )
    return row + 1 + len(core_percents[:8]) + 1


def _render_memory_section(
    stdscr: curses._CursesWindow, row: int, snapshot: ResourceSnapshot
) -> int:
    total_gib = snapshot.total_memory_bytes / 1024**3
    available_gib = snapshot.available_memory_bytes / 1024**3
    mem_used = total_gib - available_gib
    mem_percent = snapshot.memory_utilization_percent or 0.0
    mem_suffix = (
        f"{mem_used:.1f}GiB used / {total_gib:.1f}GiB total ({mem_percent:.1f}%)"
    )
    swap = psutil.swap_memory()
    _safe_addstr(stdscr, row, 0, "Memory:")
    _draw_bar(stdscr, row + 1, 2, "RAM", mem_percent, mem_suffix)
    available_text = (
        f"Available: {available_gib:.1f}GiB " f"({(100.0 - mem_percent):.1f}%)"
    )
    _safe_addstr(stdscr, row + 2, 2, available_text)
    _draw_bar(
        stdscr,
        row + 3,
        2,
        "Swap",
        swap.percent,
        f"{swap.used / 1024**3:.1f}GiB/{swap.total / 1024**3:.1f}GiB",
    )
    return row + 5


def _render_gpu_section(
    stdscr: curses._CursesWindow, row: int, snapshot: ResourceSnapshot
) -> int:
    _safe_addstr(stdscr, row, 0, "GPUs:")
    if snapshot.gpu_usage:
        for idx, gpu in enumerate(snapshot.gpu_usage, start=1):
            util_row = row + idx * 3 - 2
            util = gpu.utilization_percent or 0.0
            mem_util = gpu.memory_utilization_percent or 0.0
            power = gpu.power_watts or 0.0
            _draw_bar(
                stdscr,
                util_row,
                2,
                f"GPU[{gpu.index}] {gpu.model or 'N/A'}",
                util,
                f"util {util:.1f}% power {power:.1f}W",
            )
            _draw_bar(
                stdscr,
                util_row + 1,
                4,
                "GPU Memory",
                mem_util,
                f"mem {mem_util:.1f}%",
            )
            if gpu.guid:
                _safe_addstr(stdscr, util_row + 2, 4, f"GUID: {gpu.guid}")
        return util_row + 3
    _safe_addstr(stdscr, row + 1, 2, "No NVIDIA GPUs detected")
    return row + 3


def _render_storage_section(stdscr: curses._CursesWindow, row: int) -> None:
    _safe_addstr(stdscr, row, 0, "Storage:")
    for idx, usage in enumerate(_collect_storage(), start=1):
        storage_suffix = (
            f"{usage['used_gb']:.1f}GiB/{usage['total_gb']:.1f}GiB "
            f"(avail {usage['available_gb']:.1f}GiB)"
        )
        _draw_bar(
            stdscr,
            row + idx,
            2,
            usage["mount"],
            usage["percent"],
            storage_suffix,
        )


def _collect_storage() -> list[StorageUsage]:
    partitions = []
    seen_devices: set[str] = set()
    for partition in psutil.disk_partitions(all=False):
        if partition.device in seen_devices:
            continue
        seen_devices.add(partition.device)
        if partition.mountpoint.startswith("/System/Volumes/"):
            continue
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            continue
        partitions.append(
            StorageUsage(
                mount=partition.mountpoint,
                percent=float(usage.percent),
                used_gb=usage.used / 1024**3,
                total_gb=usage.total / 1024**3,
                available_gb=(usage.total - usage.used) / 1024**3,
            )
        )
    return partitions[:4]
