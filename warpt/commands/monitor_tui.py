"""Text-based UI for the live monitor."""

from __future__ import annotations

import curses
import platform
import subprocess
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, TypedDict

import psutil

from warpt.monitoring import ResourceSnapshot, SystemMonitorDaemon

if TYPE_CHECKING:
    _CursesWindow = Any  # curses window type
else:
    _CursesWindow = object


class StorageUsage(TypedDict):
    """Typed storage usage record for the monitor dashboard."""

    mount: str
    percent: float
    used_gb: float
    total_gb: float
    available_gb: float


class PowerMetrics:
    """Background power metrics collector for macOS."""

    def __init__(self):
        self.cpu_power_mw: float | None = None
        self.gpu_power_mw: float | None = None
        self.ane_power_mw: float | None = None
        self.package_power_mw: float | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start collecting power metrics in background."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop collecting power metrics."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _collect_loop(self) -> None:
        """Background loop to collect power metrics."""
        while self._running:
            try:
                result = subprocess.run(
                    [
                        "sudo",
                        "-n",
                        "powermetrics",
                        "--samplers",
                        "cpu_power",
                        "-i",
                        "1000",
                        "-n",
                        "1",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if result.returncode == 0:
                    self._parse_powermetrics(result.stdout)
                else:
                    with self._lock:
                        self.error = "Requires passwordless sudo for powermetrics"
            except subprocess.TimeoutExpired:
                with self._lock:
                    self.error = "powermetrics timeout"
            except FileNotFoundError:
                with self._lock:
                    self.error = "powermetrics not found"
            except Exception as e:
                with self._lock:
                    self.error = str(e)
            time.sleep(2.0)

    def _parse_powermetrics(self, output: str) -> None:
        """Parse powermetrics output."""
        with self._lock:
            self.error = None
            for line in output.splitlines():
                if "CPU Power:" in line:
                    try:
                        val = line.split(":")[1].strip().split()[0]
                        self.cpu_power_mw = float(val)
                    except (IndexError, ValueError):
                        pass
                elif "GPU Power:" in line:
                    try:
                        val = line.split(":")[1].strip().split()[0]
                        self.gpu_power_mw = float(val)
                    except (IndexError, ValueError):
                        pass
                elif "ANE Power:" in line:
                    try:
                        val = line.split(":")[1].strip().split()[0]
                        self.ane_power_mw = float(val)
                    except (IndexError, ValueError):
                        pass
                elif "Combined Power (CPU + GPU + ANE):" in line:
                    try:
                        val = line.split(":")[1].strip().split()[0]
                        self.package_power_mw = float(val)
                    except (IndexError, ValueError):
                        pass

    def get_metrics(self) -> dict[str, float | str | None]:
        """Get current power metrics."""
        with self._lock:
            return {
                "cpu_power_mw": self.cpu_power_mw,
                "gpu_power_mw": self.gpu_power_mw,
                "ane_power_mw": self.ane_power_mw,
                "package_power_mw": self.package_power_mw,
                "error": self.error,
            }


class MonitorState:
    """State tracking for the monitor TUI."""

    def __init__(self, max_history: int = 5):
        self.current_tab = 0
        self.tabs = ["Overview", "Processes", "Power"]
        self.cpu_history: deque[float] = deque(maxlen=max_history)
        self.mem_history: deque[float] = deque(maxlen=max_history)
        self.power_metrics: PowerMetrics | None = None

        # Start power metrics collection if on macOS
        if platform.system() == "Darwin":
            self.power_metrics = PowerMetrics()
            self.power_metrics.start()

    def add_snapshot(self, snapshot: ResourceSnapshot) -> None:
        """Add a snapshot to history for averaging."""
        if snapshot.cpu_utilization_percent is not None:
            self.cpu_history.append(snapshot.cpu_utilization_percent)
        if snapshot.memory_utilization_percent is not None:
            self.mem_history.append(snapshot.memory_utilization_percent)

    def get_cpu_avg(self) -> float:
        """Get 5-second average CPU utilization."""
        if self.cpu_history:
            return sum(self.cpu_history) / len(self.cpu_history)
        return 0.0

    def get_mem_avg(self) -> float:
        """Get 5-second average memory utilization."""
        if self.mem_history:
            return sum(self.mem_history) / len(self.mem_history)
        return 0.0

    def cleanup(self) -> None:
        """Clean up background threads."""
        if self.power_metrics:
            self.power_metrics.stop()


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
    stdscr: _CursesWindow,
    daemon: SystemMonitorDaemon,
    interval: float,
) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)  # 100ms timeout for responsive input
    last_snapshot: ResourceSnapshot | None = None
    state = MonitorState()
    last_update_time = time.time()
    needs_render = True

    try:
        while True:
            ch = stdscr.getch()
            if ch == ord("q"):
                break
            elif ch == ord("1"):
                state.current_tab = 0
                needs_render = True
            elif ch == ord("2"):
                state.current_tab = 1
                needs_render = True
            elif ch == ord("3"):
                state.current_tab = 2
                needs_render = True
            elif ch == curses.KEY_LEFT:
                state.current_tab = (state.current_tab - 1) % len(state.tabs)
                needs_render = True
            elif ch == curses.KEY_RIGHT:
                state.current_tab = (state.current_tab + 1) % len(state.tabs)
                needs_render = True

            # Check if it's time to update metrics
            current_time = time.time()
            if current_time - last_update_time >= interval:
                snapshot = daemon.get_latest_snapshot()
                if snapshot:
                    last_snapshot = snapshot
                    state.add_snapshot(snapshot)
                last_update_time = current_time
                needs_render = True

            # Only re-render if something changed
            if needs_render:
                stdscr.erase()
                _render_header(stdscr, state)

                if state.current_tab == 0:
                    _render_overview_tab(stdscr, last_snapshot, state)
                elif state.current_tab == 1:
                    _render_processes_tab(stdscr, last_snapshot)
                elif state.current_tab == 2:
                    _render_power_tab(stdscr, state)

                stdscr.refresh()
                needs_render = False
    finally:
        state.cleanup()


def _render_header(stdscr: _CursesWindow, state: MonitorState) -> None:
    """Render the header with tab navigation."""
    _safe_addstr(stdscr, 0, 0, "Warpt Monitor", curses.A_BOLD)
    _safe_addstr(stdscr, 0, 40, time.strftime("%Y-%m-%d %H:%M:%S"))
    _safe_addstr(stdscr, 1, 0, "Press 1-3 or ←→ to switch tabs, q to quit")

    # Tab bar
    tab_line = ""
    for idx, tab in enumerate(state.tabs):
        if idx == state.current_tab:
            tab_line += f"[{idx + 1}:{tab}] "
        else:
            tab_line += f" {idx + 1}:{tab}  "
    _safe_addstr(stdscr, 2, 0, tab_line)
    _safe_hline(stdscr, 3, 0, "=", 80)


def _render_overview_tab(
    stdscr: _CursesWindow,
    snapshot: ResourceSnapshot | None,
    state: MonitorState,
) -> None:
    """Render the main overview tab with system metrics."""
    row = 5
    if snapshot:
        row = _render_cpu_section(stdscr, row, snapshot, state)
        row = _render_memory_section(stdscr, row, snapshot, state)
        row = _render_gpu_section(stdscr, row, snapshot)
    else:
        row = _render_cpu_section_placeholder(stdscr, row, state)
        row = _render_memory_section_placeholder(stdscr, row, state)
        row = _render_gpu_section_placeholder(stdscr, row)
    _render_storage_section(stdscr, row)


def _render_processes_tab(
    stdscr: _CursesWindow, snapshot: ResourceSnapshot | None
) -> None:
    """Render the processes tab showing CPU-intensive processes."""
    row = 5
    _section_title(stdscr, row, "Top CPU-Intensive Processes")
    row += 2

    if not snapshot:
        _safe_addstr(stdscr, row, 2, "Waiting for data…")
        return

    try:
        processes: list[dict[str, Any]] = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
            try:
                info = proc.info
                if info["cpu_percent"] and info["cpu_percent"] > 1.0:
                    processes.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        processes.sort(key=lambda x: x["cpu_percent"], reverse=True)

        header = f"{'PID':<10} {'NAME':<30} {'CPU%':<10}"
        _safe_addstr(stdscr, row, 2, header)
        row += 1
        _safe_hline(stdscr, row, 2, "-", 50)
        row += 1

        for proc_info in processes[:15]:
            pid = proc_info["pid"]
            name = proc_info["name"][:28]
            cpu = proc_info["cpu_percent"]
            line = f"{pid:<10} {name:<30} {cpu:>6.1f}%"
            _safe_addstr(stdscr, row, 2, line)
            row += 1
    except Exception as e:
        _safe_addstr(stdscr, row, 2, f"Error reading processes: {e}")


def _render_power_tab(stdscr: _CursesWindow, state: MonitorState) -> None:
    """Render the power usage tab."""
    row = 5
    _section_title(stdscr, row, "Power Usage")
    row += 2

    system = platform.system()
    if system == "Darwin":
        _render_macos_power(stdscr, row, state)
    elif system == "Linux":
        _render_linux_power(stdscr, row)
    else:
        _safe_addstr(stdscr, row, 2, f"Power monitoring not supported on {system}")


def _render_macos_power(stdscr: _CursesWindow, row: int, state: MonitorState) -> None:
    """Render macOS power metrics from powermetrics (requires sudo)."""
    if not state.power_metrics:
        _safe_addstr(stdscr, row, 2, "Power metrics not initialized")
        return

    metrics = state.power_metrics.get_metrics()

    if metrics.get("error"):
        _safe_addstr(stdscr, row, 2, f"Error: {metrics['error']}", curses.A_BOLD)
        row += 2
        msg = "To enable power metrics, configure passwordless sudo:"
        _safe_addstr(stdscr, row, 2, msg)
        row += 1
        _safe_addstr(stdscr, row, 4, "1. Run: sudo visudo")
        row += 1
        _safe_addstr(stdscr, row, 4, "2. Add line:")
        row += 1
        username = platform.node()
        _safe_addstr(
            stdscr,
            row,
            6,
            f"{username} ALL=(ALL) NOPASSWD: /usr/bin/powermetrics",
        )
        return

    # Display power metrics
    has_data = False

    if metrics.get("package_power_mw") is not None:
        power_w = metrics["package_power_mw"] / 1000.0  # type: ignore
        _safe_addstr(
            stdscr,
            row,
            2,
            f"Combined Power:  {power_w:7.2f} W",
            curses.A_BOLD,
        )
        row += 2
        has_data = True

    if metrics.get("cpu_power_mw") is not None:
        power_w = metrics["cpu_power_mw"] / 1000.0  # type: ignore
        _safe_addstr(stdscr, row, 4, f"CPU Power:  {power_w:7.2f} W")
        row += 1
        has_data = True

    if metrics.get("gpu_power_mw") is not None:
        power_w = metrics["gpu_power_mw"] / 1000.0  # type: ignore
        _safe_addstr(stdscr, row, 4, f"GPU Power:  {power_w:7.2f} W")
        row += 1
        has_data = True

    if metrics.get("ane_power_mw") is not None:
        power_w = metrics["ane_power_mw"] / 1000.0  # type: ignore
        _safe_addstr(stdscr, row, 4, f"ANE Power:  {power_w:7.2f} W")
        row += 1
        has_data = True

    if not has_data:
        _safe_addstr(stdscr, row, 2, "Collecting power data...")


def _render_linux_power(stdscr: _CursesWindow, row: int) -> None:
    """Render Linux power metrics from sysfs."""
    _safe_addstr(stdscr, row, 2, "Linux Power Metrics")
    row += 1

    try:
        # Try to read from powercap interface
        power_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
        with open(power_path) as f:
            energy_uj = int(f.read().strip())
            _safe_addstr(stdscr, row, 2, f"CPU Energy: {energy_uj / 1_000_000:.2f} J")
    except (FileNotFoundError, PermissionError, ValueError):
        msg = "Power metrics unavailable (requires root or powercap access)"
        _safe_addstr(stdscr, row, 2, msg)


def _safe_addstr(
    stdscr: _CursesWindow,
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


def _safe_hline(stdscr: _CursesWindow, row: int, col: int, ch: str, width: int) -> None:
    try:
        stdscr.hline(row, col, ch, width)
    except curses.error:
        pass


def _draw_bar(
    stdscr: _CursesWindow,
    row: int,
    col: int,
    label: str,
    percent: float,
    suffix: str,
    label_width: int = 20,
    bar_width: int = 40,
) -> None:
    """Draw a progress bar with consistent alignment."""
    filled = int(bar_width * min(max(percent, 0.0), 100.0) / 100.0)
    bar = "█" * filled + "░" * (bar_width - filled)
    text = f"{label:<{label_width}} [{bar}] {suffix}"
    _safe_addstr(stdscr, row, col, text)


def _section_title(stdscr: _CursesWindow, row: int, title: str) -> None:
    """Draw a bold section title with a separator."""
    _safe_addstr(stdscr, row, 0, title, curses.A_BOLD)


def _render_cpu_section(
    stdscr: _CursesWindow,
    row: int,
    snapshot: ResourceSnapshot,
    state: MonitorState,
) -> int:
    cpu_percent = snapshot.cpu_utilization_percent or 0.0
    cpu_avg = state.get_cpu_avg()
    total_cores = psutil.cpu_count(logical=True) or 0

    _section_title(stdscr, row, "CPU")
    row += 1

    agg_label = f"Total ({total_cores} cores)"
    suffix = f"{cpu_percent:5.1f}% (5s avg: {cpu_avg:5.1f}%)"
    _draw_bar(
        stdscr, row, 2, agg_label, cpu_percent, suffix, label_width=20, bar_width=40
    )
    row += 1

    core_percents = psutil.cpu_percent(percpu=True)
    for idx, percent in enumerate(core_percents[:8], start=1):
        core_label = f"  Core {idx}"
        core_suffix = f"{percent:5.1f}%"
        _draw_bar(
            stdscr,
            row,
            2,
            core_label,
            percent,
            core_suffix,
            label_width=20,
            bar_width=32,
        )
        row += 1

    return row + 1


def _render_cpu_section_placeholder(
    stdscr: _CursesWindow, row: int, _state: MonitorState
) -> int:
    """Render CPU section with placeholder values."""
    total_cores = psutil.cpu_count(logical=True) or 0

    _section_title(stdscr, row, "CPU")
    row += 1

    agg_label = f"Total ({total_cores} cores)"
    _draw_bar(stdscr, row, 2, agg_label, 0.0, "—", label_width=20, bar_width=40)
    row += 1

    for idx in range(1, min(9, total_cores + 1)):
        core_label = f"  Core {idx}"
        _draw_bar(stdscr, row, 2, core_label, 0.0, "—", label_width=20, bar_width=32)
        row += 1

    return row + 1


def _render_memory_section(
    stdscr: _CursesWindow,
    row: int,
    snapshot: ResourceSnapshot,
    state: MonitorState,
) -> int:
    total_gib = snapshot.total_memory_bytes / 1024**3
    available_gib = snapshot.available_memory_bytes / 1024**3
    mem_used = total_gib - available_gib
    mem_percent = snapshot.memory_utilization_percent or 0.0
    mem_avg = state.get_mem_avg()

    _section_title(stdscr, row, "Memory")
    row += 1

    mem_suffix = f"{mem_used:5.1f}/{total_gib:5.1f}GiB (5s avg: {mem_avg:5.1f}%)"
    _draw_bar(
        stdscr, row, 2, "RAM", mem_percent, mem_suffix, label_width=20, bar_width=40
    )
    row += 1

    swap = psutil.swap_memory()
    swap_used_gib = swap.used / 1024**3
    swap_total_gib = swap.total / 1024**3
    swap_suffix = f"{swap_used_gib:5.1f}/{swap_total_gib:5.1f}GiB"
    _draw_bar(
        stdscr, row, 2, "Swap", swap.percent, swap_suffix, label_width=20, bar_width=40
    )
    row += 1

    return row + 1


def _render_memory_section_placeholder(
    stdscr: _CursesWindow, row: int, _state: MonitorState
) -> int:
    """Render memory section with placeholder values."""
    _section_title(stdscr, row, "Memory")
    row += 1

    _draw_bar(stdscr, row, 2, "RAM", 0.0, "—", label_width=20, bar_width=40)
    row += 1

    _draw_bar(stdscr, row, 2, "Swap", 0.0, "—", label_width=20, bar_width=40)
    row += 1

    return row + 1


def _render_gpu_section(
    stdscr: _CursesWindow, row: int, snapshot: ResourceSnapshot
) -> int:
    _section_title(stdscr, row, "GPUs")
    row += 1

    if snapshot.gpu_usage:
        for gpu in snapshot.gpu_usage:
            util = gpu.utilization_percent or 0.0
            mem_util = gpu.memory_utilization_percent or 0.0
            power = gpu.power_watts or 0.0

            gpu_label = f"GPU {gpu.index} ({gpu.model or 'N/A'})"
            gpu_suffix = f"{util:5.1f}% | {power:6.1f}W"
            _draw_bar(
                stdscr,
                row,
                2,
                gpu_label,
                util,
                gpu_suffix,
                label_width=30,
                bar_width=30,
            )
            row += 1

            mem_label = "  └─ Memory"
            mem_suffix = f"{mem_util:5.1f}%"
            _draw_bar(
                stdscr,
                row,
                2,
                mem_label,
                mem_util,
                mem_suffix,
                label_width=30,
                bar_width=30,
            )
            row += 1

            if gpu.guid:
                _safe_addstr(stdscr, row, 6, f"GUID: {gpu.guid[:32]}...")
                row += 1
        return row + 1
    _safe_addstr(stdscr, row, 2, "No NVIDIA GPUs detected")
    return row + 2


def _render_gpu_section_placeholder(stdscr: _CursesWindow, row: int) -> int:
    """Render GPU section with placeholder values."""
    _section_title(stdscr, row, "GPUs")
    row += 1

    # Show placeholder for one GPU
    _draw_bar(
        stdscr, row, 2, "GPU 0 (Detecting…)", 0.0, "—", label_width=30, bar_width=30
    )
    row += 1

    _draw_bar(stdscr, row, 2, "  └─ Memory", 0.0, "—", label_width=30, bar_width=30)
    row += 1

    return row + 1


def _render_storage_section(stdscr: _CursesWindow, row: int) -> None:
    _section_title(stdscr, row, "Storage")
    row += 1

    mounts = ["/System/Volumes/Data", "/"]
    for usage in _collect_storage(mounts):
        storage_label = usage["mount"]
        storage_suffix = (
            f"{usage['used_gb']:6.1f}/{usage['total_gb']:6.1f}GiB "
            f"(avail: {usage['available_gb']:6.1f}GiB)"
        )
        _draw_bar(
            stdscr,
            row,
            2,
            storage_label,
            usage["percent"],
            storage_suffix,
            label_width=24,
            bar_width=30,
        )
        row += 1


def _collect_storage(mountpoints: list[str] | None = None) -> list[StorageUsage]:
    def build_entry(partition: str) -> StorageUsage | None:
        try:
            usage = psutil.disk_usage(partition)
        except (PermissionError, FileNotFoundError):
            return None

        return StorageUsage(
            mount=partition,
            percent=float(usage.percent),
            used_gb=usage.used / 1024**3,
            total_gb=usage.total / 1024**3,
            available_gb=usage.free / 1024**3,
        )

    if mountpoints:
        return [entry for mount in mountpoints if (entry := build_entry(mount))]

    excluded_mounts = {
        "/System/Volumes/Preboot",
        "/System/Volumes/Update",
        "/System/Volumes/VM",
    }
    preferred_mount = "/System/Volumes/Data"
    seen_devices: list[str] = []
    device_entries: dict[str, StorageUsage] = {}

    for partition in psutil.disk_partitions(all=False):
        if partition.mountpoint in excluded_mounts:
            continue
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except (PermissionError, FileNotFoundError):
            continue

        entry = StorageUsage(
            mount=partition.mountpoint,
            percent=float(usage.percent),
            used_gb=usage.used / 1024**3,
            total_gb=usage.total / 1024**3,
            available_gb=usage.free / 1024**3,
        )

        existing = device_entries.get(partition.device)
        if existing:
            if partition.mountpoint == preferred_mount:
                device_entries[partition.device] = entry
                if partition.device not in seen_devices:
                    seen_devices.append(partition.device)
            continue

        device_entries[partition.device] = entry
        seen_devices.append(partition.device)

    return [device_entries[device] for device in seen_devices[:4]]
