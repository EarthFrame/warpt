"""Power monitoring CLI command implementation."""

from __future__ import annotations

import json
import time
from datetime import datetime

from warpt.backends.power.factory import PowerMonitor
from warpt.models.power_models import PowerSnapshot


def run_power(
    interval_seconds: float = 1.0,
    duration_seconds: float | None = None,
    show_processes: bool = True,
    top_n_processes: int = 10,
    output_format: str = "text",
    output_file: str | None = None,
    continuous: bool = False,
) -> None:
    """Run power monitoring and display results.

    Args:
        interval_seconds: Sampling interval in seconds.
        duration_seconds: Stop after this many seconds (None = until interrupted).
        show_processes: Whether to show per-process power.
        top_n_processes: Number of top processes to show.
        output_format: Output format ("text", "json").
        output_file: Optional file to write JSON output.
        continuous: Whether to run continuously (vs single snapshot).
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero")

    monitor = PowerMonitor(include_process_attribution=show_processes)

    if not monitor.initialize():
        print("No power sources available on this system.")
        print("\nPossible reasons:")
        print("  - Linux: Need read access to /sys/class/powercap/intel-rapl/")
        print("  - macOS: Need passwordless sudo for powermetrics")
        print("  - No supported GPUs detected")
        return

    sources = monitor.get_available_sources()
    print(f"Available power sources: {[s.value for s in sources]}")
    print()

    if not continuous:
        # Single snapshot mode - wait a bit longer for powermetrics
        # (powermetrics needs ~1s sample interval + collection time)
        time.sleep(max(interval_seconds, 1.5))
        snapshot = monitor.get_snapshot()
        _display_snapshot(snapshot, show_processes, top_n_processes, output_format)

        if output_file:
            _write_json_output(snapshot, output_file)

        monitor.cleanup()
        return

    # Continuous mode — wrap in CarbonTracker for energy accounting
    from warpt.carbon.tracker import CarbonTracker

    snapshots: list[PowerSnapshot] = []
    start_time = time.time()

    with CarbonTracker(label="warpt power"):
        try:
            print("Power monitoring started. Press Ctrl+C to stop.")
            print("-" * 60)

            while True:
                snapshot = monitor.get_snapshot()
                snapshots.append(snapshot)

                if output_format == "json":
                    print(json.dumps(snapshot.to_dict(), indent=2))
                else:
                    _display_snapshot_compact(snapshot, show_processes, top_n_processes)

                # Check duration limit
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    break

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\nStopping power monitor...")

        finally:
            monitor.cleanup()

            if output_file and snapshots:
                _write_json_output_list(snapshots, output_file)
                print(f"\nResults written to: {output_file}")


def _display_snapshot(
    snapshot: PowerSnapshot,
    show_processes: bool,
    top_n: int,
    output_format: str,
) -> None:
    """Display a power snapshot in detail."""
    if output_format == "json":
        print(json.dumps(snapshot.to_dict(), indent=2))
        return

    timestamp = datetime.fromtimestamp(snapshot.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Power Snapshot - {timestamp}")
    print("=" * 60)

    # Total power
    if snapshot.total_power_watts is not None:
        print(f"\nTotal System Power: {snapshot.total_power_watts:.2f} W")

    # Domain breakdown
    if snapshot.domains:
        print("\nPower by Domain:")
        print("-" * 40)
        for domain in snapshot.domains:
            src = f" ({domain.source.value})" if domain.source else ""
            print(f"  {domain.domain.value:12}: {domain.power_watts:8.2f} W{src}")

    # GPU breakdown
    if snapshot.gpus:
        print("\nGPU Power:")
        print("-" * 40)
        for gpu in snapshot.gpus:
            print(f"  GPU {gpu.index} ({gpu.name}):")
            print(f"    Power:       {gpu.power_watts:8.2f} W")
            if gpu.power_limit_watts:
                pct = (gpu.power_watts / gpu.power_limit_watts) * 100
                print(f"    Limit:       {gpu.power_limit_watts:8.2f} W ({pct:.0f}%)")
            print(f"    Utilization: {gpu.utilization_percent:8.1f}%")
            if gpu.temperature_celsius:
                print(f"    Temperature: {gpu.temperature_celsius:8.0f}°C")

    # Process breakdown
    if show_processes and snapshot.processes:
        print(f"\nTop {top_n} Processes by Power:")
        print("-" * 60)
        print(f"{'PID':<8} {'Name':<25} {'CPU W':>8} {'GPU W':>8} {'Total W':>8}")
        print("-" * 60)

        for proc in snapshot.processes[:top_n]:
            name = proc.name[:24] if len(proc.name) > 24 else proc.name
            print(
                f"{proc.pid:<8} {name:<25} {proc.cpu_power_watts:>8.2f} "
                f"{proc.gpu_power_watts:>8.2f} {proc.total_power_watts:>8.2f}"
            )


def _display_snapshot_compact(
    snapshot: PowerSnapshot,
    show_processes: bool,
    _top_n: int,
) -> None:
    """Display a compact single-line snapshot for continuous mode."""
    timestamp = datetime.fromtimestamp(snapshot.timestamp).strftime("%H:%M:%S")

    # Build component strings
    parts = [timestamp]

    if snapshot.total_power_watts is not None:
        parts.append(f"Total: {snapshot.total_power_watts:.1f}W")

    cpu_power = snapshot.get_cpu_power()
    if cpu_power is not None:
        parts.append(f"CPU: {cpu_power:.1f}W")

    gpu_power = snapshot.get_gpu_power()
    if gpu_power > 0:
        parts.append(f"GPU: {gpu_power:.1f}W")

    # Top process
    if show_processes and snapshot.processes:
        top = snapshot.processes[0]
        parts.append(f"Top: {top.name[:15]}({top.total_power_watts:.1f}W)")

    print(" | ".join(parts))


def _write_json_output(snapshot: PowerSnapshot, filename: str) -> None:
    """Write a single snapshot to JSON file."""
    with open(filename, "w") as f:
        json.dump(snapshot.to_dict(), f, indent=2)


def _write_json_output_list(snapshots: list[PowerSnapshot], filename: str) -> None:
    """Write multiple snapshots to JSON file."""
    data = {
        "start_time": datetime.fromtimestamp(snapshots[0].timestamp).isoformat(),
        "end_time": datetime.fromtimestamp(snapshots[-1].timestamp).isoformat(),
        "sample_count": len(snapshots),
        "snapshots": [s.to_dict() for s in snapshots],
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def show_power_sources() -> None:
    """Display available power sources on this system."""
    monitor = PowerMonitor(include_process_attribution=False)
    monitor.initialize()

    sources = monitor.get_available_sources()

    print("Power Monitoring Capabilities")
    print("=" * 50)
    print()

    if not sources:
        print("No power sources available.")
        print()
        print("Troubleshooting:")
        print("  Linux:")
        print("    - Check /sys/class/powercap/intel-rapl/ exists")
        print("    - May need: sudo chmod -R a+r /sys/class/powercap/")
        print("    - Or add user to powercap group")
        print()
        print("  macOS:")
        print("    - Requires passwordless sudo for powermetrics")
        print("    - Add to /etc/sudoers:")
        print("      username ALL=(ALL) NOPASSWD: /usr/bin/powermetrics")
        print()
        print("  NVIDIA GPUs:")
        print("    - Requires nvidia-ml-py package")
        print("    - pip install nvidia-ml-py")
    else:
        print(f"Available sources: {[s.value for s in sources]}")
        print()

        # Get a sample reading
        time.sleep(0.5)
        snapshot = monitor.get_snapshot()

        if snapshot.domains:
            print("Supported power domains:")
            for domain in snapshot.domains:
                print(f"  - {domain.domain.value}: {domain.power_watts:.2f} W")

        if snapshot.gpus:
            print(f"\nDetected GPUs: {len(snapshot.gpus)}")
            for gpu in snapshot.gpus:
                print(f"  - {gpu.name}: {gpu.power_watts:.2f} W")

    monitor.cleanup()
