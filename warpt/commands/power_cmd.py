"""Power monitoring CLI command implementation.

Reads power exclusively from the warpt power-daemon. If the daemon is not
running, the command reports that and exits — there is no native fallback and no
per-process attribution (the daemon reports per-component, not per-process).
"""

from __future__ import annotations

import json
import time
from datetime import datetime

from warpt.backends.power.factory import PowerMonitor
from warpt.models.power_models import PowerSnapshot


def _print_daemon_unavailable(monitor: PowerMonitor) -> None:
    """Print the daemon-unavailable message and troubleshooting hints."""
    for reason in monitor.get_unavailable_reasons():
        print(reason)
    print("\nTroubleshooting:")
    print("  - Start the power-daemon, then re-run.")
    print("  - Or set POWER_DAEMON_URL to point at a running daemon.")


def run_power(
    interval_seconds: float = 1.0,
    duration_seconds: float | None = None,
    output_format: str = "text",
    output_file: str | None = None,
    continuous: bool = False,
) -> None:
    """Run power monitoring and display results.

    Args:
        interval_seconds: Sampling interval in seconds.
        duration_seconds: Stop after this many seconds (None = until interrupted).
        output_format: Output format ("text", "json").
        output_file: Optional file to write JSON output.
        continuous: Whether to run continuously (vs single snapshot).
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero")

    monitor = PowerMonitor()

    if not monitor.initialize():
        _print_daemon_unavailable(monitor)
        return

    print(f"Power source: {[s.value for s in monitor.get_available_sources()]}")
    print()

    if not continuous:
        snapshot = monitor.get_snapshot()
        _display_snapshot(snapshot, output_format)

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
                    _display_snapshot_compact(snapshot)

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


def _display_snapshot(snapshot: PowerSnapshot, output_format: str) -> None:
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


def _display_snapshot_compact(snapshot: PowerSnapshot) -> None:
    """Display a compact single-line snapshot for continuous mode."""
    timestamp = datetime.fromtimestamp(snapshot.timestamp).strftime("%H:%M:%S")

    parts = [timestamp]

    if snapshot.total_power_watts is not None:
        parts.append(f"Total: {snapshot.total_power_watts:.1f}W")

    cpu_power = snapshot.get_cpu_power()
    if cpu_power is not None:
        parts.append(f"CPU: {cpu_power:.1f}W")

    gpu_power = snapshot.get_gpu_power()
    if gpu_power > 0:
        parts.append(f"GPU: {gpu_power:.1f}W")

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
    """Display power source availability (the daemon) on this system."""
    monitor = PowerMonitor()
    available = monitor.initialize()

    print("Power Monitoring Capabilities")
    print("=" * 50)
    print()

    if not available:
        _print_daemon_unavailable(monitor)
        return

    print(f"Available source: {[s.value for s in monitor.get_available_sources()]}")
    print()

    snapshot = monitor.get_snapshot()
    if snapshot.domains:
        print("Reported power domains:")
        for domain in snapshot.domains:
            print(f"  - {domain.domain.value}: {domain.power_watts:.2f} W")

    if snapshot.gpus:
        print(f"\nDetected GPUs: {len(snapshot.gpus)}")
        for gpu in snapshot.gpus:
            print(f"  - {gpu.name}: {gpu.power_watts:.2f} W")

    monitor.cleanup()
