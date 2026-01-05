#!/usr/bin/env python3
"""Demo script showing power monitoring capabilities.

This script demonstrates:
1. Creating a power monitor
2. Getting power readings
3. Per-process power attribution
4. Working with different power sources

Run with: python examples/power_monitoring_demo.py

Note: Full functionality requires:
- Linux: Read access to /sys/class/powercap/intel-rapl/
- macOS: Passwordless sudo for powermetrics
- NVIDIA GPU: nvidia-ml-py package installed
"""

from __future__ import annotations

import time


def main() -> None:
    """Run power monitoring demo."""
    from warpt.backends.power import create_power_monitor

    print("=" * 60)
    print("Power Monitoring Demo")
    print("=" * 60)
    print()

    # Create and initialize the power monitor
    monitor = create_power_monitor()

    # Check available sources
    sources = monitor.get_available_sources()
    print(f"Available power sources: {[s.value for s in sources]}")
    print()

    if not sources:
        print("No power sources available on this system.")
        print()
        print("To enable power monitoring:")
        print()
        print("Linux (Intel/AMD RAPL):")
        print("  # Grant read access to power cap interface")
        print("  sudo chmod -R a+r /sys/class/powercap/")
        print()
        print("macOS (powermetrics):")
        print("  # Add to /etc/sudoers (via 'sudo visudo'):")
        print("  # username ALL=(ALL) NOPASSWD: /usr/bin/powermetrics")
        print()
        print("NVIDIA GPUs:")
        print("  pip install nvidia-ml-py")
        print()
        monitor.cleanup()
        return

    # Wait a moment to collect a delta for power calculation
    print("Collecting power data...")
    time.sleep(1.5)

    # Get a power snapshot
    snapshot = monitor.get_snapshot()

    print()
    print("Power Snapshot")
    print("-" * 40)

    # Total power
    if snapshot.total_power_watts is not None:
        print(f"Total System Power: {snapshot.total_power_watts:.2f} W")

    # Per-domain breakdown
    if snapshot.domains:
        print("\nPower by Domain:")
        for domain in snapshot.domains:
            print(f"  {domain.domain.value:12}: {domain.power_watts:6.2f} W")

    # GPU power
    if snapshot.gpus:
        print("\nGPU Power:")
        for gpu in snapshot.gpus:
            print(f"  {gpu.name}: {gpu.power_watts:.2f} W")
            print(f"    Utilization: {gpu.utilization_percent:.1f}%")
            if gpu.temperature_celsius:
                print(f"    Temperature: {gpu.temperature_celsius:.0f}°C")

    # Per-process breakdown (top 5)
    if snapshot.processes:
        print("\nTop 5 Processes by Power:")
        print(f"{'PID':<8} {'Name':<20} {'CPU W':>8} {'GPU W':>8} {'Total':>8}")
        print("-" * 54)
        for proc in snapshot.processes[:5]:
            name = proc.name[:19] if len(proc.name) > 19 else proc.name
            print(
                f"{proc.pid:<8} {name:<20} "
                f"{proc.cpu_power_watts:>8.2f} "
                f"{proc.gpu_power_watts:>8.2f} "
                f"{proc.total_power_watts:>8.2f}"
            )

    # JSON export example
    print("\n" + "-" * 40)
    print("Snapshot can be exported to JSON:")
    print("  snapshot.to_dict()")

    # Clean up
    monitor.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    main()
