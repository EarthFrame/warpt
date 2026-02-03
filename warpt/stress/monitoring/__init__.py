"""Monitoring module for stress tests and benchmarks.

Provides platform-specific CPU monitoring via abstract base class
and factory pattern. Supports Linux and macOS.

Example:
    >>> from warpt.stress.monitoring import get_cpu_monitor
    >>> monitor = get_cpu_monitor()
    >>> if monitor:
    ...     temp = monitor.get_temperature()
    ...     freq = monitor.get_frequency()
    ...     power = monitor.get_power()
"""

from warpt.stress.monitoring.base import CPUMonitor
from warpt.stress.monitoring.factory import CPUMonitorFactory, get_cpu_monitor

__all__ = ["CPUMonitor", "CPUMonitorFactory", "get_cpu_monitor"]
