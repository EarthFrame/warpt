"""Factory for creating platform-specific CPU monitors."""

import platform

from warpt.stress.monitoring.base import CPUMonitor


class CPUMonitorFactory:
    """Factory for creating CPU monitors for the current platform.

    Automatically selects the appropriate platform implementation
    (Linux or Darwin/macOS) based on the running system.
    """

    @staticmethod
    def create() -> CPUMonitor | None:
        """Create a CPU monitor for the current platform.

        Returns
        -------
            CPUMonitor subclass instance, or None if platform unsupported

        Raises
        ------
            ImportError: If platform-specific module unavailable
        """
        system = platform.system()

        if system == "Linux":
            from warpt.stress.monitoring.linux import LinuxCPUMonitor

            return LinuxCPUMonitor()
        elif system == "Darwin":
            from warpt.stress.monitoring.darwin import DarwinCPUMonitor

            return DarwinCPUMonitor()
        else:
            # Unsupported platform
            return None


def get_cpu_monitor() -> CPUMonitor | None:
    """Get a CPU monitor for the current platform.

    Convenience function for easy access to platform-specific monitor.

    Returns
    -------
        CPUMonitor instance or None if unsupported
    """
    return CPUMonitorFactory.create()
