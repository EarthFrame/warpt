"""macOS CPU monitor using system_profiler and powermetrics."""

import re
import subprocess

from warpt.stress.monitoring.base import CPUMonitor


class DarwinCPUMonitor(CPUMonitor):
    """CPU monitoring for macOS systems.

    Uses:
    - sysctl: CPU frequency and core count
    - powermetrics: Power consumption and thermal info
    - pmset: Throttling state

    Note: Many operations require sudo access on macOS.
    """

    def __init__(self) -> None:
        """Initialize macOS CPU monitor."""
        self._core_count = self._get_core_count()
        self._base_freq = self._get_base_frequency_cached()

    def get_temperature(self) -> float | None:
        """Get CPU temperature using powermetrics.

        Requires sudo access. Runs a short powermetrics sample.

        Returns
        -------
            Temperature in Celsius, or None if unavailable/no permissions
        """
        try:
            result = subprocess.run(
                [
                    "sudo",
                    "powermetrics",
                    "-n",
                    "1",
                    "-i",
                    "100",
                    "--samplers",
                    "cpu_power",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Look for temperature patterns
                match = re.search(r"CPU Die Temp: (\d+(?:\.\d+)?)\s*C", result.stdout)
                if match:
                    return float(match.group(1))
        except Exception:
            pass
        return None

    def get_frequency(self) -> float | None:
        """Get current CPU frequency in MHz.

        Uses sysctl to read hw.cpufrequency (in Hz).

        Returns
        -------
            Current frequency in MHz, or None if unavailable
        """
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.cpufrequency"],
                capture_output=True,
                text=True,
                check=True,
            )
            freq_hz = int(result.stdout.strip())
            return freq_hz / 1_000_000.0
        except Exception:
            return None

    def get_utilization(self) -> float | None:
        """Get CPU utilization percentage.

        Uses top command to get average utilization.

        Returns
        -------
            Average CPU utilization percentage (0-100), or None if unavailable
        """
        try:
            result = subprocess.run(
                ["top", "-l", "1", "-n", "0"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Look for "CPU usage:" line
            match = re.search(
                r"CPU usage:\s*([\d.]+)%\s*user,\s*([\d.]+)%\s*sys,\s*([\d.]+)%\s*idle",
                result.stdout,
            )
            if match:
                user = float(match.group(1))
                sys = float(match.group(2))
                return user + sys
        except Exception:
            pass
        return None

    def is_throttling(self) -> bool:
        """Check if CPU is throttling on macOS.

        Checks pmset status or frequency limits.

        Returns
        -------
            True if throttling detected, False otherwise
        """
        try:
            current_freq = self.get_frequency()
            base_freq = self.get_base_frequency()

            if current_freq is None or base_freq is None:
                return False

            # Consider throttling if current < 95% of base
            return current_freq < (base_freq * 0.95)
        except Exception:
            return False

    def get_power(self) -> float | None:
        """Get CPU power consumption in Watts.

        Uses Apple Silicon power metrics if available.

        Returns
        -------
            Power consumption in Watts, or None if unavailable
        """
        try:
            from warpt.backends.power.sources import AppleSiliconPowerSource

            apple_pwr = AppleSiliconPowerSource()
            return apple_pwr.get_power_w()
        except Exception:
            return None

    def get_throttle_reason(self) -> str | None:
        """Get reason for CPU throttling on macOS.

        Returns
        -------
            Throttle reason string ('thermal', 'power') or None
        """
        if not self.is_throttling():
            return None

        # Try to detect thermal throttling
        temp = self.get_temperature()
        if temp is not None and temp > 85:
            return "thermal"

        # Otherwise assume power/frequency limit
        return "power"

    def get_core_count(self) -> int:
        """Get number of active CPU cores.

        Uses sysctl hw.ncpu or hw.physicalcpu.

        Returns
        -------
            Number of active cores
        """
        return self._core_count

    def get_base_frequency(self) -> float | None:
        """Get CPU base frequency in MHz.

        Uses sysctl hw.cpufrequency_max.

        Returns
        -------
            Base frequency in MHz, or None if unavailable
        """
        return self._base_freq

    def get_boost_frequency(self) -> float | None:
        """Get CPU boost frequency in MHz.

        On Apple Silicon, this is typically P-core max frequency.
        Uses sysctl hw.cpufrequency_max.

        Returns
        -------
            Boost frequency in MHz, or None if unavailable
        """
        # On macOS, boost frequency is similar to max frequency
        return self.get_base_frequency()

    def _get_core_count(self) -> int:
        """Get number of CPU cores using sysctl."""
        try:
            # Try physical core count first
            result = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                text=True,
                check=True,
            )
            return int(result.stdout.strip())
        except Exception:
            pass

        try:
            # Fallback to logical core count
            result = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"],
                capture_output=True,
                text=True,
                check=True,
            )
            return int(result.stdout.strip())
        except Exception:
            import os

            return max(1, os.cpu_count() or 1)

    def _get_base_frequency_cached(self) -> float | None:
        """Get base frequency once at initialization."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.cpufrequency_max"],
                capture_output=True,
                text=True,
                check=True,
            )
            freq_hz = int(result.stdout.strip())
            return freq_hz / 1_000_000.0
        except Exception:
            return None
