"""Linux CPU monitor using /proc/cpuinfo and thermal_zone interfaces."""

import re
from pathlib import Path

from warpt.stress.monitoring.base import CPUMonitor


class LinuxCPUMonitor(CPUMonitor):
    """CPU monitoring for Linux systems.

    Uses:
    - /proc/cpuinfo: CPU frequency and core count
    - /sys/devices/virtual/thermal/thermal_zone*: Temperature
    - /sys/module/cpufreq_powersave/parameters: Throttling detection
    """

    def __init__(self) -> None:
        """Initialize Linux CPU monitor."""
        self._core_count = self._get_core_count()

    def get_temperature(self) -> float | None:
        """Get CPU temperature from thermal zones.

        Searches /sys/devices/virtual/thermal/thermal_zone* directories
        and returns the highest temperature in Celsius.

        Returns
        -------
            Maximum CPU temperature in Celsius, or None if unavailable
        """
        max_temp = None
        thermal_zones = Path("/sys/devices/virtual/thermal").glob("thermal_zone*")
        temps: list[float] = []

        for zone in thermal_zones:
            try:
                temp_file = zone / "temp"
                if temp_file.exists():
                    # Temperature is in millidegrees Celsius
                    temp_mc = int(temp_file.read_text().strip())
                    temps.append(temp_mc / 1000.0)
            except (OSError, ValueError):
                continue

        if temps:
            max_temp = max(temps)

        return max_temp

    def get_frequency(self) -> float | None:
        """Get current CPU frequency in MHz.

        Reads from /proc/cpuinfo. Returns the frequency of the first CPU
        (assumes all CPUs have similar frequency).

        Returns
        -------
            Current frequency in MHz, or None if unavailable
        """
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
            # Look for "cpu MHz" line
            match = re.search(r"cpu MHz\s*:\s*([\d.]+)", cpuinfo)
            if match:
                return float(match.group(1))
        except (OSError, ValueError):
            pass
        return None

    def get_utilization(self) -> float | None:
        """Get CPU utilization percentage.

        Calculates from /proc/stat by comparing user+system+nice+iowait
        against total ticks across all CPUs.

        Returns
        -------
            Average CPU utilization percentage (0-100), or None if unavailable
        """
        try:
            # Read /proc/stat twice with small delay to calculate delta
            import time

            stat_before = self._read_proc_stat()
            time.sleep(0.1)
            stat_after = self._read_proc_stat()

            if stat_before is None or stat_after is None:
                return None

            # Calculate deltas
            work_before = (
                stat_before["user"] + stat_before["nice"] + stat_before["system"]
            )
            work_after = stat_after["user"] + stat_after["nice"] + stat_after["system"]
            work_delta = work_after - work_before

            total_delta = sum(stat_after.values()) - sum(stat_before.values())

            if total_delta <= 0:
                return 0.0

            utilization = (work_delta / total_delta) * 100.0
            return max(0.0, min(100.0, utilization))
        except Exception:
            return None

    def is_throttling(self) -> bool:
        """Check if CPU is throttling.

        Checks for P-state or cpufreq throttling by comparing current
        frequency to max frequency.

        Returns
        -------
            True if throttling detected, False otherwise
        """
        try:
            current_freq = self.get_frequency()
            max_freq = self.get_base_frequency()

            if current_freq is None or max_freq is None:
                return False

            # Consider throttling if current < 95% of base
            return current_freq < (max_freq * 0.95)
        except Exception:
            return False

    def get_power(self) -> float | None:
        """Get CPU power consumption in Watts.

        Uses RAPL counters if available (Intel/AMD).

        Returns
        -------
            Power consumption in Watts, or None if unavailable
        """
        try:
            from warpt.backends.power.sources import RAPLPowerSource

            rapl = RAPLPowerSource()
            return rapl.get_power_w()
        except Exception:
            return None

    def get_throttle_reason(self) -> str | None:
        """Get reason for CPU throttling.

        Returns
        -------
            Throttle reason string ('thermal', 'power', 'frequency') or None
        """
        if not self.is_throttling():
            return None

        temp = self.get_temperature()
        if temp is not None and temp > 80:
            return "thermal"

        # Check for power limit throttling
        try:
            pl1_file = Path(
                "/sys/devices/virtual/powercap/intel-rapl/constraint_0_power_limit_uw"
            )
            if pl1_file.exists():
                return "power"
        except Exception:
            pass

        return "frequency"

    def get_core_count(self) -> int:
        """Get number of active CPU cores.

        Returns
        -------
            Number of active cores
        """
        return self._core_count

    def get_base_frequency(self) -> float | None:
        """Get CPU base frequency in MHz.

        Reads from /proc/cpuinfo or cpufreq scaling_min_freq.

        Returns
        -------
            Base frequency in MHz, or None if unavailable
        """
        try:
            # Try cpufreq interface first
            cpufreq_file = Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq")
            if cpufreq_file.exists():
                # Frequency is in kHz
                freq_khz = int(cpufreq_file.read_text().strip())
                return freq_khz / 1000.0

            # Fallback: use current frequency
            return self.get_frequency()
        except (OSError, ValueError):
            return None

    def get_boost_frequency(self) -> float | None:
        """Get CPU boost (turbo) frequency in MHz.

        Reads from cpufreq scaling_max_freq.

        Returns
        -------
            Boost frequency in MHz, or None if unavailable
        """
        try:
            cpufreq_file = Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
            if cpufreq_file.exists():
                # Frequency is in kHz
                freq_khz = int(cpufreq_file.read_text().strip())
                return freq_khz / 1000.0
        except (OSError, ValueError):
            pass
        return None

    @staticmethod
    def _read_proc_stat() -> dict[str, int] | None:
        """Read /proc/stat and extract CPU times."""
        try:
            stat_line = Path("/proc/stat").read_text().split("\n")[0]
            # Format: cpu  user nice system idle iowait irq softirq ...
            parts = stat_line.split()
            return {
                "user": int(parts[1]),
                "nice": int(parts[2]),
                "system": int(parts[3]),
                "idle": int(parts[4]),
                "iowait": int(parts[5]),
            }
        except (OSError, ValueError, IndexError):
            return None

    @staticmethod
    def _get_core_count() -> int:
        """Get number of CPU cores from /proc/cpuinfo."""
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
            # Count processor entries (0-indexed, so add 1)
            cores = len(re.findall(r"processor\s*:", cpuinfo))
            return max(1, cores)
        except (OSError, ValueError):
            import os

            return max(1, os.cpu_count() or 1)
