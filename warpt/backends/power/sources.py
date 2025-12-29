"""Power monitoring sources for different platforms."""

import platform
import re
import subprocess
import time
from pathlib import Path

from warpt.backends.power.base import PowerSource


class AppleSiliconPowerSource(PowerSource):
    """Power source for Apple Silicon using powermetrics.

    Uses the macOS 'powermetrics' tool to measure combined CPU+GPU+ANE power.
    Requires sudo access. Works on Apple Silicon (M1/M2/M3) and Intel Macs.

    Patterns matched (in priority order):
    1. "Combined Power (CPU + GPU + ANE): XXX mW" (M-series Macs)
    2. "CPU Power: XXX mW" (fallback for older/Intel Macs)
    """

    def __init__(self) -> None:
        """Initialize Apple Silicon power source."""
        self._name = "apple_silicon"
        # Pattern for M-series Macs with separate GPU/ANE
        self._pattern = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\): (\d+) mW")
        # Fallback pattern for older/Intel Macs
        self._cpu_pattern = re.compile(r"CPU Power: (\d+) mW")

    @property
    def name(self) -> str:
        """Return the name of the power source."""
        return self._name

    def get_power_w(self) -> float | None:
        """Get current power in Watts using powermetrics.

        Note:
            Requires sudo access. Prompts for password if not cached in sudo.
            Uses a 100ms sample window for consistent measurement.

        Returns
        -------
            Power in Watts, or None if unavailable/failed
        """
        if platform.system() != "Darwin":
            return None

        try:
            # We use a very short sample interval to get a snapshot
            # -n 1: one sample
            # -i 100: 100ms interval
            # --samplers cpu_power: only CPU power (includes GPU/ANE on M-series)
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
            if result.returncode != 0:
                return None

            # Try primary pattern first (M-series Macs)
            match = self._pattern.search(result.stdout)
            if match:
                mw = float(match.group(1))
                return mw / 1000.0

            # Fallback for older macOS versions or Intel Macs
            cpu_match = self._cpu_pattern.search(result.stdout)
            if cpu_match:
                return float(cpu_match.group(1)) / 1000.0

        except (subprocess.SubprocessError, FileNotFoundError, PermissionError):
            pass

        return None

    def check_permissions(self) -> bool:
        """Check if we can run powermetrics with sudo.

        First tries non-interactive check with `sudo -n`. If that fails,
        prompts user for sudo password (cached for future use).

        Returns
        -------
            True if powermetrics can be executed, False otherwise
        """
        if platform.system() != "Darwin":
            return False

        try:
            # Try non-interactive check first
            # Uses 'sudo -n' (non-interactive) to check if we HAVE permission
            # without prompting
            result = subprocess.run(
                ["sudo", "-n", "true"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return True

            # If 'sudo -n' failed, try interactive sudo
            # This will prompt for password if not cached
            print(
                "\nPower monitoring on macOS requires sudo access "
                "for the 'powermetrics' tool.\n"
            )
            result = subprocess.run(
                ["sudo", "true"],
                check=False,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False


class NvidiaPowerSource(PowerSource):
    """Power source for NVIDIA GPUs using NVML."""

    def __init__(self, index: int = 0) -> None:
        """Initialize NVIDIA power source for a specific GPU index."""
        from warpt.backends.nvidia import NvidiaBackend

        self._backend = NvidiaBackend()
        self._index = index
        self._name = f"nvidia_gpu_{index}"

    @property
    def name(self) -> str:
        """Return the name of the power source."""
        return self._name

    def get_power_w(self) -> float | None:
        """Get current GPU power in Watts."""
        try:
            return self._backend.get_power_usage(self._index)
        except Exception:
            return None

    def check_permissions(self) -> bool:
        """Check if NVML is available and working."""
        try:
            return self._backend.is_available()
        except Exception:
            return False


class RAPLPowerSource(PowerSource):
    """Power source for Intel/AMD CPUs using RAPL (Linux only).

    RAPL (Running Average Power Limit) provides hardware-based power measurement
    via the Linux kernel's powercap interface. Works on:
    - Intel: Sandy Bridge and later
    - AMD: Ryzen and EPYC with support

    Reads energy counters from /sys/class/powercap/intel-rapl:*/energy_uj
    and calculates instantaneous power by measuring counter deltas.

    Note on counter behavior:
    - Counter is 64-bit and wraps (~500 years of continuous operation)
    - Reads are non-blocking and take microseconds
    - Requires no special privileges if readable (often via user groups)
    - Energy values are cumulative since system boot or counter reset
    """

    def __init__(self) -> None:
        """Initialize RAPL power source.

        Discovers all RAPL energy files available on the system.
        """
        self._name = "rapl"
        self._last_energy: int | None = None
        self._last_time: float | None = None
        self._energy_files = list(
            Path("/sys/class/powercap").glob("intel-rapl:*/energy_uj")
        )

    @property
    def name(self) -> str:
        """Return the name of the power source."""
        return self._name

    def get_power_w(self) -> float | None:
        """Calculate power in Watts from RAPL energy counters.

        Uses delta-encoding: power = (energy_now - energy_last) / time_elapsed

        Returns
        -------
            Instantaneous power in Watts, or None if:
            - No RAPL files available
            - First sample (need baseline)
            - Read error occurs

        Note:
            Counter wrapping is handled by checking for large negative deltas
            and treating them as 0 (wrap is extremely rare in practice).
        """
        if not self._energy_files:
            return None

        current_time = time.time()
        try:
            total_uj = 0
            for f in self._energy_files:
                total_uj += int(f.read_text().strip())

            if self._last_energy is not None and self._last_time is not None:
                delta_e = total_uj - self._last_energy
                delta_t = current_time - self._last_time

                # Handle counter rollover (energy_uj is typically 64-bit)
                # Negative delta indicates wrap occurred; treat as no power
                if delta_e < 0:
                    delta_e = 0

                if delta_t > 0:
                    power_w = (delta_e / 1_000_000.0) / delta_t
                    self._last_energy = total_uj
                    self._last_time = current_time
                    return power_w

            # First sample: initialize baseline
            self._last_energy = total_uj
            self._last_time = current_time
        except (OSError, ValueError):
            pass

        return None

    def check_permissions(self) -> bool:
        """Check if RAPL energy files are readable.

        Returns
        -------
            True if RAPL files exist and are readable, False otherwise

        Note:
            On most systems, RAPL files require membership in the 'root'
            or special group, or can be read as regular user on recent kernels.
        """
        if not self._energy_files:
            return False

        try:
            # Try reading the first file
            self._energy_files[0].read_text()
            return True
        except (OSError, PermissionError):
            return False
