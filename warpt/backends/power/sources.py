"""Power monitoring sources for different platforms."""

import platform
import re
import subprocess
import time
from pathlib import Path

from warpt.backends.power.base import PowerSource


class AppleSiliconPowerSource(PowerSource):
    """Power source for Apple Silicon using powermetrics."""

    def __init__(self) -> None:
        """Initialize Apple Silicon power source."""
        self._name = "apple_silicon"
        self._pattern = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\): (\d+) mW")

    @property
    def name(self) -> str:
        """Return the name of the power source."""
        return self._name

    def get_power_w(self) -> float | None:
        """Get current power in Watts using powermetrics.

        Note: Requires sudo or appropriate permissions.
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

            # Look for "Combined Power (CPU + GPU + ANE): XXX mW"
            match = self._pattern.search(result.stdout)
            if match:
                mw = float(match.group(1))
                return mw / 1000.0

            # Fallback for older macOS/non-M-series or different powermetrics versions
            # Look for "CPU Power: XXX mW"
            cpu_match = re.search(r"CPU Power: (\d+) mW", result.stdout)
            if cpu_match:
                return float(cpu_match.group(1)) / 1000.0

        except (subprocess.SubprocessError, FileNotFoundError, PermissionError):
            pass

        return None

    def check_permissions(self) -> bool:
        """Check if we can run powermetrics with sudo."""
        if platform.system() != "Darwin":
            return False

        try:
            # Try a simple sudo command that doesn't do much but verify access
            # We use -v to validate credentials, but it might prompt for password
            # if they aren't cached.
            # Using 'sudo -n' (non-interactive) to check if we HAVE permission
            # without prompting.
            result = subprocess.run(
                ["sudo", "-n", "true"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return True

            # If 'sudo -n' failed, it means we might need a password.
            # We should probably let the user know and let them try a normal sudo.
            # For the early check, we'll try a very fast powermetrics run
            # which will prompt the user for a password if needed.
            print(
                "\nPower monitoring on macOS requires sudo permissions "
                "for 'powermetrics'."
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
    """Power source for Intel/AMD CPUs using RAPL (Linux only)."""

    def __init__(self) -> None:
        """Initialize RAPL power source."""
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
        """Calculate power in Watts from RAPL energy counters."""
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
                if delta_e < 0:
                    delta_e = 0

                if delta_t > 0:
                    power_w = (delta_e / 1_000_000.0) / delta_t
                    self._last_energy = total_uj
                    self._last_time = current_time
                    return power_w

            self._last_energy = total_uj
            self._last_time = current_time
        except (OSError, ValueError):
            pass

        return None

    def check_permissions(self) -> bool:
        """Check if RAPL energy files are readable."""
        if not self._energy_files:
            return False

        try:
            # Try reading the first file
            self._energy_files[0].read_text()
            return True
        except (OSError, PermissionError):
            return False
