"""macOS power monitoring backend using powermetrics.

On macOS, power information comes from several sources:
1. powermetrics - Detailed CPU/GPU/ANE power (requires sudo)
2. IOKit - Battery information and system power
3. ioreg - Hardware power management info

The powermetrics tool provides the most accurate power measurements
but requires either:
- Running as root
- Passwordless sudo configured for powermetrics

Power domains available on Apple Silicon:
- CPU (efficiency + performance cores)
- GPU (integrated)
- ANE (Apple Neural Engine)
- Combined (total SoC power)

On Intel Macs:
- CPU Package power
- DRAM power (sometimes)
- iGPU power (if present)
"""

from __future__ import annotations

import plistlib
import re
import subprocess
import threading
import time
from typing import Any

from warpt.backends.power.base import PowerBackend
from warpt.models.power_models import DomainPower, PowerDomain, PowerSource


class MacOSPowerBackend(PowerBackend):
    """Backend for reading power on macOS via powermetrics.

    This backend runs powermetrics in the background and parses
    its output to extract power measurements.
    """

    def __init__(self, use_plist: bool = True, sample_interval_ms: int = 1000) -> None:
        """Initialize the macOS power backend.

        Args:
            use_plist: Use plist output format (more reliable parsing).
            sample_interval_ms: Sampling interval in milliseconds.
        """
        self._use_plist = use_plist
        self._sample_interval_ms = sample_interval_ms
        self._lock = threading.Lock()
        self._latest_readings: list[DomainPower] = []
        self._error: str | None = None
        self._sudo_available: bool | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._process: subprocess.Popen | None = None

    def is_available(self) -> bool:
        """Check if powermetrics is available.

        Returns:
            True if powermetrics exists and sudo is available.
        """
        # Check if powermetrics exists
        try:
            result = subprocess.run(
                ["which", "powermetrics"],
                capture_output=True,
                timeout=2.0,
            )
            if result.returncode != 0:
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

        # Check if we can run with sudo -n (non-interactive)
        return self._check_sudo()

    def _check_sudo(self) -> bool:
        """Check if passwordless sudo is available for powermetrics.

        Returns:
            True if sudo -n works for powermetrics.
        """
        if self._sudo_available is not None:
            return self._sudo_available

        try:
            # Try a quick sample to see if sudo works
            result = subprocess.run(
                [
                    "sudo",
                    "-n",
                    "powermetrics",
                    "--samplers",
                    "cpu_power",
                    "-n",
                    "1",
                    "-i",
                    "100",
                ],
                capture_output=True,
                timeout=3.0,
            )
            self._sudo_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._sudo_available = False

        return self._sudo_available

    def get_source(self) -> PowerSource:
        """Get the power source type.

        Returns:
            PowerSource.POWERMETRICS
        """
        return PowerSource.POWERMETRICS

    def initialize(self) -> bool:
        """Start background powermetrics collection.

        Launches a single persistent powermetrics process that streams
        plist samples. This avoids repeated fork() calls which deadlock
        on macOS when Accelerate/GCD BLAS threads are active.

        Returns:
            True if collection started successfully.
        """
        if self._running:
            return True

        if not self.is_available():
            self._error = "powermetrics not available (requires sudo)"
            return False

        self._running = True
        self._stop_event.clear()

        # Start persistent powermetrics process (single fork, before BLAS)
        self._process = subprocess.Popen(
            [
                "sudo",
                "-n",
                "powermetrics",
                "--samplers",
                "cpu_power",
                "-f",
                "plist",
                "-i",
                str(self._sample_interval_ms),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()

        # Wait briefly for first reading
        time.sleep(0.1)
        return True

    def _collection_loop(self) -> None:
        """Read streaming plist samples from the persistent powermetrics process."""
        buf = b""
        while not self._stop_event.is_set():
            if self._process is None or self._process.stdout is None:
                break
            if self._process.poll() is not None:
                break
            chunk = self._process.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            # powermetrics separates plist samples with \x00
            while b"\x00" in buf:
                raw, buf = buf.split(b"\x00", 1)
                if raw.strip():
                    try:
                        data = plistlib.loads(raw)
                        readings = self._parse_plist_data(data)
                        with self._lock:
                            self._latest_readings = readings
                    except Exception:
                        pass

    def _collect_single_sample(self) -> list[DomainPower]:
        """Collect a single powermetrics sample.

        Returns:
            List of DomainPower readings.
        """
        try:
            if self._use_plist:
                return self._collect_plist_sample()
            else:
                return self._collect_text_sample()
        except Exception as e:
            with self._lock:
                self._error = str(e)
            return []

    def _collect_plist_sample(self) -> list[DomainPower]:
        """Collect sample using plist output format.

        Returns:
            List of DomainPower readings.
        """
        try:
            result = subprocess.run(
                [
                    "sudo",
                    "-n",
                    "powermetrics",
                    "--samplers",
                    "cpu_power",
                    "-f",
                    "plist",
                    "-n",
                    "1",
                    "-i",
                    str(self._sample_interval_ms),
                ],
                capture_output=True,
                timeout=self._sample_interval_ms / 1000.0 + 3.0,
            )

            if result.returncode != 0:
                self._error = "powermetrics failed"
                return []

            # Parse plist output
            try:
                data = plistlib.loads(result.stdout)
                return self._parse_plist_data(data)
            except plistlib.InvalidFileException:
                # Fall back to text parsing
                return self._parse_text_output(result.stdout.decode(errors="ignore"))

        except subprocess.TimeoutExpired:
            self._error = "powermetrics timeout"
            return []
        except FileNotFoundError:
            self._error = "powermetrics not found"
            return []

    def _collect_text_sample(self) -> list[DomainPower]:
        """Collect sample using text output format.

        Returns:
            List of DomainPower readings.
        """
        try:
            result = subprocess.run(
                [
                    "sudo",
                    "-n",
                    "powermetrics",
                    "--samplers",
                    "cpu_power",
                    "-n",
                    "1",
                    "-i",
                    str(self._sample_interval_ms),
                ],
                capture_output=True,
                text=True,
                timeout=self._sample_interval_ms / 1000.0 + 3.0,
            )

            if result.returncode != 0:
                self._error = "powermetrics failed"
                return []

            return self._parse_text_output(result.stdout)

        except subprocess.TimeoutExpired:
            self._error = "powermetrics timeout"
            return []

    def _parse_plist_data(self, data: dict[str, Any]) -> list[DomainPower]:
        """Parse plist-format powermetrics output.

        Args:
            data: Parsed plist dictionary.

        Returns:
            List of DomainPower readings.
        """
        readings: list[DomainPower] = []
        processor = data.get("processor", {})

        # CPU Power
        cpu_power_mw = processor.get("cpu_power")
        if cpu_power_mw is not None:
            readings.append(
                DomainPower(
                    domain=PowerDomain.CORE,
                    power_watts=cpu_power_mw / 1000.0,
                    source=PowerSource.POWERMETRICS,
                    metadata={"raw_mw": cpu_power_mw},
                )
            )

        # GPU Power
        gpu_power_mw = processor.get("gpu_power")
        if gpu_power_mw is not None:
            readings.append(
                DomainPower(
                    domain=PowerDomain.GPU,
                    power_watts=gpu_power_mw / 1000.0,
                    source=PowerSource.POWERMETRICS,
                    metadata={"raw_mw": gpu_power_mw, "type": "integrated"},
                )
            )

        # ANE Power (Apple Neural Engine)
        ane_power_mw = processor.get("ane_power")
        if ane_power_mw is not None:
            readings.append(
                DomainPower(
                    domain=PowerDomain.ANE,
                    power_watts=ane_power_mw / 1000.0,
                    source=PowerSource.POWERMETRICS,
                    metadata={"raw_mw": ane_power_mw},
                )
            )

        # Combined/Package Power
        combined_power_mw = processor.get("combined_power")
        if combined_power_mw is not None:
            readings.append(
                DomainPower(
                    domain=PowerDomain.PACKAGE,
                    power_watts=combined_power_mw / 1000.0,
                    source=PowerSource.POWERMETRICS,
                    metadata={"raw_mw": combined_power_mw},
                )
            )

        # DRAM power (if available)
        dram_power_mw = processor.get("dram_power")
        if dram_power_mw is not None:
            readings.append(
                DomainPower(
                    domain=PowerDomain.DRAM,
                    power_watts=dram_power_mw / 1000.0,
                    source=PowerSource.POWERMETRICS,
                    metadata={"raw_mw": dram_power_mw},
                )
            )

        return readings

    def _parse_text_output(self, output: str) -> list[DomainPower]:
        """Parse text-format powermetrics output.

        Args:
            output: Raw text output from powermetrics.

        Returns:
            List of DomainPower readings.
        """
        readings: list[DomainPower] = []

        # Regex patterns for different power lines
        combined_re = r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW"
        patterns = [
            (r"CPU Power:\s*([\d.]+)\s*mW", PowerDomain.CORE),
            (r"GPU Power:\s*([\d.]+)\s*mW", PowerDomain.GPU),
            (r"ANE Power:\s*([\d.]+)\s*mW", PowerDomain.ANE),
            (combined_re, PowerDomain.PACKAGE),
            (r"Package Power:\s*([\d.]+)\s*mW", PowerDomain.PACKAGE),
            (r"DRAM Power:\s*([\d.]+)\s*mW", PowerDomain.DRAM),
        ]

        for pattern, domain in patterns:
            match = re.search(pattern, output)
            if match:
                power_mw = float(match.group(1))
                readings.append(
                    DomainPower(
                        domain=domain,
                        power_watts=power_mw / 1000.0,
                        source=PowerSource.POWERMETRICS,
                        metadata={"raw_mw": power_mw},
                    )
                )

        return readings

    def get_power_readings(self) -> list[DomainPower]:
        """Get the latest power readings.

        Returns:
            List of DomainPower objects with current measurements.
        """
        if not self._running:
            # Try to get a single sample synchronously
            return self._collect_single_sample()

        with self._lock:
            readings = self._latest_readings.copy()

        # If background thread hasn't collected yet, do synchronous sample
        if not readings:
            return self._collect_single_sample()

        return readings

    def get_error(self) -> str | None:
        """Get any error message from the last collection attempt.

        Returns:
            Error message or None if no error.
        """
        with self._lock:
            return self._error

    def cleanup(self) -> None:
        """Stop background collection and clean up."""
        self._running = False
        self._stop_event.set()
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._latest_readings = []


class BatteryInfo:
    """Utility class to read battery information on macOS."""

    @staticmethod
    def get_battery_info() -> dict[str, Any] | None:
        """Get battery information via ioreg.

        Returns:
            Dictionary with battery info or None if not available.
        """
        try:
            result = subprocess.run(
                ["ioreg", "-r", "-c", "AppleSmartBattery", "-a"],
                capture_output=True,
                timeout=2.0,
            )
            if result.returncode != 0:
                return None

            data = plistlib.loads(result.stdout)
            if not data:
                return None

            # Navigate to battery data
            battery = data[0] if isinstance(data, list) else data

            return {
                "is_charging": battery.get("IsCharging", False),
                "current_capacity": battery.get("CurrentCapacity"),
                "max_capacity": battery.get("MaxCapacity"),
                "design_capacity": battery.get("DesignCapacity"),
                "cycle_count": battery.get("CycleCount"),
                "voltage_mv": battery.get("Voltage"),
                "amperage_ma": battery.get("Amperage"),
                "temperature_c": battery.get("Temperature", 0) / 100.0,
                "time_remaining_min": battery.get("TimeRemaining"),
                "fully_charged": battery.get("FullyCharged", False),
                "external_connected": battery.get("ExternalConnected", False),
            }

        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            plistlib.InvalidFileException,
        ):
            return None

    @staticmethod
    def get_power_draw_watts() -> float | None:
        """Calculate current power draw from battery.

        Returns:
            Power draw in watts (positive = discharging, negative = charging).
        """
        info = BatteryInfo.get_battery_info()
        if not info:
            return None

        voltage = info.get("voltage_mv")
        amperage = info.get("amperage_ma")

        if voltage is None or amperage is None:
            return None

        # P = V * I (convert mV * mA to W)
        power_watts = (float(voltage) / 1000.0) * (float(amperage) / 1000.0)
        return float(power_watts)
