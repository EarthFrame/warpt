"""Tenstorrent accelerator power monitoring backend using Linux sysfs.

Provides per-device power readings, power limits, temperature, voltage,
and current for Tenstorrent Wormhole and Blackhole accelerators. All
data is read from the standard Linux hwmon interface exposed by the
Tenstorrent kernel-mode driver (tt-kmd).

Unit conversions:

- ``power1_input``  — microwatts  → Watts (÷ 1 000 000)
- ``power1_max``    — microwatts  → Watts (÷ 1 000 000)
- ``temp1_input``   — millidegrees Celsius → °C (÷ 1000)
- ``in0_input``     — millivolts  → Volts (÷ 1000)
- ``curr1_input``   — milliamps   → Amps  (÷ 1000)
"""

from __future__ import annotations

from typing import Any

from warpt.backends.power.base import PowerBackend
from warpt.models.power_models import (
    DomainPower,
    GPUPowerInfo,
    PowerDomain,
    PowerSource,
)

# Re-use sysfs helpers from the accelerator backend to avoid duplication.
try:
    from warpt.backends.tenstorrent import (
        _discover_devices,
        _read_sysfs,
        _read_sysfs_int,
    )

    _TT_SYSFS_AVAILABLE = True
except ImportError:
    _TT_SYSFS_AVAILABLE = False


class TenstorrentPowerBackend(PowerBackend):
    """Power monitoring backend for Tenstorrent accelerators.

    Reads power, temperature, voltage, and current from the Linux hwmon
    sysfs interface.
    """

    def __init__(self) -> None:
        """Initialize the Tenstorrent power backend."""
        self._initialized = False
        self._devices: list[dict] = []

    def is_available(self) -> bool:
        """Check if Tenstorrent accelerators are present.

        Returns
        -------
        bool
            ``True`` if the sysfs helpers are importable and at least one
            device is detected.
        """
        if not _TT_SYSFS_AVAILABLE:
            return False
        try:
            devices = _discover_devices()
            return len(devices) > 0
        except Exception:
            return False

    def get_source(self) -> PowerSource:
        """Return the power source identifier.

        Returns
        -------
        PowerSource
            ``PowerSource.ESTIMATED`` — readings come from on-board
            sensors exposed via sysfs, not a calibrated external meter.
            There is no dedicated ``PowerSource`` variant for Tenstorrent
            sysfs; ``ESTIMATED`` is the closest match.
        """
        return PowerSource.ESTIMATED

    def initialize(self) -> bool:
        """Initialize the backend by discovering devices.

        Returns
        -------
        bool
            ``True`` if at least one device was found.
        """
        if self._initialized:
            return bool(self._devices)

        if not _TT_SYSFS_AVAILABLE:
            return False

        try:
            self._devices = _discover_devices()
            self._initialized = True
            return bool(self._devices)
        except Exception:
            return False

    def get_power_readings(self) -> list[DomainPower]:
        """Get per-device power readings as ``DomainPower`` objects.

        Returns
        -------
        list[DomainPower]
            One entry per device, using ``PowerDomain.GPU``.
        """
        if not self._initialized:
            if not self.initialize():
                return []

        readings: list[DomainPower] = []

        for idx, dev in enumerate(self._devices):
            hwmon_dir = dev.get("hwmon_dir")
            if hwmon_dir is None:
                continue

            try:
                raw_power = _read_sysfs_int(hwmon_dir / "power1_input")
                if raw_power is None:
                    continue
                power_watts = raw_power / 1_000_000.0

                # Read chip name for metadata
                chip_name = _read_sysfs(hwmon_dir / "name") or "tenstorrent"
                card_type = _read_sysfs(dev["tt_dir"] / "tt_card_type")

                metadata: dict[str, Any] = {
                    "gpu_index": idx,
                    "gpu_name": f"Tenstorrent {card_type or 'accelerator'}",
                    "chip_name": chip_name,
                    "pci_bdf": dev["pci_bdf"],
                    "raw_uw": raw_power,
                }

                readings.append(
                    DomainPower(
                        domain=PowerDomain.GPU,
                        power_watts=power_watts,
                        energy_joules=None,
                        source=PowerSource.ESTIMATED,
                        metadata=metadata,
                    )
                )
            except Exception:
                continue

        return readings

    def get_gpu_power_info(self) -> list[GPUPowerInfo]:
        """Get detailed power information for all Tenstorrent devices.

        Returns
        -------
        list[GPUPowerInfo]
            One entry per device with power, power limit, and temperature.
        """
        if not self._initialized:
            if not self.initialize():
                return []

        gpus: list[GPUPowerInfo] = []

        for idx, dev in enumerate(self._devices):
            hwmon_dir = dev.get("hwmon_dir")
            if hwmon_dir is None:
                continue

            try:
                # Power draw
                raw_power = _read_sysfs_int(hwmon_dir / "power1_input")
                if raw_power is None:
                    continue
                power_watts = raw_power / 1_000_000.0

                # Power limit (maximum rated power)
                power_limit_watts: float | None = None
                raw_max = _read_sysfs_int(hwmon_dir / "power1_max")
                if raw_max is not None:
                    power_limit_watts = raw_max / 1_000_000.0

                # Temperature
                temperature: float | None = None
                raw_temp = _read_sysfs_int(hwmon_dir / "temp1_input")
                if raw_temp is not None:
                    temperature = raw_temp / 1000.0

                # Voltage and current for metadata
                metadata: dict[str, Any] = {
                    "pci_bdf": dev["pci_bdf"],
                }

                raw_voltage = _read_sysfs_int(hwmon_dir / "in0_input")
                if raw_voltage is not None:
                    metadata["voltage_v"] = raw_voltage / 1000.0

                raw_current = _read_sysfs_int(hwmon_dir / "curr1_input")
                if raw_current is not None:
                    metadata["current_a"] = raw_current / 1000.0

                # Build model name
                card_type = _read_sysfs(dev["tt_dir"] / "tt_card_type")
                chip_name = _read_sysfs(hwmon_dir / "name")
                name = f"Tenstorrent {card_type or 'accelerator'}"
                if chip_name:
                    name += f" ({chip_name})"

                gpus.append(
                    GPUPowerInfo(
                        index=idx,
                        name=name,
                        power_watts=power_watts,
                        power_limit_watts=power_limit_watts,
                        # Sysfs does not expose utilisation metrics.
                        utilization_percent=0.0,
                        memory_utilization_percent=0.0,
                        temperature_celsius=temperature,
                        processes=[],
                        metadata=metadata,
                    )
                )
            except Exception:
                continue

        return gpus

    def get_total_gpu_power(self) -> float:
        """Get total power consumption across all Tenstorrent devices.

        Returns
        -------
        float
            Total power in Watts.
        """
        if not self._initialized:
            if not self.initialize():
                return 0.0

        total = 0.0
        for dev in self._devices:
            hwmon_dir = dev.get("hwmon_dir")
            if hwmon_dir is None:
                continue
            try:
                raw = _read_sysfs_int(hwmon_dir / "power1_input")
                if raw is not None:
                    total += raw / 1_000_000.0
            except Exception:
                continue
        return total

    def cleanup(self) -> None:
        """Clean up resources."""
        self._initialized = False
        self._devices = []
