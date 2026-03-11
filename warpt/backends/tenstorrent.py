"""Tenstorrent accelerator backend using Linux sysfs telemetry.

This backend reads device information, power, temperature, and clock
frequencies from the sysfs virtual filesystem exposed by the Tenstorrent
kernel-mode driver (tt-kmd). No third-party SDK is required — only the
standard ``pathlib`` and ``os`` modules.

Two sysfs subsystems are used:

- ``/sys/class/tenstorrent/tenstorrent!N/`` — device identity, clocks,
  firmware versions, serial numbers.
- ``/sys/class/hwmon/hwmonX/`` — temperature, power, voltage, current.
  The correct hwmon directory is located by following the PCI device
  symlink from the tenstorrent driver entry.
"""

from __future__ import annotations

from pathlib import Path

from warpt.backends.base import AcceleratorBackend
from warpt.models.list_models import GPUInfo

# Root sysfs path for the Tenstorrent driver
_TT_CLASS = Path("/sys/class/tenstorrent")


def _read_sysfs(path: Path) -> str | None:
    """Read and strip a sysfs attribute file.

    Parameters
    ----------
    path : Path
        Absolute path to the sysfs attribute file.

    Returns
    -------
    str or None
        The stripped file contents, or ``None`` if the file does not exist
        or cannot be read.
    """
    try:
        return path.read_text().strip()
    except (FileNotFoundError, OSError, PermissionError):
        return None


def _read_sysfs_int(path: Path) -> int | None:
    """Read a sysfs attribute file and parse as an integer.

    Parameters
    ----------
    path : Path
        Absolute path to the sysfs attribute file.

    Returns
    -------
    int or None
        The parsed integer value, or ``None`` on failure.
    """
    raw = _read_sysfs(path)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _discover_devices() -> list[dict]:
    """Discover all Tenstorrent devices via sysfs.

    Each returned dict contains:

    - ``name``      - kernel entry name (e.g. ``"tenstorrent!0"``)
    - ``tt_dir``    - resolved Path to the tenstorrent driver directory
    - ``hwmon_dir`` - resolved Path to the matching hwmon directory, or ``None``
    - ``pci_bdf``   - PCI bus-device-function address string

    Returns
    -------
    list[dict]
        One dict per detected device, sorted by entry name.
    """
    if not _TT_CLASS.exists():
        return []

    devices: list[dict] = []
    try:
        entries = sorted(_TT_CLASS.iterdir())
    except OSError:
        return []

    for entry in entries:
        if not entry.is_dir() or not entry.name.startswith("tenstorrent!"):
            continue

        try:
            tt_dir = entry.resolve()
        except OSError:
            continue

        # PCI device directory is two levels above:
        #   .../DDDD:BB:DD.F/tenstorrent/tenstorrent!N
        pci_device_dir = tt_dir.parent.parent

        # Locate the sibling hwmon directory for this PCI device
        hwmon_base = pci_device_dir / "hwmon"
        hwmon_dir: Path | None = None
        if hwmon_base.is_dir():
            try:
                for h in sorted(hwmon_base.iterdir()):
                    if h.is_dir() and h.name.startswith("hwmon"):
                        hwmon_dir = h
                        break
            except OSError:
                pass

        devices.append(
            {
                "name": entry.name,
                "tt_dir": tt_dir,
                "hwmon_dir": hwmon_dir,
                "pci_bdf": pci_device_dir.name,
            }
        )

    return devices


class TenstorrentBackend(AcceleratorBackend):
    """Backend for Tenstorrent accelerators (Wormhole, Blackhole).

    All telemetry is read from the Linux sysfs virtual filesystem.
    The Tenstorrent kernel-mode driver (tt-kmd) must be installed
    and loaded for the sysfs entries to exist.
    """

    def __init__(self) -> None:
        """Initialize the Tenstorrent backend by discovering devices."""
        self._devices: list[dict] = _discover_devices()

    def is_available(self) -> bool:
        """Check if Tenstorrent accelerators are present.

        Returns
        -------
        bool
            ``True`` if at least one Tenstorrent device is detected.
        """
        return self.get_device_count() > 0

    def get_device_count(self) -> int:
        """Get the number of Tenstorrent accelerators.

        Returns
        -------
        int
            Number of Tenstorrent devices detected via sysfs.
        """
        return len(self._devices)

    def list_devices(self) -> list[GPUInfo]:
        """List all Tenstorrent accelerators with their specifications.

        Returns
        -------
        list[GPUInfo]
            One ``GPUInfo`` per detected device. Fields that are
            unavailable via sysfs (e.g. ``memory_gb``,
            ``compute_capability``) are set to ``0`` or ``None``.
        """
        device_info: list[GPUInfo] = []
        driver_version = self.get_driver_version()

        for i, dev in enumerate(self._devices):
            tt_dir: Path = dev["tt_dir"]
            hwmon_dir: Path | None = dev["hwmon_dir"]

            # Device identification
            card_type = _read_sysfs(tt_dir / "tt_card_type") or "unknown"
            serial = _read_sysfs(tt_dir / "tt_serial")
            asic_id = _read_sysfs(tt_dir / "tt_asic_id")

            # Chip architecture from hwmon name (e.g. "wormhole", "blackhole")
            chip_name: str | None = None
            if hwmon_dir is not None:
                chip_name = _read_sysfs(hwmon_dir / "name")

            # Build a human-readable model string
            model = self._build_model_string(card_type, chip_name)

            # Use board serial as UUID (stable across reboots).
            # Fall back to ASIC ID if serial is unavailable.
            uuid = serial or asic_id

            # Clock frequencies
            ai_clk = _read_sysfs_int(tt_dir / "tt_aiclk")
            arc_clk = _read_sysfs_int(tt_dir / "tt_arcclk")
            axi_clk = _read_sysfs_int(tt_dir / "tt_axiclk")

            # Firmware versions
            fw_bundle = _read_sysfs(tt_dir / "tt_fw_bundle_ver")
            arc_fw = _read_sysfs(tt_dir / "tt_arc_fw_ver")
            eth_fw = _read_sysfs(tt_dir / "tt_eth_fw_ver")
            m3app_fw = _read_sysfs(tt_dir / "tt_m3app_fw_ver")

            device_info.append(
                GPUInfo(
                    index=i,
                    model=model,
                    # Sysfs does not expose device memory capacity.
                    # Set to 0; consumers should check extra_metrics
                    # for details.
                    memory_gb=0,
                    uuid=uuid,
                    compute_capability=None,
                    pcie_gen=None,
                    driver_version=driver_version,
                    extra_metrics={
                        "card_type": card_type,
                        "chip_name": chip_name,
                        "serial": serial,
                        "asic_id": asic_id,
                        "pci_bdf": dev["pci_bdf"],
                        "ai_clk_mhz": ai_clk,
                        "arc_clk_mhz": arc_clk,
                        "axi_clk_mhz": axi_clk,
                        "fw_bundle_version": fw_bundle,
                        "arc_fw_version": arc_fw,
                        "eth_fw_version": eth_fw,
                        "m3app_fw_version": m3app_fw,
                    },
                )
            )

        return device_info

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    def get_temperature(self, index: int) -> float | None:
        """Get accelerator temperature in degrees Celsius.

        Parameters
        ----------
        index : int
            Device index (0-based).

        Returns
        -------
        float or None
            Temperature in °C, or ``None`` if unavailable.
        """
        hwmon_dir = self._get_hwmon_dir(index)
        if hwmon_dir is None:
            return None
        raw = _read_sysfs_int(hwmon_dir / "temp1_input")
        if raw is None:
            return None
        # temp1_input is in millidegrees Celsius
        return raw / 1000.0

    def get_memory_usage(self, index: int) -> dict | None:  # noqa: ARG002
        """Get device memory usage.

        Tenstorrent sysfs does not expose memory utilisation metrics.

        Parameters
        ----------
        index : int
            Device index (0-based).

        Returns
        -------
        None
            Always returns ``None``.
        """
        return None

    def get_utilization(self, index: int) -> dict | None:  # noqa: ARG002
        """Get device utilization percentages.

        Tenstorrent sysfs does not expose utilisation metrics.

        Parameters
        ----------
        index : int
            Device index (0-based).

        Returns
        -------
        None
            Always returns ``None``.
        """
        return None

    def get_pytorch_device_string(self, device_id: int) -> str:
        """Get the PyTorch device string for a Tenstorrent accelerator.

        Tenstorrent's tt-metalium runtime registers the ``"tt"`` device
        type with PyTorch.

        Parameters
        ----------
        device_id : int
            Device index (0-based).

        Returns
        -------
        str
            PyTorch device string, e.g. ``"tt:0"``.
        """
        return f"tt:{device_id}"

    def get_power_usage(self, index: int) -> float | None:
        """Get current power draw in Watts.

        Parameters
        ----------
        index : int
            Device index (0-based).

        Returns
        -------
        float or None
            Power in Watts, or ``None`` if unavailable.
        """
        hwmon_dir = self._get_hwmon_dir(index)
        if hwmon_dir is None:
            return None
        raw = _read_sysfs_int(hwmon_dir / "power1_input")
        if raw is None:
            return None
        # power1_input is in microwatts
        return raw / 1_000_000.0

    def get_throttle_reasons(self, index: int) -> list[str]:  # noqa: ARG002
        """Get active throttle reasons.

        Tenstorrent sysfs does not expose throttling information.

        Parameters
        ----------
        index : int
            Device index (0-based).

        Returns
        -------
        list[str]
            Always returns an empty list.
        """
        return []

    def get_driver_version(self) -> str | None:
        """Get the Tenstorrent firmware bundle version.

        The firmware bundle version is the closest equivalent to a
        "driver version" in the Tenstorrent sysfs interface. It is
        read from the first detected device.

        Returns
        -------
        str or None
            Firmware bundle version string, or ``None`` if unavailable.
        """
        if not self._devices:
            return None
        tt_dir: Path = self._devices[0]["tt_dir"]
        return _read_sysfs(tt_dir / "tt_fw_bundle_ver")

    def shutdown(self) -> None:
        """Clean up backend resources.

        Sysfs does not require explicit cleanup. This method clears the
        cached device list.
        """
        self._devices = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_hwmon_dir(self, index: int) -> Path | None:
        """Get the hwmon directory for a device by index.

        Parameters
        ----------
        index : int
            Device index (0-based).

        Returns
        -------
        Path or None
            Path to the hwmon directory, or ``None`` if the index is
            out of range or the hwmon directory was not found.
        """
        if index < 0 or index >= len(self._devices):
            return None
        return self._devices[index].get("hwmon_dir")

    @staticmethod
    def _build_model_string(
        card_type: str,
        chip_name: str | None,
    ) -> str:
        """Build a human-readable model name.

        Parameters
        ----------
        card_type : str
            Product name (e.g. ``"n150"``).
        chip_name : str or None
            Chip architecture (e.g. ``"blackhole"``).

        Returns
        -------
        str
            Model string, e.g. ``"Tenstorrent n150 (blackhole)"``.
        """
        base = f"Tenstorrent {card_type}"
        if chip_name:
            base += f" ({chip_name})"
        return base
