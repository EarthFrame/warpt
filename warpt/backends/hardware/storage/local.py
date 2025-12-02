"""Local storage backend for detecting block devices.

Supports detection of:
- NVMe SSDs
- SATA SSDs
- HDDs (spinning disk)
- USB storage devices

Platform support:
- Linux: /sys/block, lsblk
- macOS: diskutil, system_profiler
"""

import json
import platform
import subprocess
from pathlib import Path
from typing import TypeVar

from warpt.backends.hardware.storage.base import (
    BusType,
    LocalBlockDeviceInfo,
    StorageBackend,
    StorageDeviceInfo,
    StorageType,
)

T = TypeVar("T")


class LocalStorageBackend(StorageBackend):
    """Backend for local block storage devices."""

    @property
    def storage_type(self) -> str:
        """Return storage type identifier."""
        return "local"

    def is_available(self) -> bool:
        """Check if local storage detection is available."""
        # Local storage is always available on supported platforms
        return platform.system() in ("Linux", "Darwin", "Windows")

    def list_devices(self) -> list[StorageDeviceInfo]:
        """List all local block devices.

        Returns:
            List of StorageDeviceInfo for each detected device
        """
        system = platform.system()

        if system == "Linux":
            return self._list_linux_devices()
        elif system == "Darwin":
            return self._list_macos_devices()
        else:
            # Windows or unsupported - return empty for now
            return []

    def get_device_info(self, device_path: str) -> StorageDeviceInfo | None:
        """Get info for a specific device.

        Args:
            device_path: Device path (e.g., /dev/nvme0n1, /dev/sda)

        Returns:
            StorageDeviceInfo if found, None otherwise
        """
        devices = self.list_devices()
        for device in devices:
            if device.device_path == device_path:
                return device
        return None

    # -------------------------------------------------------------------------
    # Linux Implementation
    # -------------------------------------------------------------------------

    def _list_linux_devices(self) -> list[StorageDeviceInfo]:
        """List block devices on Linux using lsblk."""
        devices: list[StorageDeviceInfo] = []

        # Try lsblk JSON output first (most reliable)
        try:
            result = subprocess.run(
                [
                    "lsblk",
                    "-J",  # JSON output
                    "-b",  # bytes for size
                    "-o",
                    "NAME,SIZE,TYPE,TRAN,MODEL,SERIAL,REV,ROTA,RM,MOUNTPOINTS,FSTYPE",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                devices = self._parse_lsblk_json(result.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            # Fallback to /sys/block parsing
            devices = self._parse_sys_block()

        return devices

    def _parse_lsblk_json(self, json_output: str) -> list[StorageDeviceInfo]:
        """Parse lsblk JSON output.

        Args:
            json_output: JSON string from lsblk -J

        Returns:
            List of StorageDeviceInfo
        """
        devices: list[StorageDeviceInfo] = []
        data = json.loads(json_output)

        for block in data.get("blockdevices", []):
            # Skip partitions, only report whole devices
            if block.get("type") != "disk":
                continue

            device_path = f"/dev/{block['name']}"
            size_bytes = int(block.get("size", 0))
            transport = block.get("tran", "")
            model = block.get("model", "").strip() if block.get("model") else "Unknown"
            serial = block.get("serial", "").strip() if block.get("serial") else None
            firmware = block.get("rev", "").strip() if block.get("rev") else None
            is_rotational = block.get("rota", False)
            is_removable = block.get("rm", False)

            # Determine device type and bus
            device_type, bus_type = self._classify_linux_device(
                block["name"], transport, is_rotational
            )

            # Collect mount points from partitions
            mount_points = self._collect_mount_points(block)
            filesystem = self._get_primary_filesystem(block)

            # Check if system disk (has / mount)
            is_system = "/" in mount_points

            vendor = self._extract_vendor(model)

            device_data = {
                "device_path": device_path,
                "model": model,
                "serial": serial,
                "firmware": firmware,
                "manufacturer": vendor,
                "device_type": device_type,
                "bus_type": bus_type,
                "capacity_bytes": size_bytes,
                "capacity_gb": size_bytes // (10**9),
                "mount_points": mount_points,
                "filesystem": filesystem,
                "is_system_disk": is_system,
                "is_removable": is_removable,
            }

            devices.append(LocalBlockDeviceInfo.model_validate(device_data))

        return devices

    def _classify_linux_device(
        self, name: str, transport: str, is_rotational: bool
    ) -> tuple[StorageType, BusType]:
        """Classify a Linux block device by type and bus.

        Args:
            name: Device name (e.g., nvme0n1, sda)
            transport: Transport type from lsblk (nvme, sata, usb, etc.)
            is_rotational: True if spinning disk

        Returns:
            Tuple of (StorageType, BusType)
        """
        # NVMe devices
        if name.startswith("nvme"):
            return StorageType.NVME_SSD, BusType.PCIE

        # USB devices
        if transport == "usb":
            return StorageType.USB, BusType.USB

        # SATA/SAS devices
        if transport in ("sata", "sas", ""):
            if is_rotational:
                return StorageType.HDD, BusType.SATA
            else:
                return StorageType.SATA_SSD, BusType.SATA

        return StorageType.UNKNOWN_BLOCK, BusType.UNKNOWN

    def _collect_mount_points(self, block: dict) -> list[str]:
        """Collect all mount points from device and its partitions.

        Args:
            block: lsblk block device dict

        Returns:
            List of mount point paths
        """
        mounts = []

        # Check device-level mounts
        device_mounts = block.get("mountpoints", [])
        if device_mounts:
            mounts.extend([m for m in device_mounts if m])

        # Check partition mounts
        for child in block.get("children", []):
            child_mounts = child.get("mountpoints", [])
            if child_mounts:
                mounts.extend([m for m in child_mounts if m])

        return mounts

    def _get_primary_filesystem(self, block: dict) -> str | None:
        """Get the primary filesystem type.

        Args:
            block: lsblk block device dict

        Returns:
            Filesystem type or None
        """
        # Check partitions first (more common)
        for child in block.get("children", []):
            fstype = child.get("fstype")
            if isinstance(fstype, str):
                return fstype

        # Check device-level filesystem
        fstype = block.get("fstype")
        if isinstance(fstype, str):
            return fstype
        return None

    def _parse_sys_block(self) -> list[StorageDeviceInfo]:
        """Fallback: Parse /sys/block directly.

        Returns:
            List of StorageDeviceInfo
        """
        devices: list[StorageDeviceInfo] = []
        sys_block = Path("/sys/block")

        if not sys_block.exists():
            return devices

        for device_dir in sys_block.iterdir():
            name = device_dir.name

            # Skip virtual devices (loop, ram, dm-*)
            if name.startswith(("loop", "ram", "dm-", "sr", "fd")):
                continue

            device_path = f"/dev/{name}"
            size_sectors = int(self._read_sys_value(device_dir / "size", 0))
            sector_size = 512
            size_bytes = size_sectors * sector_size

            # Skip zero-size devices
            if size_bytes == 0:
                continue

            model = str(
                self._read_sys_value(device_dir / "device" / "model", "Unknown")
            ).strip()
            rotational = int(
                self._read_sys_value(device_dir / "queue" / "rotational", 1)
            )
            removable = int(self._read_sys_value(device_dir / "removable", 0))

            # Classify device
            if name.startswith("nvme"):
                device_type = StorageType.NVME_SSD
                bus_type = BusType.PCIE
            elif rotational == 1:
                device_type = StorageType.HDD
                bus_type = BusType.SATA
            else:
                device_type = StorageType.SATA_SSD
                bus_type = BusType.SATA

            device_data = {
                "device_path": device_path,
                "model": model,
                "manufacturer": self._extract_vendor(model),
                "device_type": device_type,
                "bus_type": bus_type,
                "capacity_bytes": size_bytes,
                "capacity_gb": size_bytes // (10**9),
                "is_removable": bool(removable),
            }

            devices.append(LocalBlockDeviceInfo.model_validate(device_data))

        return devices

    def _read_sys_value(self, path: Path, default: T) -> int | str | T:
        """Read a value from /sys filesystem.

        Args:
            path: Path to sysfs file
            default: Default value if read fails

        Returns:
            File contents or default
        """
        try:
            content = path.read_text().strip()
            # Try to convert to int if possible
            try:
                return int(content)
            except ValueError:
                return content
        except (OSError, PermissionError):
            return default

    # -------------------------------------------------------------------------
    # macOS Implementation
    # -------------------------------------------------------------------------

    def _list_macos_devices(self) -> list[StorageDeviceInfo]:
        """List block devices on macOS using diskutil."""
        devices: list[StorageDeviceInfo] = []

        try:
            # Get list of all disks
            result = subprocess.run(
                ["diskutil", "list", "-plist"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                return devices

            # Parse plist output
            import plistlib

            plist_data = plistlib.loads(result.stdout)

            # Use AllDisksAndPartitions which gives us structure info
            all_disks = plist_data.get("AllDisksAndPartitions", [])

            for disk_entry in all_disks:
                disk_id = disk_entry.get("DeviceIdentifier", "")
                content = disk_entry.get("Content", "")

                # Only process physical disks (have GUID partition scheme)
                # Skip synthesized APFS containers
                if content != "GUID_partition_scheme":
                    continue

                device_info = self._get_macos_device_info(disk_id)
                if device_info:
                    devices.append(device_info)

        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        except Exception:
            # plistlib or other parsing error
            pass

        return devices

    def _get_macos_device_info(self, disk_name: str) -> StorageDeviceInfo | None:
        """Get detailed info for a macOS disk.

        Args:
            disk_name: Disk identifier (e.g., disk0, disk1)

        Returns:
            StorageDeviceInfo or None if unavailable
        """
        try:
            result = subprocess.run(
                ["diskutil", "info", "-plist", disk_name],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None

            import plistlib

            info = plistlib.loads(result.stdout)

            device_path = f"/dev/{disk_name}"
            size_bytes = info.get("IOKitSize", info.get("TotalSize", 0))
            model = info.get("MediaName", "Unknown")
            is_internal = info.get("Internal", False)
            is_removable = info.get("Removable", False)
            protocol = info.get("DeviceTreePath", "")
            bus_protocol = info.get("BusProtocol", "")

            # Determine device type
            device_type, bus_type = self._classify_macos_device(
                protocol, info.get("SolidState", True), is_internal, bus_protocol
            )

            # Get mount points from partitions (physical disks don't mount directly)
            mount_points = self._get_macos_mount_points(disk_name)

            # Check if system disk (has / mount)
            is_system = "/" in mount_points

            # Extract vendor from model name
            vendor = self._extract_vendor(model)

            # Extract SMART data if available
            smart_data = info.get("SMARTDeviceSpecificKeysMayVaryNotGuaranteed", {})
            wear_level = None
            if smart_data:
                available_spare = smart_data.get("AVAILABLE_SPARE")
                if available_spare is not None:
                    # Available spare is inverted: 100 = new, lower = more worn
                    wear_level = 100 - available_spare

            device_data = {
                "device_path": device_path,
                "model": model,
                "serial": info.get("SerialNumber"),
                "firmware": info.get("FirmwareRevision"),
                "manufacturer": vendor,
                "device_type": device_type,
                "bus_type": bus_type,
                "capacity_bytes": size_bytes,
                "capacity_gb": size_bytes // (10**9),
                "mount_points": mount_points,
                "filesystem": None,  # Physical disks don't have filesystems directly
                "is_system_disk": is_system,
                "is_removable": is_removable,
                "wear_level_percent": wear_level,
                "health_status": "healthy" if smart_data else None,
                "extra_info": {
                    "internal": is_internal,
                    "device_node": info.get("DeviceNode"),
                    "ejectable": info.get("Ejectable", False),
                    "bus_protocol": bus_protocol,
                    "smart_available": bool(smart_data),
                },
            }

            return LocalBlockDeviceInfo.model_validate(device_data)

        except Exception:
            return None

    def _get_macos_mount_points(self, disk_name: str) -> list[str]:
        """Get mount points for all partitions/volumes backed by a macOS disk.

        On APFS, volumes are in synthesized containers. We need to find
        containers that have this disk as their physical store.

        Args:
            disk_name: Disk identifier (e.g., disk0)

        Returns:
            List of mount point paths
        """
        mount_points: list[str] = []
        try:
            # Get full disk list to find APFS relationships
            result = subprocess.run(
                ["diskutil", "list", "-plist"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                return mount_points

            import plistlib

            plist_data = plistlib.loads(result.stdout)

            # Find partitions that belong to this physical disk
            physical_partitions = set()
            for disk_entry in plist_data.get("AllDisksAndPartitions", []):
                if disk_entry.get("DeviceIdentifier") == disk_name:
                    for partition in disk_entry.get("Partitions", []):
                        physical_partitions.add(partition.get("DeviceIdentifier"))

            # Now find APFS containers backed by these partitions
            for disk_entry in plist_data.get("AllDisksAndPartitions", []):
                # Check if this container is backed by our physical disk
                physical_stores = disk_entry.get("APFSPhysicalStores", [])
                for store in physical_stores:
                    store_id = store.get("DeviceIdentifier", "")
                    if store_id in physical_partitions:
                        # This APFS container is on our disk
                        for volume in disk_entry.get("APFSVolumes", []):
                            mount = volume.get("MountPoint")
                            if mount:
                                mount_points.append(mount)
                            # Also check for mounted snapshots (like macOS root)
                            for snapshot in volume.get("MountedSnapshots", []):
                                snap_mount = snapshot.get("SnapshotMountPoint")
                                if snap_mount:
                                    mount_points.append(snap_mount)

        except Exception:
            pass

        # Deduplicate while preserving order
        seen = set()
        unique_mounts = []
        for m in mount_points:
            if m not in seen:
                seen.add(m)
                unique_mounts.append(m)
        return unique_mounts

    def _classify_macos_device(
        self, protocol: str, is_ssd: bool, is_internal: bool, bus_protocol: str = ""
    ) -> tuple[StorageType, BusType]:
        """Classify a macOS disk device.

        Args:
            protocol: Device tree path or protocol string
            is_ssd: True if SSD
            is_internal: True if internal drive
            bus_protocol: Bus protocol string (e.g., "Apple Fabric", "USB")

        Returns:
            Tuple of (StorageType, BusType)
        """
        protocol_lower = protocol.lower()
        bus_lower = bus_protocol.lower()

        # NVMe detection - Apple ANS controllers are NVMe
        if "nvme" in protocol_lower or "AppleANS" in protocol:
            return StorageType.NVME_SSD, BusType.PCIE

        # Apple Fabric is the internal interconnect for M-series
        if "apple fabric" in bus_lower:
            return StorageType.NVME_SSD, BusType.PCIE

        # USB detection
        if "usb" in protocol_lower or "usb" in bus_lower:
            return StorageType.USB, BusType.USB

        # Thunderbolt
        if "thunderbolt" in bus_lower:
            if is_ssd:
                return StorageType.NVME_SSD, BusType.PCIE
            return StorageType.HDD, BusType.PCIE

        # Internal Apple SSDs (M-series, T2, etc.)
        if is_internal and is_ssd:
            return StorageType.NVME_SSD, BusType.PCIE

        # SATA
        if is_ssd:
            return StorageType.SATA_SSD, BusType.SATA
        else:
            return StorageType.HDD, BusType.SATA

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _extract_vendor(self, model: str) -> str | None:
        """Extract vendor from model string.

        Args:
            model: Device model string

        Returns:
            Vendor name or None
        """
        if not model or model == "Unknown":
            return None

        # Common vendor prefixes
        vendors = [
            "Samsung",
            "WDC",
            "Western Digital",
            "Seagate",
            "Toshiba",
            "Kingston",
            "Crucial",
            "SanDisk",
            "Intel",
            "Micron",
            "SK hynix",
            "HGST",
            "Hitachi",
            "Apple",
            "KIOXIA",
            "Sabrent",
            "ADATA",
            "PNY",
            "Corsair",
            "Patriot",
            "Team",
            "Transcend",
            "Phison",
            "Silicon Power",
            "HP",
            "Lenovo",
            "Dell",
        ]

        model_upper = model.upper()
        for vendor in vendors:
            if vendor.upper() in model_upper:
                return vendor

        # Try first word as vendor
        parts = model.split()
        if parts:
            return parts[0]

        return None
