"""PCI backend - identifies devices on the PCIe bus using lspci."""

import shutil
import subprocess
from enum import Enum


class PCIVendor(str, Enum):
    """Common PCI vendor IDs."""

    NVIDIA = "10de"
    AMD = "1002"
    INTEL = "8086"


class PCIDevice:
    """Represents a device on the PCIe bus."""

    def __init__(
        self,
        slot: str,
        vendor_id: str,
        device_id: str,
        class_name: str,
        vendor_name: str,
        device_name: str,
    ):
        self.slot = slot
        self.vendor_id = vendor_id
        self.device_id = device_id
        self.class_name = class_name
        self.vendor_name = vendor_name
        self.device_name = device_name

    def __repr__(self) -> str:
        """Return a string representation of the PCI device."""
        return (
            f"PCIDevice(slot='{self.slot}', vendor='{self.vendor_id}', "
            f"device='{self.device_id}', name='{self.vendor_name} "
            f"{self.device_name}')"
        )


class PCIBackend:
    """Backend for PCIe device discovery using lspci.

    Used to detect hardware presence even if vendor-specific drivers or
    libraries are missing or failing.
    """

    def __init__(self) -> None:
        self.lspci_path = shutil.which("lspci")

    def is_available(self) -> bool:
        """Check if lspci is available on the system."""
        return self.lspci_path is not None

    def list_devices(self) -> list[PCIDevice]:
        """List all PCI devices using lspci -nn.

        Returns:
            List of PCIDevice objects.
        """
        if not self.is_available() or self.lspci_path is None:
            return []

        try:
            # -nn shows both text and numeric IDs
            # Example line: 01:00.0 VGA compatible controller [0300]:
            # NVIDIA Corporation AD102 [GeForce RTX 4090] [10de:2684] (rev a1)
            result = subprocess.run(
                [self.lspci_path, "-nn"],
                capture_output=True,
                text=True,
                check=True,
            )
            return self._parse_lspci_output(result.stdout)
        except (subprocess.CalledProcessError, OSError):
            return []

    def get_gpus(self) -> list[PCIDevice]:
        """Find all GPU/VGA devices on the PCI bus.

        Looks for common display controller class names and vendor IDs.

        Returns:
            List of detected GPU devices.
        """
        devices = self.list_devices()
        gpus = []

        gpu_classes = [
            "VGA compatible controller",
            "Display controller",
            "3D controller",
        ]

        for dev in devices:
            is_gpu_class = any(cls in dev.class_name for cls in gpu_classes)
            is_gpu_vendor = dev.vendor_id in [
                PCIVendor.NVIDIA,
                PCIVendor.AMD,
                PCIVendor.INTEL,
            ]

            if is_gpu_class or is_gpu_vendor:
                # Filter out non-GPU devices from these vendors if necessary,
                # but for now we focus on display controllers.
                if is_gpu_class:
                    gpus.append(dev)

        return gpus

    def _parse_lspci_output(self, output: str) -> list[PCIDevice]:
        """Parse lspci -nn output.

        Example line format:
        01:00.0 VGA compatible controller [0300]:
        NVIDIA Corporation Device [10de:2684] (rev a1)
        """
        devices = []
        for line in output.splitlines():
            if not line.strip():
                continue

            try:
                # Split slot from the rest
                slot, rest = line.split(" ", 1)

                # Find IDs in [vendor:device] format
                if "[" not in rest or ":" not in rest or "]" not in rest:
                    continue

                # The last [] usually contains the vendor:device IDs
                # But some lines have multiple []. We want the one with ':'
                parts = rest.split("[")
                id_part = ""
                for p in reversed(parts):
                    if ":" in p and "]" in p:
                        id_part = p.split("]")[0]
                        break

                if not id_part or ":" not in id_part:
                    continue

                vendor_id, device_id = id_part.split(":")

                # Try to extract class and names
                # format: <class> [<class_id>]: <vendor_and_device_names>
                # [<vendor_id>:<device_id>]
                class_and_names = rest.split(":", 1)
                if len(class_and_names) < 2:
                    continue

                class_part = class_and_names[0].strip()
                # Remove [class_id] if present
                if "[" in class_part:
                    class_part = class_part.split("[")[0].strip()

                names_part = class_and_names[1].strip()

                # Find the start of the [vendor_id:device_id] part
                # It's usually the last occurrence of '[' followed by digits and ':'
                full_name = names_part
                id_search_str = f"[{vendor_id}:{device_id}]"
                if id_search_str in names_part:
                    full_name = names_part.split(id_search_str)[0].strip()
                elif "[" in names_part:
                    # Fallback: split at the last '[' if we can't find the exact ID
                    full_name = names_part.rsplit("[", 1)[0].strip()

                # Split vendor and device if possible
                # Vendors often have multiple words
                # (e.g., "Advanced Micro Devices, Inc.")
                # We'll try to match against known vendor names or just
                # take the first word
                vendor_name = ""
                device_name = ""

                lower_full_name = full_name.lower()
                if "nvidia" in lower_full_name:
                    vendor_name = "NVIDIA"
                    device_name = (
                        full_name.replace("NVIDIA Corporation", "")
                        .replace("NVIDIA", "")
                        .strip()
                    )
                elif (
                    "advanced micro devices" in lower_full_name
                    or "amd" in lower_full_name
                ):
                    vendor_name = "AMD"
                    device_name = (
                        full_name.replace("Advanced Micro Devices, Inc.", "")
                        .replace("AMD/ATI", "")
                        .replace("AMD", "")
                        .strip()
                    )
                elif "intel" in lower_full_name:
                    vendor_name = "Intel"
                    device_name = (
                        full_name.replace("Intel Corporation", "")
                        .replace("Intel", "")
                        .strip()
                    )
                else:
                    name_parts = full_name.split(" ", 1)
                    vendor_name = name_parts[0]
                    device_name = name_parts[1] if len(name_parts) > 1 else ""

                devices.append(
                    PCIDevice(
                        slot=slot,
                        vendor_id=vendor_id.lower(),
                        device_id=device_id.lower(),
                        class_name=class_part,
                        vendor_name=vendor_name,
                        device_name=device_name,
                    )
                )
            except Exception:
                continue

        return devices
