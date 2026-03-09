"""AMD GPU backend using amdsmi (AMD System Management Interface).

This backend collects GPU information, telemetry, and power data
for AMD GPUs via the amdsmi Python library (ROCm SMI).
"""

from __future__ import annotations

from warpt.backends.base import AcceleratorBackend
from warpt.models.list_models import GPUInfo

# Optional amdsmi import — only loaded when this backend is tried
try:
    import amdsmi

    AMDSMI_AVAILABLE = True
except ImportError:
    amdsmi = None  # type: ignore[assignment]
    AMDSMI_AVAILABLE = False


class AmdBackend(AcceleratorBackend):
    """Backend for AMD GPU information using amdsmi."""

    def __init__(self):
        """Initialize AMD SMI library.

        Raises
        ------
        RuntimeError
            If the amdsmi library is not installed.
        """
        if not AMDSMI_AVAILABLE:
            raise RuntimeError("amdsmi library is not available")
        amdsmi.amdsmi_init(amdsmi.AmdSmiInitFlags.INIT_AMD_GPUS)
        self._processor_handles: list = []
        try:
            self._processor_handles = amdsmi.amdsmi_get_processor_handles()
        except Exception:
            self._processor_handles = []

    def is_available(self) -> bool:
        """Check if AMD GPUs are available.

        Returns
        -------
        bool
            True if at least one AMD GPU is detected.
        """
        try:
            return self.get_device_count() > 0
        except Exception:
            return False

    def get_device_count(self) -> int:
        """Get the number of AMD GPUs.

        Returns
        -------
        int
            Number of AMD GPUs detected.
        """
        return len(self._processor_handles)

    def _get_handle(self, index: int):
        """Get processor handle for a GPU by index.

        Parameters
        ----------
        index : int
            GPU index (0-based).

        Returns
        -------
        amdsmi_processor_handle
            Processor handle for the given index.
        """
        return self._processor_handles[index]

    def list_devices(self) -> list[GPUInfo]:
        """List all AMD GPUs with their specifications.

        Returns
        -------
        list[GPUInfo]
            List of GPU information objects.
        """
        devices: list[GPUInfo] = []
        driver_version = self.get_driver_version()

        for i in range(self.get_device_count()):
            handle = self._get_handle(i)

            # Get ASIC info for model name and identity
            model_name = "AMD GPU"
            asic_serial = None
            target_gfx_version = None
            num_compute_units = None
            try:
                asic_info = amdsmi.amdsmi_get_gpu_asic_info(handle)
                market_name = asic_info.get("market_name", "N/A")
                if market_name and market_name != "N/A":
                    model_name = market_name
                serial = asic_info.get("asic_serial", "N/A")
                if serial and serial != "N/A":
                    asic_serial = serial
                gfx_ver = asic_info.get("target_graphics_version", "N/A")
                if gfx_ver and gfx_ver != "N/A":
                    target_gfx_version = gfx_ver
                cu_count = asic_info.get("num_compute_units", "N/A")
                if isinstance(cu_count, int):
                    num_compute_units = cu_count
            except Exception:
                pass

            # Get UUID
            uuid = None
            try:
                uuid = amdsmi.amdsmi_get_gpu_device_uuid(handle)
            except Exception:
                pass

            # Get VRAM total in GB (amdsmi reports MB)
            memory_gb = 0
            try:
                vram_usage = amdsmi.amdsmi_get_gpu_vram_usage(handle)
                vram_total_mb = vram_usage.get("vram_total", 0)
                if isinstance(vram_total_mb, int):
                    memory_gb = vram_total_mb // 1024
            except Exception:
                pass

            # Get PCIe generation
            pcie_gen = None
            try:
                pcie_info = amdsmi.amdsmi_get_pcie_info(handle)
                static = pcie_info.get("pcie_static", {})
                gen = static.get("pcie_interface_version", "N/A")
                if isinstance(gen, int):
                    pcie_gen = gen
            except Exception:
                pass

            # Build extra_metrics with vendor-specific data
            extra_metrics: dict = {}
            if target_gfx_version:
                extra_metrics["target_graphics_version"] = target_gfx_version
            if num_compute_units is not None:
                extra_metrics["num_compute_units"] = num_compute_units
            if asic_serial:
                extra_metrics["asic_serial"] = asic_serial

            # VRAM type and vendor
            try:
                vram_info = amdsmi.amdsmi_get_gpu_vram_info(handle)
                vram_type_raw = vram_info.get("vram_type")
                if vram_type_raw is not None:
                    try:
                        extra_metrics["vram_type"] = (
                            amdsmi.AmdSmiVramType(vram_type_raw).name
                        )
                    except (ValueError, AttributeError):
                        extra_metrics["vram_type"] = str(vram_type_raw)
                vram_vendor = vram_info.get("vram_vendor", "N/A")
                if vram_vendor and vram_vendor != "N/A":
                    extra_metrics["vram_vendor"] = vram_vendor
                bw = vram_info.get("vram_max_bandwidth", "N/A")
                if isinstance(bw, int):
                    extra_metrics["vram_max_bandwidth_gbps"] = bw
            except Exception:
                pass

            # Board info
            try:
                board_info = amdsmi.amdsmi_get_gpu_board_info(handle)
                product_name = board_info.get("product_name", "N/A")
                if product_name and product_name != "N/A":
                    extra_metrics["product_name"] = product_name
                manufacturer = board_info.get("manufacturer_name", "N/A")
                if manufacturer and manufacturer != "N/A":
                    extra_metrics["manufacturer_name"] = manufacturer
            except Exception:
                pass

            devices.append(
                GPUInfo(
                    index=i,
                    model=model_name,
                    memory_gb=memory_gb,
                    uuid=uuid,
                    compute_capability=None,
                    pcie_gen=pcie_gen,
                    driver_version=driver_version,
                    extra_metrics=extra_metrics if extra_metrics else None,
                )
            )

        return devices

    def get_temperature(self, index: int) -> float | None:
        """Get GPU temperature in degrees Celsius.

        Reads the edge sensor first, falling back to hotspot if unavailable.
        amdsmi reports temperature in millidegrees Celsius.

        Parameters
        ----------
        index : int
            GPU index (0-based).

        Returns
        -------
        float or None
            Temperature in degrees Celsius, or None if unavailable.
        """
        handle = self._get_handle(index)
        # Try edge temperature first, then hotspot as fallback
        for sensor in (
            amdsmi.AmdSmiTemperatureType.EDGE,
            amdsmi.AmdSmiTemperatureType.HOTSPOT,
        ):
            try:
                temp_mc = amdsmi.amdsmi_get_temp_metric(
                    handle,
                    sensor,
                    amdsmi.AmdSmiTemperatureMetric.CURRENT,
                )
                return float(temp_mc) / 1000.0
            except Exception:
                continue
        return None

    def get_memory_usage(self, index: int) -> dict | None:
        """Get current GPU memory usage.

        amdsmi reports VRAM usage in megabytes; this method converts
        to bytes for consistency with the warpt interface.

        Parameters
        ----------
        index : int
            GPU index (0-based).

        Returns
        -------
        dict or None
            Dictionary with keys ``total``, ``used``, ``free``
            (all in bytes), or None if unavailable.
        """
        try:
            handle = self._get_handle(index)
            vram_usage = amdsmi.amdsmi_get_gpu_vram_usage(handle)
            total_mb = vram_usage.get("vram_total", 0)
            used_mb = vram_usage.get("vram_used", 0)
            if not isinstance(total_mb, int) or not isinstance(used_mb, int):
                return None
            total_bytes = total_mb * 1024 * 1024
            used_bytes = used_mb * 1024 * 1024
            free_bytes = total_bytes - used_bytes
            return {
                "total": total_bytes,
                "used": used_bytes,
                "free": max(free_bytes, 0),
            }
        except Exception:
            return None

    def get_utilization(self, index: int) -> dict | None:
        """Get GPU utilization percentages.

        Maps ``gfx_activity`` to GPU compute utilization and
        ``umc_activity`` to memory bandwidth utilization.

        Parameters
        ----------
        index : int
            GPU index (0-based).

        Returns
        -------
        dict or None
            Dictionary with keys ``gpu`` and ``memory`` (both 0-100),
            or None if unavailable.
        """
        try:
            handle = self._get_handle(index)
            activity = amdsmi.amdsmi_get_gpu_activity(handle)
            gfx = activity.get("gfx_activity", "N/A")
            umc = activity.get("umc_activity", "N/A")
            gpu_util = float(gfx) if isinstance(gfx, int | float) else 0.0
            mem_util = float(umc) if isinstance(umc, int | float) else 0.0
            return {
                "gpu": gpu_util,
                "memory": mem_util,
            }
        except Exception:
            return None

    def get_pytorch_device_string(self, device_id: int) -> str:
        """Get PyTorch device string for AMD GPUs.

        ROCm-enabled PyTorch uses the ``cuda`` device string because
        HIP maps to CUDA semantics transparently.

        Parameters
        ----------
        device_id : int
            GPU index (0-based).

        Returns
        -------
        str
            PyTorch device string (e.g., ``'cuda:0'``).
        """
        return f"cuda:{device_id}"

    def get_power_usage(self, index: int) -> float | None:
        """Get current GPU power usage in Watts.

        Tries ``current_socket_power`` first, then
        ``average_socket_power``, then ``socket_power``.
        amdsmi reports power in watts.

        Parameters
        ----------
        index : int
            GPU index (0-based).

        Returns
        -------
        float or None
            Power usage in Watts, or None if unavailable.
        """
        try:
            handle = self._get_handle(index)
            power_info = amdsmi.amdsmi_get_power_info(handle)
            for key in (
                "current_socket_power",
                "average_socket_power",
                "socket_power",
            ):
                val = power_info.get(key, "N/A")
                if isinstance(val, int | float) and val > 0:
                    return float(val)
            return None
        except Exception:
            return None

    def get_throttle_reasons(self, index: int) -> list[str]:
        """Get current GPU throttling reasons.

        AMD SMI exposes violation status via ``amdsmi_get_violation_status``
        which returns ``active_`` prefixed keys. This API is MI300+ only;
        on older hardware the call may fail and an empty list is returned.

        Parameters
        ----------
        index : int
            GPU index (0-based).

        Returns
        -------
        list[str]
            Active throttle reasons, empty list if not throttling.
        """
        reasons: list[str] = []
        try:
            handle = self._get_handle(index)
            violation = amdsmi.amdsmi_get_violation_status(handle)
            if isinstance(violation, dict):
                _mapping = {
                    "active_ppt_pwr": "power_limit",
                    "active_socket_thrm": "thermal",
                    "active_prochot_thrm": "thermal",
                    "active_vr_thrm": "vr_thermal",
                    "active_hbm_thrm": "hbm_thermal",
                }
                for key, label in _mapping.items():
                    val = violation.get(key, 0)
                    if isinstance(val, int | float) and val > 0:
                        if label not in reasons:
                            reasons.append(label)
        except Exception:
            pass
        return reasons

    def get_driver_version(self) -> str | None:
        """Get AMD GPU driver version.

        Returns
        -------
        str or None
            Driver version string, or None if unavailable.
        """
        if not self._processor_handles:
            return None
        try:
            handle = self._get_handle(0)
            driver_info = amdsmi.amdsmi_get_gpu_driver_info(handle)
            version = driver_info.get("driver_version", "N/A")
            if version and version != "N/A":
                return str(version)
            return None
        except Exception:
            return None

    def shutdown(self):
        """Cleanup and shutdown AMD SMI."""
        try:
            amdsmi.amdsmi_shut_down()
        except Exception:
            pass
