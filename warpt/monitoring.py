"""Simple resource monitoring daemon for CPU, GPU, and memory."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psutil

try:
    import pynvml
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None

_NVML_INITIALIZED = False


def _ensure_nvml_initialized() -> bool:
    """Initialize NVML the first time it is needed.

    Returns
    -------
        True if NVML is available and initialized, False otherwise.
    """
    global _NVML_INITIALIZED

    if not pynvml:
        return False

    if _NVML_INITIALIZED:
        return True

    try:
        pynvml.nvmlInit()
        _NVML_INITIALIZED = True
        return True
    except pynvml.NVMLError:
        return False


@dataclass
class GPUUsage:
    """Per-GPU utilization, power, and identity metrics."""

    index: int
    model: str | None
    utilization_percent: float | None
    memory_utilization_percent: float | None
    power_watts: float | None
    guid: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of GPU metrics."""
        return {
            "index": self.index,
            "model": self.model,
            "utilization_percent": self.utilization_percent,
            "memory_utilization_percent": self.memory_utilization_percent,
            "power_watts": self.power_watts,
            "guid": self.guid,
        }


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource metrics."""

    timestamp: float
    cpu_utilization_percent: float | None
    cpu_power_watts: float | None
    total_memory_bytes: int
    available_memory_bytes: int
    wired_memory_bytes: int | None
    memory_utilization_percent: float | None
    gpu_usage: list[GPUUsage]

    def to_dict(self) -> dict[str, Any]:
        """Convert the snapshot to a JSON-serializable dictionary."""
        return {
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "cpu_utilization_percent": self.cpu_utilization_percent,
            "cpu_power_watts": self.cpu_power_watts,
            "total_memory_bytes": self.total_memory_bytes,
            "available_memory_bytes": self.available_memory_bytes,
            "wired_memory_bytes": self.wired_memory_bytes,
            "memory_utilization_percent": self.memory_utilization_percent,
            "gpu_usage": [gpu.to_dict() for gpu in self.gpu_usage],
        }


SnapshotListener = Callable[[ResourceSnapshot], None]


class SystemMonitorDaemon:
    """Daemon that polls CPU, memory, and GPU metrics at fixed intervals."""

    def __init__(
        self,
        interval_seconds: float = 1.0,
        include_gpu: bool = True,
        snapshot_listener: SnapshotListener | None = None,
    ):
        """Create a monitor daemon thread.

        Args:
            interval_seconds: Sampling interval in seconds. Must be positive.
            include_gpu: Whether to collect NVIDIA GPU metrics via NVML.
            snapshot_listener: Optional callback invoked for each snapshot.
        """
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")

        self.interval_seconds = interval_seconds
        self.include_gpu = include_gpu
        self._snapshot_listener = snapshot_listener

        self._stop_event = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread_started = False
        self._nvml_ready = include_gpu and _ensure_nvml_initialized()
        self._latest_snapshot: ResourceSnapshot | None = None

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread_started:
            return
        self._thread_started = True
        self._thread.start()

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        if not self._thread_started:
            return
        self._stop_event.set()
        self._thread.join()

    def get_latest_snapshot(self) -> ResourceSnapshot | None:
        """Return the most recent resource snapshot.

        Returns
        -------
            Latest ResourceSnapshot or None if no snapshot has been collected.
        """
        with self._snapshot_lock:
            return self._latest_snapshot

    def _run(self) -> None:
        psutil.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            start = time.time()
            snapshot = self._collect_snapshot()
            with self._snapshot_lock:
                self._latest_snapshot = snapshot
            if self._snapshot_listener:
                self._snapshot_listener(snapshot)
            elapsed = time.time() - start
            remaining = self.interval_seconds - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

    def _collect_snapshot(self) -> ResourceSnapshot:
        cpu_utilization = psutil.cpu_percent(interval=None)
        virtual_memory = psutil.virtual_memory()

        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_utilization_percent=cpu_utilization,
            cpu_power_watts=None,
            total_memory_bytes=virtual_memory.total,
            available_memory_bytes=virtual_memory.available,
            wired_memory_bytes=getattr(virtual_memory, "wired", None),
            memory_utilization_percent=virtual_memory.percent,
            gpu_usage=self._collect_gpu_usage(),
        )

    def _collect_gpu_usage(self) -> list[GPUUsage]:
        if not self.include_gpu or not self._nvml_ready or not pynvml:
            return []

        stats: list[GPUUsage] = []

        try:
            device_count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            return []

        for index in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                model = pynvml.nvmlDeviceGetName(handle)
                guid = pynvml.nvmlDeviceGetUUID(handle)
            except pynvml.NVMLError:
                continue

            if isinstance(model, bytes):
                model = model.decode(errors="ignore").strip()

            if isinstance(guid, bytes):
                guid = guid.decode(errors="ignore").strip()

            stats.append(
                GPUUsage(
                    index=index,
                    model=model,
                    utilization_percent=float(util.gpu),
                    memory_utilization_percent=float(util.memory),
                    power_watts=float(power_mw) / 1000.0,
                    guid=guid,
                )
            )

        return stats


def get_gpu_guid(index: int) -> str | None:
    """Return the GUID for the requested NVIDIA GPU.

    Args:
        index: GPU index (zero-based).

    Returns
    -------
        GPU GUID string if available, otherwise None.
    """
    if not pynvml or not _ensure_nvml_initialized():
        return None

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        guid = pynvml.nvmlDeviceGetUUID(handle)
    except pynvml.NVMLError:
        return None

    if isinstance(guid, bytes):
        guid_str = guid.decode(errors="ignore").strip()
    else:
        guid_str = str(guid).strip()

    return guid_str or None
