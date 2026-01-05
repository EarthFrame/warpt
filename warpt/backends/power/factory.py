"""Factory for creating platform-appropriate power monitors."""

from __future__ import annotations

import platform
import threading
import time
from typing import TYPE_CHECKING

import psutil

from warpt.backends.power.base import PowerBackend
from warpt.backends.power.linux_rapl import LinuxRAPLBackend
from warpt.backends.power.macos_power import MacOSPowerBackend
from warpt.backends.power.nvidia_power import NvidiaPowerBackend
from warpt.models.power_models import (
    DomainPower,
    GPUPowerInfo,
    PowerDomain,
    PowerSnapshot,
    PowerSource,
    ProcessPower,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class PowerMonitor:
    """Unified power monitor that combines all available backends.

    Automatically detects and uses available power sources:
    - Linux: RAPL for CPU, NVML for NVIDIA GPUs
    - macOS: powermetrics for CPU/GPU, NVML for discrete NVIDIA GPUs
    - Windows: NVML for GPUs (CPU power not directly available)

    Provides system-wide measurements, per-component breakdown, and
    per-process power attribution.
    """

    def __init__(self, include_process_attribution: bool = True) -> None:
        """Initialize the power monitor.

        Args:
            include_process_attribution: Whether to calculate per-process power.
        """
        self._platform = platform.system()
        self._include_process_attribution = include_process_attribution
        self._backends: list[PowerBackend] = []
        self._nvidia_backend: NvidiaPowerBackend | None = None
        self._initialized = False
        self._lock = threading.Lock()

        # For process CPU tracking: pid -> (cpu_time, wall_time)
        self._last_process_times: dict[int, tuple[float, float]] = {}

    def initialize(self) -> bool:
        """Initialize all available power backends.

        Returns:
            True if at least one backend was initialized.
        """
        if self._initialized:
            return bool(self._backends)

        self._backends = []

        # Platform-specific CPU power backend
        if self._platform == "Linux":
            rapl = LinuxRAPLBackend()
            if rapl.is_available():
                rapl.initialize()
                self._backends.append(rapl)

        elif self._platform == "Darwin":
            macos = MacOSPowerBackend()
            if macos.is_available():
                macos.initialize()
                self._backends.append(macos)

        # NVIDIA GPU backend (cross-platform)
        nvidia = NvidiaPowerBackend()
        if nvidia.is_available():
            nvidia.initialize()
            self._backends.append(nvidia)
            self._nvidia_backend = nvidia

        self._initialized = True
        return bool(self._backends)

    def get_available_sources(self) -> list[PowerSource]:
        """Get list of available power sources.

        Returns:
            List of PowerSource enum values.
        """
        if not self._initialized:
            self.initialize()

        return [backend.get_source() for backend in self._backends]

    def get_snapshot(self) -> PowerSnapshot:
        """Get a complete power snapshot.

        Returns:
            PowerSnapshot with all available measurements.
        """
        if not self._initialized:
            self.initialize()

        timestamp = time.time()
        domains: list[DomainPower] = []
        gpus: list[GPUPowerInfo] = []

        # Collect readings from all backends
        for backend in self._backends:
            readings = backend.get_power_readings()
            domains.extend(readings)

        # Get detailed GPU info if NVIDIA backend available
        if self._nvidia_backend:
            gpus = self._nvidia_backend.get_gpu_power_info()

        # Calculate total power
        total_power = self._calculate_total_power(domains, gpus)

        # Get per-process attribution
        processes: list[ProcessPower] = []
        if self._include_process_attribution:
            cpu_power = self._get_cpu_power_from_domains(domains)
            processes = self._get_process_power_attribution(cpu_power)

        return PowerSnapshot(
            timestamp=timestamp,
            total_power_watts=total_power,
            domains=domains,
            gpus=gpus,
            processes=processes,
            platform=self._platform,
            available_sources=self.get_available_sources(),
        )

    def _calculate_total_power(
        self,
        domains: list[DomainPower],
        gpus: list[GPUPowerInfo],
    ) -> float | None:
        """Calculate total system power from available measurements.

        Args:
            domains: CPU/system power domains.
            gpus: GPU power info (discrete GPUs not in package).

        Returns:
            Total power in watts or None if not measurable.
        """
        total = 0.0
        has_measurement = False

        # Sum CPU package power (or core if package not available)
        package_power = None
        core_power = None
        for domain in domains:
            if domain.domain == PowerDomain.PACKAGE:
                package_power = domain.power_watts
            elif domain.domain == PowerDomain.CORE:
                core_power = domain.power_watts

        if package_power is not None:
            total += package_power
            has_measurement = True
        elif core_power is not None:
            total += core_power
            has_measurement = True

        # Add discrete GPU power (integrated GPU already in package on some systems)
        for gpu in gpus:
            # Only count if this is a discrete GPU (not integrated)
            if not gpu.metadata.get("integrated", False):
                total += gpu.power_watts
                has_measurement = True

        return total if has_measurement else None

    def _get_cpu_power_from_domains(self, domains: list[DomainPower]) -> float:
        """Extract CPU power from domain readings.

        Args:
            domains: List of domain power readings.

        Returns:
            CPU power in watts (0 if not available).
        """
        for domain in domains:
            if domain.domain == PowerDomain.PACKAGE:
                return domain.power_watts
            elif domain.domain == PowerDomain.CORE:
                return domain.power_watts
        return 0.0

    def _get_process_power_attribution(
        self, total_cpu_power: float
    ) -> list[ProcessPower]:
        """Calculate per-process power attribution.

        Power is attributed based on CPU utilization (for CPU power)
        and GPU memory usage (for GPU power).

        Args:
            total_cpu_power: Total CPU power consumption.

        Returns:
            List of ProcessPower objects sorted by total power.
        """
        processes: list[ProcessPower] = []

        # Get GPU process usage if available
        gpu_process_usage: dict[int, dict] = {}
        if self._nvidia_backend:
            gpu_process_usage = self._nvidia_backend.get_process_gpu_usage()

        # Get all process CPU usage
        try:
            proc_attrs = ["pid", "name", "cpu_percent", "memory_info"]
            all_procs = list(psutil.process_iter(proc_attrs))
        except Exception:
            all_procs = []

        # Calculate total CPU utilization for normalization
        total_cpu_percent = sum((p.info.get("cpu_percent") or 0) for p in all_procs)

        for proc in all_procs:
            try:
                info = proc.info
                pid = info["pid"]
                name = info["name"] or f"pid_{pid}"
                cpu_percent = info.get("cpu_percent") or 0

                # Skip idle processes
                if cpu_percent < 0.1 and pid not in gpu_process_usage:
                    continue

                # Calculate CPU power attribution
                cpu_power = 0.0
                if total_cpu_percent > 0 and cpu_percent > 0:
                    cpu_power = total_cpu_power * (cpu_percent / total_cpu_percent)

                # Get GPU power attribution
                gpu_power = 0.0
                gpu_percent = 0.0
                if pid in gpu_process_usage:
                    gpu_info = gpu_process_usage[pid]
                    gpu_power = gpu_info.get("estimated_power_watts", 0)
                    gpu_percent = gpu_info.get("estimated_gpu_util", 0)

                # Get memory usage
                memory_mb = 0.0
                mem_info = info.get("memory_info")
                if mem_info:
                    memory_mb = mem_info.rss / (1024 * 1024)

                processes.append(
                    ProcessPower(
                        pid=pid,
                        name=name,
                        cpu_power_watts=cpu_power,
                        gpu_power_watts=gpu_power,
                        cpu_percent=cpu_percent,
                        gpu_percent=gpu_percent,
                        memory_mb=memory_mb,
                    )
                )

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort by total power descending
        processes.sort(key=lambda p: p.total_power_watts, reverse=True)

        return processes[:50]  # Top 50 processes

    def cleanup(self) -> None:
        """Clean up all backends."""
        for backend in self._backends:
            backend.cleanup()
        self._backends = []
        self._nvidia_backend = None
        self._initialized = False


class PowerMonitorDaemon:
    """Background daemon that continuously monitors power."""

    def __init__(
        self,
        interval_seconds: float = 1.0,
        include_process_attribution: bool = True,
        snapshot_listener: Callable[[PowerSnapshot], None] | None = None,
    ) -> None:
        """Create a power monitor daemon.

        Args:
            interval_seconds: Sampling interval in seconds.
            include_process_attribution: Whether to calculate per-process power.
            snapshot_listener: Optional callback for each snapshot.
        """
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")

        self.interval_seconds = interval_seconds
        self._monitor = PowerMonitor(
            include_process_attribution=include_process_attribution
        )
        self._snapshot_listener = snapshot_listener
        self._stop_event = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread_started = False
        self._latest_snapshot: PowerSnapshot | None = None

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread_started:
            return
        self._monitor.initialize()
        self._thread_started = True
        self._thread.start()

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        if not self._thread_started:
            return
        self._stop_event.set()
        self._thread.join()
        self._monitor.cleanup()

    def get_latest_snapshot(self) -> PowerSnapshot | None:
        """Get the most recent power snapshot.

        Returns:
            Latest PowerSnapshot or None if no snapshot has been collected.
        """
        with self._snapshot_lock:
            return self._latest_snapshot

    def _run(self) -> None:
        """Background collection loop."""
        while not self._stop_event.is_set():
            start = time.time()
            snapshot = self._monitor.get_snapshot()

            with self._snapshot_lock:
                self._latest_snapshot = snapshot

            if self._snapshot_listener:
                self._snapshot_listener(snapshot)

            elapsed = time.time() - start
            remaining = self.interval_seconds - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)


class PowerMonitorFactory:
    """Factory for creating power monitors."""

    @staticmethod
    def create() -> PowerMonitor:
        """Create a power monitor for the current platform.

        Returns:
            Initialized PowerMonitor instance.
        """
        monitor = PowerMonitor()
        monitor.initialize()
        return monitor

    @staticmethod
    def create_daemon(
        interval_seconds: float = 1.0,
        include_processes: bool = True,
    ) -> PowerMonitorDaemon:
        """Create a background power monitoring daemon.

        Args:
            interval_seconds: Sampling interval.
            include_processes: Whether to calculate per-process power.

        Returns:
            PowerMonitorDaemon instance (not yet started).
        """
        return PowerMonitorDaemon(
            interval_seconds=interval_seconds,
            include_process_attribution=include_processes,
        )


def create_power_monitor() -> PowerMonitor:
    """Create and return an initialized power monitor.

    Returns:
        Initialized PowerMonitor instance.
    """
    return PowerMonitorFactory.create()
