"""Power monitoring daemon for background sampling."""

import platform
import threading
import time
from typing import Any

from warpt.backends.power.base import BasePowerSource
from warpt.utils.logger import Logger


class PowerMonitorDaemon:
    """Daemon that samples power sources in the background.

    Collects power samples and calculates total energy consumption (Joules)
    and average power (Watts) over a period of time.
    """

    def __init__(self, interval: float = 0.1) -> None:
        """Initialize the power monitor daemon.

        Args:
            interval: Sampling interval in seconds (default 0.1s / 10Hz).
        """
        self._interval = interval
        self._sources: list[BasePowerSource] = []
        self._samples: dict[str, list[float]] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._logger = Logger.get("power.daemon")
        self._start_time: float | None = None
        self._end_time: float | None = None

        self._discover_sources()

    def _discover_sources(self) -> None:
        """Discover available local power sources.

        Attempts to discover power sources for the current platform.
        On macOS: Apple Silicon power metrics
        On Linux: Intel/AMD RAPL counters
        On any platform: NVIDIA GPU power via NVML
        """
        src: BasePowerSource

        # Apple Silicon (macOS)
        if platform.system() == "Darwin":
            from warpt.backends.power.sources import AppleSiliconPowerSource

            src = AppleSiliconPowerSource()
            if src.check_permissions():
                self._sources.append(src)
                self._logger.info(f"Discovered power source: {src.name}")
            else:
                self._logger.debug(
                    "Apple Silicon power source unavailable "
                    "(requires sudo or permissions)"
                )

        # Intel/AMD RAPL (Linux only)
        if platform.system() == "Linux":
            from warpt.backends.power.sources import RAPLPowerSource

            rapl_src = RAPLPowerSource()
            # Verify we can actually read the counters (requires kernel support)
            if rapl_src._energy_files and rapl_src.check_permissions():
                self._sources.append(rapl_src)
                self._logger.info(f"Discovered power source: {rapl_src.name}")
            else:
                self._logger.debug(
                    "RAPL power source unavailable "
                    "(requires kernel support or permissions)"
                )

        # NVIDIA GPUs (cross-platform)
        try:
            from warpt.backends.nvidia import NvidiaBackend

            nv = NvidiaBackend()
            if nv.is_available():
                from warpt.backends.power.sources import NvidiaPowerSource

                for i in range(nv.get_device_count()):
                    src = NvidiaPowerSource(i)
                    if src.check_permissions():
                        self._sources.append(src)
                        self._logger.info(f"Discovered power source: {src.name}")
        except Exception as e:
            self._logger.debug(f"NVIDIA power source unavailable: {e}")

        # Initialize sample storage for all discovered sources
        for src in self._sources:
            self._samples[src.name] = []

        if not self._sources:
            self._logger.warning(
                "No power sources discovered. Power monitoring will be unavailable."
            )

    def start(self) -> None:
        """Start the monitoring thread."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._end_time = None
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self._logger.info(
            f"Power monitor daemon started with {len(self._sources)} sources"
        )

    def check_permissions(self) -> bool:
        """Verify that all discovered sources have sufficient permissions.

        Returns
        -------
            True if all sources are authorized, False if any require intervention.
        """
        if not self._sources:
            return True

        for src in self._sources:
            if not src.check_permissions():
                self._logger.error(f"Permission denied for power source: {src.name}")
                return False
        return True

    def stop(self) -> None:
        """Stop the monitoring thread."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._end_time = time.time()
        self._logger.info("Power monitor daemon stopped")

    def _monitor_loop(self) -> None:
        """Background loop for sampling power sources."""
        while self._running:
            start_tick = time.time()
            for src in self._sources:
                p = src.get_power_w()
                if p is not None:
                    self._samples[src.name].append(p)

            # Wait for next interval, accounting for sampling time
            elapsed = time.time() - start_tick
            sleep_time = max(0, self._interval - elapsed)
            time.sleep(sleep_time)

    def get_results(self) -> dict[str, Any]:
        """Calculate and return power/energy results.

        Returns
        -------
            Dictionary with energy (Joules), average power (Watts), and source details.
        """
        results: dict[str, Any] = {
            "sources": {},
            "total_energy_j": 0.0,
            "avg_power_w": 0.0,
            "duration_s": 0.0,
        }

        if self._start_time is None:
            return results

        duration = (self._end_time or time.time()) - self._start_time
        results["duration_s"] = duration

        total_avg_power = 0.0
        for name, samples in self._samples.items():
            if not samples:
                continue

            avg_p = sum(samples) / len(samples)
            energy = avg_p * duration  # Joules = Watts * Seconds
            results["sources"][name] = {
                "avg_power_w": avg_p,
                "energy_j": energy,
                "sample_count": len(samples),
            }
            results["total_energy_j"] += energy
            total_avg_power += avg_p

        results["avg_power_w"] = total_avg_power
        return results
