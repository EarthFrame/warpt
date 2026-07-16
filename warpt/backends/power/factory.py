"""Daemon-backed power monitor.

The out-of-process Rust ``power-daemon`` is the single source of power/energy
readings. There is no native fallback: if the daemon is not reachable, the
monitor has no backend and reports nothing. Callers decide what to do with that
(disable tracking, error out, etc.).
"""

from __future__ import annotations

import platform
import sys
import threading
import time

from warpt.backends.power.daemon_source import DaemonPowerBackend
from warpt.models.power_models import (
    DomainPower,
    GPUPowerInfo,
    PowerSnapshot,
    PowerSource,
)
from warpt.utils.logger import Logger

# Health-check retry policy for the power-daemon: one retry after a short pause.
_HEALTH_RETRIES = 1
_HEALTH_RETRY_PAUSE_S = 0.5
_LOG = "power.monitor"

_DAEMON_UNREACHABLE = (
    "power-daemon not reachable — start it (or set POWER_DAEMON_URL). "
    "warpt now reads power exclusively from the daemon; there is no native "
    "fallback."
)


def _warn(message: str) -> None:
    """Emit a WARNING via the logger, falling back to stderr if unconfigured."""
    if Logger.is_configured():
        Logger.get(_LOG).warning(message)
    else:
        print(f"[warpt] WARNING {message}", file=sys.stderr)


def _daemon_available_with_retry(daemon: DaemonPowerBackend) -> bool:
    """Health-check the daemon, retrying once after a short pause.

    Each failed attempt is a WARNING (transient). Whether giving up is an
    ERROR is left to the caller, which knows the user-facing context.
    """
    for attempt in range(_HEALTH_RETRIES + 1):
        if daemon.is_available():
            return True
        if attempt < _HEALTH_RETRIES:
            _warn(
                f"power-daemon unreachable, retrying in {_HEALTH_RETRY_PAUSE_S}s "
                f"({attempt + 1}/{_HEALTH_RETRIES})"
            )
            time.sleep(_HEALTH_RETRY_PAUSE_S)
    return False


class PowerMonitor:
    """Power monitor backed solely by the out-of-process Rust power-daemon."""

    def __init__(self) -> None:
        self._platform = platform.system()
        self._daemon_backend: DaemonPowerBackend | None = None
        self._initialized = False
        self._lock = threading.Lock()
        self._unavailable_reason: str | None = None

    def initialize(self) -> bool:
        """Health-check the daemon (with one retry) and adopt it as the source.

        Returns:
            True if the daemon is reachable, False otherwise (no native fallback).
        """
        if self._initialized:
            return self._daemon_backend is not None

        daemon = DaemonPowerBackend()
        if _daemon_available_with_retry(daemon):
            self._daemon_backend = daemon
            self._unavailable_reason = None
        else:
            self._daemon_backend = None
            self._unavailable_reason = _DAEMON_UNREACHABLE
        self._initialized = True
        return self._daemon_backend is not None

    def get_unavailable_reasons(self) -> list[str]:
        """Human-readable reason the daemon is unavailable (empty if it's up)."""
        return [self._unavailable_reason] if self._unavailable_reason else []

    def get_available_sources(self) -> list[PowerSource]:
        """Return the active power source(s): just the daemon, or none."""
        if not self._initialized:
            self.initialize()
        return [PowerSource.DAEMON] if self._daemon_backend is not None else []

    def is_daemon_active(self) -> bool:
        """Return True if the power-daemon is the active source."""
        return self._daemon_backend is not None

    def get_snapshot(self) -> PowerSnapshot:
        """Get a complete power snapshot from the daemon (one fetch)."""
        if not self._initialized:
            self.initialize()

        timestamp = time.time()
        domains: list[DomainPower] = []
        gpus: list[GPUPowerInfo] = []
        total_power: float | None = None

        if self._daemon_backend is not None:
            # One daemon fetch yields readings, GPU info, and the authoritative
            # total (which includes components warpt doesn't model as domains).
            domains, gpus, total_power = self._daemon_backend.read_snapshot()

        return PowerSnapshot(
            timestamp=timestamp,
            total_power_watts=total_power,
            domains=domains,
            gpus=gpus,
            processes=[],
            platform=self._platform,
            available_sources=self.get_available_sources(),
        )

    def cleanup(self) -> None:
        """Release the daemon backend."""
        if self._daemon_backend is not None:
            self._daemon_backend.cleanup()
        self._daemon_backend = None
        self._initialized = False


def create_power_monitor() -> PowerMonitor:
    """Create and return an initialized daemon-backed power monitor."""
    monitor = PowerMonitor()
    monitor.initialize()
    return monitor
