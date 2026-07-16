"""CarbonTracker context manager for automatic energy tracking.

Energy is read exclusively from the out-of-process Rust power-daemon: warpt no
longer measures power itself, it asks the daemon and trusts the answer. If the
daemon is unreachable at start, tracking is disabled (the wrapped workload still
runs). If the daemon drops mid-run, tracking is terminated and the session is
recorded with a ``terminated`` status rather than a guessed energy number.
"""

from __future__ import annotations

import platform
import sys
import threading
import time
import uuid

from warpt.backends.power.daemon_client import (
    PowerClient,
    PowerClientError,
    PowerReading,
    counter_delta_joules,
)
from warpt.backends.power.factory import PowerMonitor
from warpt.carbon.calculator import CarbonCalculator
from warpt.carbon.store import EnergyStore
from warpt.models.carbon_models import CarbonSession
from warpt.utils.logger import Logger

# Mid-run reconnect policy: one retry after a short pause before giving up.
_RECONNECT_RETRIES = 1
_RECONNECT_PAUSE_S = 0.5
_LOG = "carbon.tracker"

_TERMINATED_LOST = (
    "terminated due to loss of connection, please try connection to daemon again"
)
_TERMINATED_RESET = (
    "terminated: daemon counter reset mid-session "
    "(daemon restarted), please re-run with a stable daemon"
)


def _warn(message: str) -> None:
    """Emit a WARNING (transient) via the logger, falling back to stderr."""
    if Logger.is_configured():
        Logger.get(_LOG).warning(message)
    else:
        print(f"[warpt] WARNING {message}", file=sys.stderr)


def _error(message: str) -> None:
    """Emit an ERROR (terminal) via the logger, falling back to stderr."""
    if Logger.is_configured():
        Logger.get(_LOG).error(message)
    else:
        print(f"[warpt] ERROR {message}", file=sys.stderr)


class CarbonTracker:
    """Context manager that tracks energy via the Rust power-daemon.

    Wraps existing command logic. If the daemon is unavailable, it becomes a
    silent no-op for tracking purposes and never interferes with the workload.

    Parameters
    ----------
    label : str
        Human-readable label for the session (e.g. "warpt stress").
    interval : float
        Sampling interval in seconds (for avg/peak power display only).
    region : str
        Grid region for CO2 calculation.

    Examples
    --------
    >>> with CarbonTracker(label="warpt stress"):
    ...     # run some workload
    ...     pass
    """

    def __init__(
        self,
        label: str,
        interval: float = 1.0,
        region: str = "US",
    ) -> None:
        self._label = label
        self._interval = interval
        self._region = region
        self._session_id = str(uuid.uuid4())
        self._monitor: PowerMonitor | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float, float, float, float]] = []
        self._sources: list[str] = ["daemon"]
        self._noop = False
        self._daemon_lost = False
        self._daemon_client: PowerClient | None = None
        self._start_daemon_reading: PowerReading | None = None

    def __enter__(self) -> CarbonTracker:
        """Gate on the daemon, snapshot the start counter, then start sampling."""
        try:
            self._monitor = PowerMonitor()
            if not self._monitor.initialize():
                # Daemon unreachable after retry → disable tracking, NOT the
                # workload. initialize() already warned on the retry attempt.
                _error("energy tracking disabled: power-daemon not reachable")
                self._noop = True
                return self
        except Exception:
            self._noop = True
            return self

        # Snapshot the daemon's energy counter at the start of the session.
        try:
            self._daemon_client = PowerClient()
            self._start_daemon_reading = self._daemon_client.current()
        except PowerClientError:
            _error("energy tracking disabled: could not read daemon counter")
            self._noop = True
            return self

        # Create the session record now that we know the daemon is live.
        self._session = CarbonSession(
            id=self._session_id,
            label=self._label,
            start_time=time.time(),
            region=self._region,
            platform=platform.system().lower(),
            sources=self._sources,
        )
        EnergyStore().create_session(self._session)

        # Background sampling (for avg/peak power display; energy comes from the
        # counter delta, not these samples).
        self._running = True
        self._thread = threading.Thread(
            target=self._sample_loop, daemon=True, name="carbon-tracker"
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop sampling, compute energy from the daemon counter, finalize."""
        if self._noop:
            return

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

        # If the daemon dropped mid-run, record a terminated session — never a
        # guessed energy number.
        if self._daemon_lost:
            self._finalize_terminated(_TERMINATED_LOST)
            return

        delta_j, failure = self._daemon_counter_delta()
        if failure is not None:
            self._finalize_terminated(failure)
            return

        calc = CarbonCalculator(region=self._region)
        energy_kwh = calc.energy_from_counter(delta_j)
        co2_grams = calc.co2_from_energy(energy_kwh)
        cost_usd = calc.cost_from_energy(energy_kwh)

        end_time = time.time()
        duration_s = end_time - self._session.start_time
        powers = [w for _, w, _, _ in self._samples]
        avg_power = sum(powers) / len(powers) if powers else 0.0
        peak_power = max(powers) if powers else 0.0

        self._session.end_time = end_time
        self._session.duration_s = duration_s
        self._session.energy_kwh = energy_kwh
        self._session.co2_grams = co2_grams
        self._session.cost_usd = cost_usd
        self._session.metadata = {
            "avg_power_w": round(avg_power, 2),
            "peak_power_w": round(peak_power, 2),
            "sample_count": len(self._samples),
            "energy_source": "daemon-counter",
            "status": "completed",
        }
        self._session.samples = [
            {
                "timestamp": t,
                "power_watts": round(w, 2),
                "cpu_watts": round(c, 2),
                "gpu_watts": round(g, 2),
            }
            for t, w, c, g in self._samples
        ]
        EnergyStore().update_session(self._session)

        humanized = calc.humanize(co2_grams)
        print(
            f"\n[carbon] {duration_s:.1f}s | "
            f"{avg_power:.1f}W avg | "
            f"{energy_kwh * 1_000_000:.1f} mWh | "
            f"{co2_grams:.2f}g CO2 | "
            f"${cost_usd:.4f} | "
            f"{humanized} | "
            f"via rust daemon counter",
            file=sys.stderr,
        )

    def _finalize_terminated(self, status: str) -> None:
        """Record a session that ended because the daemon connection failed."""
        end_time = time.time()
        powers = [w for _, w, _, _ in self._samples]
        avg_power = sum(powers) / len(powers) if powers else 0.0
        peak_power = max(powers) if powers else 0.0

        self._session.end_time = end_time
        self._session.duration_s = end_time - self._session.start_time
        self._session.metadata = {
            "avg_power_w": round(avg_power, 2),
            "peak_power_w": round(peak_power, 2),
            "sample_count": len(self._samples),
            "energy_source": "daemon-counter",
            "status": status,
        }
        EnergyStore().update_session(self._session)
        _error(f"energy tracking {status}")

    def _daemon_counter_delta(self) -> tuple[float, str | None]:
        """Return (energy_joules, None) on success, or (0.0, status) on failure.

        Reads the daemon counter one last time and subtracts the start reading.
        A failed read or a counter reset (daemon restarted) is a terminal
        failure, not something to estimate around.
        """
        if self._daemon_client is None or self._start_daemon_reading is None:
            return 0.0, _TERMINATED_LOST
        try:
            end = self._daemon_client.current()
        except PowerClientError:
            return 0.0, _TERMINATED_LOST

        delta = counter_delta_joules(self._start_daemon_reading, end)
        if delta is None:
            return 0.0, _TERMINATED_RESET
        return delta, None

    def _reconnect(self) -> bool:
        """Try to reconnect to the daemon once after a short pause."""
        if self._daemon_client is None:
            return False
        for attempt in range(_RECONNECT_RETRIES):
            _warn(
                f"power-daemon read failed, retrying in {_RECONNECT_PAUSE_S}s "
                f"({attempt + 1}/{_RECONNECT_RETRIES})"
            )
            time.sleep(_RECONNECT_PAUSE_S)
            if self._daemon_client.healthz():
                return True
        return False

    def _sample_loop(self) -> None:
        """Sample daemon power at the configured interval (avg/peak display)."""
        while self._running:
            recorded = False
            try:
                if self._monitor is None:
                    break
                snapshot = self._monitor.get_snapshot()
                total = snapshot.total_power_watts
                if total is not None and total > 0:
                    cpu = snapshot.get_cpu_power() or 0.0
                    gpu = snapshot.get_gpu_power()
                    self._samples.append((snapshot.timestamp, total, cpu, gpu))
                    recorded = True
            except Exception:
                recorded = False

            # A read that returned nothing may be a transient blip or a real
            # loss — retry once, and terminate tracking if it doesn't recover.
            if not recorded and not self._reconnect():
                self._daemon_lost = True
                _error("power-daemon connection lost — energy tracking terminated")
                break

            time.sleep(self._interval)
