"""Continuous lifetime energy odometer backed by the Rust power-daemon.

Unlike ``CarbonTracker`` (a session "trip meter" that measures one wrapped
workload and terminates on counter resets), this is an **odometer**: a
running lifetime total that survives both warpt-daemon restarts and Rust
power-daemon restarts. On a counter reset it re-baselines and keeps the
accumulated total — it never guesses at unobserved energy, so gaps where
the counter reset mid-outage are undercounted, not estimated.

The cumulative total persists to ``{warpt_dir}/energy_odometer.json`` so the
number keeps climbing across restarts. Totals count energy observed since
tracking first began on the node.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from warpt.backends.power.daemon_client import (
    PowerClient,
    PowerClientError,
    PowerReading,
    counter_delta_joules,
)
from warpt.carbon.calculator import CarbonCalculator
from warpt.utils.logger import Logger

STATE_FILENAME = "energy_odometer.json"
DEFAULT_POLL_INTERVAL_S = 60.0


class ContinuousEnergyTracker:
    """Thread-safe lifetime energy/CO2/cost odometer for one node.

    Parameters
    ----------
    warpt_dir
        Directory for the persisted odometer state (e.g. ``~/.warpt``).
    region
        Grid region code for CO2 intensity lookup.
    cost_per_kwh
        Electricity rate in USD per kWh.
    client
        Optional ``PowerClient`` override (for tests).
    """

    def __init__(
        self,
        warpt_dir: str,
        region: str = "US",
        cost_per_kwh: float = 0.12,
        client: PowerClient | None = None,
    ) -> None:
        self._state_path = Path(warpt_dir).expanduser() / STATE_FILENAME
        self._client = client or PowerClient()
        self._calc = CarbonCalculator(region=region)
        self._cost_per_kwh = cost_per_kwh
        self._lock = threading.Lock()
        self._log = Logger.get("carbon.continuous")

        self._cumulative_joules = 0.0
        self._last_reading: PowerReading | None = None
        self._load_state()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------ read

    def read(self) -> dict[str, Any]:
        """Advance the odometer from the daemon counter and return totals.

        Reads the daemon's cumulative counter, credits the delta since the
        last read (re-baselining without crediting when the counter reset or
        went backwards), persists, and returns the running totals. If the
        daemon is unreachable the last-known totals are returned unchanged —
        this never raises.

        Returns
        -------
            ``{"energy_kwh", "co2_grams", "cost_usd", "daemon_reachable"}``
        """
        with self._lock:
            try:
                now = self._client.current()
            except PowerClientError:
                return self._totals(daemon_reachable=False)
            except Exception:
                # Malformed daemon payload — treat like unreachable.
                self._log.debug("Unexpected power-daemon read failure", exc_info=True)
                return self._totals(daemon_reachable=False)

            if self._last_reading is not None:
                delta = counter_delta_joules(self._last_reading, now)
                if delta is None:
                    # Daemon restarted (or counter went backwards): keep the
                    # running total, start counting from the fresh baseline.
                    self._log.info(
                        "power-daemon counter reset detected; re-baselining "
                        "(cumulative total preserved)"
                    )
                else:
                    self._cumulative_joules += delta

            self._last_reading = now
            self._save_state()
            return self._totals(daemon_reachable=True)

    def last_totals(self) -> dict[str, Any]:
        """Return the running totals without contacting the daemon.

        For latency-sensitive callers (the health endpoint) that want the
        odometer value but must not block on an HTTP read.

        Returns
        -------
            ``{"energy_kwh", "co2_grams", "cost_usd"}``
        """
        with self._lock:
            totals = self._totals(daemon_reachable=True)
        totals.pop("daemon_reachable")
        return totals

    def _totals(self, daemon_reachable: bool) -> dict[str, Any]:
        """Convert the cumulative joules into the reporting dict."""
        energy_kwh = self._calc.energy_from_counter(self._cumulative_joules)
        return {
            "energy_kwh": round(energy_kwh, 6),
            "co2_grams": round(self._calc.co2_from_energy(energy_kwh), 2),
            "cost_usd": round(
                self._calc.cost_from_energy(energy_kwh, rate=self._cost_per_kwh), 4
            ),
            "daemon_reachable": daemon_reachable,
        }

    # ------------------------------------------------------------ background

    def start(self, interval_s: float = DEFAULT_POLL_INTERVAL_S) -> None:
        """Start a background thread that advances the odometer periodically.

        Keeps the persisted total fresh even when nothing else calls
        ``read()``, and bounds how much energy a Rust-daemon restart can
        drop to one polling interval.

        Parameters
        ----------
        interval_s
            Seconds between odometer reads.
        """
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            args=(interval_s,),
            name="energy-odometer",
            daemon=True,
        )
        self._thread.start()
        self._log.info("Energy odometer started (interval=%.0fs)", interval_s)

    def stop(self) -> None:
        """Stop the background poll thread (final state already persisted)."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
        self._log.info("Energy odometer stopped.")

    def _poll_loop(self, interval_s: float) -> None:
        while not self._stop_event.is_set():
            try:
                self.read()
            except Exception:
                # Belt and braces: read() shouldn't raise, but the odometer
                # must never take the daemon down.
                self._log.exception("Odometer poll failed")
            self._stop_event.wait(interval_s)

    # ----------------------------------------------------------- persistence

    def _load_state(self) -> None:
        """Load persisted odometer state; start fresh on any problem."""
        try:
            data = json.loads(self._state_path.read_text())
            self._cumulative_joules = float(data["cumulative_joules"])
            self._last_reading = PowerReading(
                timestamp=0.0,
                watts=0.0,
                joules_since_reset=float(data["last_joules_since_reset"]),
                watt_hours_since_reset=0.0,
                reset_time=float(data["last_reset_time"]),
                hostname="",
            )
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            self._log.warning(
                "Corrupt odometer state at %s; starting fresh", self._state_path
            )
            self._cumulative_joules = 0.0
            self._last_reading = None

    def _save_state(self) -> None:
        """Atomically persist the cumulative total and last counter reading."""
        assert self._last_reading is not None
        state = {
            "cumulative_joules": self._cumulative_joules,
            "last_joules_since_reset": self._last_reading.joules_since_reset,
            "last_reset_time": self._last_reading.reset_time,
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=self._state_path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(state, f)
                os.replace(tmp_path, self._state_path)
            except BaseException:
                os.unlink(tmp_path)
                raise
        except OSError:
            self._log.warning("Could not persist odometer state", exc_info=True)


def peek_persisted_totals(
    warpt_dir: str, region: str = "US", cost_per_kwh: float = 0.12
) -> dict[str, Any] | None:
    """Read the last persisted odometer totals without contacting the daemon.

    For out-of-process callers (``warpt daemon status``) that must not race
    the live daemon's odometer or its state file.

    Parameters
    ----------
    warpt_dir
        The warpt data directory.
    region
        Grid region code for CO2 intensity lookup.
    cost_per_kwh
        Electricity rate in USD per kWh.

    Returns
    -------
        ``{"energy_kwh", "co2_grams", "cost_usd"}`` or ``None`` when no
        odometer state exists yet.
    """
    state_path = Path(warpt_dir).expanduser() / STATE_FILENAME
    try:
        data = json.loads(state_path.read_text())
        cumulative_joules = float(data["cumulative_joules"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    calc = CarbonCalculator(region=region)
    energy_kwh = calc.energy_from_counter(cumulative_joules)
    return {
        "energy_kwh": round(energy_kwh, 6),
        "co2_grams": round(calc.co2_from_energy(energy_kwh), 2),
        "cost_usd": round(calc.cost_from_energy(energy_kwh, rate=cost_per_kwh), 4),
    }
