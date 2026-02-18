"""CarbonTracker context manager for automatic energy tracking."""

from __future__ import annotations

import platform
import sys
import threading
import time
import uuid

from warpt.backends.power.factory import PowerMonitor
from warpt.carbon.calculator import CarbonCalculator
from warpt.carbon.store import EnergyStore
from warpt.models.carbon_models import CarbonSession


class CarbonTracker:
    """Context manager that samples power in a background thread.

    Wraps existing command logic to automatically track energy, CO2,
    and cost. If no power sources are available, becomes a silent no-op.

    Parameters
    ----------
    label : str
        Human-readable label for the session (e.g. "warpt stress").
    interval : float
        Sampling interval in seconds.
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
        self._sources: list[str] = []
        self._noop = False

    def __enter__(self) -> CarbonTracker:
        """Start power sampling in a background thread."""
        try:
            self._monitor = PowerMonitor(include_process_attribution=False)
            if not self._monitor.initialize():
                self._noop = True
                return self

            self._sources = [s.value for s in self._monitor.get_available_sources()]
        except Exception:
            self._noop = True
            return self

        # Create initial session record
        self._session = CarbonSession(
            id=self._session_id,
            label=self._label,
            start_time=time.time(),
            region=self._region,
            platform=platform.system().lower(),
            sources=self._sources,
        )

        store = EnergyStore()
        store.create_session(self._session)

        # Start background sampling
        self._running = True
        self._thread = threading.Thread(
            target=self._sample_loop, daemon=True, name="carbon-tracker"
        )
        self._thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop sampling, calculate results, and finalize the session."""
        if self._noop:
            return

        # Stop the sampling thread
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

        # Cleanup monitor
        if self._monitor is not None:
            try:
                self._monitor.cleanup()
            except Exception:
                pass

        # Calculate energy/CO2/cost
        calc = CarbonCalculator(region=self._region)
        power_samples = [(t, w) for t, w, _c, _g in self._samples]
        energy_kwh = calc.energy_from_samples(power_samples)
        co2_grams = calc.co2_from_energy(energy_kwh)
        cost_usd = calc.cost_from_energy(energy_kwh)

        end_time = time.time()
        duration_s = end_time - self._session.start_time

        # Compute metadata
        powers = [w for _, w, _, _ in self._samples]
        avg_power = sum(powers) / len(powers) if powers else 0.0
        peak_power = max(powers) if powers else 0.0

        # Build sample dicts for storage
        sample_dicts = [
            {
                "timestamp": t,
                "power_watts": round(w, 2),
                "cpu_watts": round(c, 2),
                "gpu_watts": round(g, 2),
            }
            for t, w, c, g in self._samples
        ]

        # Finalize session
        self._session.end_time = end_time
        self._session.duration_s = duration_s
        self._session.energy_kwh = energy_kwh
        self._session.co2_grams = co2_grams
        self._session.cost_usd = cost_usd
        self._session.metadata = {
            "avg_power_w": round(avg_power, 2),
            "peak_power_w": round(peak_power, 2),
            "sample_count": len(self._samples),
        }
        self._session.samples = sample_dicts

        store = EnergyStore()
        store.update_session(self._session)

        # Print one-line summary to stderr
        humanized = calc.humanize(co2_grams)
        print(
            f"\n[carbon] {duration_s:.1f}s | "
            f"{avg_power:.1f}W avg | "
            f"{energy_kwh * 1_000_000:.1f} mWh | "
            f"{co2_grams:.2f}g CO2 | "
            f"${cost_usd:.4f} | "
            f"{humanized}",
            file=sys.stderr,
        )

    def _sample_loop(self) -> None:
        """Continuously sample power readings at the configured interval."""
        while self._running:
            try:
                if self._monitor is None:
                    break
                snapshot = self._monitor.get_snapshot()
                total = snapshot.total_power_watts
                cpu = snapshot.get_cpu_power() or 0.0
                gpu = snapshot.get_gpu_power()
                if total is not None and total > 0:
                    self._samples.append((snapshot.timestamp, total, cpu, gpu))
            except Exception:
                pass
            time.sleep(self._interval)
