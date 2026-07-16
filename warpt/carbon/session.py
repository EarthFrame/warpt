"""Manual energy-tracking sessions backed by the Rust power-daemon's counter.

``start`` bookmarks the counter to disk, ``status`` polls the live delta, and
``stop`` records ``end - start`` — no background process, just reads at the edges.
"""

from __future__ import annotations

import json
import platform
import time
import uuid
from pathlib import Path

from warpt.backends.power.daemon_client import (
    PowerClient,
    PowerClientError,
    PowerReading,
    counter_delta_joules,
)
from warpt.carbon.calculator import CarbonCalculator
from warpt.carbon.store import EnergyStore
from warpt.models.carbon_models import CarbonSession

# The open-session bookmark. Its existence means a session is in progress;
# there is no PID file because there is no background process.
SESSIONFILE = Path.home() / ".warpt" / "carbon.session"

_TERMINATED_LOST = (
    "terminated due to loss of connection, please try connection to daemon again"
)


def _reading_to_dict(reading: PowerReading) -> dict:
    """Serialize a PowerReading for the bookmark file."""
    return {
        "timestamp": reading.timestamp,
        "watts": reading.watts,
        "joules_since_reset": reading.joules_since_reset,
        "watt_hours_since_reset": reading.watt_hours_since_reset,
        "reset_time": reading.reset_time,
        "hostname": reading.hostname,
    }


def _reading_from_dict(data: dict) -> PowerReading:
    """Rebuild a PowerReading from the bookmark file."""
    return PowerReading(
        timestamp=data["timestamp"],
        watts=data["watts"],
        joules_since_reset=data["joules_since_reset"],
        watt_hours_since_reset=data["watt_hours_since_reset"],
        reset_time=data["reset_time"],
        hostname=data["hostname"],
    )


_REQUIRED_BOOKMARK_KEYS = {"session_id", "start_time", "start_reading"}


def _read_open_session() -> dict | None:
    """Return the open-session bookmark, or None if no valid session is active.

    A missing, corrupt, or schema-stale file (e.g. left by an older warpt
    version) is treated as "no session" and removed, so it can't wedge start.
    """
    if not SESSIONFILE.exists():
        return None
    try:
        data: dict = json.loads(SESSIONFILE.read_text())
    except (json.JSONDecodeError, OSError):
        SESSIONFILE.unlink(missing_ok=True)
        return None
    if not _REQUIRED_BOOKMARK_KEYS.issubset(data):
        SESSIONFILE.unlink(missing_ok=True)
        return None
    return data


def start_session(
    label: str = "manual",
    region: str = "US",
    intensity: float | None = None,
    kwh_price: float = 0.12,
) -> str:
    """Open a tracking session by bookmarking the daemon's energy counter.

    Parameters
    ----------
    label : str
        Session label.
    region : str
        Grid region for CO2 calculation.
    intensity : float | None
        Explicit gCO2/kWh override (used when region is ``CUSTOM``).
    kwh_price : float
        Electricity price in USD per kWh.

    Returns
    -------
    str
        The session ID of the opened session.

    Raises
    ------
    RuntimeError
        If a session is already open, or the power-daemon is unreachable.
    """
    if _read_open_session() is not None:
        raise RuntimeError(
            "Carbon tracking already active. Stop it with 'warpt carbon stop'."
        )

    client = PowerClient()
    try:
        start_reading = client.current()
    except PowerClientError as exc:
        raise RuntimeError(f"power-daemon not reachable: {exc}") from exc

    session_id = str(uuid.uuid4())
    start_time = time.time()

    SESSIONFILE.parent.mkdir(parents=True, exist_ok=True)
    SESSIONFILE.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "label": label,
                "region": region,
                "intensity": intensity,
                "kwh_price": kwh_price,
                "start_time": start_time,
                "start_reading": _reading_to_dict(start_reading),
            }
        )
    )

    # Record the session as in-progress; stop() finalizes it.
    EnergyStore().create_session(
        CarbonSession(
            id=session_id,
            label=label,
            start_time=start_time,
            region=region,
            platform=platform.system().lower(),
            sources=["daemon"],
        )
    )
    return session_id


def stop_session() -> CarbonSession | None:
    """Close the open session: read the end counter, finalize, clear bookmark.

    Returns
    -------
    CarbonSession | None
        The finalized session, or None if no session was active.
    """
    info = _read_open_session()
    if info is None:
        return None

    store = EnergyStore()
    session = store.get_session(info["session_id"])
    if session is None:
        session = CarbonSession(
            id=info["session_id"],
            label=info.get("label", "manual"),
            start_time=info.get("start_time", time.time()),
            region=info.get("region", "US"),
            platform=platform.system().lower(),
            sources=["daemon"],
        )
        store.create_session(session)

    _finalize_session(
        session,
        store,
        start_reading=_reading_from_dict(info["start_reading"]),
        region=info.get("region", "US"),
        intensity=info.get("intensity"),
        kwh_price=info.get("kwh_price", 0.12),
    )
    SESSIONFILE.unlink(missing_ok=True)
    return store.get_session(session.id)


def session_status() -> dict | None:
    """Report the open session's live status by polling the daemon.

    Returns
    -------
    dict | None
        Status info (with live current power and energy-so-far), or None if no
        session is active.
    """
    info = _read_open_session()
    if info is None:
        return None

    status: dict = {
        "running": True,
        "session_id": info.get("session_id"),
        "label": info.get("label", "manual"),
        "region": info.get("region", "US"),
    }
    start_time = info.get("start_time")
    if start_time is not None:
        status["start_time"] = start_time
        status["elapsed_s"] = round(time.time() - start_time, 1)

    # Poll the daemon live for current power and energy consumed so far.
    try:
        now = PowerClient().current()
        status["current_power_w"] = round(now.watts, 2)
        delta_j = counter_delta_joules(_reading_from_dict(info["start_reading"]), now)
        if delta_j is not None:
            status["energy_kwh_so_far"] = delta_j / 3_600_000.0
    except PowerClientError:
        status["daemon_unreachable"] = True

    return status


def _finalize_session(
    session: CarbonSession,
    store: EnergyStore,
    start_reading: PowerReading,
    region: str,
    intensity: float | None = None,
    kwh_price: float = 0.12,
) -> None:
    """Compute final energy from the daemon counter delta and persist."""
    calc = CarbonCalculator(region=region, intensity=intensity)
    end_time = time.time()
    duration_s = end_time - session.start_time

    delta_j: float | None
    try:
        end = PowerClient().current()
        delta_j = counter_delta_joules(start_reading, end)
    except PowerClientError:
        delta_j = None

    session.end_time = end_time
    session.duration_s = duration_s

    if delta_j is None:
        # Daemon connection lost or counter reset mid-session — record a
        # terminated session rather than a guessed energy number.
        session.metadata = {
            "energy_source": "daemon-counter",
            "status": _TERMINATED_LOST,
        }
        store.update_session(session)
        return

    energy_kwh = calc.energy_from_counter(delta_j)
    # Average power follows directly from energy over time (J/s = W) — no
    # sampling needed. Peak power is not available without continuous sampling.
    avg_power_w = (delta_j / duration_s) if duration_s > 0 else 0.0

    session.energy_kwh = energy_kwh
    session.co2_grams = calc.co2_from_energy(energy_kwh)
    session.cost_usd = calc.cost_from_energy(energy_kwh, rate=kwh_price)
    session.metadata = {
        "avg_power_w": round(avg_power_w, 2),
        "energy_source": "daemon-counter",
        "status": "completed",
        "intensity_gco2_kwh": intensity or calc.intensity,
        "kwh_price_usd": kwh_price,
    }
    store.update_session(session)
