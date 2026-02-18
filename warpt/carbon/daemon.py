"""Background daemon for manual energy tracking mode."""

from __future__ import annotations

import json
import os
import platform
import signal
import threading
import time
import uuid
from pathlib import Path

from warpt.carbon.calculator import CarbonCalculator
from warpt.carbon.store import EnergyStore
from warpt.models.carbon_models import CarbonSession

PIDFILE = Path.home() / ".warpt" / "carbon.pid"
SESSIONFILE = Path.home() / ".warpt" / "carbon.session"


def _is_process_alive(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _cleanup_stale_files() -> None:
    """Remove PID/session files if the tracked process is no longer alive."""
    if PIDFILE.exists():
        try:
            pid = int(PIDFILE.read_text().strip())
            if not _is_process_alive(pid):
                PIDFILE.unlink(missing_ok=True)
                SESSIONFILE.unlink(missing_ok=True)
        except (ValueError, OSError):
            PIDFILE.unlink(missing_ok=True)
            SESSIONFILE.unlink(missing_ok=True)


def start_daemon(
    label: str = "manual",
    interval: float = 1.0,
    region: str = "US",
) -> str:
    """Fork a background daemon process for energy tracking.

    Parameters
    ----------
    label : str
        Session label.
    interval : float
        Sampling interval in seconds.
    region : str
        Grid region for CO2 calculation.

    Returns
    -------
    str
        The session ID of the started daemon.

    Raises
    ------
    RuntimeError
        If a daemon is already running.
    """
    _cleanup_stale_files()

    if PIDFILE.exists():
        pid = int(PIDFILE.read_text().strip())
        if _is_process_alive(pid):
            raise RuntimeError(
                f"Carbon daemon already running (PID {pid}). "
                "Stop it with 'warpt carbon stop'."
            )

    session_id = str(uuid.uuid4())

    # Ensure .warpt directory exists
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)

    # Fork to background
    pid = os.fork()
    if pid > 0:
        # Parent process — write PID and session info, then return
        PIDFILE.write_text(str(pid))
        SESSIONFILE.write_text(
            json.dumps({"session_id": session_id, "label": label, "region": region})
        )
        return session_id

    # Child process — detach and run daemon main loop
    os.setsid()

    # Redirect stdio to /dev/null
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)

    try:
        _daemon_main(session_id, label, interval, region)
    except Exception:
        pass
    finally:
        PIDFILE.unlink(missing_ok=True)
        SESSIONFILE.unlink(missing_ok=True)

    os._exit(0)


def stop_daemon() -> CarbonSession | None:
    """Stop a running daemon and return the finalized session.

    Returns
    -------
    CarbonSession | None
        The finalized session data, or None if no daemon was running.
    """
    _cleanup_stale_files()

    if not PIDFILE.exists():
        return None

    try:
        pid = int(PIDFILE.read_text().strip())
    except (ValueError, OSError):
        PIDFILE.unlink(missing_ok=True)
        SESSIONFILE.unlink(missing_ok=True)
        return None

    if not _is_process_alive(pid):
        PIDFILE.unlink(missing_ok=True)
        SESSIONFILE.unlink(missing_ok=True)
        return None

    # Read session info before stopping
    session_id = None
    if SESSIONFILE.exists():
        try:
            info = json.loads(SESSIONFILE.read_text())
            session_id = info.get("session_id")
        except (json.JSONDecodeError, OSError):
            pass

    # Send SIGTERM and wait for graceful shutdown
    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass

    # Wait for process to exit (up to 10s)
    for _ in range(100):
        if not _is_process_alive(pid):
            break
        time.sleep(0.1)

    # Clean up files
    PIDFILE.unlink(missing_ok=True)
    SESSIONFILE.unlink(missing_ok=True)

    # Return the finalized session from the store
    if session_id:
        store = EnergyStore()
        return store.get_session(session_id)

    return None


def daemon_status() -> dict | None:
    """Check daemon status.

    Returns
    -------
    dict | None
        Status info dict, or None if no daemon is running.
    """
    _cleanup_stale_files()

    if not PIDFILE.exists():
        return None

    try:
        pid = int(PIDFILE.read_text().strip())
    except (ValueError, OSError):
        return None

    if not _is_process_alive(pid):
        PIDFILE.unlink(missing_ok=True)
        SESSIONFILE.unlink(missing_ok=True)
        return None

    info: dict = {"pid": pid, "running": True}

    if SESSIONFILE.exists():
        try:
            session_info = json.loads(SESSIONFILE.read_text())
            info["session_id"] = session_info.get("session_id")
            info["label"] = session_info.get("label", "manual")
            info["region"] = session_info.get("region", "US")
        except (json.JSONDecodeError, OSError):
            pass

    # Try to read the session from the store for live stats
    if "session_id" in info:
        store = EnergyStore()
        session = store.get_session(info["session_id"])
        if session:
            info["start_time"] = session.start_time
            elapsed = time.time() - session.start_time
            info["elapsed_s"] = round(elapsed, 1)
            info["sample_count"] = session.metadata.get("sample_count", 0)
            info["avg_power_w"] = session.metadata.get("avg_power_w", 0.0)
            info["peak_power_w"] = session.metadata.get("peak_power_w", 0.0)

    return info


def _daemon_main(
    session_id: str,
    label: str,
    interval: float,
    region: str,
) -> None:
    """Run the main daemon loop.

    Samples power readings and periodically updates the session file.
    On SIGTERM, finalizes the session and exits.
    """
    from warpt.backends.power.factory import PowerMonitor

    shutdown = threading.Event()

    def _handle_sigterm(_signum, _frame):
        shutdown.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    monitor = PowerMonitor(include_process_attribution=False)
    if not monitor.initialize():
        return

    sources = [s.value for s in monitor.get_available_sources()]
    start_time = time.time()
    samples: list[tuple[float, float, float, float]] = []

    # Create initial session
    session = CarbonSession(
        id=session_id,
        label=label,
        start_time=start_time,
        region=region,
        platform=platform.system().lower(),
        sources=sources,
    )
    store = EnergyStore()
    store.create_session(session)

    # Sampling loop
    update_counter = 0
    while not shutdown.is_set():
        try:
            snapshot = monitor.get_snapshot()
            total = snapshot.total_power_watts
            cpu = snapshot.get_cpu_power() or 0.0
            gpu = snapshot.get_gpu_power()
            if total is not None and total > 0:
                samples.append((snapshot.timestamp, total, cpu, gpu))
        except Exception:
            pass

        # Periodically update session on disk (every 10 samples)
        update_counter += 1
        if update_counter >= 10:
            _update_session_on_disk(session, samples, store)
            update_counter = 0

        shutdown.wait(timeout=interval)

    # Final update
    monitor.cleanup()
    _finalize_session(session, samples, region, store)


def _update_session_on_disk(
    session: CarbonSession,
    samples: list[tuple[float, float, float, float]],
    store: EnergyStore,
) -> None:
    """Write intermediate session state to disk."""
    powers = [w for _, w, _, _ in samples]
    session.metadata = {
        "avg_power_w": round(sum(powers) / len(powers), 2) if powers else 0.0,
        "peak_power_w": round(max(powers), 2) if powers else 0.0,
        "sample_count": len(samples),
    }
    store.update_session(session)


def _finalize_session(
    session: CarbonSession,
    samples: list[tuple[float, float, float, float]],
    region: str,
    store: EnergyStore,
) -> None:
    """Calculate final energy/CO2/cost and persist the completed session."""
    calc = CarbonCalculator(region=region)
    power_samples = [(t, w) for t, w, _c, _g in samples]
    energy_kwh = calc.energy_from_samples(power_samples)
    co2_grams = calc.co2_from_energy(energy_kwh)
    cost_usd = calc.cost_from_energy(energy_kwh)

    end_time = time.time()
    powers = [w for _, w, _, _ in samples]

    session.end_time = end_time
    session.duration_s = end_time - session.start_time
    session.energy_kwh = energy_kwh
    session.co2_grams = co2_grams
    session.cost_usd = cost_usd
    session.metadata = {
        "avg_power_w": round(sum(powers) / len(powers), 2) if powers else 0.0,
        "peak_power_w": round(max(powers), 2) if powers else 0.0,
        "sample_count": len(samples),
    }
    session.samples = [
        {
            "timestamp": t,
            "power_watts": round(w, 2),
            "cpu_watts": round(c, 2),
            "gpu_watts": round(g, 2),
        }
        for t, w, c, g in samples
    ]
    store.update_session(session)
