"""DaemonProcess — orchestrates VitalsNurse, ChargeNurse, and CaseFile."""

from __future__ import annotations

import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from warpt.daemon.agents.attending import Attending
from warpt.daemon.agents.chart_nurse import ChartNurse
from warpt.daemon.agents.ollama_client import OllamaClient
from warpt.daemon.agents.scribe import Scribe
from warpt.daemon.casefile import CaseFile
from warpt.daemon.charge_nurse import ChargeNurse
from warpt.daemon.config import load_config
from warpt.daemon.vitals_nurse import VitalsNurse
from warpt.utils.logger import Logger

DEFAULT_WARPT_DIR = os.path.expanduser("~/.warpt")


class DaemonProcess:
    """Manages the daemon lifecycle: wiring, PID file, start/stop.

    Parameters
    ----------
    warpt_dir
        Directory for PID file and database. Defaults to ``~/.warpt``.
    """

    def __init__(self, warpt_dir: str = DEFAULT_WARPT_DIR) -> None:
        self._warpt_dir = Path(warpt_dir)
        self._pid_path = self._warpt_dir / "daemon.pid"
        self._db_path = str(self._warpt_dir / "warpt.db")
        self._casefile: CaseFile | None = None
        self._vitals_nurse: VitalsNurse | None = None
        self._charge_nurse: ChargeNurse | None = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        """Run the daemon in the current process (blocking)."""
        if self.is_running():
            raise RuntimeError("Daemon already running.")
        self._warpt_dir.mkdir(parents=True, exist_ok=True)
        self._write_pid()

        log = Logger.get("daemon")
        log.info("Daemon starting...")

        config = load_config(str(self._warpt_dir))
        self._casefile = CaseFile(self._db_path)
        self._vitals_nurse = VitalsNurse(casefile=self._casefile)

        pipeline_fn = None
        if config.get("intelligence_enabled"):
            pipeline_fn = self._build_pipeline(config, log)

        self._charge_nurse = ChargeNurse(
            casefile=self._casefile, pipeline_fn=pipeline_fn
        )
        self._vitals_nurse.set_on_threshold_breach(self._charge_nurse.handle_breach)
        log.info("Wired VitalsNurse -> ChargeNurse")
        self._vitals_nurse.start()

        log.info("Daemon ready, waiting for stop signal")
        self._stop_event.wait()
        self._shutdown()

    def _build_pipeline(self, config: dict, log: Any) -> Any:
        """Create intelligence agents and return the pipeline closure."""
        ollama_url = config.get("ollama_url", "http://localhost:11434")
        models = config.get("models", {})

        chart_client = OllamaClient(
            model=models.get("chart_nurse", "llama3:8b"), ollama_url=ollama_url
        )
        attending_client = OllamaClient(
            model=models.get("attending", "llama3:70b"), ollama_url=ollama_url
        )

        chart_nurse = ChartNurse(casefile=self._casefile, ollama_client=chart_client)
        attending = Attending(
            casefile=self._casefile,
            ollama_client=attending_client,
            vitals_nurse=self._vitals_nurse,
            config=config,
        )
        scribe = Scribe(casefile=self._casefile)
        log.info("Intelligence pipeline enabled")

        def pipeline_fn(case_id: int, event: dict) -> None:
            gpu_guid = event.get("gpu_guid", "")
            metric = event.get("metric", "")
            value = event.get("value", 0.0)
            try:
                chart_result = chart_nurse.analyze(gpu_guid, metric, value)
                attending.diagnose(chart_result, case_id)
                scribe.report(case_id)
            except Exception:
                log.exception("Pipeline failed for case #%s", case_id)

        return pipeline_fn

    def stop(self) -> None:
        """Signal the daemon to stop."""
        self._stop_event.set()

    @property
    def pid_path(self) -> Path:
        """Path to the PID file."""
        return self._pid_path

    def is_running(self) -> bool:
        """Check if a daemon process is running via PID file."""
        if not self._pid_path.exists():
            return False
        try:
            pid = int(self._pid_path.read_text().strip())
            os.kill(pid, 0)
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            return False

    def get_status(self) -> dict[str, Any]:
        """Return daemon status information."""
        running = self.is_running()
        status: dict[str, Any] = {"running": running}
        if running:
            status["pid"] = int(self._pid_path.read_text().strip())
        if self._warpt_dir.exists():
            db_path = self._warpt_dir / "warpt.db"
            if db_path.exists():
                try:
                    cf = CaseFile(str(db_path))
                    rows = cf.query("SELECT count(*) FROM vitals")
                    status["vitals_count"] = rows[0][0]
                    rows = cf.query("SELECT count(*) FROM events")
                    status["events_count"] = rows[0][0]
                    rows = cf.query("SELECT count(*) FROM cases WHERE status = 'open'")
                    status["open_cases"] = rows[0][0]
                    rows = cf.query("SELECT max(ts) FROM vitals")
                    status["last_heartbeat"] = rows[0][0]
                    cf.close()
                except Exception:
                    pass
        return status

    def _write_pid(self) -> None:
        """Write the current process PID to the PID file."""
        self._pid_path.write_text(str(os.getpid()))

    def _remove_pid(self) -> None:
        """Remove the PID file if it exists."""
        self._pid_path.unlink(missing_ok=True)

    def _shutdown(self) -> None:
        """Clean up resources."""
        log = Logger.get("daemon")
        log.info("Daemon shutting down...")
        if self._charge_nurse:
            self._charge_nurse.shutdown()
        if self._vitals_nurse:
            self._vitals_nurse.stop()
        if self._casefile:
            self._casefile.close()
        self._remove_pid()
        log.info("Daemon stopped.")


def send_stop(warpt_dir: str = DEFAULT_WARPT_DIR) -> str:
    """Send a stop signal to a running daemon.

    Returns
    -------
        Status message.
    """
    pid_path = Path(warpt_dir) / "daemon.pid"
    if not pid_path.exists():
        return "Daemon not running (no PID file)."
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        pid_path.unlink(missing_ok=True)
        return f"Daemon (PID {pid}) stopped."
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        return "Daemon not running (stale PID file removed)."
    except ValueError:
        pid_path.unlink(missing_ok=True)
        return "Invalid PID file removed."


if __name__ == "__main__":
    import signal as _sig

    Logger.configure(level=os.environ.get("WARPT_LOG_LEVEL", "INFO"))

    warpt_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WARPT_DIR
    dp = DaemonProcess(warpt_dir=warpt_dir)

    def _handle_term(_signum: int, _frame: Any) -> None:
        dp.stop()

    _sig.signal(_sig.SIGTERM, _handle_term)
    _sig.signal(_sig.SIGINT, _handle_term)
    dp.run()
