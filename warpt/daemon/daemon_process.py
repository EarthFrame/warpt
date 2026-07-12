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
from warpt.daemon.agents.pipeline import run_intelligence_pipeline
from warpt.daemon.agents.scribe import Scribe
from warpt.daemon.agents.tools import build_default_registry
from warpt.daemon.casefile import CaseFile, read_only_snapshot
from warpt.daemon.charge_nurse import ChargeNurse
from warpt.daemon.config import load_config
from warpt.daemon.health import HealthServer
from warpt.daemon.janitor import Janitor
from warpt.daemon.llm.registry import provider_for_agent
from warpt.daemon.remediation import AuditLog, PolicyEngine
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
        self._node_reporter: Any = None
        self._health_server: HealthServer | None = None
        self._janitor: Janitor | None = None
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

        self._janitor = Janitor(self._casefile, config)
        self._janitor.start()

        http_cfg = config.get("daemon_http", {}) or {}
        if http_cfg.get("enabled"):
            try:
                self._health_server = HealthServer(
                    status_fn=self._live_status,
                    ready_fn=lambda: (
                        self._vitals_nurse.is_healthy() if self._vitals_nurse else False
                    ),
                    host=http_cfg.get("host", "127.0.0.1"),
                    port=int(http_cfg.get("port", 8788)),
                )
                self._health_server.start()
            except Exception:
                log.exception("Health endpoint failed to start; continuing")

        # Fleet reporting is additive — node autonomy never depends on it.
        if (config.get("fleet", {}) or {}).get("enabled"):
            try:
                from warpt.fleet.node_reporter import NodeReporter

                self._node_reporter = NodeReporter(
                    casefile=self._casefile,
                    config=config,
                    warpt_dir=str(self._warpt_dir),
                )
                self._node_reporter.start()
            except Exception:
                log.exception(
                    "Fleet reporter failed to start; node continues unaffected"
                )

        log.info("Daemon ready, waiting for stop signal")
        self._stop_event.wait()
        self._shutdown()

    def _build_pipeline(self, config: dict, log: Any) -> Any:
        """Create intelligence agents and return the pipeline closure."""
        chart_provider = provider_for_agent(config, "chart_nurse")
        attending_provider = provider_for_agent(config, "attending")
        log.info(
            "LLM providers: chart_nurse=%s(%s) attending=%s(%s)",
            chart_provider.name,
            chart_provider.model,
            attending_provider.name,
            attending_provider.model,
        )

        chart_nurse = ChartNurse(casefile=self._casefile, provider=chart_provider)

        policy_engine = PolicyEngine(config)
        audit_log = AuditLog(self._casefile)
        tool_registry = build_default_registry(
            casefile=self._casefile,
            vitals_nurse=self._vitals_nurse,
            policy_engine=policy_engine,
            audit_log=audit_log,
            config=config,
        )
        log.info("Attending tools: %s", ", ".join(tool_registry.names()))

        attending_agent = Attending(
            casefile=self._casefile,
            provider=attending_provider,
            vitals_nurse=self._vitals_nurse,
            config=config,
            tool_registry=tool_registry,
            policy_engine=policy_engine,
            audit_log=audit_log,
        )
        scribe = Scribe(casefile=self._casefile)
        log.info("Intelligence pipeline enabled")

        def pipeline_fn(case_id: int, event: dict) -> None:
            run_intelligence_pipeline(
                case_id=case_id,
                event=event,
                chart_nurse=chart_nurse,
                attending=attending_agent,
                scribe=scribe,
                casefile=self._casefile,
                log=log,
            )

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
                # Read through a lock-tolerant snapshot so querying status never
                # contends with the live daemon holding the write lock.
                try:
                    with read_only_snapshot(str(db_path)) as cf:
                        status["vitals_count"] = cf.query(
                            "SELECT count(*) FROM vitals"
                        )[0][0]
                        status["events_count"] = cf.query(
                            "SELECT count(*) FROM events"
                        )[0][0]
                        status["open_cases"] = cf.query(
                            "SELECT count(*) FROM cases WHERE status = 'open'"
                        )[0][0]
                        status["last_heartbeat"] = cf.query(
                            "SELECT max(ts) FROM vitals"
                        )[0][0]
                except Exception:
                    pass
        return status

    def _live_status(self) -> dict[str, Any]:
        """Rich in-process status for the health endpoint (exact, lock-free)."""
        status: dict[str, Any] = {"running": True, "pid": os.getpid()}
        if self._vitals_nurse:
            status["vitals_nurse"] = self._vitals_nurse.get_health()
        if self._casefile:
            try:
                status["vitals_count"] = self._casefile.query(
                    "SELECT count(*) FROM vitals"
                )[0][0]
                status["events_count"] = self._casefile.query(
                    "SELECT count(*) FROM events"
                )[0][0]
                status["open_cases"] = self._casefile.query(
                    "SELECT count(*) FROM cases WHERE status = 'open'"
                )[0][0]
                status["last_heartbeat"] = str(
                    self._casefile.query("SELECT max(ts) FROM vitals")[0][0]
                )
            except Exception:
                status["db_error"] = True
        status["fleet_reporter"] = self._node_reporter is not None
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
        if self._health_server:
            self._health_server.stop()
        if self._node_reporter:
            self._node_reporter.stop()
        if self._janitor:
            self._janitor.stop()
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
