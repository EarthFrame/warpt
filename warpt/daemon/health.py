"""In-process health/status endpoint for the node daemon.

A tiny stdlib HTTP server (no new dependencies) bound to localhost by
default. Because it runs *inside* the daemon it reads live state directly —
no second DuckDB connection, no lock contention (this supersedes the Phase-0
``read_only_snapshot`` hack for anything running on the node itself).

Endpoints:

- ``GET /healthz`` — liveness (the daemon process is up).
- ``GET /readyz``  — readiness (the monitor subprocess is healthy).
- ``GET /status``  — rich JSON: vitals-nurse health, DB row counts, open
  cases, fleet-reporter presence.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from warpt.utils.logger import Logger

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8788


class HealthServer:
    """Serves daemon liveness/readiness/status over local HTTP.

    Parameters
    ----------
    status_fn
        Zero-arg callable returning the rich status dict (called per
        ``/status`` request, in-process).
    ready_fn
        Zero-arg callable returning readiness (monitor subprocess healthy).
    host
        Bind address; keep loopback unless fronted by real auth.
    port
        Bind port.
    """

    def __init__(
        self,
        status_fn: Callable[[], dict[str, Any]],
        ready_fn: Callable[[], bool],
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self._status_fn = status_fn
        self._ready_fn = ready_fn
        self._host = host
        self._port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._log = Logger.get("daemon.health")

    def start(self) -> None:
        """Start serving in a daemon thread."""
        status_fn, ready_fn, log = self._status_fn, self._ready_fn, self._log

        class Handler(BaseHTTPRequestHandler):
            """Request handler bound to the daemon's live state."""

            def do_GET(self) -> None:
                """Route the three health endpoints."""
                try:
                    if self.path == "/healthz":
                        self._reply(200, {"status": "ok"})
                    elif self.path == "/readyz":
                        ready = ready_fn()
                        self._reply(
                            200 if ready else 503,
                            {"status": "ok" if ready else "degraded"},
                        )
                    elif self.path == "/status":
                        self._reply(200, status_fn())
                    else:
                        self._reply(404, {"error": "not found"})
                except Exception as e:  # never take the daemon down
                    log.exception("Health endpoint error")
                    self._reply(500, {"error": str(e)})

            def _reply(self, code: int, body: dict[str, Any]) -> None:
                data = json.dumps(body, default=str).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args: Any) -> None:
                """Silence per-request stderr logging."""

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="daemon-health", daemon=True
        )
        self._thread.start()
        self._log.info("Health endpoint on http://%s:%d", self._host, self._port)

    def stop(self) -> None:
        """Shut the server down."""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
