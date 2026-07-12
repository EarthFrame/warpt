"""Central FastAPI app — ingest, fleet queries, live activity stream.

REST for queries, Server-Sent Events for the live agent-activity stream the
dashboard renders. Every ``/api`` route is auth-gated; ``/healthz`` and
``/readyz`` are open for orchestration probes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from warpt.fleet.central.auth import BearerAuth
from warpt.fleet.central.store import FleetStore
from warpt.fleet.messages import IngestBatch, IngestResponse

_STREAM_POLL_S = 2.0


def create_app(store: FleetStore, token: str | None = None) -> FastAPI:
    """Build the central API app.

    Parameters
    ----------
    store
        The fleet store to serve from.
    token
        Shared bearer token; ``None`` disables auth (dev only).
    """
    app = FastAPI(title="warpt fleet central", version="1")
    auth = Depends(BearerAuth(token))

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        """Readiness probe — checks the fleet store."""
        ok = store.healthy()
        return {"status": "ok" if ok else "degraded", "store": ok}

    @app.post("/api/v1/ingest", response_model=IngestResponse, dependencies=[auth])
    def ingest(batch: IngestBatch) -> IngestResponse:
        """Accept a batch of node messages."""
        accepted = store.ingest([m.model_dump() for m in batch.messages])
        return IngestResponse(accepted=accepted)

    @app.get("/api/v1/nodes", dependencies=[auth])
    def list_nodes() -> list[dict[str, Any]]:
        """Fleet node inventory with health rollups."""
        return store.list_nodes()

    @app.get("/api/v1/cases", dependencies=[auth])
    def cases(
        status: str | None = None,
        severity: str | None = None,
        node_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Fleet-wide cases (e.g. ``?status=open&severity=critical``)."""
        return store.get_cases(
            status=status, severity=severity, node_id=node_id, limit=limit
        )

    @app.get("/api/v1/vitals/{node_id}", dependencies=[auth])
    def vitals(node_id: str, hours: float = 1.0) -> list[dict[str, Any]]:
        """Recent vitals for one node."""
        return store.get_vitals(node_id, hours=hours)

    @app.get("/api/v1/events", dependencies=[auth])
    def events(node_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        """Recent events, fleet-wide or per node."""
        return store.get_events(node_id=node_id, limit=limit)

    @app.get("/api/v1/activity", dependencies=[auth])
    def activity(since_id: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        """Agent-activity entries (poll form of the stream)."""
        return store.get_activity(since_id=since_id, limit=limit)

    @app.get("/api/v1/stream", dependencies=[auth])
    async def stream() -> StreamingResponse:
        """Live agent-activity stream over Server-Sent Events."""

        async def event_source():
            last_id = store.max_activity_id()
            yield 'event: hello\ndata: {"status": "connected"}\n\n'
            while True:
                await asyncio.sleep(_STREAM_POLL_S)
                entries = store.get_activity(since_id=last_id)
                for entry in entries:
                    last_id = max(last_id, entry["id"])
                    yield f"data: {json.dumps(entry, default=str)}\n\n"
                if not entries:
                    yield ": keepalive\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    # Dashboard (Phase 4): mounted last so /api keeps precedence.
    dashboard_dir = Path(__file__).parent / "dashboard"
    if dashboard_dir.exists():
        app.mount("/", StaticFiles(directory=str(dashboard_dir), html=True), name="ui")

    return app
