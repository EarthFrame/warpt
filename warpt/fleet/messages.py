"""Schema-versioned message shapes for node → central transport.

Every message carries ``schema_version`` so central can accept mixed-version
fleets during rolling upgrades. Bump the version on any breaking payload
change and keep central able to ingest at least one version back.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

MessageKind = Literal["heartbeat", "vitals", "event", "case", "activity"]


class FleetMessage(BaseModel):
    """One unit of node → central data.

    Parameters
    ----------
    schema_version
        Message schema version (see ``SCHEMA_VERSION``).
    node_id
        Stable node identity (generated once per node).
    hostname
        Node hostname for display.
    kind
        Payload discriminator.
    ts
        ISO-8601 timestamp of the underlying observation.
    payload
        Kind-specific body.
    """

    schema_version: int = SCHEMA_VERSION
    node_id: str
    hostname: str = ""
    kind: MessageKind
    ts: str
    payload: dict[str, Any] = Field(default_factory=dict)


class IngestBatch(BaseModel):
    """A batch of messages POSTed to ``/api/v1/ingest``."""

    messages: list[FleetMessage]


class IngestResponse(BaseModel):
    """Central's acknowledgement of an ingest batch."""

    accepted: int
