"""Tests for lifetime-energy totals flowing node → central (fleet)."""

from __future__ import annotations

from warpt.daemon.casefile import CaseFile
from warpt.fleet.central.store import FleetStore
from warpt.fleet.node_reporter import NodeReporter

ENERGY = {
    "energy_kwh": 12.5,
    "co2_grams": 4812.5,
    "cost_usd": 1.5,
    "daemon_reachable": True,
}


class FakeTracker:
    """ContinuousEnergyTracker stand-in returning fixed totals."""

    def __init__(self, totals: dict | None = None, fail: bool = False) -> None:
        self._totals = totals or dict(ENERGY)
        self._fail = fail

    def read(self) -> dict:
        """Return the configured totals (or raise when failing)."""
        if self._fail:
            raise RuntimeError("odometer exploded")
        return dict(self._totals)


def make_reporter(tmp_path, tracker=None) -> NodeReporter:
    """Build a NodeReporter over an in-memory CaseFile."""
    return NodeReporter(
        casefile=CaseFile(":memory:"),
        config={"fleet": {}},
        warpt_dir=str(tmp_path),
        energy_tracker=tracker,
    )


def heartbeat_of(messages: list[dict]) -> dict:
    """Return the single heartbeat message from a collect batch."""
    beats = [m for m in messages if m["kind"] == "heartbeat"]
    assert len(beats) == 1
    return beats[0]


# --------------------------------------------------------------- node side


def test_heartbeat_carries_energy_totals(tmp_path):
    """Heartbeats include the odometer totals when a tracker is set."""
    reporter = make_reporter(tmp_path, tracker=FakeTracker())
    beat = heartbeat_of(reporter._collect({}))
    assert beat["payload"]["status"] == "ok"
    assert beat["payload"]["energy"] == ENERGY


def test_heartbeat_without_tracker_has_no_energy(tmp_path):
    """Without a tracker the heartbeat has no energy field."""
    reporter = make_reporter(tmp_path, tracker=None)
    beat = heartbeat_of(reporter._collect({}))
    assert "energy" not in beat["payload"]


def test_failing_tracker_never_blocks_heartbeat(tmp_path):
    """A crashing odometer never stops the heartbeat from sending."""
    reporter = make_reporter(tmp_path, tracker=FakeTracker(fail=True))
    beat = heartbeat_of(reporter._collect({}))
    assert beat["payload"]["status"] == "ok"
    assert "energy" not in beat["payload"]


# ------------------------------------------------------------ central side


def make_heartbeat(energy: dict | None, ts: str = "2026-07-17T12:00:00") -> dict:
    """Build a heartbeat ingest message, optionally carrying energy."""
    payload: dict = {"status": "ok"}
    if energy is not None:
        payload["energy"] = energy
    return {
        "schema_version": 1,
        "node_id": "node-a",
        "hostname": "gpu-box-1",
        "kind": "heartbeat",
        "ts": ts,
        "payload": payload,
    }


def test_store_records_energy_from_heartbeat():
    """Central stores heartbeat energy and exposes it via list_nodes."""
    store = FleetStore("sqlite:///:memory:")
    accepted = store.ingest([make_heartbeat(ENERGY)])
    assert accepted == 1

    (node,) = store.list_nodes()
    assert node["energy_kwh"] == 12.5
    assert node["co2_grams"] == 4812.5
    assert node["cost_usd"] == 1.5
    assert node["energy_updated"] is not None


def test_store_keeps_latest_energy():
    """A newer heartbeat replaces the stored totals."""
    store = FleetStore("sqlite:///:memory:")
    store.ingest([make_heartbeat(ENERGY)])
    newer = {**ENERGY, "energy_kwh": 13.0, "co2_grams": 5005.0, "cost_usd": 1.56}
    store.ingest([make_heartbeat(newer, ts="2026-07-17T12:05:00")])

    (node,) = store.list_nodes()
    assert node["energy_kwh"] == 13.0


def test_heartbeat_without_energy_leaves_totals_untouched():
    """A heartbeat without energy must not wipe stored totals."""
    store = FleetStore("sqlite:///:memory:")
    store.ingest([make_heartbeat(ENERGY)])
    store.ingest([make_heartbeat(None, ts="2026-07-17T12:05:00")])

    (node,) = store.list_nodes()
    # A plain heartbeat must not wipe previously reported totals.
    assert node["energy_kwh"] == 12.5


def test_node_without_energy_reports_null():
    """Nodes that never reported energy expose null totals."""
    store = FleetStore("sqlite:///:memory:")
    store.ingest([make_heartbeat(None)])

    (node,) = store.list_nodes()
    assert node["energy_kwh"] is None
    assert node["energy_updated"] is None
