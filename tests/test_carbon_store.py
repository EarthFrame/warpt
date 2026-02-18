"""Tests for carbon session store."""

import pytest

from warpt.carbon.store import EnergyStore
from warpt.models.carbon_models import CarbonSession


@pytest.fixture
def tmp_store(tmp_path):
    """Create a store backed by a temporary directory."""
    return EnergyStore(base_dir=tmp_path / "sessions")


@pytest.fixture
def sample_session():
    """Create a sample completed session."""
    return CarbonSession(
        id="test-session-001",
        label="test",
        start_time=1000.0,
        end_time=1060.0,
        duration_s=60.0,
        energy_kwh=0.001,
        co2_grams=0.39,
        cost_usd=0.00012,
        region="US",
        platform="linux",
        sources=["rapl"],
        metadata={
            "avg_power_w": 60.0,
            "peak_power_w": 80.0,
            "sample_count": 60,
        },
        samples=[
            {
                "timestamp": 1000.0 + i,
                "power_watts": 60.0,
                "cpu_watts": 60.0,
                "gpu_watts": 0.0,
            }
            for i in range(5)
        ],
    )


class TestEnergyStore:
    """Tests for JSON file-based session storage."""

    def test_create_and_get_session(self, tmp_store, sample_session):
        """Create a session and read it back."""
        tmp_store.create_session(sample_session)
        loaded = tmp_store.get_session("test-session-001")
        assert loaded is not None
        assert loaded.id == "test-session-001"
        assert loaded.label == "test"
        assert loaded.energy_kwh == 0.001

    def test_get_nonexistent_session(self, tmp_store):
        """Return None for missing sessions."""
        assert tmp_store.get_session("nonexistent") is None

    def test_update_session(self, tmp_store, sample_session):
        """Update overwrites session data on disk."""
        tmp_store.create_session(sample_session)
        sample_session.energy_kwh = 0.002
        sample_session.co2_grams = 0.78
        tmp_store.update_session(sample_session)

        loaded = tmp_store.get_session("test-session-001")
        assert loaded.energy_kwh == 0.002
        assert loaded.co2_grams == 0.78

    def test_delete_session(self, tmp_store, sample_session):
        """Delete removes the session file."""
        tmp_store.create_session(sample_session)
        tmp_store.delete_session("test-session-001")
        assert tmp_store.get_session("test-session-001") is None

    def test_delete_nonexistent_session(self, tmp_store):
        """Delete of missing session does not raise."""
        tmp_store.delete_session("nonexistent")

    def test_get_sessions_empty(self, tmp_store):
        """Empty store returns empty list."""
        sessions = tmp_store.get_sessions()
        assert sessions == []

    def test_get_sessions_returns_sorted(self, tmp_store):
        """Sessions are returned newest-first."""
        for i in range(3):
            s = CarbonSession(
                id=f"session-{i}",
                label="test",
                start_time=1000.0 + i * 100,
            )
            tmp_store.create_session(s)

        sessions = tmp_store.get_sessions()
        assert len(sessions) == 3
        assert sessions[0].id == "session-2"
        assert sessions[2].id == "session-0"

    def test_get_sessions_with_limit(self, tmp_store):
        """Limit caps the number of returned sessions."""
        for i in range(5):
            s = CarbonSession(
                id=f"session-{i}",
                label="test",
                start_time=1000.0 + i,
            )
            tmp_store.create_session(s)

        sessions = tmp_store.get_sessions(limit=2)
        assert len(sessions) == 2

    def test_get_sessions_with_since(self, tmp_store):
        """Since filter excludes older sessions."""
        for i in range(5):
            s = CarbonSession(
                id=f"session-{i}",
                label="test",
                start_time=1000.0 + i * 100,
            )
            tmp_store.create_session(s)

        sessions = tmp_store.get_sessions(since=1250.0)
        assert len(sessions) == 2

    def test_get_sessions_ignores_malformed_json(self, tmp_store):
        """Malformed JSON files are silently skipped."""
        s = CarbonSession(id="good", label="test", start_time=1000.0)
        tmp_store.create_session(s)

        tmp_store._ensure_dir()
        bad_path = tmp_store._base_dir / "bad.json"
        bad_path.write_text("not json")

        sessions = tmp_store.get_sessions()
        assert len(sessions) == 1
        assert sessions[0].id == "good"


class TestEnergyStoreTotals:
    """Tests for aggregation / get_totals."""

    def test_totals_empty(self, tmp_store):
        """Empty store returns zero summary."""
        summary = tmp_store.get_totals()
        assert summary.total_sessions == 0
        assert summary.total_energy_kwh == 0.0
        assert summary.humanized == "No sessions recorded"

    def test_totals_single_session(self, tmp_store, sample_session):
        """Single session totals match session values."""
        tmp_store.create_session(sample_session)
        summary = tmp_store.get_totals()

        assert summary.total_sessions == 1
        assert summary.total_energy_kwh == 0.001
        assert summary.total_co2_grams == 0.39
        assert summary.total_cost_usd == 0.00012
        assert summary.avg_power_watts == 60.0

    def test_totals_multiple_sessions(self, tmp_store):
        """Multiple sessions are aggregated correctly."""
        for i in range(3):
            s = CarbonSession(
                id=f"session-{i}",
                label="test",
                start_time=1000.0 + i * 100,
                end_time=1060.0 + i * 100,
                duration_s=60.0,
                energy_kwh=0.001,
                co2_grams=0.39,
                cost_usd=0.00012,
                region="US",
                metadata={"avg_power_w": 60.0},
            )
            tmp_store.create_session(s)

        summary = tmp_store.get_totals()
        assert summary.total_sessions == 3
        assert abs(summary.total_energy_kwh - 0.003) < 1e-9

    def test_totals_with_since_filter(self, tmp_store):
        """Since filter limits aggregation window."""
        for i in range(3):
            s = CarbonSession(
                id=f"session-{i}",
                label="test",
                start_time=1000.0 + i * 1000,
                energy_kwh=0.001,
                co2_grams=0.39,
                cost_usd=0.00012,
                region="US",
            )
            tmp_store.create_session(s)

        summary = tmp_store.get_totals(since=1500.0)
        assert summary.total_sessions == 2


class TestCarbonSessionSerialization:
    """Tests for CarbonSession to_dict / from_dict round-trip."""

    def test_round_trip(self, sample_session):
        """Serialize and deserialize preserves all fields."""
        d = sample_session.to_dict()
        restored = CarbonSession.from_dict(d)
        assert restored.id == sample_session.id
        assert restored.energy_kwh == sample_session.energy_kwh
        assert restored.region == sample_session.region
        assert len(restored.samples) == len(sample_session.samples)

    def test_to_dict_none_values(self):
        """None fields serialize as null."""
        s = CarbonSession(id="test", label="test", start_time=1000.0)
        d = s.to_dict()
        assert d["end_time"] is None
        assert d["energy_kwh"] is None
