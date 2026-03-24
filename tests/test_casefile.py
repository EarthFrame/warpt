"""Tests for the CaseFile DuckDB schema and migration system."""

from __future__ import annotations

from pytest import approx

from warpt.daemon.casefile import CaseFile


def test_schema_creates_all_tables() -> None:
    """CaseFile creates all 5 tables on initialization."""
    cf = CaseFile(":memory:")

    tables = cf.query(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    )
    table_names = [row[0] for row in tables]

    assert "cases" in table_names
    assert "events" in table_names
    assert "gpu_profiles" in table_names
    assert "schema_migrations" in table_names
    assert "vitals" in table_names

    cf.close()


def test_migration_version_recorded() -> None:
    """After init, schema_migrations contains version 1."""
    cf = CaseFile(":memory:")

    rows = cf.query("SELECT version, description FROM schema_migrations")
    assert len(rows) == 1
    assert rows[0][0] == 1
    assert rows[0][1] == "Initial Case File schema"

    cf.close()


def test_schema_creation_is_idempotent(tmp_path) -> None:
    """Opening CaseFile twice on the same database does not error or duplicate."""
    db_path = str(tmp_path / "warpt.db")

    cf1 = CaseFile(db_path)
    cf1.close()

    cf2 = CaseFile(db_path)
    rows = cf2.query("SELECT version FROM schema_migrations")
    assert len(rows) == 1
    assert rows[0][0] == 1

    cf2.close()


def test_creates_parent_directory(tmp_path) -> None:
    """CaseFile creates the parent directory if it does not exist."""
    db_path = str(tmp_path / "nested" / "deep" / "warpt.db")

    cf = CaseFile(db_path)
    assert (tmp_path / "nested" / "deep").is_dir()

    cf.close()


def test_vitals_struct_array_round_trips() -> None:
    """GPU data inserted as STRUCT[] can be queried back via UNNEST."""
    cf = CaseFile(":memory:")

    cf.execute(
        """
        INSERT INTO vitals (
            ts, mem_total_bytes, mem_available_bytes, collection_type, gpus
        ) VALUES (
            '2026-03-23 14:00:00',
            8589934592,
            2684354560,
            'heartbeat',
            [
                {
                    gpu_guid: 'GPU-aaa',
                    gpu_index: 0::TINYINT,
                    utilization_pct: 82.3,
                    mem_utilization_pct: 45.1,
                    power_w: 280.5,
                    temperature_c: NULL,
                    mem_used_bytes: NULL,
                    mem_total_bytes: NULL,
                    throttle_reasons: NULL
                },
                {
                    gpu_guid: 'GPU-bbb',
                    gpu_index: 1::TINYINT,
                    utilization_pct: 10.0,
                    mem_utilization_pct: 5.0,
                    power_w: 50.0,
                    temperature_c: NULL,
                    mem_used_bytes: NULL,
                    mem_total_bytes: NULL,
                    throttle_reasons: NULL
                }
            ]
        )
        """
    )

    rows = cf.query(
        "SELECT UNNEST(gpus).gpu_guid, UNNEST(gpus).utilization_pct "
        "FROM vitals"
    )

    assert len(rows) == 2
    assert rows[0][0] == "GPU-aaa"
    assert rows[0][1] == approx(82.3, abs=0.01)
    assert rows[1][0] == "GPU-bbb"
    assert rows[1][1] == approx(10.0, abs=0.01)

    cf.close()
