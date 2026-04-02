"""Tests for warpt.daemon.gpu_fields — canonical GPU metric field mapping."""

from warpt.daemon.casefile import CaseFile
from warpt.daemon.gpu_fields import DB_TO_SNAPSHOT, GPU_FIELDS, SNAPSHOT_TO_DB


def test_mapping_covers_all_struct_fields():
    """Every DuckDB GPU struct field has a corresponding GpuField entry."""
    cf = CaseFile(":memory:")
    # Insert a dummy row so we can inspect the struct columns
    cf.execute("""
        INSERT INTO vitals (
            ts, mem_total_bytes, mem_available_bytes, gpus, collection_type
        ) VALUES (
            current_timestamp, 0, 0,
            [{'gpu_guid': 'x', 'gpu_index': 0, 'utilization_pct': 0,
              'mem_utilization_pct': 0, 'power_w': 0, 'temperature_c': 0,
              'mem_used_bytes': 0, 'mem_total_bytes': 0,
              'throttle_reasons': NULL}],
            'test'
        )
        """)
    rows = cf.query("""
        SELECT UNNEST(struct_keys(gpus[1]))
        FROM vitals
        """)
    struct_cols = {r[0] for r in rows}
    mapped_cols = {f.db_column for f in GPU_FIELDS}

    # throttle_reasons is a VARCHAR[] — not a simple metric, excluded from mapping
    assert struct_cols - {"throttle_reasons"} == mapped_cols
    cf.close()


def test_snapshot_to_db_and_reverse_are_consistent():
    """SNAPSHOT_TO_DB and DB_TO_SNAPSHOT are exact inverses."""
    assert len(SNAPSHOT_TO_DB) == len(DB_TO_SNAPSHOT) == len(GPU_FIELDS)
    for f in GPU_FIELDS:
        assert SNAPSHOT_TO_DB[f.snapshot_key] == f.db_column
        assert DB_TO_SNAPSHOT[f.db_column] == f.snapshot_key


def test_no_duplicate_keys():
    """No duplicate snapshot_key or db_column across GPU_FIELDS."""
    snap_keys = [f.snapshot_key for f in GPU_FIELDS]
    db_cols = [f.db_column for f in GPU_FIELDS]
    assert len(snap_keys) == len(set(snap_keys))
    assert len(db_cols) == len(set(db_cols))
