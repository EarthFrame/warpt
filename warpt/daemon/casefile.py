"""DuckDB Case File — connection management, schema creation, migrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from warpt.utils.logger import Logger

_SCHEMA_V1 = """\
-- schema_migrations
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER   NOT NULL,
    applied_at  TIMESTAMP NOT NULL DEFAULT current_timestamp,
    description VARCHAR   NOT NULL,
    PRIMARY KEY (version)
);

-- gpu_profiles
CREATE TABLE IF NOT EXISTS gpu_profiles (
    gpu_guid            VARCHAR   NOT NULL,
    model               VARCHAR   NOT NULL,
    vendor              VARCHAR   NOT NULL DEFAULT 'nvidia',
    memory_total_bytes  BIGINT,
    compute_capability  VARCHAR,
    pcie_gen            TINYINT,
    driver_version      VARCHAR,
    power_limit_w       FLOAT,
    first_seen_at       TIMESTAMP NOT NULL DEFAULT current_timestamp,
    last_seen_at        TIMESTAMP NOT NULL DEFAULT current_timestamp,
    specs               JSON,
    PRIMARY KEY (gpu_guid)
);

-- vitals
CREATE TABLE IF NOT EXISTS vitals (
    ts                  TIMESTAMP NOT NULL,
    cpu_utilization_pct FLOAT,
    cpu_power_w         FLOAT,
    mem_total_bytes     BIGINT    NOT NULL,
    mem_available_bytes BIGINT    NOT NULL,
    mem_wired_bytes     BIGINT,
    mem_utilization_pct FLOAT,
    gpus                STRUCT(
                            gpu_guid            VARCHAR,
                            gpu_index           TINYINT,
                            utilization_pct     FLOAT,
                            mem_utilization_pct FLOAT,
                            power_w             FLOAT,
                            temperature_c       FLOAT,
                            mem_used_bytes      BIGINT,
                            mem_total_bytes     BIGINT,
                            throttle_reasons    VARCHAR[]
                        )[],
    total_power_w       FLOAT,
    collection_type     VARCHAR   NOT NULL,
    PRIMARY KEY (ts)
);

-- cases
CREATE SEQUENCE IF NOT EXISTS case_id_seq START 1;

CREATE TABLE IF NOT EXISTS cases (
    case_id    BIGINT    NOT NULL DEFAULT nextval('case_id_seq'),
    title      VARCHAR   NOT NULL,
    status     VARCHAR   NOT NULL DEFAULT 'open',
    opened_at  TIMESTAMP NOT NULL DEFAULT current_timestamp,
    closed_at  TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
    observation            TEXT,
    hypothesis             TEXT,
    confidence_pct         DOUBLE,
    recommended_action     VARCHAR,
    reasoning_chain        TEXT,
    historical_context     TEXT,
    baseline_deviation_pct FLOAT,
    stress_tests_ordered   JSON,
    stress_test_results    JSON,
    report_content         TEXT,
    diagnostician_model    VARCHAR,
    historian_model        VARCHAR,
    tags  JSON,
    notes VARCHAR,
    PRIMARY KEY (case_id)
);

CREATE INDEX IF NOT EXISTS idx_cases_status ON cases (status);

-- events
CREATE SEQUENCE IF NOT EXISTS event_id_seq START 1;

CREATE TABLE IF NOT EXISTS events (
    event_id     BIGINT    NOT NULL DEFAULT nextval('event_id_seq'),
    ts           TIMESTAMP NOT NULL,
    kind         VARCHAR   NOT NULL,
    severity     VARCHAR   NOT NULL DEFAULT 'info',
    gpu_guid     VARCHAR,
    summary      VARCHAR   NOT NULL,
    metadata     JSON,
    case_id      BIGINT,
    triggered_by VARCHAR,
    PRIMARY KEY (event_id)
);

CREATE INDEX IF NOT EXISTS idx_events_case_id ON events (case_id);
CREATE INDEX IF NOT EXISTS idx_events_gpu_guid ON events (gpu_guid);

-- Record migration
INSERT INTO schema_migrations (version, description)
    SELECT 1, 'Initial Case File schema'
    WHERE NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = 1);
"""

_MIGRATIONS = {
    1: _SCHEMA_V1,
}


class CaseFile:
    """DuckDB Case File for the warpt daemon.

    Manages the database connection, schema creation, and forward-only
    migrations. All daemon database access goes through this class.

    Parameters
    ----------
    db_path
        Path to the DuckDB file, or ``":memory:"`` for in-memory databases.
    """

    def __init__(self, db_path: str = "~/.warpt/warpt.db") -> None:
        if db_path != ":memory:":
            resolved = Path(db_path).expanduser()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            db_path = str(resolved)

        self._conn = duckdb.connect(db_path)
        self._log = Logger.get("daemon.casefile")
        self._log.info("CaseFile opened: %s", db_path)
        self._apply_migrations()

    def _current_version(self) -> int:
        """Return the highest applied migration version, or 0 if none."""
        try:
            result = self._conn.execute(
                "SELECT MAX(version) FROM schema_migrations"
            ).fetchone()
            return result[0] if result and result[0] is not None else 0
        except duckdb.CatalogException:
            return 0

    def _apply_migrations(self) -> None:
        """Apply all pending migrations in order."""
        current = self._current_version()
        for version in sorted(_MIGRATIONS):
            if version > current:
                self._conn.execute(_MIGRATIONS[version])
                self._log.info("Applied migration v%d", version)

    def query(self, sql: str, params: list[Any] | None = None) -> list[tuple]:
        """Execute a query and return all rows.

        Parameters
        ----------
        sql
            SQL query string.
        params
            Optional query parameters.

        Returns
        -------
            List of result tuples.
        """
        if params:
            return self._conn.execute(sql, params).fetchall()
        return self._conn.execute(sql).fetchall()

    def execute(self, sql: str, params: list[Any] | None = None) -> None:
        """Execute a statement without returning results.

        Parameters
        ----------
        sql
            SQL statement.
        params
            Optional query parameters.
        """
        if params:
            self._conn.execute(sql, params)
        else:
            self._conn.execute(sql)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        self._log.info("CaseFile closed.")
