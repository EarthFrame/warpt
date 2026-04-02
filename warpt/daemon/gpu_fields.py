"""Canonical GPU metric field mapping between snapshot JSON and DuckDB struct."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuField:
    """Maps a GPU metric between snapshot JSON and DuckDB struct."""

    snapshot_key: str  # Field name in GPUUsage.to_dict() / threshold config
    db_column: str  # Field name in DuckDB gpus STRUCT


GPU_FIELDS: tuple[GpuField, ...] = (
    GpuField("guid", "gpu_guid"),
    GpuField("index", "gpu_index"),
    GpuField("utilization_percent", "utilization_pct"),
    GpuField("memory_utilization_percent", "mem_utilization_pct"),
    GpuField("power_watts", "power_w"),
    GpuField("temperature_c", "temperature_c"),
    GpuField("memory_used_bytes", "mem_used_bytes"),
    GpuField("memory_total_bytes", "mem_total_bytes"),
)

# Lookup dicts
SNAPSHOT_TO_DB: dict[str, str] = {f.snapshot_key: f.db_column for f in GPU_FIELDS}
DB_TO_SNAPSHOT: dict[str, str] = {f.db_column: f.snapshot_key for f in GPU_FIELDS}
