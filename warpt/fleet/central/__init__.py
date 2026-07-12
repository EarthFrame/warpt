"""Central control plane — ingest API, fleet store, dashboard API.

Requires the ``fleet`` pip extra (FastAPI, uvicorn, SQLAlchemy). The fleet
store is a real multi-writer database (Postgres/Timescale in production via
``--db postgresql://...``; SQLite for dev/small fleets) — node-local DuckDB
stays on the node.
"""
