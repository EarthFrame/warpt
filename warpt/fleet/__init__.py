"""Fleet control plane — node → central reporting and the central API.

Node autonomy is sacred: everything in this package is **additive**. The
``NodeReporter`` pushes buffered copies of node-local data to a central
ingest API; when central is unreachable the node keeps observing, diagnosing,
and storing cases locally, and backfills on reconnect. Central is a consumer,
never a controller.

Node side (``node_reporter``) needs no extra dependencies. Central side
(``warpt.fleet.central``) requires the ``fleet`` pip extra (FastAPI,
uvicorn, SQLAlchemy).
"""
