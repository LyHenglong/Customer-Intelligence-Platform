"""Backwards-compatible entry point: the serving API now lives in src/api/
(app assembly in src/api/app.py, endpoints in src/api/routers/). Kept so
`uvicorn src.model.api:app` and existing deployment configs keep working."""

from src.api.app import app  # noqa: F401
