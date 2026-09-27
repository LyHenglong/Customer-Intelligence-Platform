"""FastAPI app assembly: CORS, auth, routers, and the startup lifecycle.

Run locally:
    uvicorn src.api.app:app --reload
(src.model.api:app still works - it re-exports this app.)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import observability, state
from src.api.routers import assistant, dashboard, models
from src.api.security import auth_enabled, require_api_key

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

observability.configure_logging()
log = logging.getLogger("api")

# Eager warming is fine on a 3GB local container but OOM-killed Render's
# free tier in a boot loop (the RAG embedder + reranker alone are
# ~300-500MB), so render.yaml turns it off and everything lazy-loads on its
# first request instead. Read at import time.
_WARM_MODELS_ON_STARTUP = os.environ.get("WARM_MODELS_ON_STARTUP", "true").lower() not in ("false", "0", "")

# Comma-separated browser origins allowed to call the API directly. Unset
# means no cross-origin access - the frontend's server-side proxy route
# does not need CORS at all.
_CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()]


def load_models() -> None:
    state.load_models()


def _warm_rag_models() -> None:
    """Loads the RAG embedder and reranker before the scoring pass claims
    most of the memory: a cross-encoder loaded under memory pressure can
    come back with its weights stranded on the meta device (see
    src/ai/rag/reranker.py). Best-effort - RAG degrades to no retrieval
    evidence rather than taking the API down."""
    try:
        from src.ai.rag.embeddings import embed_query

        t0 = time.monotonic()
        embed_query("warm-up query")
        log.info("RAG embedder ready in %.1fs", time.monotonic() - t0)
    except Exception:
        log.exception("RAG embedder warm failed; retrieval will retry per-request")

    try:
        from src.ai.rag.reranker import _get_reranker

        t0 = time.monotonic()
        _get_reranker()
        log.info("RAG reranker ready in %.1fs", time.monotonic() - t0)
    except Exception:
        log.exception("RAG reranker warm failed; retrieval will retry per-request")


def _warm_scored_cache() -> None:
    """Pays the ~35-110s full-population scoring pass at startup instead of
    on the first visitor's request. It takes the same lock as a request-path
    miss, so a visitor arriving mid-warm waits for this pass rather than
    starting a second one. Failures are swallowed: this is an optimisation,
    and the request path will simply pay the cost itself."""
    if state.models["churn"] is None:
        log.info("skipping scored-cache warm: no churn model loaded")
        return
    try:
        t0 = time.monotonic()
        rows = len(state.get_scored_customers())
        log.info("scored-cache warm complete: %d customers in %.1fs", rows, time.monotonic() - t0)
    except Exception:
        log.exception("scored-cache warm failed; first request will pay the scoring cost")


def _warm_all() -> None:
    _warm_rag_models()
    _warm_scored_cache()


def start_cache_warm() -> None:
    if not _WARM_MODELS_ON_STARTUP:
        log.info("WARM_MODELS_ON_STARTUP=false: skipping eager warm, everything lazy-loads on first request")
        return
    # A daemon thread, so the warm can never delay startup or the healthcheck.
    threading.Thread(target=_warm_all, name="startup-warm", daemon=True).start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Looked up as module globals at call time, so tests can patch them.
    load_models()
    start_cache_warm()
    if not auth_enabled():
        log.warning("API_KEYS is not set: API key auth is disabled")
    yield


app = FastAPI(
    title="Telecom Churn & Recommendation API",
    version="1.1.0",
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
)
observability.install(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(models.router)
app.include_router(dashboard.router)
app.include_router(assistant.router)
