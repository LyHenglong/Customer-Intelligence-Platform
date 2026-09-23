"""
FastAPI serving layer for the churn classifier, content-based recommender,
the AI Agent Layer that narrates their output (src/agents/), and the
dashboard-facing analytics endpoints a frontend needs instead of reading
Postgres directly (src/model/dashboard_queries.py). Loads the latest
versioned artifact of each model at startup (by filename timestamp - see
train_churn.py / train_recommender.py) and serves from those in-memory
artifacts. Churn predictions and recommender index hits need no database
at all; /recommend falls back to a single warehouse lookup only when a
customer isn't in the recommender's bounded reference set (see the
endpoint for why that is the common case), and /explain-churn always
touches the warehouse (it needs the customer's full feature row) plus
Postgres-backed LLM output caching (see src/agents/cache.py).

Run locally:
    uvicorn src.model.api:app --reload

Endpoints:
    GET  /health
    GET  /model-info
    POST /predict-churn
    POST /recommend
    GET  /explain-churn/{customer_id}
    POST /outreach-draft/{customer_id}
    GET  /overview/stats
    GET  /overview/segment-rates
    GET  /overview/revenue-at-risk-by-segment
    GET  /at-risk
    GET  /customers
    GET  /customers/{customer_id}
    GET  /model-history
    GET  /pipeline-status
    POST /assistant/query
    GET  /assistant/trace/{trace_id}
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model import dashboard_queries
from src.model.train_churn import ALL_FEATURES as CHURN_FEATURES
from src.model.train_churn import ID_COL as CHURN_ID_COL
from src.model.registry import resolve_model_path
from src.warehouse import get_pg_conn
from src.model.train_recommender import recommend_for_customer, recommend_for_profile
from src.model.train_recommender import PROFILE_NUMERIC as REC_PROFILE_NUMERIC
from src.model.train_recommender import PROFILE_CATEGORICAL as REC_PROFILE_CATEGORICAL
from src.model.train_recommender import SERVICE_COLUMNS as REC_SERVICE_COLUMNS
from src.model.explain_churn import compute_shap_details, format_risk_factors_text
from src.agents.groq_client import AgentCallFailed
from src.agents.explanation_agent import explain_churn as ai_explain_churn
from src.agents.outreach_agent import draft_outreach as ai_draft_outreach
from src.agents.cache import get_or_generate, get_latest_retrain_summary
from src.ai.graph import run_query as ai_run_query
from src.ai.observability.tracing import get_trace as ai_get_trace
from src.ai.schemas import AssistantResponse, CustomerLookupResult, CustomerSearchFilters, CustomerSearchResult
from src.ai.tools.customer_tool import customer_lookup as ai_customer_lookup
from src.ai.tools.customer_tool import customer_search as ai_customer_search

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api")

MODELS_DIR = Path(__file__).resolve().parents[2] / "models_store"

app = FastAPI(title="Telecom Churn & Recommendation API", version="1.0.0")

# CORS_ALLOWED_ORIGINS: comma-separated origins the new frontend(s) are
# served from (e.g. "http://localhost:3000,https://telecom-churn-
# frontend.onrender.com"). Read once at import time, same staleness
# property as MODELS_DIR above - a change needs a container restart, not
# just an env var update. Unset (default) -> empty list -> no cross-
# origin browser access permitted, a safe default for a still-
# unauthenticated API (see README's "Production hardening" section).
_CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_state: dict = {"churn": None, "churn_version": None, "recommender": None, "recommender_version": None}

# Cache for dashboard_queries.score_all_customers (~1M-row streaming pass
# on every call - too expensive to run on every request). Hand-rolled
# in-process TTL cache, not Redis/a materialized view: matches this
# project's consistent "avoid unnecessary infrastructure" choices
# elsewhere (e.g. pgvector instead of a second DB). 600s staleness is the
# accepted tolerance for this computation.
#
# The lock spans the recompute itself, not just the check-and-return: a
# check-then-release-then-recompute pattern would let two concurrent
# cache-miss requests both kick off a full 1M-row streaming score at
# once, doubling peak memory on a container already sized once for real
# (see docker-compose.yml's api mem_limit, raised after a reproduced
# OOM). Holding the lock across recompute serializes that instead -
# bounded added latency for the second request, not doubled memory.
# Single-worker uvicorn (no --workers flag in docker/Dockerfile.api) - this
# lock only needs to arbitrate threads within one process, not processes.
#
# Multi-replica caveat, not silently glossed over: if render.yaml's
# numInstances is ever enabled, each replica holds its own independent
# copy of this cache and independently pays the recompute cost -
# correctness-neutral (every replica reads the same Postgres data and
# model artifact, so they converge to identical scores) but resource-
# wasteful. Not fixed here: real added infrastructure (a shared cache)
# for a problem that doesn't exist at Render's free-tier scale (capped
# at one instance regardless).
_scored_cache: dict = {"data": None, "expires_at": 0.0, "version": None}
_scored_lock = threading.Lock()
# An hour, not the original 10 minutes. The scores this caches only move
# when a batch is ingested (a manually triggered DAG run here) or a model
# is promoted - and a model promotion already invalidates this cache
# independently, since entries are keyed by churn_version. A 10-minute TTL
# therefore bought no extra freshness in practice, while guaranteeing that
# any visitor arriving after a quiet spell paid a ~35-110s scoring pass and
# saturated the single uvicorn worker while doing it.
_SCORED_CACHE_TTL_SECONDS = 3600

# /at-risk's SHAP + recommendation loop is real per-request CPU work,
# independent of the (cached) population-scoring pass above. An
# unauthenticated public REST endpoint has no natural bound on max_rows
# and could be hit with max_rows=1000000 repeatedly, so this is a
# server-side clamp.
_AT_RISK_MAX_ROWS = 500


@app.on_event("startup")
def load_models() -> None:
    churn_path = resolve_model_path("churn_model", "churn_model_*.joblib")
    if churn_path:
        _state["churn"] = joblib.load(churn_path)
        _state["churn_version"] = churn_path.stem.replace("churn_model_", "")
        log.info("Loaded churn model %s", churn_path.name)
    else:
        log.warning("No churn model artifact found in %s", MODELS_DIR)

    rec_path = resolve_model_path("recommender", "recommender_*.joblib")
    if rec_path:
        _state["recommender"] = joblib.load(rec_path)
        _state["recommender_version"] = rec_path.stem.replace("recommender_", "")
        log.info("Loaded recommender artifact %s", rec_path.name)
    else:
        log.warning("No recommender artifact found in %s", MODELS_DIR)


def _warm_scored_cache() -> None:
    """Pays the full-population scoring cost at startup instead of making
    the first visitor wait for it.

    /overview/stats and /at-risk both need every customer scored, which is
    a single-threaded ~35-110s pass over 1,000,000 rows (the spread is CPU
    contention - it is slowest when Airflow is running alongside). On a
    cold cache that cost landed on whoever opened the dashboard first, and
    since it saturates the one uvicorn worker, every other request queued
    behind it too. The page said "Scoring the full customer population..."
    the whole time, which is accurate and still looks broken.

    Runs on a daemon thread so it cannot delay startup or the healthcheck
    (an unhealthy container would just get restarted, restarting the warm
    with it). It takes the same lock as a request-path miss, so a visitor
    arriving mid-warm waits for this pass rather than starting a second
    one."""
    if _state["churn"] is None:
        log.info("skipping scored-cache warm: no churn model loaded")
        return
    try:
        t0 = time.monotonic()
        rows = len(_get_scored_customers())
        log.info("scored-cache warm complete: %d customers in %.1fs", rows, time.monotonic() - t0)
    except Exception:
        # Deliberately swallowed: this is an optimisation. A failure here
        # must not stop the API serving - the request path will simply pay
        # the cost itself, exactly as it did before.
        log.exception("scored-cache warm failed; first request will pay the scoring cost")


@app.on_event("startup")
def start_cache_warm() -> None:
    threading.Thread(target=_warm_scored_cache, name="scored-cache-warm", daemon=True).start()


def _get_scored_customers() -> pd.DataFrame:
    """Cached wrapper around dashboard_queries.score_all_customers - see
    the module-level comment on _scored_cache/_scored_lock for why."""
    with _scored_lock:
        now = time.monotonic()
        if (
            _scored_cache["data"] is not None
            and _scored_cache["version"] == _state["churn_version"]
            and now < _scored_cache["expires_at"]
        ):
            return _scored_cache["data"]
        df = dashboard_queries.score_all_customers(_state["churn"], _state["churn_version"])
        _scored_cache.update(data=df, expires_at=now + _SCORED_CACHE_TTL_SECONDS, version=_state["churn_version"])
        return df


class ChurnRequest(BaseModel):
    age: Optional[float] = None
    annual_income: Optional[float] = None
    dependents: Optional[float] = None
    tenure: Optional[float] = None
    tenure_years: Optional[float] = None
    monthlycharges: Optional[float] = None
    totalcharges: Optional[float] = None
    num_services: Optional[float] = None
    total_active_services: Optional[float] = None
    cost_per_active_service: Optional[float] = None
    customer_satisfaction: Optional[float] = None
    num_complaints: Optional[float] = None
    num_service_calls: Optional[float] = None
    late_payments: Optional[float] = None
    avg_monthly_gb: Optional[float] = None
    avg_gb_per_service: Optional[float] = None
    days_since_last_interaction: Optional[float] = None
    credit_score: Optional[float] = None
    complaints_per_tenure_month: Optional[float] = None
    annual_spend_to_income_ratio: Optional[float] = None

    senior_citizen: Optional[float] = None
    paperless_billing: Optional[float] = None
    has_phone_service: Optional[float] = None
    has_internet_service: Optional[float] = None
    has_online_security: Optional[float] = None
    has_online_backup: Optional[float] = None
    has_device_protection: Optional[float] = None
    has_tech_support: Optional[float] = None
    has_streaming_tv: Optional[float] = None
    has_streaming_movies: Optional[float] = None
    is_month_to_month: Optional[float] = None
    is_disengaged: Optional[float] = None

    gender: Optional[str] = None
    education: Optional[str] = None
    marital_status: Optional[str] = None
    contract: Optional[str] = None
    payment_method: Optional[str] = None
    tenure_bucket: Optional[str] = None


class ChurnResponse(BaseModel):
    churn_probability: float = Field(..., description="Predicted probability of churn (class 1)")
    churn_prediction: int = Field(..., description="0 = predicted retained, 1 = predicted churn")
    threshold_used: float = Field(..., description="Decision threshold applied to churn_probability")
    model_version: str


class RecommendRequest(BaseModel):
    customer_id: str
    top_n: int = 3


class RecommendResponse(BaseModel):
    customer_id: str
    recommendations: list[dict]
    model_version: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "churn_model_loaded": _state["churn"] is not None,
        "recommender_loaded": _state["recommender"] is not None,
    }


@app.get("/model-info")
def model_info():
    return {
        "churn_model_version": _state["churn_version"],
        "recommender_version": _state["recommender_version"],
    }


@app.post("/predict-churn", response_model=ChurnResponse)
def predict_churn(req: ChurnRequest):
    if _state["churn"] is None:
        raise HTTPException(status_code=503, detail="Churn model not loaded")

    payload = req.model_dump()
    row = {f: payload.get(f) for f in CHURN_FEATURES}
    X = pd.DataFrame([row], columns=CHURN_FEATURES)

    pipeline = _state["churn"]["pipeline"]
    proba = float(pipeline.predict_proba(X)[0, 1])
    # Threshold is saved with the artifact (chosen at training time to hit a
    # target recall - see train_churn.py). Older artifacts saved before that
    # existed fall back to 0.5.
    threshold = _state["churn"].get("threshold", 0.5)
    pred = int(proba >= threshold)

    return ChurnResponse(
        churn_probability=round(proba, 4),
        churn_prediction=pred,
        threshold_used=round(threshold, 4),
        model_version=_state["churn_version"],
    )


def _fetch_profile_from_warehouse(customer_id: str):
    """Pulls one customer's profile + current subscriptions from the mart.

    Only called on an index miss (see /recommend). Keeping this off the hot
    path preserves the "no DB dependency at request time" property for the
    common case, while still letting the endpoint answer for customers the
    recommender's reference set doesn't happen to contain.
    """
    columns = REC_PROFILE_NUMERIC + REC_PROFILE_CATEGORICAL + REC_SERVICE_COLUMNS
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(columns)} FROM marts.customer_360 WHERE customer_id = %s",
                (customer_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return None, None
    profile = pd.DataFrame([row], columns=columns)
    own_services = [int(profile[s].iloc[0]) for s in REC_SERVICE_COLUMNS]
    return profile, own_services


def _recommend_with_fallback(customer_id: str, top_n: int = 3) -> list[dict]:
    """Shared by /recommend and /outreach-draft - the same fast-path/
    warehouse-fallback logic, one place, so they can never disagree about
    what's recommended for the same customer."""
    if _state["recommender"] is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded")

    artifact = _state["recommender"]
    try:
        # Fast path: customer is in the k-NN reference set, answered purely
        # from the in-memory artifact.
        return recommend_for_customer(artifact, customer_id, top_n=top_n)
    except KeyError:
        # Miss. The reference set is deliberately bounded (~154K customers)
        # while the warehouse holds 1M, so this is the *common* case, not an
        # edge case - answering it with a 404 left ~85% of customers with no
        # recommendation at all. Look the profile up and match it against
        # the index instead.
        try:
            profile, own_services = _fetch_profile_from_warehouse(customer_id)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"customer not in recommender index and warehouse lookup failed: {exc}",
            )
        if profile is None:
            raise HTTPException(
                status_code=404, detail=f"customer_id {customer_id!r} not found"
            )
        return recommend_for_profile(artifact, profile, own_services, top_n=top_n)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    recs = _recommend_with_fallback(req.customer_id, req.top_n)
    return RecommendResponse(
        customer_id=req.customer_id,
        recommendations=recs,
        model_version=_state["recommender_version"],
    )


class ExplainChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_factors: list[dict] = Field(
        ..., description="Raw SHAP attribution: [{feature, shap_value, direction}], always present"
    )
    explanation: str = Field(..., description="Plain-English explanation - AI-generated when available")
    source: str = Field(..., description='"llm" (AI Agent Layer) or "fallback" (raw SHAP text, LLM unavailable)')
    model_version: str


def _fetch_churn_row_from_warehouse(customer_id: str):
    """Pulls one customer's full churn-feature row from the mart, for the
    /explain-churn endpoint - the analog of _fetch_profile_from_warehouse
    above, for the churn model's feature set rather than the recommender's."""
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(CHURN_FEATURES)} FROM marts.customer_360 WHERE {CHURN_ID_COL} = %s",
                (customer_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return pd.DataFrame([row], columns=CHURN_FEATURES)


def _get_explanation(customer_id: str) -> Optional[dict]:
    """Shared by /explain-churn and /outreach-draft - the exact same
    fetch-row -> score -> SHAP -> get_or_generate(...) sequence, so both
    endpoints can never produce two different explanations for the same
    customer/model-version. Returns None if the customer isn't found (the
    caller decides the right HTTP status for its own context)."""
    X = _fetch_churn_row_from_warehouse(customer_id)
    if X is None:
        return None

    artifact = _state["churn"]
    pipeline = artifact["pipeline"]
    explain_pipeline = artifact.get("base_pipeline", pipeline)

    X_for_predict = X.copy()
    for c in [f for f in CHURN_FEATURES if X_for_predict[f].dtype == bool]:
        X_for_predict[c] = X_for_predict[c].astype(float)
    churn_probability = float(pipeline.predict_proba(X_for_predict)[0, 1])

    shap_details = compute_shap_details(explain_pipeline, X, top_k=5)[0]
    fallback_text = format_risk_factors_text(shap_details)

    try:
        explanation = get_or_generate(
            customer_id, "explanation", _state["churn_version"],
            lambda: ai_explain_churn(churn_probability, shap_details),
        )
        source = "llm"
    except AgentCallFailed as exc:
        log.warning("explanation agent unavailable for %s, falling back to raw SHAP: %s", customer_id, exc)
        explanation = fallback_text
        source = "fallback"

    return {
        "churn_probability": churn_probability,
        "risk_factors": shap_details,
        "explanation": explanation,
        "source": source,
    }


@app.get("/explain-churn/{customer_id}", response_model=ExplainChurnResponse)
def explain_churn_endpoint(customer_id: str):
    """AI Agent Layer: a plain-English explanation of why this customer is
    flagged as at-risk, grounded in the churn model's own SHAP attribution
    (see src/model/explain_churn.py). This is a presentation layer over an
    already-final prediction - it never changes the churn probability or
    which features the model used, only narrates them.

    Falls back to the raw SHAP factor list (as compact text) if the LLM
    call fails for any reason - the endpoint always returns 200 with real
    data, it just degrades from AI prose to the underlying numbers rather
    than ever raising a 5xx for an LLM outage."""
    if _state["churn"] is None:
        raise HTTPException(status_code=503, detail="Churn model not loaded")

    result = _get_explanation(customer_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"customer_id {customer_id!r} not found")

    return ExplainChurnResponse(
        customer_id=customer_id,
        churn_probability=round(result["churn_probability"], 4),
        risk_factors=result["risk_factors"],
        explanation=result["explanation"],
        source=result["source"],
        model_version=_state["churn_version"],
    )


class OutreachDraftResponse(BaseModel):
    customer_id: str
    explanation: str
    explanation_source: str = Field(..., description='"llm" or "fallback"')
    recommended_service: Optional[str] = None
    draft: Optional[str] = Field(None, description="Retention outreach draft - null if no service was recommended")
    draft_source: Optional[str] = Field(None, description='"llm" or "fallback", null iff draft is null')
    model_version: str


@app.post("/outreach-draft/{customer_id}", response_model=OutreachDraftResponse)
def outreach_draft(customer_id: str):
    """AI Agent Layer: same explanation (via _get_explanation, shared with
    /explain-churn) plus a drafted retention message for the
    recommender's top suggestion, same Postgres-cached-LLM-then-fallback
    pattern as everywhere else in this API. Soft-fails to
    recommended_service/draft: null (200, not 404/503) when the
    recommender has nothing to suggest - that's not an error, the
    customer just already has every service."""
    if _state["churn"] is None:
        raise HTTPException(status_code=503, detail="Churn model not loaded")

    explanation_result = _get_explanation(customer_id)
    if explanation_result is None:
        raise HTTPException(status_code=404, detail=f"customer_id {customer_id!r} not found")

    recommended_service = None
    draft = None
    draft_source = None
    if _state["recommender"] is not None:
        try:
            recs = _recommend_with_fallback(customer_id, top_n=1)
        except HTTPException:
            recs = []  # recommender miss for this customer - soft-fail, not this endpoint's error to raise
        if recs:
            recommended_service = recs[0]["service"]
            try:
                draft = get_or_generate(
                    customer_id, "outreach", _state["churn_version"],
                    lambda: ai_draft_outreach(explanation_result["explanation"], recommended_service),
                )
                draft_source = "llm"
            except AgentCallFailed as exc:
                log.warning("outreach agent unavailable for %s, falling back to canned message: %s", customer_id, exc)
                draft = f"We'd like to offer you {recommended_service} to improve your experience with us."
                draft_source = "fallback"

    return OutreachDraftResponse(
        customer_id=customer_id,
        explanation=explanation_result["explanation"],
        explanation_source=explanation_result["source"],
        recommended_service=recommended_service,
        draft=draft,
        draft_source=draft_source,
        model_version=_state["churn_version"],
    )


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class OverviewStatsResponse(BaseModel):
    total_customers: int
    historical_churn_rate: float
    at_risk_count: int
    revenue_at_risk: float
    model_auc: Optional[float] = None
    model_version: Optional[str] = None
    threshold_used: float
    top_feature_importances: list[FeatureImportance] = Field(default_factory=list)


@app.get("/overview/stats", response_model=OverviewStatsResponse)
def overview_stats(threshold: Optional[float] = None):
    """Headline KPIs, sourced from dashboard_queries so this endpoint and
    any other caller of those functions can never silently disagree."""
    if _state["churn"] is None:
        raise HTTPException(status_code=503, detail="Churn model not loaded")

    stats = dashboard_queries.load_overall_stats()
    effective_threshold = threshold if threshold is not None else _state["churn"].get("threshold", 0.5)

    scored = _get_scored_customers()
    at_risk = scored[scored["churn_probability"] >= effective_threshold]

    # The metadata entry for the model actually loaded, not simply the
    # newest file in models_store. Those diverge whenever the registry
    # resolves something other than latest-on-disk - most obviously when
    # MLflow's "champion" alias points at an older artifact, which is
    # exactly what happens if a retrain saves successfully but fails to
    # register (see src/model/registry.py). Reporting the newest file's
    # AUC beside _state's version produced a self-contradictory response:
    # one model's version label next to another model's score. None is
    # returned rather than a fallback, because a wrong AUC is worse than
    # an absent one.
    metadata_history = dashboard_queries.load_all_churn_metadata()
    serving_metadata = next(
        (m for m in metadata_history if m.get("version") == _state["churn_version"]), None
    )
    model_auc = serving_metadata.get("roc_auc") if serving_metadata else None

    pipeline = _state["churn"]["pipeline"]
    explain_pipeline = _state["churn"].get("base_pipeline", pipeline)
    importances = dashboard_queries.column_importances(explain_pipeline)
    top_importances = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:12]

    return OverviewStatsResponse(
        total_customers=stats["total_customers"],
        historical_churn_rate=round(stats["churn_rate"], 4),
        at_risk_count=int(len(at_risk)),
        revenue_at_risk=round(float(at_risk["monthlycharges"].sum()), 2),
        model_auc=round(model_auc, 4) if model_auc is not None else None,
        model_version=_state["churn_version"],
        threshold_used=round(effective_threshold, 4),
        top_feature_importances=[
            FeatureImportance(feature=f, importance=round(v, 4)) for f, v in top_importances
        ],
    )


class SegmentBucket(BaseModel):
    key: str
    churn_rate: float
    n_customers: int


class SegmentRatesResponse(BaseModel):
    column: str
    buckets: list[SegmentBucket]


@app.get("/overview/segment-rates", response_model=SegmentRatesResponse)
def overview_segment_rates(column: str):
    """Churn rate by segment (contract, tenure_bucket, total_active_services,
    education, marital_status, payment_method, gender - the same allow-
    list dashboard_queries.load_segment_rates itself enforces)."""
    try:
        df = dashboard_queries.load_segment_rates(column)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return SegmentRatesResponse(
        column=column,
        buckets=[
            SegmentBucket(key=str(row[column]), churn_rate=round(row["churn_rate"], 4), n_customers=int(row["n_customers"]))
            for _, row in df.iterrows()
        ],
    )


class RevenueAtRiskBucket(BaseModel):
    segment: str
    revenue_at_risk: float


class RevenueAtRiskResponse(BaseModel):
    segment_column: str
    buckets: list[RevenueAtRiskBucket]


@app.get("/overview/revenue-at-risk-by-segment", response_model=RevenueAtRiskResponse)
def overview_revenue_at_risk_by_segment(
    threshold: Optional[float] = None, segment_column: str = "contract",
):
    """Monthly revenue at risk, broken down by segment, over the ENTIRE
    at-risk population - so these buckets sum to /overview/stats's
    revenue_at_risk headline rather than to some capped subset of it.

    No max_rows here on purpose, unlike /at-risk: the segment columns ride
    along on the cached scored frame (see dashboard_queries
    .score_all_customers), so this is an in-process groupby over columns
    already in memory - no per-row Postgres fetch, and nothing that grows
    with the number of rows displayed."""
    if _state["churn"] is None:
        raise HTTPException(status_code=503, detail="Churn model not loaded")
    if segment_column not in ("contract", "tenure_bucket", "education", "marital_status", "payment_method", "gender"):
        raise HTTPException(status_code=422, detail=f"unexpected segment_column {segment_column!r}")

    effective_threshold = threshold if threshold is not None else _state["churn"].get("threshold", 0.5)

    scored = _get_scored_customers()
    at_risk = scored[scored["churn_probability"] >= effective_threshold]
    if at_risk.empty:
        return RevenueAtRiskResponse(segment_column=segment_column, buckets=[])

    grouped = at_risk.groupby(segment_column, observed=True)["monthlycharges"].sum()

    return RevenueAtRiskResponse(
        segment_column=segment_column,
        buckets=[RevenueAtRiskBucket(segment=str(k), revenue_at_risk=round(float(v), 2)) for k, v in grouped.items()],
    )


class AtRiskCustomer(BaseModel):
    customer_id: str
    churn_probability: float
    contract: Optional[str] = None
    tenure: Optional[int] = None
    monthlycharges: Optional[float] = None
    key_risk_factors: list[dict] = Field(default_factory=list)
    recommended_action: Optional[str] = None


class AtRiskListResponse(BaseModel):
    customers: list[AtRiskCustomer]
    total_at_risk: int
    threshold: float
    max_rows_used: int
    offset: int
    model_version: Optional[str] = None


@app.get("/at-risk", response_model=AtRiskListResponse)
def at_risk(threshold: Optional[float] = None, max_rows: int = 100, offset: int = 0):
    """Top customers by churn probability above threshold, with
    per-customer SHAP risk factors and a recommended retention action.
    max_rows is server-clamped (an unauthenticated REST endpoint has no
    natural bound otherwise) - the response's own max_rows_used tells the
    real page size actually returned."""
    if _state["churn"] is None:
        raise HTTPException(status_code=503, detail="Churn model not loaded")

    effective_threshold = threshold if threshold is not None else _state["churn"].get("threshold", 0.5)
    max_rows = max(1, min(max_rows, _AT_RISK_MAX_ROWS))
    offset = max(0, offset)

    scored = _get_scored_customers()
    at_risk_all = scored[scored["churn_probability"] >= effective_threshold].sort_values(
        "churn_probability", ascending=False
    )
    page = at_risk_all.iloc[offset: offset + max_rows]

    if page.empty:
        return AtRiskListResponse(
            customers=[], total_at_risk=int(len(at_risk_all)), threshold=round(effective_threshold, 4),
            max_rows_used=max_rows, offset=offset, model_version=_state["churn_version"],
        )

    full_rows = dashboard_queries.load_customers_by_id(tuple(page["customer_id"]))
    full_rows = full_rows.set_index("customer_id")

    pipeline = _state["churn"]["pipeline"]
    explain_pipeline = _state["churn"].get("base_pipeline", pipeline)

    customers = []
    for _, scored_row in page.iterrows():
        cid = scored_row["customer_id"]
        if cid not in full_rows.index:
            continue
        full_row = full_rows.loc[[cid]]
        try:
            shap_details = compute_shap_details(explain_pipeline, full_row[CHURN_FEATURES], top_k=3)[0]
        except Exception as exc:
            log.warning("SHAP failed for %s, omitting risk factors: %s", cid, exc)
            shap_details = []
        recommended_action = None
        if _state["recommender"] is not None:
            try:
                # Whole row, not a hand-picked column subset -
                # recommend_for_profile selects the columns it actually
                # needs internally.
                own = [int(full_row[s].iloc[0]) for s in REC_SERVICE_COLUMNS]
                recs = recommend_for_profile(_state["recommender"], full_row, own, top_n=1)
                if recs:
                    recommended_action = recs[0]["service"]
            except Exception as exc:
                log.warning("recommendation failed for %s, omitting: %s", cid, exc)

        customers.append(AtRiskCustomer(
            customer_id=cid,
            churn_probability=round(float(scored_row["churn_probability"]), 4),
            contract=full_row["contract"].iloc[0] if "contract" in full_row else None,
            tenure=int(full_row["tenure"].iloc[0]) if "tenure" in full_row else None,
            monthlycharges=float(full_row["monthlycharges"].iloc[0]) if "monthlycharges" in full_row else None,
            key_risk_factors=shap_details,
            recommended_action=recommended_action,
        ))

    return AtRiskListResponse(
        customers=customers,
        total_at_risk=int(len(at_risk_all)),
        threshold=round(effective_threshold, 4),
        max_rows_used=max_rows,
        offset=offset,
        model_version=_state["churn_version"],
    )


@app.get("/customers", response_model=CustomerSearchResult)
def search_customers(
    limit: int = 25,
    offset: int = 0,
    min_churn_probability: Optional[float] = None,
    max_churn_probability: Optional[float] = None,
    min_monthly_charges: Optional[float] = None,
    max_monthly_charges: Optional[float] = None,
    min_satisfaction: Optional[float] = None,
    max_satisfaction: Optional[float] = None,
    min_complaints: Optional[float] = None,
    contract: Optional[str] = None,
    tenure_bucket: Optional[str] = None,
):
    """Thin wrapper over src/ai/tools/customer_tool.py's customer_search -
    already paginated/filtered/bounded (see that function's own docstring
    for the total_matched caveat when a probability filter is combined
    with a low-selectivity SQL filter), reused as-is rather than
    reimplemented."""
    filters = CustomerSearchFilters(
        min_churn_probability=min_churn_probability,
        max_churn_probability=max_churn_probability,
        min_monthly_charges=min_monthly_charges,
        max_monthly_charges=max_monthly_charges,
        min_satisfaction=min_satisfaction,
        max_satisfaction=max_satisfaction,
        min_complaints=min_complaints,
        contract=contract,
        tenure_bucket=tenure_bucket,
    )
    return ai_customer_search(filters, limit=limit, offset=offset)


@app.get("/customers/{customer_id}", response_model=CustomerLookupResult)
def get_customer(customer_id: str):
    """Thin wrapper over src/ai/tools/customer_tool.py's customer_lookup.

    Deliberately returns 200 with found=false on a miss, not 404 - this
    matches customer_lookup()'s own typed contract (found: bool is
    already how "not found" is represented, shared with src/ai/graph.py's
    internal usage) rather than discarding that information behind an
    exception. This is an intentional inconsistency with /explain-churn's
    404-on-miss convention, not an oversight - do not "fix" it into
    matching /explain-churn without also changing customer_lookup()'s own
    contract, which src/ai/graph.py also depends on."""
    return ai_customer_lookup(customer_id)


@app.get("/model-history")
def model_history():
    """Raw metadata list, no strict response_model on purpose - real
    models_store/*.json shape has drifted across versions (e.g. not every
    file has every field), and over-constraining risks a 500 on an old
    version's file that simply predates a newer field. Mirrors
    /assistant/trace/{id}'s same "return the raw dict" choice below.

    serving_version names which of these is actually loaded. Callers must
    not assume that is the last entry: the registry can resolve an older
    artifact than the newest on disk (an MLflow "champion" alias pinned to
    a previous version, a retrain that saved but failed to register), and
    labelling the newest file "current production model" then misreports
    the live threshold, precision and confusion matrix."""
    return {
        "versions": dashboard_queries.load_all_churn_metadata(),
        "serving_version": _state["churn_version"],
    }


@app.get("/pipeline-status")
def pipeline_status():
    """Ingestion, drift, and retrain-summary status, sourced from
    dashboard_queries plus src.agents.cache's already-existing
    get_latest_retrain_summary."""
    ingestion_log = dashboard_queries.load_ingestion_log()
    drift_df = dashboard_queries.load_latest_drift()
    metadata_history = dashboard_queries.load_all_churn_metadata()
    batches_ingested = int(len(ingestion_log))

    return {
        "ingestion_log": ingestion_log.to_dict(orient="records"),
        "batches_ingested": batches_ingested,
        "total_simulated_batches": dashboard_queries.TOTAL_SIMULATED_BATCHES,
        "batches_to_next_retrain": max(
            0, dashboard_queries.RETRAIN_EVERY_N_BATCHES - (batches_ingested % dashboard_queries.RETRAIN_EVERY_N_BATCHES)
        ),
        "retrain_every_n_batches": dashboard_queries.RETRAIN_EVERY_N_BATCHES,
        "model_versions_trained": len(metadata_history),
        "drift": drift_df.to_dict(orient="records"),
        "latest_retrain_summary": get_latest_retrain_summary(),
    }


class AssistantQueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None  # accepted, not yet used - see AI_Customer_Intelligence_Claude_Code_Plan.md section 27


@app.post("/assistant/query", response_model=AssistantResponse)
def assistant_query(req: AssistantQueryRequest):
    """AI decision assistant (src/ai/graph.py): routes the question to the
    structured tools, controlled SQL, and RAG layers, aggregates their
    output as evidence, and only then asks the LLM to turn that evidence
    into an answer - never the other way around. Independent of the
    startup-loaded churn/recommender artifacts above (_state); the tool
    layer loads and caches its own copies (src/ai/tools/_artifacts.py)."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    return ai_run_query(req.query)


@app.get("/assistant/trace/{trace_id}")
def assistant_trace(trace_id: str):
    """Debugging/observability endpoint (AI_Customer_Intelligence_Claude_Code_Plan.md
    section 28) over the trace src/ai/graph.py wrote for a given request
    (src/ai/observability/tracing.py). 404s rather than 200-with-null on a
    miss, unlike most read paths in this API, since there's no meaningful
    partial answer for "this trace_id doesn't exist"."""
    trace = ai_get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"trace_id {trace_id!r} not found")
    return trace
