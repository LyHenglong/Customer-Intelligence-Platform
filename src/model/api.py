"""
FastAPI serving layer for the churn classifier, content-based recommender,
and the AI Agent Layer that narrates their output (src/agents/). Loads the
latest versioned artifact of each model at startup (by filename timestamp
- see train_churn.py / train_recommender.py) and serves from those
in-memory artifacts. Churn predictions and recommender index hits need no
database at all; /recommend falls back to a single warehouse lookup only
when a customer isn't in the recommender's bounded reference set (see the
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
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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
from src.agents.cache import get_or_generate
from src.ai.graph import run_query as ai_run_query
from src.ai.observability.tracing import get_trace as ai_get_trace
from src.ai.schemas import AssistantResponse

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api")

MODELS_DIR = Path(__file__).resolve().parents[2] / "models_store"

app = FastAPI(title="Telecom Churn & Recommendation API", version="1.0.0")

_state: dict = {"churn": None, "churn_version": None, "recommender": None, "recommender_version": None}


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


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if _state["recommender"] is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded")

    artifact = _state["recommender"]
    try:
        # Fast path: customer is in the k-NN reference set, answered purely
        # from the in-memory artifact.
        recs = recommend_for_customer(artifact, req.customer_id, top_n=req.top_n)
    except KeyError:
        # Miss. The reference set is deliberately bounded (~154K customers)
        # while the warehouse holds 1M, so this is the *common* case, not an
        # edge case - answering it with a 404 left ~85% of customers with no
        # recommendation at all. Look the profile up and match it against
        # the index instead.
        try:
            profile, own_services = _fetch_profile_from_warehouse(req.customer_id)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"customer not in recommender index and warehouse lookup failed: {exc}",
            )
        if profile is None:
            raise HTTPException(
                status_code=404, detail=f"customer_id {req.customer_id!r} not found"
            )
        recs = recommend_for_profile(artifact, profile, own_services, top_n=req.top_n)

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

    X = _fetch_churn_row_from_warehouse(customer_id)
    if X is None:
        raise HTTPException(status_code=404, detail=f"customer_id {customer_id!r} not found")

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

    return ExplainChurnResponse(
        customer_id=customer_id,
        churn_probability=round(churn_probability, 4),
        risk_factors=shap_details,
        explanation=explanation,
        source=source,
        model_version=_state["churn_version"],
    )


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
