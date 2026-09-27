"""Prediction endpoints and the AI Agent Layer that narrates them.

/predict-churn and recommender index hits are answered purely from the
in-memory artifacts. /recommend falls back to one warehouse lookup when a
customer isn't in the recommender's bounded reference set (the common case,
see _recommend_with_fallback), and /explain-churn always reads the
customer's feature row plus the Postgres-backed LLM cache
(src/agents/cache.py).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.agents.cache import get_or_generate
from src.agents.explanation_agent import explain_churn as ai_explain_churn
from src.agents.groq_client import AgentCallFailed
from src.agents.outreach_agent import draft_outreach as ai_draft_outreach
from src.api import state
from src.api.schemas import (
    ChurnRequest,
    ChurnResponse,
    ExplainChurnResponse,
    OutreachDraftResponse,
    RecommendRequest,
    RecommendResponse,
)
from src.api.security import limit_llm_calls
from src.model.explain_churn import compute_shap_details, format_risk_factors_text
from src.model.train_churn import ALL_FEATURES as CHURN_FEATURES
from src.model.train_churn import ID_COL as CHURN_ID_COL
from src.model.train_recommender import PROFILE_CATEGORICAL as REC_PROFILE_CATEGORICAL
from src.model.train_recommender import PROFILE_NUMERIC as REC_PROFILE_NUMERIC
from src.model.train_recommender import SERVICE_COLUMNS as REC_SERVICE_COLUMNS
from src.model.train_recommender import recommend_for_customer, recommend_for_profile
from src.warehouse import get_pg_conn

log = logging.getLogger("api")

router = APIRouter(tags=["models"])


@router.get("/health")
def health():
    return {
        "status": "ok",
        "churn_model_loaded": state.models["churn"] is not None,
        "recommender_loaded": state.models["recommender"] is not None,
    }


@router.get("/model-info")
def model_info():
    return {
        "churn_model_version": state.models["churn_version"],
        "recommender_version": state.models["recommender_version"],
    }


@router.post("/predict-churn", response_model=ChurnResponse)
def predict_churn(req: ChurnRequest):
    artifact = state.require_churn_model()

    payload = req.model_dump()
    X = pd.DataFrame([{f: payload.get(f) for f in CHURN_FEATURES}], columns=CHURN_FEATURES)

    proba = float(artifact["pipeline"].predict_proba(X)[0, 1])
    threshold = state.churn_threshold()

    return ChurnResponse(
        churn_probability=round(proba, 4),
        churn_prediction=int(proba >= threshold),
        threshold_used=round(threshold, 4),
        model_version=state.models["churn_version"],
    )


def _fetch_profile_from_warehouse(customer_id: str):
    """One customer's profile + current subscriptions from the mart. Only
    called on a recommender index miss, keeping the common fast path free
    of any database dependency."""
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
    """Shared by /recommend and /outreach-draft so they can never disagree
    about what's recommended for the same customer."""
    artifact = state.models["recommender"]
    if artifact is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded")

    try:
        return recommend_for_customer(artifact, customer_id, top_n=top_n)
    except KeyError:
        # The reference set is deliberately bounded (~154K customers) while
        # the warehouse holds 1M, so a miss is the common case - look the
        # profile up and match it against the index instead of 404ing.
        try:
            profile, own_services = _fetch_profile_from_warehouse(customer_id)
        except Exception as exc:
            log.warning("warehouse profile lookup failed for %s: %s", customer_id, exc)
            raise HTTPException(
                status_code=503, detail="customer not in recommender index and warehouse lookup failed",
            ) from exc
        if profile is None:
            raise HTTPException(status_code=404, detail=f"customer_id {customer_id!r} not found") from None
        return recommend_for_profile(artifact, profile, own_services, top_n=top_n)


@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    recs = _recommend_with_fallback(req.customer_id, req.top_n)
    return RecommendResponse(
        customer_id=req.customer_id,
        recommendations=recs,
        model_version=state.models["recommender_version"],
    )


def _fetch_churn_row_from_warehouse(customer_id: str) -> Optional[pd.DataFrame]:
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
    """Shared by /explain-churn and /outreach-draft - the same fetch -> score
    -> SHAP -> cached-LLM sequence, so both endpoints can never produce two
    different explanations for the same customer and model version. None if
    the customer isn't found."""
    X = _fetch_churn_row_from_warehouse(customer_id)
    if X is None:
        return None

    artifact = state.models["churn"]
    X_for_predict = X.copy()
    for c in [f for f in CHURN_FEATURES if X_for_predict[f].dtype == bool]:
        X_for_predict[c] = X_for_predict[c].astype(float)
    churn_probability = float(artifact["pipeline"].predict_proba(X_for_predict)[0, 1])

    shap_details = compute_shap_details(state.explain_pipeline(artifact), X, top_k=5)[0]

    try:
        explanation = get_or_generate(
            customer_id, "explanation", state.models["churn_version"],
            lambda: ai_explain_churn(churn_probability, shap_details),
        )
        source = "llm"
    except AgentCallFailed as exc:
        log.warning("explanation agent unavailable for %s, falling back to raw SHAP: %s", customer_id, exc)
        explanation = format_risk_factors_text(shap_details)
        source = "fallback"

    return {
        "churn_probability": churn_probability,
        "risk_factors": shap_details,
        "explanation": explanation,
        "source": source,
    }


@router.get(
    "/explain-churn/{customer_id}",
    response_model=ExplainChurnResponse,
    dependencies=[Depends(limit_llm_calls)],
)
def explain_churn_endpoint(customer_id: str):
    """A plain-English explanation of why this customer is flagged, grounded
    in the churn model's own SHAP attribution. Narrates an already-final
    prediction, never changes it, and degrades to the raw SHAP factors
    rather than returning a 5xx when the LLM is unavailable."""
    state.require_churn_model()

    result = _get_explanation(customer_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"customer_id {customer_id!r} not found")

    return ExplainChurnResponse(
        customer_id=customer_id,
        churn_probability=round(result["churn_probability"], 4),
        risk_factors=result["risk_factors"],
        explanation=result["explanation"],
        source=result["source"],
        model_version=state.models["churn_version"],
    )


@router.post(
    "/outreach-draft/{customer_id}",
    response_model=OutreachDraftResponse,
    dependencies=[Depends(limit_llm_calls)],
)
def outreach_draft(customer_id: str):
    """The /explain-churn explanation plus a drafted retention message for
    the recommender's top suggestion. recommended_service/draft are null
    (200, not an error) when the customer already has every service."""
    state.require_churn_model()

    explanation_result = _get_explanation(customer_id)
    if explanation_result is None:
        raise HTTPException(status_code=404, detail=f"customer_id {customer_id!r} not found")

    recommended_service = None
    draft = None
    draft_source = None
    if state.models["recommender"] is not None:
        try:
            recs = _recommend_with_fallback(customer_id, top_n=1)
        except HTTPException:
            recs = []  # recommender miss for this customer - soft-fail, not this endpoint's error
        if recs:
            recommended_service = recs[0]["service"]
            try:
                draft = get_or_generate(
                    customer_id, "outreach", state.models["churn_version"],
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
        model_version=state.models["churn_version"],
    )
