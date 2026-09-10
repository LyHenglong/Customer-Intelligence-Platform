"""
FastAPI serving layer for the churn classifier and content-based
recommender. Loads the latest versioned artifact of each model at startup
(by filename timestamp - see train_churn.py / train_recommender.py) and
serves from those in-memory artifacts. Churn predictions and recommender
index hits need no database at all; /recommend falls back to a single
warehouse lookup only when a customer isn't in the recommender's bounded
reference set (see the endpoint for why that is the common case).

Run locally:
    uvicorn src.model.api:app --reload

Endpoints:
    GET  /health
    GET  /model-info
    POST /predict-churn
    POST /recommend
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
from src.warehouse import get_pg_conn
from src.model.train_recommender import recommend_for_customer, recommend_for_profile
from src.model.train_recommender import PROFILE_NUMERIC as REC_PROFILE_NUMERIC
from src.model.train_recommender import PROFILE_CATEGORICAL as REC_PROFILE_CATEGORICAL
from src.model.train_recommender import SERVICE_COLUMNS as REC_SERVICE_COLUMNS

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api")

MODELS_DIR = Path(__file__).resolve().parents[2] / "models_store"

app = FastAPI(title="Telecom Churn & Recommendation API", version="1.0.0")

_state: dict = {"churn": None, "churn_version": None, "recommender": None, "recommender_version": None}


def _latest(pattern: str) -> Optional[Path]:
    matches = sorted(MODELS_DIR.glob(pattern))
    return matches[-1] if matches else None


@app.on_event("startup")
def load_models() -> None:
    churn_path = _latest("churn_model_*.joblib")
    if churn_path:
        _state["churn"] = joblib.load(churn_path)
        _state["churn_version"] = churn_path.stem.replace("churn_model_", "")
        log.info("Loaded churn model %s", churn_path.name)
    else:
        log.warning("No churn model artifact found in %s", MODELS_DIR)

    rec_path = _latest("recommender_*.joblib")
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
