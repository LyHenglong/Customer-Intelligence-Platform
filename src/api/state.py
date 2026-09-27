"""The in-memory model artifacts every router serves from.

Loaded once at startup (src/api/app.py) from whatever src/model/registry.py
resolves - the MLflow "champion" alias when configured, otherwise the
newest versioned file in models_store/. Churn predictions and recommender
index hits need no database at all.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import HTTPException

from src.model import dashboard_queries
from src.model.registry import resolve_model_path

log = logging.getLogger("api")

MODELS_DIR = Path(__file__).resolve().parents[2] / "models_store"

models: dict = {"churn": None, "churn_version": None, "recommender": None, "recommender_version": None}


def load_models() -> None:
    churn_path = resolve_model_path("churn_model", "churn_model_*.joblib")
    if churn_path:
        models["churn"] = joblib.load(churn_path)
        models["churn_version"] = churn_path.stem.replace("churn_model_", "")
        log.info("Loaded churn model %s", churn_path.name)
    else:
        log.warning("No churn model artifact found in %s", MODELS_DIR)

    rec_path = resolve_model_path("recommender", "recommender_*.joblib")
    if rec_path:
        models["recommender"] = joblib.load(rec_path)
        models["recommender_version"] = rec_path.stem.replace("recommender_", "")
        log.info("Loaded recommender artifact %s", rec_path.name)
    else:
        log.warning("No recommender artifact found in %s", MODELS_DIR)


def require_churn_model() -> dict:
    if models["churn"] is None:
        raise HTTPException(status_code=503, detail="Churn model not loaded")
    return models["churn"]


def churn_threshold(override: float | None = None) -> float:
    """The decision threshold saved with the artifact (chosen at training
    time to hit a target recall - see train_churn.py), unless the caller
    overrides it. Artifacts saved before thresholds existed fall back to 0.5."""
    if override is not None:
        return override
    return models["churn"].get("threshold", 0.5)


def explain_pipeline(artifact: dict):
    """The uncalibrated pipeline SHAP should explain - calibration wraps the
    base model, and TreeExplainer needs the tree model itself."""
    return artifact.get("base_pipeline", artifact["pipeline"])


def get_scored_customers() -> pd.DataFrame:
    """The scored population, from the cache shared with the AI tool layer
    (src/ai/tools/churn_tool.py reads the same frame rather than re-scoring
    the whole warehouse). TTL, locking and the multi-replica caveat are
    documented in dashboard_queries."""
    return dashboard_queries.get_scored_customers(models["churn"], models["churn_version"])
