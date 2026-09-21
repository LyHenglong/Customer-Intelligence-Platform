"""Shared, cached loader for the current churn/recommender joblib artifacts.

Mirrors src/model/api.py's load_models() and src/dashboard/app.py's
load_latest_artifact() - a third process (this AI tool layer) needs the
same artifacts, so this factors the pattern into one place rather than a
third copy. Resolution itself (MLflow "champion" alias, falling back to
glob-latest-by-timestamp) lives in src/model/registry.py, shared by all
three call sites.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib

from src.model.registry import resolve_model_path

MODELS_DIR = Path(__file__).resolve().parents[3] / "models_store"


@lru_cache(maxsize=1)
def load_churn_artifact() -> tuple[Optional[dict], Optional[str]]:
    path = resolve_model_path("churn_model", "churn_model_*.joblib", models_dir=MODELS_DIR)
    if path is None:
        return None, None
    return joblib.load(path), path.stem.replace("churn_model_", "")


@lru_cache(maxsize=1)
def load_recommender_artifact() -> tuple[Optional[dict], Optional[str]]:
    path = resolve_model_path("recommender", "recommender_*.joblib", models_dir=MODELS_DIR)
    if path is None:
        return None, None
    return joblib.load(path), path.stem.replace("recommender_", "")


def clear_artifact_cache() -> None:
    """Test-only. lru_cache would otherwise pin whichever artifact loaded
    first for the rest of the process, which breaks tests that point
    MODELS_DIR at different fixture directories across test cases."""
    load_churn_artifact.cache_clear()
    load_recommender_artifact.cache_clear()
