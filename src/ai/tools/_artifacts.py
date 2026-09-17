"""Shared, cached loader for the latest churn/recommender joblib artifacts.

Mirrors src/model/api.py's load_models()/_latest() glob-by-timestamp
pattern and src/dashboard/app.py's load_latest_artifact() - a third
process (this AI tool layer) needs the same artifacts, so this factors
the pattern into one place rather than a third copy.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib

MODELS_DIR = Path(__file__).resolve().parents[3] / "models_store"


def _latest_path(pattern: str) -> Optional[Path]:
    matches = sorted(MODELS_DIR.glob(pattern))
    return matches[-1] if matches else None


@lru_cache(maxsize=1)
def load_churn_artifact() -> tuple[Optional[dict], Optional[str]]:
    path = _latest_path("churn_model_*.joblib")
    if path is None:
        return None, None
    return joblib.load(path), path.stem.replace("churn_model_", "")


@lru_cache(maxsize=1)
def load_recommender_artifact() -> tuple[Optional[dict], Optional[str]]:
    path = _latest_path("recommender_*.joblib")
    if path is None:
        return None, None
    return joblib.load(path), path.stem.replace("recommender_", "")


def clear_artifact_cache() -> None:
    """Test-only. lru_cache would otherwise pin whichever artifact loaded
    first for the rest of the process, which breaks tests that point
    MODELS_DIR at different fixture directories across test cases."""
    load_churn_artifact.cache_clear()
    load_recommender_artifact.cache_clear()
