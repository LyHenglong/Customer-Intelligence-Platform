"""
Unit tests for src/model/registry.py's resolve_model_path(): the shared
resolution logic behind all three model-loading call sites (src/model/
api.py, src/ai/tools/_artifacts.py, src/dashboard/app.py).

Deliberately built on synthetic in-memory fixtures and a mocked
mlflow.tracking.MlflowClient rather than a live MLflow server, so these
run in CI with no server, no Postgres backend store, and no real
artifacts (same philosophy as tests/test_model.py).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.model import registry


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    return tmp_path


def _touch(path: Path) -> Path:
    path.write_text("fake artifact")
    return path


# --------------------------------------------------------------------------
# No MLflow configured
# --------------------------------------------------------------------------

def test_resolve_falls_back_to_glob_latest_when_tracking_uri_unset(models_dir, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    older = _touch(models_dir / "churn_model_20260101T000000Z.joblib")
    newer = _touch(models_dir / "churn_model_20260102T000000Z.joblib")

    result = registry.resolve_model_path("churn_model", "churn_model_*.joblib", models_dir=models_dir)

    assert result == newer
    assert result != older


def test_resolve_returns_none_when_no_artifacts_exist(models_dir, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    result = registry.resolve_model_path("churn_model", "churn_model_*.joblib", models_dir=models_dir)

    assert result is None


def test_telco_benchmark_artifacts_can_never_hijack_the_production_model(models_dir, monkeypatch):
    """src/model/train_churn_telco.py trains on 7,043 real Telco rows with a
    completely different schema, purely as a signal benchmark. Because the
    production glob takes the newest match, a name like
    "churn_model_telco_*.joblib" would silently be served against
    marts.customer_360's 1,000,000 customers the moment the benchmark ran.
    The "telco_churn_" prefix is what prevents that, so it is pinned here
    rather than left to a future rename."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    production = _touch(models_dir / "churn_model_20260101T000000Z.joblib")
    # Deliberately newer - if the prefix stopped protecting us, recency
    # would hand this one the production slot.
    _touch(models_dir / "telco_churn_20260601T000000Z.joblib")

    result = registry.resolve_model_path("churn_model", "churn_model_*.joblib", models_dir=models_dir)

    assert result == production


# --------------------------------------------------------------------------
# MLflow configured
# --------------------------------------------------------------------------

def test_resolve_falls_back_to_glob_when_mlflow_client_raises(models_dir, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    newer = _touch(models_dir / "churn_model_20260102T000000Z.joblib")

    class RaisingClient:
        def __init__(self, tracking_uri=None):
            pass

        def get_model_version_by_alias(self, name, alias):
            raise RuntimeError("connection refused")

    import mlflow
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", RaisingClient)

    result = registry.resolve_model_path("churn_model", "churn_model_*.joblib", models_dir=models_dir)

    assert result == newer


def test_resolve_falls_back_to_glob_when_champion_file_missing_locally(models_dir, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    newer = _touch(models_dir / "churn_model_20260102T000000Z.joblib")

    class MissingFileClient:
        def __init__(self, tracking_uri=None):
            pass

        def get_model_version_by_alias(self, name, alias):
            return SimpleNamespace(version="3", tags={"model_file": "churn_model_does_not_exist.joblib"})

    import mlflow
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", MissingFileClient)

    result = registry.resolve_model_path("churn_model", "churn_model_*.joblib", models_dir=models_dir)

    assert result == newer


def test_resolve_prefers_champion_over_newer_untagged_artifact(models_dir, monkeypatch):
    """Proves champion-pinning actually overrides "newest wins" - an older
    aliased version should be returned even when a newer, unregistered
    file also exists in models_dir."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    champion = _touch(models_dir / "churn_model_20260101T000000Z.joblib")
    _touch(models_dir / "churn_model_20260102T000000Z.joblib")  # newer, but not champion

    class ChampionClient:
        def __init__(self, tracking_uri=None):
            pass

        def get_model_version_by_alias(self, name, alias):
            assert name == "churn_model"
            assert alias == "champion"
            return SimpleNamespace(version="1", tags={"model_file": champion.name})

    import mlflow
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", ChampionClient)

    result = registry.resolve_model_path("churn_model", "churn_model_*.joblib", models_dir=models_dir)

    assert result == champion
