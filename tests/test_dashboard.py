"""
Regression tests for dashboard functions that read the churn artifact's
pipeline object directly.

Why this file exists: adding probability calibration
(`CalibratedClassifierCV`, see train_churn.py) silently broke the
dashboard's SHAP risk-factor column and feature-importance fallback - both
called `pipeline.named_steps[...]`, which only exists on a raw sklearn
`Pipeline`. `CalibratedClassifierCV` has no such attribute, so
`column_importances` was catching the resulting `AttributeError` and
returning `{}`, and the dashboard rendered an empty "Risk factors" column
with no error at all. The fix stores the pre-calibration pipeline in the
artifact under `base_pipeline` and routes explainability through it. These
tests build both pipeline shapes directly (no DB, no saved artifact) and
pin the exact contract that broke, so it cannot regress silently again.

Deliberately built on synthetic in-memory data, like test_model.py, so
this runs in CI with no database and no artifact checked out.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from src.dashboard.app import column_importances
from src.model.train_churn import (
    ALL_FEATURES,
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_pipeline,
)

N_ROWS = 300


def _synthetic_frame(n=N_ROWS, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC_FEATURES})
    for c in BOOLEAN_FEATURES:
        df[c] = rng.integers(0, 2, size=n).astype(float)
    for c in CATEGORICAL_FEATURES:
        df[c] = rng.choice(["a", "b", "c"], size=n)
    # A weak but real signal, so the fitted model has non-trivial
    # feature_importances_ rather than a degenerate all-zero split.
    df[TARGET] = (df[NUMERIC_FEATURES[0]] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return df


@pytest.fixture(scope="module")
def fitted_pipelines():
    """One base (raw sklearn Pipeline) fit, and that same fit wrapped in
    CalibratedClassifierCV - exactly the two shapes train_and_save() now
    produces and saves as "base_pipeline" / "pipeline"."""
    df = _synthetic_frame()
    X, y = df[ALL_FEATURES], df[TARGET]

    base = build_pipeline()
    base.fit(X.iloc[:200], y.iloc[:200])

    calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid", cv="prefit")
    calibrated.fit(X.iloc[200:], y.iloc[200:])

    return base, calibrated


def test_column_importances_works_on_the_raw_pipeline(fitted_pipelines):
    base, _ = fitted_pipelines
    importances = column_importances(base)

    assert importances != {}
    assert set(importances.keys()) <= set(ALL_FEATURES)
    # Every raw feature contributes a non-negative share of total gain.
    assert all(v >= 0 for v in importances.values())


def test_column_importances_degrades_gracefully_on_a_calibrated_wrapper(fitted_pipelines):
    """Pins the exact failure mode this file exists to catch:
    CalibratedClassifierCV has no .named_steps, so calling this function
    on it directly must not raise - it must return {} so callers can fall
    back, not crash the page."""
    _, calibrated = fitted_pipelines
    assert column_importances(calibrated) == {}


def test_base_pipeline_is_what_dashboard_code_must_use_for_explainability(fitted_pipelines):
    """Documents the routing rule the dashboard now follows: explainability
    functions (column_importances, SHAP TreeExplainer) must be given
    artifact["base_pipeline"], never artifact["pipeline"], once the
    artifact is calibrated."""
    base, calibrated = fitted_pipelines
    artifact = {"pipeline": calibrated, "base_pipeline": base}

    explain_pipeline = artifact.get("base_pipeline", artifact["pipeline"])
    assert column_importances(explain_pipeline) != {}


def test_artifact_without_base_pipeline_falls_back_to_pipeline(fitted_pipelines):
    """An artifact trained before calibration was added has no
    "base_pipeline" key - in that case "pipeline" already IS the raw
    Pipeline, so the same .get() fallback must still resolve to something
    with real feature importances, not silently produce {}."""
    base, _ = fitted_pipelines
    old_style_artifact = {"pipeline": base}  # no "base_pipeline" key at all

    explain_pipeline = old_style_artifact.get("base_pipeline", old_style_artifact["pipeline"])
    assert column_importances(explain_pipeline) != {}


def test_calibration_does_not_change_predict_proba_availability(fitted_pipelines):
    """The serving path (score_all_customers, /predict-churn) calls
    pipeline.predict_proba directly and must keep working on the
    calibrated wrapper - this is the one call CalibratedClassifierCV
    *does* support natively, unlike .named_steps."""
    _, calibrated = fitted_pipelines
    df = _synthetic_frame(n=10, seed=1)
    proba = calibrated.predict_proba(df[ALL_FEATURES])
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
