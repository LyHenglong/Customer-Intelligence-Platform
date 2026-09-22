"""
Shared per-customer SHAP explanation logic.

Shared by the FastAPI /explain-churn/{customer_id} endpoint and the
/at-risk endpoint's per-row SHAP output, so both produce identical
explanations from identical code rather than maintaining two
implementations that could silently drift apart - the exact kind of
duplication this project has avoided everywhere else
(recommend_for_profile, stream_query, etc.).

Callers must pass the *raw* (uncalibrated) pipeline - artifact["base_pipeline"]
if present, else artifact["pipeline"] for pre-calibration artifacts - never
the calibrated CalibratedClassifierCV wrapper directly. See train_churn.py
and the "Probability calibration" section of the README for why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.model.train_churn import ALL_FEATURES as CHURN_FEATURES
from src.model.train_churn import CATEGORICAL_FEATURES as CHURN_CATEGORICAL

# Keyed by id(pipeline). Safe because a pipeline object is only ever
# loaded once per process (src/model/api.py's load_models(), run once at
# startup), so id() is stable for the process's lifetime.
_explainer_cache: dict[int, "shap.TreeExplainer"] = {}


def _feature_name_to_column(name: str) -> str:
    """Maps a preprocessor's expanded output name (e.g. 'num__num_complaints',
    'cat__contract_two_year') back to the real customer_360 column name."""
    if name.startswith("num__"):
        return name[len("num__"):]
    if name.startswith("cat__"):
        rest = name[len("cat__"):]
        return next((c for c in CHURN_CATEGORICAL if rest.startswith(c + "_")), rest)
    return name


def get_explainer(pipeline) -> "shap.TreeExplainer":
    key = id(pipeline)
    if key not in _explainer_cache:
        _explainer_cache[key] = shap.TreeExplainer(pipeline.named_steps["model"])
    return _explainer_cache[key]


def compute_shap_details(pipeline, customers_df: pd.DataFrame, top_k: int = 5) -> list[list[dict]]:
    """Returns, per row of customers_df, the top_k |SHAP value| features as
    structured dicts: {"feature": str, "shap_value": float, "direction": str}.

    Structured rather than pre-formatted text (contrast with this
    module's own `format_risk_factors_text` output) because the LLM
    explanation agent needs the real numbers as grounding input in its
    prompt, not a display string built for a table cell.
    """
    X = customers_df[CHURN_FEATURES].copy()
    for c in [f for f in CHURN_FEATURES if customers_df[f].dtype == bool]:
        X[c] = X[c].astype(float)

    preproc = pipeline.named_steps["preprocess"]
    X_transformed = preproc.transform(X)
    X_transformed = X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed
    feature_names = preproc.get_feature_names_out()

    explainer = get_explainer(pipeline)
    shap_values = explainer.shap_values(X_transformed)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    results = []
    for i in range(len(customers_df)):
        contributions = sv[i]
        ranked_idx = np.argsort(np.abs(contributions))[::-1][:top_k]
        row = [
            {
                "feature": _feature_name_to_column(feature_names[idx]),
                "shap_value": float(contributions[idx]),
                "direction": "increases risk" if contributions[idx] > 0 else "decreases risk",
            }
            for idx in ranked_idx
        ]
        results.append(row)
    return results


def format_risk_factors_text(shap_details: list[dict]) -> str:
    """Compact display format: '▲ feature · ▼ feature'. Also the LLM
    explanation agent's fallback text when the Groq call is unavailable."""
    parts = []
    for d in shap_details:
        sign = "▲" if d["shap_value"] > 0 else "▼"
        parts.append(f"{sign} {d['feature'].replace('_', ' ')}")
    return " · ".join(parts)
