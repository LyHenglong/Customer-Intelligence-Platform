"""churn_analysis tool - population-level churn statistics.

Streams marts.customer_360 through the churn model in bounded batches -
the same memory-safety pattern as src/model/dashboard_queries.py's
score_all_customers, reimplemented independently here rather than
imported, since this tool also accepts optional SQL filters to narrow
the population before scoring (mirroring customer_search's SQL-first
approach) - a capability score_all_customers doesn't have, since it
always scores the full, unfiltered population.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai.schemas import ChurnAnalysisResult
from src.ai.tools._artifacts import load_churn_artifact
from src.model.train_churn import ALL_FEATURES as CHURN_FEATURES
from src.warehouse import stream_query

_ALLOWED_FILTER_COLUMNS = {"contract", "tenure_bucket"}


def churn_analysis(filters: dict | None = None, batch_size: int = 25_000) -> ChurnAnalysisResult:
    filters = filters or {}
    unknown = set(filters) - _ALLOWED_FILTER_COLUMNS
    if unknown:
        raise ValueError(
            f"unsupported churn_analysis filter(s) {unknown}; allowed: {sorted(_ALLOWED_FILTER_COLUMNS)}"
        )

    where_clause, params = "", None
    if filters:
        where_clause = f"WHERE {' AND '.join(f'{col} = %s' for col in filters)}"
        params = list(filters.values())

    churn_artifact, churn_version = load_churn_artifact()
    score_cols = list(dict.fromkeys(["customer_id", "churn"] + CHURN_FEATURES))

    probas: list[np.ndarray] = []
    actual_churn: list[np.ndarray] = []

    def _score_batch(batch: pd.DataFrame) -> pd.DataFrame:
        if churn_artifact is not None:
            X = batch[CHURN_FEATURES].copy()
            for c in [f for f in CHURN_FEATURES if X[f].dtype == bool]:
                X[c] = X[c].astype(float)
            probas.append(churn_artifact["pipeline"].predict_proba(X)[:, 1].astype(np.float32))
        actual_churn.append(batch["churn"].to_numpy())
        # Nothing is retained past this batch - stream_query concatenates
        # whatever comes back, and only the reduced scores/labels matter.
        return batch.iloc[0:0]

    stream_query(
        f"SELECT {', '.join(score_cols)} FROM marts.customer_360 {where_clause}",
        columns=score_cols, batch_rows=batch_size, transform=_score_batch, params=params,
    )

    actual = np.concatenate(actual_churn) if actual_churn else np.array([])
    population_size = int(len(actual))
    current_churn_rate = float(actual.mean()) if population_size else 0.0

    if not probas or churn_artifact is None:
        return ChurnAnalysisResult(
            population_size=population_size,
            current_churn_rate=round(current_churn_rate, 4),
            predicted_high_risk_count=0,
            mean_churn_probability=0.0,
            median_churn_probability=0.0,
            model_version=churn_version,
            threshold=0.0,
            filters_applied=filters,
        )

    all_proba = np.concatenate(probas)
    threshold = float(churn_artifact.get("threshold", 0.5))
    return ChurnAnalysisResult(
        population_size=population_size,
        current_churn_rate=round(current_churn_rate, 4),
        predicted_high_risk_count=int((all_proba >= threshold).sum()),
        mean_churn_probability=round(float(all_proba.mean()), 4),
        median_churn_probability=round(float(np.median(all_proba)), 4),
        model_version=churn_version,
        threshold=round(threshold, 4),
        filters_applied=filters,
    )
