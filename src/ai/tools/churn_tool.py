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

import logging
import threading
import time

import numpy as np
import pandas as pd

from src.ai.schemas import ChurnAnalysisResult
from src.ai.tools._artifacts import load_churn_artifact
from src.model import dashboard_queries
from src.model.train_churn import ALL_FEATURES as CHURN_FEATURES
from src.warehouse import stream_query

log = logging.getLogger("ai.tools.churn_tool")

_ALLOWED_FILTER_COLUMNS = {"contract", "tenure_bucket"}

# One unfiltered call streams and scores all 1M rows of customer_360 -
# measured at 50-120s end to end, which is most of an assistant request's
# latency and is paid again on every question that routes to ML_ANALYSIS,
# for a number that only changes when a batch lands or a model is
# promoted. Same hand-rolled TTL cache as src/model/api.py's
# _get_scored_customers (and the same reasoning for not reaching for
# Redis), keyed by filters so a segment question and the population
# baseline it's compared against cache independently.
#
# The lock spans the recompute, not just the lookup: two concurrent
# cache-miss requests would otherwise both start a full scoring pass and
# double peak memory in a container already sized once for one.
_CACHE_TTL_SECONDS = 600
_cache: dict[tuple, tuple[float, ChurnAnalysisResult]] = {}
_cache_lock = threading.Lock()


def churn_analysis(filters: dict | None = None, batch_size: int = 25_000) -> ChurnAnalysisResult:
    """Cached for _CACHE_TTL_SECONDS - see the note above. Call
    clear_cache() first if you need a guaranteed-fresh read."""
    key = (tuple(sorted((filters or {}).items())), batch_size)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None and time.monotonic() < cached[0]:
            return cached[1]
        result = _compute_churn_analysis(filters=filters, batch_size=batch_size)
        _cache[key] = (time.monotonic() + _CACHE_TTL_SECONDS, result)
        return result


def clear_cache() -> None:
    """Drops every memoized result - for tests, and for any caller that
    knows a retrain just invalidated these numbers."""
    with _cache_lock:
        _cache.clear()


def _from_scored_cache(filters: dict, churn_artifact, churn_version) -> ChurnAnalysisResult | None:
    """Answers from the already-scored population if it is in memory.

    The serving process keeps every customer scored in
    dashboard_queries' cache (warmed at API startup, an hour's TTL), and
    since that frame carries the categorical segment columns plus the
    churn label, every figure this tool returns can be computed from it
    with a boolean mask. Re-streaming and re-scoring 1,000,000 rows from
    Postgres to recompute what is already in RAM cost 209s of a measured
    211s assistant request.

    Returns None rather than computing the frame when the cache is cold:
    the caller then does its own filtered query, which for a narrow
    segment is far cheaper than a full population pass, and nobody waits
    on somebody else's multi-minute scoring run."""
    if churn_artifact is None:
        return None
    frame = dashboard_queries.peek_scored_customers(churn_version)
    if frame is None or "churn" not in frame.columns:
        return None

    for column, value in filters.items():
        if column not in frame.columns:
            return None  # not a column that rides along; fall back to SQL
        frame = frame[frame[column] == value]

    population_size = int(len(frame))
    if population_size == 0:
        return ChurnAnalysisResult(
            population_size=0, current_churn_rate=0.0, predicted_high_risk_count=0,
            mean_churn_probability=0.0, median_churn_probability=0.0,
            model_version=churn_version, threshold=0.0, filters_applied=filters,
        )

    proba = frame["churn_probability"].to_numpy()
    threshold = float(churn_artifact.get("threshold", 0.5))
    return ChurnAnalysisResult(
        population_size=population_size,
        current_churn_rate=round(float(frame["churn"].mean()), 4),
        predicted_high_risk_count=int((proba >= threshold).sum()),
        mean_churn_probability=round(float(proba.mean()), 4),
        median_churn_probability=round(float(np.median(proba)), 4),
        model_version=churn_version,
        threshold=round(threshold, 4),
        filters_applied=filters,
    )


def _compute_churn_analysis(filters: dict | None = None, batch_size: int = 25_000) -> ChurnAnalysisResult:
    filters = filters or {}
    unknown = set(filters) - _ALLOWED_FILTER_COLUMNS
    if unknown:
        raise ValueError(
            f"unsupported churn_analysis filter(s) {unknown}; allowed: {sorted(_ALLOWED_FILTER_COLUMNS)}"
        )

    artifact_for_cache, version_for_cache = load_churn_artifact()
    from_cache = _from_scored_cache(filters, artifact_for_cache, version_for_cache)
    if from_cache is not None:
        log.info("churn_analysis served from the shared scored-population cache (filters=%s)", filters or "none")
        return from_cache

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
