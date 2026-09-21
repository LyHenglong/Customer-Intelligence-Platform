"""
Population Stability Index (PSI) drift detection between ingested batches.

Why this exists: the retraining trigger in the DAG was originally a pure
batch counter ("retrain every 3rd batch"), which retrains on a fixed
schedule whether or not anything about the incoming data has actually
changed. PSI compares the distribution of each feature in a newly-arrived
batch against a reference batch and quantifies how far it has moved, so
retraining can also be triggered by evidence rather than only by a count.

PSI is the standard drift metric in credit risk and churn modeling:

    PSI = sum over bins of  (actual% - expected%) * ln(actual% / expected%)

It is unitless and comes with widely-used interpretation bands
(Siddiqi, Credit Risk Scorecards):

    PSI < 0.10          stable        - no action
    0.10 <= PSI < 0.25  moderate      - investigate
    PSI >= 0.25         significant   - retrain

Two deliberate choices:

1. Bin edges come from the *reference* distribution only (quantile-based),
   never recomputed on the new batch. Re-binning per batch would make every
   batch look identical to itself and hide exactly the shift being measured.
2. NULLs are their own bucket rather than dropped. A feature whose missing
   rate jumps from 3% to 40% has drifted in the way that matters most
   operationally, and dropping NULLs would score that as perfectly stable.

Honest note on this dataset: the 13 batches are sequential slices of a
single pre-shuffled synthetic file (see README), so the *expected* result
here is near-zero PSI on every feature. That is a correct negative result,
not a broken detector - tests/test_drift.py asserts the detector fires on
data that genuinely has shifted.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.monitoring.alerting import send_slack_alert
from src.warehouse import get_pg_conn, stream_query

log = logging.getLogger(__name__)

# Smoothing floor. A bin that is empty in one distribution but populated in
# the other would otherwise produce ln(0) = -inf and make PSI meaningless.
EPSILON = 1e-6

MODERATE_THRESHOLD = 0.10
SIGNIFICANT_THRESHOLD = 0.25

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "tenure",
    "monthlycharges",
    "totalcharges",
    "num_services",
    "customer_satisfaction",
    "num_complaints",
    "num_service_calls",
    "late_payments",
    "avg_monthly_gb",
    "days_since_last_interaction",
    "credit_score",
    "total_active_services",
]

CATEGORICAL_FEATURES = [
    "gender",
    "education",
    "marital_status",
    "contract",
    "payment_method",
]

DRIFT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def severity(psi_value: float) -> str:
    """Maps a PSI value onto the standard interpretation bands."""
    if psi_value >= SIGNIFICANT_THRESHOLD:
        return "significant"
    if psi_value >= MODERATE_THRESHOLD:
        return "moderate"
    return "stable"


def _psi_from_proportions(expected_pct: np.ndarray, actual_pct: np.ndarray) -> float:
    expected_pct = np.clip(expected_pct, EPSILON, None)
    actual_pct = np.clip(actual_pct, EPSILON, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def numeric_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """PSI for a continuous feature, using reference-derived quantile bins.

    Quantile bins rather than equal-width: equal-width bins on a skewed
    feature (income, totalcharges) put almost all mass in one bin, which
    makes PSI insensitive to real movement in the tail.
    """
    expected = pd.to_numeric(expected, errors="coerce")
    actual = pd.to_numeric(actual, errors="coerce")

    exp_null = float(expected.isna().mean()) if len(expected) else 0.0
    act_null = float(actual.isna().mean()) if len(actual) else 0.0

    exp_valid = expected.dropna()
    act_valid = actual.dropna()
    if exp_valid.empty or act_valid.empty:
        # Nothing to compare on the value axis; the missingness shift is the
        # only signal available.
        return _psi_from_proportions(
            np.array([exp_null, 1 - exp_null]), np.array([act_null, 1 - act_null])
        )

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(exp_valid, quantiles))
    if len(edges) < 2:
        # Constant reference feature: it can only drift by ceasing to be
        # constant, so compare "equal to that value" vs "not equal".
        only = edges[0]
        exp_pct = np.array([1.0, 0.0])
        act_pct = np.array(
            [float((act_valid == only).mean()), float((act_valid != only).mean())]
        )
    else:
        # Open the outer edges so values beyond the reference range - the
        # most obvious kind of drift - land in the end bins instead of being
        # dropped as out-of-range.
        edges = edges.astype(float)
        edges[0], edges[-1] = -np.inf, np.inf
        exp_counts = np.histogram(exp_valid, bins=edges)[0]
        act_counts = np.histogram(act_valid, bins=edges)[0]
        exp_pct = exp_counts / max(exp_counts.sum(), 1)
        act_pct = act_counts / max(act_counts.sum(), 1)

    # Rescale the value bins by the non-null share and append NULL as its
    # own bucket, so both axes of change live in one number.
    exp_full = np.append(exp_pct * (1 - exp_null), exp_null)
    act_full = np.append(act_pct * (1 - act_null), act_null)
    return _psi_from_proportions(exp_full, act_full)


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """PSI for a discrete feature, over the union of observed categories."""
    exp_counts = expected.fillna("__NULL__").astype(str).value_counts()
    act_counts = actual.fillna("__NULL__").astype(str).value_counts()

    categories = sorted(set(exp_counts.index) | set(act_counts.index))
    exp_pct = np.array([exp_counts.get(c, 0) for c in categories], dtype=float)
    act_pct = np.array([act_counts.get(c, 0) for c in categories], dtype=float)
    exp_pct /= max(exp_pct.sum(), 1)
    act_pct /= max(act_pct.sum(), 1)
    return _psi_from_proportions(exp_pct, act_pct)


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: Optional[Iterable[str]] = None,
    categorical_features: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Scores every feature present in both frames.

    Pure function - no I/O - which is what makes it directly testable
    without a database.
    """
    numeric_features = list(
        numeric_features if numeric_features is not None else NUMERIC_FEATURES
    )
    categorical_features = list(
        categorical_features if categorical_features is not None else CATEGORICAL_FEATURES
    )

    rows = []
    for feature in numeric_features:
        if feature in reference.columns and feature in current.columns:
            rows.append(
                {
                    "feature": feature,
                    "feature_type": "numeric",
                    "psi": numeric_psi(reference[feature], current[feature]),
                }
            )
    for feature in categorical_features:
        if feature in reference.columns and feature in current.columns:
            rows.append(
                {
                    "feature": feature,
                    "feature_type": "categorical",
                    "psi": categorical_psi(reference[feature], current[feature]),
                }
            )

    result = pd.DataFrame(rows, columns=["feature", "feature_type", "psi"])
    if result.empty:
        result["severity"] = pd.Series(dtype=str)
        return result
    result["severity"] = result["psi"].map(severity)
    return result.sort_values("psi", ascending=False, ignore_index=True)


def load_batch_features(batch_file: str) -> pd.DataFrame:
    """Reads one batch's drift features out of customers_cleaned.

    Streamed through a server-side cursor (see src/warehouse.py) and limited
    to the drift feature list - a batch is ~77K rows, and pulling all 37
    columns of every batch is exactly the kind of unbounded read that
    OOM-killed earlier versions of this project on a 3.8GB VM.
    """
    columns = DRIFT_FEATURES
    query = (
        f"SELECT {', '.join(columns)} FROM public.customers_cleaned "
        "WHERE source_batch = %(batch)s"
    )
    return stream_query(query, columns=columns, params={"batch": batch_file})


def ensure_drift_table() -> None:
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS public.feature_drift (
                    reference_batch TEXT NOT NULL,
                    current_batch   TEXT NOT NULL,
                    feature         TEXT NOT NULL,
                    feature_type    TEXT NOT NULL,
                    psi             DOUBLE PRECISION NOT NULL,
                    severity        TEXT NOT NULL,
                    computed_at     TIMESTAMP NOT NULL DEFAULT now(),
                    PRIMARY KEY (reference_batch, current_batch, feature)
                );
                """
            )
        conn.commit()
    finally:
        conn.close()


def persist_drift(reference_batch: str, current_batch: str, results: pd.DataFrame) -> None:
    """Upserts one batch-pair's drift scores.

    ON CONFLICT rather than plain INSERT because Airflow tasks retry, and a
    retried drift task should correct its row, not duplicate it.
    """
    if results.empty:
        return
    ensure_drift_table()
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            for row in results.itertuples(index=False):
                cur.execute(
                    """
                    INSERT INTO public.feature_drift
                        (reference_batch, current_batch, feature, feature_type, psi, severity, computed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (reference_batch, current_batch, feature) DO UPDATE
                       SET psi = EXCLUDED.psi,
                           severity = EXCLUDED.severity,
                           feature_type = EXCLUDED.feature_type,
                           computed_at = EXCLUDED.computed_at;
                    """,
                    (
                        reference_batch,
                        current_batch,
                        row.feature,
                        row.feature_type,
                        float(row.psi),
                        row.severity,
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def detect_drift_for_batch(current_batch: str, reference_batch: Optional[str] = None) -> dict:
    """Full pipeline step: pick a reference, score, persist, summarize.

    The reference defaults to the first batch ever ingested - a fixed
    baseline. Comparing each batch only against its immediate predecessor
    would miss slow drift, where every consecutive pair looks fine but the
    distribution has walked a long way from where the model was trained.
    """
    if reference_batch is None:
        conn = get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT batch_file FROM ingestion_log ORDER BY loaded_at ASC LIMIT 1"
                )
                row = cur.fetchone()
        finally:
            conn.close()
        reference_batch = row[0] if row else None

    if reference_batch is None or reference_batch == current_batch:
        log.info("No prior batch to compare %s against - establishing baseline.", current_batch)
        return {
            "reference_batch": reference_batch,
            "current_batch": current_batch,
            "features_scored": 0,
            "max_psi": 0.0,
            "drifted_features": [],
            "drift_detected": False,
            "note": "baseline batch - nothing to compare against yet",
        }

    reference = load_batch_features(reference_batch)
    current = load_batch_features(current_batch)
    results = compute_drift(reference, current)
    persist_drift(reference_batch, current_batch, results)

    drifted = results.loc[results["severity"] == "significant", "feature"].tolist()
    max_psi = float(results["psi"].max()) if not results.empty else 0.0
    log.info(
        "Drift %s vs %s: max PSI %.4f across %d features; %d significant",
        current_batch,
        reference_batch,
        max_psi,
        len(results),
        len(drifted),
    )

    if drifted:
        send_slack_alert(
            f":warning: *Feature drift detected* - batch `{current_batch}` vs. "
            f"baseline `{reference_batch}`: {len(drifted)} feature(s) at "
            f"\"significant\" severity (max PSI {max_psi:.4f}): {', '.join(drifted)}"
        )

    return {
        "reference_batch": reference_batch,
        "current_batch": current_batch,
        "features_scored": int(len(results)),
        "max_psi": max_psi,
        "drifted_features": drifted,
        "drift_detected": bool(drifted),
    }


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    batch = sys.argv[1] if len(sys.argv) > 1 else None
    if batch is None:
        conn = get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT batch_file FROM ingestion_log ORDER BY loaded_at DESC LIMIT 1"
                )
                batch = cur.fetchone()[0]
        finally:
            conn.close()
    print(json.dumps(detect_drift_for_batch(batch), indent=2))
