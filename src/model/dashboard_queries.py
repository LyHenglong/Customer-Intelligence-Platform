"""Shared warehouse/filesystem read functions for the churn-analytics
dashboard-facing endpoints in src/model/api.py. Originally extracted out
of the (now-retired) Streamlit dashboard so both frontends showed
identical numbers from one code path rather than reimplementing the same
queries independently; kept as its own module since api.py's callers
apply their own caching where it's needed (see api.py's
_get_scored_customers for score_all_customers specifically, since that
one is expensive).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.train_churn import ALL_FEATURES as CHURN_FEATURES
from src.model.train_churn import CATEGORICAL_FEATURES as CHURN_CATEGORICAL
from src.warehouse import get_pg_conn, stream_query

MODELS_DIR = Path(__file__).resolve().parents[2] / "models_store"
RETRAIN_EVERY_N_BATCHES = int(os.environ.get("RETRAIN_EVERY_N_BATCHES", "3"))
TOTAL_SIMULATED_BATCHES = 13

# signup_date deliberately excluded: not used anywhere downstream, and
# pulling a raw timestamp for 300K+ rows for nothing is pure memory waste.
_DASHBOARD_COLUMNS = [c for c in ["customer_id"] + CHURN_FEATURES + [
    "churn", "contract", "tenure", "monthlycharges", "tenure_bucket", "total_active_services",
] if c != "signup_date"]
_DASHBOARD_COLUMNS = list(dict.fromkeys(_DASHBOARD_COLUMNS))  # de-dupe, keep order


def score_all_customers(_churn_artifact, churn_version: str, batch_size: int = 25_000) -> pd.DataFrame:
    """One streaming pass over customer_360, scoring every customer and
    keeping only a compact per-customer result.

    Loading the entire mart into a DataFrame and scoring it in place is
    fine at 300K rows and fatal at 1M: the frame alone is ~620MB, and the
    preprocessor materializes a dense (n x 54) float64 matrix on top of
    it (412MB at 1M) - inside a memory-limited container that is an OOM
    kill.

    Instead, rows are streamed from Postgres in batches, scored, and
    reduced immediately to three columns. Batches are kept small (25K) on
    purpose: each fetchmany materializes batch_rows x n_columns individual
    Python objects before pandas builds columnar arrays, so the batch
    size sets the transient peak far more than the retained result does.
    The retained result is ~40MB at 1M customers regardless of how wide
    the mart gets, and peak memory is bounded by one batch rather than by
    the table size. Full rows for the handful of customers actually
    displayed are fetched by id later (see load_customers_by_id).
    """
    score_cols = ["customer_id"] + CHURN_FEATURES
    score_cols = list(dict.fromkeys(score_cols))
    pipeline = _churn_artifact["pipeline"]

    ids, scores, charges = [], [], []

    def _score_batch(batch: pd.DataFrame) -> pd.DataFrame:
        X = batch[CHURN_FEATURES].copy()
        for c in CHURN_FEATURES:
            if X[c].dtype == bool:
                X[c] = X[c].astype(float)
        ids.append(batch["customer_id"].to_numpy())
        scores.append(pipeline.predict_proba(X)[:, 1].astype(np.float32))
        charges.append(batch["monthlycharges"].to_numpy(dtype=np.float32))
        # Return an empty frame: stream_query concatenates whatever comes
        # back, and we deliberately keep none of the raw rows.
        return batch.iloc[0:0]

    stream_query(
        f"SELECT {', '.join(score_cols)} FROM marts.customer_360",
        columns=score_cols,
        batch_rows=batch_size,
        transform=_score_batch,
    )

    return pd.DataFrame({
        "customer_id": np.concatenate(ids),
        "churn_probability": np.concatenate(scores),
        "monthlycharges": np.concatenate(charges),
    })


def load_overall_stats() -> dict:
    """Headline counts straight from SQL - exact, and a few bytes over the
    wire instead of a million rows."""
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*), AVG(churn::int) FROM marts.customer_360")
            total, churn_rate = cur.fetchone()
        return {"total_customers": int(total), "churn_rate": float(churn_rate)}
    finally:
        conn.close()


def load_segment_rates(column: str) -> pd.DataFrame:
    """Churn rate by segment, computed as a SQL GROUP BY.

    Postgres aggregates a million rows far more cheaply than shipping them
    all to pandas to do the same thing, and the result is a handful of rows.
    """
    if column not in set(CHURN_CATEGORICAL) | {"total_active_services"}:
        raise ValueError(f"unexpected segment column {column!r}")  # guards the f-string below
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {column}::text, AVG(churn::int), COUNT(*) "
                f"FROM marts.customer_360 GROUP BY {column} ORDER BY {column}"
            )
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=[column, "churn_rate", "n_customers"]).astype(
            {"churn_rate": float, "n_customers": int}
        )
    finally:
        conn.close()


def load_customers_by_id(customer_ids: tuple[str, ...]) -> pd.DataFrame:
    """Full rows for just the customers being displayed."""
    if not customer_ids:
        return pd.DataFrame(columns=_DASHBOARD_COLUMNS)
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(_DASHBOARD_COLUMNS)} FROM marts.customer_360 "
                f"WHERE customer_id = ANY(%s)",
                (list(customer_ids),),
            )
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=_DASHBOARD_COLUMNS)
    finally:
        conn.close()


def load_ingestion_log() -> pd.DataFrame:
    conn = get_pg_conn()
    try:
        return pd.read_sql(
            "SELECT batch_file, rows_loaded, loaded_at, status FROM ingestion_log ORDER BY loaded_at", conn
        )
    finally:
        conn.close()


def load_latest_drift() -> pd.DataFrame:
    """Per-feature PSI for the most recently scored batch.

    Returns an empty frame (rather than raising) when the table doesn't
    exist yet - a warehouse that hasn't run the drift task since this
    feature was added is a normal state, not an error worth breaking the
    whole view over.
    """
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.feature_drift')")
            if cur.fetchone()[0] is None:
                return pd.DataFrame()
        return pd.read_sql(
            """
            SELECT feature, feature_type, psi, severity, reference_batch, current_batch
              FROM public.feature_drift
             WHERE current_batch = (
                   SELECT current_batch FROM public.feature_drift
                    ORDER BY computed_at DESC LIMIT 1
             )
             ORDER BY psi DESC
            """,
            conn,
        )
    finally:
        conn.close()


def load_all_churn_metadata() -> list[dict]:
    """All churn_model_*.json metadata files, oldest to newest - lets
    callers show a real retraining trend, not just the latest snapshot."""
    records = []
    for path in sorted(MODELS_DIR.glob("churn_model_*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return records


def column_importances(pipeline) -> dict:
    """Maps the preprocessor's expanded output names (e.g.
    'num__num_complaints', 'cat__contract_two_year') back to real
    customer_360 column names, so risk-factor labels are meaningful
    instead of truncated prefixes like 'num'/'cat'."""
    try:
        preproc = pipeline.named_steps["preprocess"]
        clf = pipeline.named_steps["model"]
        names = preproc.get_feature_names_out()
        raw_importances = clf.feature_importances_
    except (KeyError, AttributeError):
        return {}

    grouped: dict[str, float] = {}
    for name, imp in zip(names, raw_importances):
        if name.startswith("num__"):
            col = name[len("num__"):]
        elif name.startswith("cat__"):
            rest = name[len("cat__"):]
            col = next((c for c in CHURN_CATEGORICAL if rest.startswith(c + "_")), rest)
        else:
            col = name
        grouped[col] = grouped.get(col, 0.0) + float(imp)
    return grouped
