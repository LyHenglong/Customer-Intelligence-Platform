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
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.train_churn import ALL_FEATURES as CHURN_FEATURES
from src.model.train_churn import CATEGORICAL_FEATURES as CHURN_CATEGORICAL
from src.model.train_churn import TARGET as CHURN_TARGET
from src.warehouse import get_pg_conn, stream_query

log = logging.getLogger("model.dashboard_queries")

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
    reduced immediately to a few columns. Batches are kept small (25K) on
    purpose: each fetchmany materializes batch_rows x n_columns individual
    Python objects before pandas builds columnar arrays, so the batch
    size sets the transient peak far more than the retained result does.
    The retained result is ~45MB at 1M customers regardless of how wide
    the mart gets, and peak memory is bounded by one batch rather than by
    the table size. Full rows for the handful of customers actually
    displayed are fetched by id later (see load_customers_by_id).

    The segment columns (CHURN_CATEGORICAL - contract, tenure_bucket,
    gender, ...) are retained as pandas `category` dtype rather than
    discarded: they're already being streamed here as model features, and
    at <=6 distinct values each they cost ~1MB per column at 1M rows
    (int8 codes + a tiny dictionary). That's what lets
    /overview/revenue-at-risk-by-segment aggregate the *whole* at-risk
    population in-process instead of re-fetching full rows from Postgres
    for a capped subset.
    """
    # "churn" is the label, not a feature, so it has to be asked for
    # explicitly. It rides along (1M int8 ~ 1MB) so callers can compute the
    # observed churn rate for any slice of this frame without going back to
    # the warehouse - see src/ai/tools/churn_tool.py.
    score_cols = ["customer_id", CHURN_TARGET] + CHURN_FEATURES
    score_cols = list(dict.fromkeys(score_cols))
    pipeline = _churn_artifact["pipeline"]

    ids, scores, charges, labels = [], [], [], []
    segments: dict[str, list] = {c: [] for c in CHURN_CATEGORICAL}

    def _score_batch(batch: pd.DataFrame) -> pd.DataFrame:
        X = batch[CHURN_FEATURES].copy()
        for c in CHURN_FEATURES:
            if X[c].dtype == bool:
                X[c] = X[c].astype(float)
        ids.append(batch["customer_id"].to_numpy())
        scores.append(pipeline.predict_proba(X)[:, 1].astype(np.float32))
        charges.append(batch["monthlycharges"].to_numpy(dtype=np.float32))
        labels.append(pd.to_numeric(batch[CHURN_TARGET], errors="coerce").fillna(0).to_numpy(dtype=np.int8))
        for c in CHURN_CATEGORICAL:
            segments[c].append(batch[c].to_numpy())
        # Return an empty frame: stream_query concatenates whatever comes
        # back, and we deliberately keep none of the raw rows.
        return batch.iloc[0:0]

    stream_query(
        f"SELECT {', '.join(score_cols)} FROM marts.customer_360",
        columns=score_cols,
        batch_rows=batch_size,
        transform=_score_batch,
    )

    frame = pd.DataFrame({
        "customer_id": np.concatenate(ids),
        "churn_probability": np.concatenate(scores),
        "monthlycharges": np.concatenate(charges),
        "churn": np.concatenate(labels),
    })
    for c in CHURN_CATEGORICAL:
        frame[c] = pd.Categorical(np.concatenate(segments[c]))
    return frame


# One scored population, shared by every caller that needs it.
#
# There used to be two: src/model/api.py cached this frame for the
# dashboard endpoints, while src/ai/tools/churn_tool.py ran its own
# independent streaming pass for the assistant. So an assistant question
# re-scored 1,000,000 customers that were already scored and sitting in
# this process's memory - measured at 209s of a 211s request.
#
# The lock spans the recompute rather than just the lookup, so two
# concurrent misses serialise instead of both starting a full pass and
# doubling peak memory in a container already sized once for one.
_SCORED_CACHE_TTL_SECONDS = int(os.environ.get("SCORED_CACHE_TTL_SECONDS", "3600"))
_scored_cache: dict = {"data": None, "expires_at": 0.0, "version": None}
_scored_lock = threading.Lock()


def get_scored_customers(churn_artifact, churn_version: str) -> pd.DataFrame:
    """Cached score_all_customers. Entries are keyed by churn_version, so
    promoting a model invalidates this without anyone having to remember
    to clear it."""
    with _scored_lock:
        now = time.monotonic()
        cached = _scored_cache
        if cached["data"] is not None and cached["version"] == churn_version and now < cached["expires_at"]:
            return cached["data"]
        frame = None
        if _USE_PRECOMPUTED_SCORES:
            frame = load_precomputed_scores(churn_version)
            if frame is None:
                log.warning(
                    "USE_PRECOMPUTED_SCORES=true but public.churn_scores has no rows for "
                    "version %s - falling back to live scoring (run "
                    "scripts/precompute_churn_scores.py against this warehouse)",
                    churn_version,
                )
        if frame is None:
            frame = score_all_customers(churn_artifact, churn_version)
        _scored_cache.update(data=frame, expires_at=now + _SCORED_CACHE_TTL_SECONDS, version=churn_version)
        return frame


def ensure_churn_scores_table() -> None:
    """Creates public.churn_scores if the warehouse predates it - same
    on-demand pattern as src/agents/cache.py's ensure_tables()."""
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS public.churn_scores (
                    customer_id         TEXT PRIMARY KEY,
                    model_version       TEXT NOT NULL,
                    churn_probability   DOUBLE PRECISION NOT NULL,
                    monthlycharges      NUMERIC(10, 2),
                    churn               INTEGER,
                    gender              TEXT,
                    education           TEXT,
                    marital_status      TEXT,
                    contract            TEXT,
                    payment_method      TEXT,
                    tenure_bucket       TEXT,
                    computed_at         TIMESTAMP NOT NULL DEFAULT now()
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_churn_scores_model_version "
                "ON public.churn_scores (model_version)"
            )
        conn.commit()
    finally:
        conn.close()


def store_scored_customers(frame: pd.DataFrame, churn_version: str) -> None:
    """Persists score_all_customers()'s output to public.churn_scores, so
    a memory-constrained deployment (see load_precomputed_scores) can
    serve it from a cheap SQL SELECT instead of re-running the ~1M-row
    scoring pass in-process on every cache miss.

    Meant to be run out-of-band (scripts/precompute_churn_scores.py, by
    hand or after a retrain) against whichever warehouse the deployment
    actually serves from - never called from a request path. Old rows
    for a different model_version are deleted first so the table never
    silently mixes two models' scores together.
    """
    from psycopg2.extras import execute_values

    ensure_churn_scores_table()
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM public.churn_scores WHERE model_version != %s", (churn_version,))
            rows = [
                (
                    r.customer_id, churn_version, float(r.churn_probability), float(r.monthlycharges),
                    int(r.churn), r.gender, r.education, r.marital_status, r.contract,
                    r.payment_method, r.tenure_bucket,
                )
                for r in frame.itertuples(index=False)
            ]
            execute_values(
                cur,
                """
                INSERT INTO public.churn_scores
                    (customer_id, model_version, churn_probability, monthlycharges, churn,
                     gender, education, marital_status, contract, payment_method, tenure_bucket)
                VALUES %s
                ON CONFLICT (customer_id) DO UPDATE SET
                    model_version = EXCLUDED.model_version,
                    churn_probability = EXCLUDED.churn_probability,
                    monthlycharges = EXCLUDED.monthlycharges,
                    churn = EXCLUDED.churn,
                    gender = EXCLUDED.gender,
                    education = EXCLUDED.education,
                    marital_status = EXCLUDED.marital_status,
                    contract = EXCLUDED.contract,
                    payment_method = EXCLUDED.payment_method,
                    tenure_bucket = EXCLUDED.tenure_bucket,
                    computed_at = now()
                """,
                rows,
                page_size=5000,
            )
        conn.commit()
    finally:
        conn.close()


def load_precomputed_scores(churn_version: str) -> pd.DataFrame | None:
    """The public.churn_scores fast path for get_scored_customers.

    Returns None - never raises - whenever the precomputed path can't
    serve the request: the table doesn't exist yet, or holds no rows for
    this exact model_version (e.g. right after a retrain, before the next
    precompute run). Callers fall back to score_all_customers() in both
    cases, same as any other cache miss.

    Reads via stream_query (a server-side cursor), not a plain
    cur.fetchall() - this table holds up to 1,000,000 rows, and a plain
    cursor buffers the *entire* result client-side before pandas sees a
    single row (see src/warehouse.py's own module docstring - this is the
    exact memory trap that module exists to avoid). An earlier version of
    this function used fetchall() directly and reproduced the same OOM
    this whole precomputed-scores path was built to eliminate.
    """
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.churn_scores')")
            if cur.fetchone()[0] is None:
                return None
    finally:
        conn.close()

    columns = [
        "customer_id", "churn_probability", "monthlycharges", "churn",
        "gender", "education", "marital_status", "contract", "payment_method", "tenure_bucket",
    ]
    frame = stream_query(
        f"SELECT {', '.join(columns)} FROM public.churn_scores WHERE model_version = %s",
        columns=columns,
        params=(churn_version,),
    )
    if frame.empty:
        return None

    frame["churn_probability"] = frame["churn_probability"].astype(np.float32)
    frame["monthlycharges"] = frame["monthlycharges"].astype(np.float32)
    frame["churn"] = frame["churn"].astype(np.int8)
    for c in CHURN_CATEGORICAL:
        frame[c] = pd.Categorical(frame[c])
    return frame


# Memory-constrained deployments (Render's free tier) opt into reading
# get_scored_customers from the public.churn_scores precomputed table
# instead of running the in-process scoring pass - see render.yaml and
# load_precomputed_scores' docstring. Defaults to false, so local/Docker
# usage (and any deployment with enough memory to just run the real thing)
# is completely unaffected.
_USE_PRECOMPUTED_SCORES = os.environ.get("USE_PRECOMPUTED_SCORES", "false").lower() not in ("false", "0", "")


def peek_scored_customers(churn_version: str) -> pd.DataFrame | None:
    """The cached frame if it is present and still valid, else None -
    never triggers a recompute.

    Lets a caller that merely *prefers* the fast path (the assistant's
    churn_analysis) use it when it is already warm and fall back to its
    own query when it is not, without blocking on somebody else's
    multi-minute scoring pass."""
    with _scored_lock:
        cached = _scored_cache
        if cached["data"] is not None and cached["version"] == churn_version and time.monotonic() < cached["expires_at"]:
            return cached["data"]
    return None


def clear_scored_cache() -> None:
    with _scored_lock:
        _scored_cache.update(data=None, expires_at=0.0, version=None)


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


def _raw_importances(clf):
    """Gain, not split counts.

    LightGBM's `feature_importances_` defaults to importance_type="split" -
    how many times a feature was chosen for a split, which is badly biased
    toward high-cardinality continuous columns simply because they offer
    more candidate split points. On this dataset that put `credit_score`
    top (univariate AUC 0.51 - indistinguishable from noise) while
    `contract`, which separates churn 4.95x, did not even make the top six.
    The dashboard and the AI assistant were both telling people to act on
    the wrong lever.

    Gain measures each feature's actual contribution to loss reduction and
    ranks `contract` first at ~29%, matching the univariate evidence in
    report/findings.md section 2b.

    Read from the fitted booster rather than set at training time on
    purpose: this corrects every artifact already in models_store without
    retraining anything, since it is a reporting bug, not a model bug.

    Falls back to `feature_importances_` for non-LightGBM estimators -
    scikit-learn's tree ensembles already report mean impurity decrease,
    which is a gain measure, so only LightGBM needs the special case."""
    booster = getattr(clf, "booster_", None)
    if booster is not None:
        try:
            return booster.feature_importance(importance_type="gain")
        except Exception:
            log.warning("gain importances unavailable, falling back to the estimator's default")
    return clf.feature_importances_


def column_importances(pipeline) -> dict:
    """Maps the preprocessor's expanded output names (e.g.
    'num__num_complaints', 'cat__contract_two_year') back to real
    customer_360 column names, so risk-factor labels are meaningful
    instead of truncated prefixes like 'num'/'cat'."""
    try:
        preproc = pipeline.named_steps["preprocess"]
        clf = pipeline.named_steps["model"]
        names = preproc.get_feature_names_out()
        raw_importances = _raw_importances(clf)
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
