"""
Content-based service recommender: represents each customer as a profile
vector (demographics + account + usage, standardized/one-hot encoded,
deliberately excluding service subscription flags), fits a k-NN index over
those profiles, and at recommendation time looks at what similar customers
subscribe to that the target customer does not - recommending the
most-common un-subscribed services among the neighborhood.

Saves a single versioned artifact (fitted NearestNeighbors index + customer
id list + service subscription matrix) that the FastAPI /recommend
endpoint loads once and queries per-request with no live DB dependency.

Usage:
    python -m src.model.train_recommender
"""

from __future__ import annotations

import json
import logging
import os

# Silences a harmless joblib/loky warning on Windows, where physical-core
# detection shells out to a command that isn't available in this environment.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
# mlflow prints a run-summary line containing an emoji on run-context exit.
# Windows' console defaults to a codepage that can't encode it, raising
# UnicodeEncodeError from inside mlflow's own code - harmless (caught by
# the try/except around the mlflow.* calls below regardless), but it
# misleadingly logs as "registration failed" even though registration
# already succeeded (see src/model/train_churn.py's identical fix).
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.warehouse import stream_query

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_recommender")

# Model registry (src/model/registry.py). Best-effort: unset when running
# outside Docker/without an mlflow service, in which case training behaves
# exactly as before this was added - see the try/except around the
# mlflow.* calls in train_and_save() below.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
# mlflow's own default (MLFLOW_HTTP_REQUEST_TIMEOUT) is 120s - too long for
# calls that must never block a retrain (see src/model/registry.py's same
# setdefault, where a real hang was reproduced with the 120s default).
# 30s here (vs. registry.py's 5s) because this path uploads the joblib
# artifact itself, not just a lookup. setdefault: respects an operator's
# own explicit value.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
# mlflow's default retry count (5, with exponential backoff) can compound
# a single unreachable-mlflow case into minutes of retries even with the
# timeout above capped - see src/model/train_churn.py's identical fix.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")

MODELS_DIR = Path(__file__).resolve().parents[2] / "models_store"

PROFILE_NUMERIC = [
    "age", "annual_income", "tenure", "monthlycharges", "totalcharges",
    "customer_satisfaction", "avg_monthly_gb", "credit_score",
]
PROFILE_CATEGORICAL = ["gender", "education", "marital_status", "contract", "payment_method"]

SERVICE_COLUMNS = [
    "has_phone_service", "has_internet_service", "has_online_security",
    "has_online_backup", "has_device_protection", "has_tech_support",
    "has_streaming_tv", "has_streaming_movies",
]


def service_display_name(service_column: str) -> str:
    """'has_streaming_tv' -> 'Streaming Tv'. The single place this
    conversion happens - the dashboard's action-list column and the AI
    outreach agent's prompt both call this, rather than each having their
    own copy of the same replace/title() chain."""
    return service_column.replace("has_", "").replace("_", " ").title()


ID_COL = "customer_id"
N_NEIGHBORS = 20

# Size of the k-NN reference set. Unlike the churn model's cap, this one is
# not only about training memory - it bounds the *artifact*, which is
# loaded into memory by both the API (400MB limit) and the dashboard.
#
# The profile matrix is one float32 row per reference customer, and the
# fitted NearestNeighbors index holds its own copy, so cost scales linearly
# with this number. Measured: 150,000 profiles produce a 9.9MB artifact
# that the API loads at 206MB resident (of a 400MB limit). Extrapolating,
# all 1,000,000 would be roughly 66MB on disk and would push the API's
# resident set past its limit - and would be rebuilt from scratch on every
# retrain.
#
# Capping costs little here because this is a *reference set*, not a
# registry: recommendations come from the 20 nearest neighbours, and a
# 150K-profile sample of a 1M-customer population already provides dense
# coverage of the profile space. Customers outside the set are served by
# projecting their profile into it (recommend_for_profile), so coverage is
# 100% regardless of this value - see the /recommend endpoint.
MAX_REFERENCE_ROWS = int(os.environ.get("MAX_REFERENCE_ROWS", "150000"))


def get_pg_conn():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB", "warehouse"),
        user=os.environ.get("POSTGRES_USER"),
        password=os.environ.get("POSTGRES_PASSWORD"),
    )


def load_customer_360(max_rows: int | None = MAX_REFERENCE_ROWS) -> pd.DataFrame:
    # Server-side cursor via stream_query, NOT pd.read_sql(chunksize=...):
    # psycopg2's default client-side cursor buffers the whole result set in
    # libpq first, so read_sql's chunking bounds pandas' peak but not the
    # process's. At 1M rows that fails outright with "out of memory for
    # query result". See src/warehouse.py.
    columns = [ID_COL] + PROFILE_NUMERIC + PROFILE_CATEGORICAL + SERVICE_COLUMNS

    def _downcast(batch: pd.DataFrame) -> pd.DataFrame:
        for c in SERVICE_COLUMNS:
            batch[c] = batch[c].astype(int)
        for c in PROFILE_CATEGORICAL:
            batch[c] = batch[c].astype("category")
        return batch

    # TABLESAMPLE in Postgres rather than trimming in pandas: sampling after
    # loading all 1M rows would defeat the point of the cap, which is to
    # bound peak memory during the retrain (see MAX_REFERENCE_ROWS).
    # REPEATABLE makes the reference set stable across retrains, so a
    # customer's recommendations don't churn just because the sample moved.
    sample_clause = ""
    if max_rows is not None:
        conn = get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM marts.customer_360")
                total = cur.fetchone()[0]
        finally:
            conn.close()
        if total > max_rows:
            pct = min(100.0, 100.0 * max_rows / total * 1.05)  # slight over-draw, trimmed below
            sample_clause = f" TABLESAMPLE BERNOULLI ({pct:.4f}) REPEATABLE (42)"
            log.info(
                "customer_360 has %d rows; sampling ~%d as the reference set (see MAX_REFERENCE_ROWS)",
                total, max_rows,
            )

    df = stream_query(
        f"SELECT {', '.join(columns)} FROM marts.customer_360{sample_clause}",
        columns=columns,
        transform=_downcast,
    )
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, PROFILE_NUMERIC),
        ("cat", categorical_transformer, PROFILE_CATEGORICAL),
    ])


def train_and_save() -> dict:
    log.info("Loading customer_360 from warehouse...")
    df = load_customer_360()
    log.info("Loaded %d rows", len(df))

    preprocessor = build_preprocessor()
    X_profile = preprocessor.fit_transform(df[PROFILE_NUMERIC + PROFILE_CATEGORICAL])
    X_profile = np.asarray(X_profile.todense()) if hasattr(X_profile, "todense") else np.asarray(X_profile)
    # float32, not float64: this matrix is one row per customer and is both
    # held in memory and serialized into the artifact that the API and
    # dashboard load. At 1M customers that is ~240MB in float64 versus
    # ~120MB in float32, and float32 is far more precision than a
    # nearest-neighbour distance ranking over standardized features needs.
    X_profile = X_profile.astype(np.float32)

    service_matrix = df[SERVICE_COLUMNS].to_numpy()
    customer_ids = df[ID_COL].to_numpy()

    k = min(N_NEIGHBORS + 1, len(df))  # +1 because a customer is its own nearest neighbor
    log.info("Fitting NearestNeighbors (k=%d) on %d profiles...", k, len(df))
    nn_model = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto", n_jobs=1)
    nn_model.fit(X_profile)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = MODELS_DIR / f"recommender_{version}.joblib"
    joblib.dump({
        "preprocessor": preprocessor,
        "nn_model": nn_model,
        "X_profile": X_profile,
        "customer_ids": customer_ids,
        "service_matrix": service_matrix,
        "service_columns": SERVICE_COLUMNS,
        "profile_numeric": PROFILE_NUMERIC,
        "profile_categorical": PROFILE_CATEGORICAL,
    }, artifact_path, compress=3)

    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_customers": len(df),
        "n_neighbors": k,
        "service_columns": SERVICE_COLUMNS,
        "artifact_file": artifact_path.name,
    }
    meta_path = MODELS_DIR / f"recommender_{version}.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    log.info("Saved recommender artifact to %s", artifact_path)
    log.info("Saved metadata to %s", meta_path)

    # Registry bookkeeping (src/model/registry.py) - a side channel, never
    # a training-blocking dependency; the joblib artifact above is already
    # saved regardless of whether any of this succeeds. No quality-floor
    # gate exists for the recommender today, so always alias the newest
    # version as champion - matches today's implicit "newest wins"
    # behavior exactly, rather than inventing a new gate not asked for.
    if MLFLOW_TRACKING_URI:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("recommender")
            with mlflow.start_run(run_name=version):
                mlflow.log_params({"n_neighbors": k})
                mlflow.log_metrics({"n_customers": metadata["n_customers"]})
                mlflow.log_artifact(str(artifact_path))
                mlflow.log_artifact(str(meta_path))
                run_id = mlflow.active_run().info.run_id
                mv = mlflow.register_model(f"runs:/{run_id}/{artifact_path.name}", "recommender")
                client = mlflow.tracking.MlflowClient()
                client.set_model_version_tag("recommender", mv.version, "model_file", artifact_path.name)
                client.set_registered_model_alias("recommender", "champion", mv.version)
                log.info("Registered recommender v%s as champion", mv.version)
        except Exception as exc:
            log.warning("MLflow logging/registration failed (training itself succeeded): %s", exc)

    return metadata


def _rank_unsubscribed(artifact: dict, distances, neighbor_idxs, own_services, top_n: int) -> list[dict]:
    """Shared ranking step: score each service by how common it is among
    the given neighbours, then return only the ones this customer lacks."""
    service_matrix = artifact["service_matrix"]
    service_columns = artifact["service_columns"]
    neighbor_services = service_matrix[neighbor_idxs]  # (k, n_services)

    # inverse-distance-weighted subscription rate per service among neighbors
    weights = 1.0 / (distances + 1e-6)
    weights = weights / weights.sum()
    weighted_rate = (neighbor_services * weights[:, None]).sum(axis=0)

    candidates = [
        {"service": service_columns[i], "score": round(float(weighted_rate[i]), 4)}
        for i in range(len(service_columns))
        if own_services[i] == 0
    ]
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_n]


def recommend_for_profile(
    artifact: dict, profile: pd.DataFrame, own_services, top_n: int = 3
) -> list[dict]:
    """Recommend for *any* customer, whether or not they are in the index.

    The fitted k-NN index is a **reference set**, not a registry of every
    customer: a profile is transformed through the saved preprocessor and
    matched against it. That decouples coverage from index size - the index
    can stay small enough to fit in memory (and in the artifact the API and
    dashboard load) while still serving the entire customer base.

    `profile` is a single-row DataFrame with the PROFILE_NUMERIC and
    PROFILE_CATEGORICAL columns; `own_services` is that customer's current
    subscription flags, in SERVICE_COLUMNS order.
    """
    preprocessor = artifact["preprocessor"]
    nn_model: NearestNeighbors = artifact["nn_model"]

    x = preprocessor.transform(profile[artifact["profile_numeric"] + artifact["profile_categorical"]])
    x = np.asarray(x.todense()) if hasattr(x, "todense") else np.asarray(x)
    x = x.astype(np.float32)

    distances, neighbor_idxs = nn_model.kneighbors(x)
    return _rank_unsubscribed(artifact, distances[0], neighbor_idxs[0], np.asarray(own_services), top_n)


def recommend_for_customer(artifact: dict, customer_id: str, top_n: int = 3) -> list[dict]:
    """Recommend for a customer already present in the index, by id.

    Kept for the id-only API path. For customers outside the reference set,
    use recommend_for_profile instead."""
    customer_ids = artifact["customer_ids"]
    idx_matches = np.where(customer_ids == customer_id)[0]
    if len(idx_matches) == 0:
        raise KeyError(f"customer_id {customer_id!r} not found in recommender artifact")
    idx = idx_matches[0]

    nn_model: NearestNeighbors = artifact["nn_model"]
    X_profile = artifact["X_profile"]
    service_matrix = artifact["service_matrix"]
    service_columns = artifact["service_columns"]

    distances, neighbor_idxs = nn_model.kneighbors(X_profile[idx: idx + 1])
    neighbor_idxs = neighbor_idxs[0]
    distances = distances[0]

    # drop the customer itself (distance 0 to self)
    mask = neighbor_idxs != idx
    neighbor_idxs = neighbor_idxs[mask]
    distances = distances[mask]

    own_services = service_matrix[idx]
    return _rank_unsubscribed(artifact, distances, neighbor_idxs, own_services, top_n)


if __name__ == "__main__":
    meta = train_and_save()
    log.info("Sanity check: recommending for first customer in the dataset...")
    artifact = joblib.load(MODELS_DIR / meta["artifact_file"])
    sample_id = artifact["customer_ids"][0]
    recs = recommend_for_customer(artifact, sample_id)
    log.info("Recommendations for %s: %s", sample_id, recs)
