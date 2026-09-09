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
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_recommender")

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

ID_COL = "customer_id"
N_NEIGHBORS = 20


def get_pg_conn():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB", "warehouse"),
        user=os.environ.get("POSTGRES_USER"),
        password=os.environ.get("POSTGRES_PASSWORD"),
    )


def load_customer_360() -> pd.DataFrame:
    # Raw psycopg2 connection, not a SQLAlchemy Engine - see the comment in
    # train_churn.py's load_customer_360 for why.
    cols = ", ".join([ID_COL] + PROFILE_NUMERIC + PROFILE_CATEGORICAL + SERVICE_COLUMNS)
    conn = get_pg_conn()
    try:
        # Chunked read for the same reason as train_churn.load_customer_360:
        # a single read_sql spikes memory well above the final DataFrame
        # while pandas holds the full raw row-tuple form.
        frames = []
        for chunk in pd.read_sql(f"SELECT {cols} FROM marts.customer_360", conn, chunksize=25_000):
            for c in SERVICE_COLUMNS:
                chunk[c] = chunk[c].astype(int)
            for c in PROFILE_CATEGORICAL:
                chunk[c] = chunk[c].astype("category")
            frames.append(chunk)
        df = pd.concat(frames, ignore_index=True)
    finally:
        conn.close()
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
    return metadata


def recommend_for_customer(artifact: dict, customer_id: str, top_n: int = 3) -> list[dict]:
    """Given a loaded recommender artifact (see load_latest_recommender in
    api.py), returns up to top_n un-subscribed services ranked by how
    common they are among the customer's nearest profile-neighbors."""
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


if __name__ == "__main__":
    meta = train_and_save()
    log.info("Sanity check: recommending for first customer in the dataset...")
    artifact = joblib.load(MODELS_DIR / meta["artifact_file"])
    sample_id = artifact["customer_ids"][0]
    recs = recommend_for_customer(artifact, sample_id)
    log.info("Recommendations for %s: %s", sample_id, recs)
