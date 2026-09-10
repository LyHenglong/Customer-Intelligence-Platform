"""
Trains a churn classifier on the customer_360 mart, evaluates it with
precision/recall/F1 (churn is imbalanced - accuracy alone would be
misleading), and saves a timestamp-versioned model artifact (never
overwriting a previous one).

Model choice (LightGBM) and the decision threshold below were settled by
a dedicated experiment comparing RandomForest/XGBoost/LightGBM/Logistic
Regression, SMOTE vs class_weight, and 5-fold CV - see
notebooks/model_dev_offline.py. Summary: LightGBM gives a small, real,
cross-validation-confirmed edge over RandomForest (AUC 0.683 +/- 0.002 vs
0.677); nothing tried closes the gap to a "strong" model, which looks
like a genuine ceiling in this synthetic data rather than a modeling gap.

Usage:
    python -m src.model.train_churn
"""

from __future__ import annotations

import json
import logging
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Business-chosen target: in a retention context, catching more churners is
# usually worth some false positives, so the decision threshold is picked to
# guarantee at least this much recall on churned customers (see
# notebooks/model_dev_offline.py Step 2) rather than defaulting to 0.5 or
# optimizing for F1 alone.
TARGET_RECALL = 0.60

# Cap on rows pulled into memory for training, overridable via env var.
#
# This is a deliberate engineering trade-off, not a limitation of the
# pipeline: ingestion, DuckDB processing, Postgres and dbt all handle the
# full 1,000,000 rows without difficulty. It is specifically the in-memory
# scikit-learn/LightGBM training step that is capped, because it runs
# inside an Airflow worker on an 8GB host (see README's RAM constraint) and
# was OOM-killed at ~460K rows before this cap existed.
#
# The cost of capping is small but not quite zero. Measured test AUC across
# training sizes: 0.652 and 0.677 @ ~150K rows (two different samples),
# 0.663 @ 231K, 0.683 @ 1M. There is no clean monotonic trend - run-to-run
# sampling variance (~0.03) is comparable to the spread across sizes - but
# the largest run did produce the best number, so capping likely costs a
# little accuracy at the top end. That trade is taken deliberately: an
# OOM-killed retrain task breaks the pipeline, a 0.02 AUC difference on
# synthetic data does not. Raise it (or set it to None) on a larger machine.
#
# Sized against the real constraint, which is not the container's mem_limit
# but Docker Desktop's WSL2 VM: 3.8GB total for every container combined,
# with swap already saturated. Retraining at 400K rows was still SIGKILLed
# inside a 1.6GB container because the VM itself had nothing left to give.
MAX_TRAINING_ROWS = int(os.environ.get("MAX_TRAINING_ROWS", "150000"))

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_churn")

MODELS_DIR = Path(__file__).resolve().parents[2] / "models_store"

NUMERIC_FEATURES = [
    "age", "annual_income", "dependents", "tenure", "tenure_years",
    "monthlycharges", "totalcharges", "num_services", "total_active_services",
    "cost_per_active_service", "customer_satisfaction", "num_complaints",
    "num_service_calls", "late_payments", "avg_monthly_gb", "avg_gb_per_service",
    "days_since_last_interaction", "credit_score", "complaints_per_tenure_month",
    "annual_spend_to_income_ratio",
]
BOOLEAN_FEATURES = [
    "senior_citizen", "paperless_billing", "has_phone_service",
    "has_internet_service", "has_online_security", "has_online_backup",
    "has_device_protection", "has_tech_support", "has_streaming_tv",
    "has_streaming_movies", "is_month_to_month", "is_disengaged",
]
CATEGORICAL_FEATURES = [
    "gender", "education", "marital_status", "contract", "payment_method",
    "tenure_bucket",
]
TARGET = "churn"
ID_COL = "customer_id"

ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


def get_pg_conn():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB", "warehouse"),
        user=os.environ.get("POSTGRES_USER"),
        password=os.environ.get("POSTGRES_PASSWORD"),
    )


def load_customer_360(max_rows: int | None = None) -> pd.DataFrame:
    # Raw psycopg2 connection, not a SQLAlchemy Engine/Connection: pandas'
    # read_sql "is this a SQLAlchemy connectable" detection breaks against
    # some pandas/SQLAlchemy version pairs (hit this with pandas 2.2.3 +
    # the SQLAlchemy 1.4.x bundled inside the Airflow image) and falls back
    # to a legacy DBAPI path that expects exactly what a raw psycopg2
    # connection provides.
    cols = ", ".join([ID_COL] + ALL_FEATURES + [TARGET])

    # TABLESAMPLE, applied in Postgres rather than pandas, when a cap is
    # set: sampling after loading everything would defeat the purpose of
    # the cap, which exists to bound *peak* memory (see MAX_TRAINING_ROWS).
    limit_clause = ""
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
            limit_clause = f" TABLESAMPLE BERNOULLI ({pct:.4f}) REPEATABLE (42)"
            log.info("customer_360 has %d rows; sampling ~%d for training (see MAX_TRAINING_ROWS)", total, max_rows)

    conn = get_pg_conn()
    try:
        # Chunked read, not one big read_sql: psycopg2/pandas materialize the
        # entire result as raw Python row-tuples before building columnar
        # arrays, which spikes memory far above the final DataFrame's size.
        # This exact pattern OOM-killed the Airflow retrain task (SIGKILL,
        # return code -9) at ~460K rows inside a 900MB container.
        frames = []
        for chunk in pd.read_sql(
            f"SELECT {cols} FROM marts.customer_360{limit_clause}", conn, chunksize=25_000
        ):
            for c in BOOLEAN_FEATURES:
                chunk[c] = chunk[c].astype("float")  # bool -> 0.0/1.0, NaN-safe
            # category dtype for the low-cardinality string columns: as
            # object dtype these cost ~65MB each at 1M rows.
            for c in CATEGORICAL_FEATURES:
                chunk[c] = chunk[c].astype("category")
            frames.append(chunk)
        df = pd.concat(frames, ignore_index=True)
    finally:
        conn.close()

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return df


def build_pipeline() -> Pipeline:
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES + BOOLEAN_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    # LightGBM, not RandomForest: a dedicated model-comparison experiment
    # (notebooks/model_dev_offline.py) found it gives a small but real edge
    # (higher AUC, confirmed by 5-fold CV) and trains ~6x faster at this
    # data size. class_weight="balanced" beat SMOTE oversampling in that
    # same experiment (statistically indistinguishable AUC, but SMOTE needs
    # careful threshold recalibration that class_weight doesn't).
    clf = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def choose_threshold(y_test, y_proba, target_recall: float = TARGET_RECALL) -> tuple[float, float, float]:
    """Finds the highest decision threshold that still clears target_recall
    on the held-out test set. Returns (threshold, precision, recall)."""
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    candidates = [i for i in range(len(thresholds)) if recalls[i] >= target_recall]
    if not candidates:
        # target recall unreachable even at the lowest threshold - fall back
        # to the lowest threshold available (maximizes recall)
        idx = 0
        return float(thresholds[idx]), float(precisions[idx]), float(recalls[idx])
    idx = candidates[-1]  # highest threshold that still clears target_recall
    return float(thresholds[idx]), float(precisions[idx]), float(recalls[idx])


def train_and_save(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        log.info("Loading customer_360 from warehouse...")
        df = load_customer_360(max_rows=MAX_TRAINING_ROWS)
    else:
        df = df.copy()
        for c in BOOLEAN_FEATURES:
            df[c] = df[c].astype("float")
    log.info("Loaded %d rows", len(df))

    class_counts = df[TARGET].value_counts().to_dict()
    class_pct = (df[TARGET].value_counts(normalize=True) * 100).round(2).to_dict()
    log.info("Class balance: counts=%s pct=%s", class_counts, class_pct)

    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    pipeline = build_pipeline()
    log.info("Training LightGBM on %d rows...", len(X_train))
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    threshold, threshold_precision, threshold_recall = choose_threshold(y_test, y_proba)
    log.info(
        "Decision threshold: %.3f (target recall >= %.2f) -> precision=%.3f recall=%.3f",
        threshold, TARGET_RECALL, threshold_precision, threshold_recall,
    )
    y_pred = (y_proba >= threshold).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    auc = roc_auc_score(y_test, y_proba)

    log.info("Classification report (at chosen threshold):\n%s", classification_report(y_test, y_pred))
    log.info("Confusion matrix: %s", cm)
    log.info("ROC AUC: %.4f", auc)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = MODELS_DIR / f"churn_model_{version}.joblib"
    joblib.dump({"pipeline": pipeline, "features": ALL_FEATURES, "threshold": threshold}, model_path)

    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "LightGBM",
        "threshold": threshold,
        "threshold_rationale": f"highest threshold that still guarantees recall >= {TARGET_RECALL} on the test set - maximizes precision subject to that recall floor (retention use case: catching churners is worth some false positives)",
        "n_rows_total": len(df),
        "n_rows_train": len(X_train),
        "n_rows_test": len(X_test),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "class_pct": {str(k): float(v) for k, v in class_pct.items()},
        "precision_churn": report["1"]["precision"],
        "recall_churn": report["1"]["recall"],
        "f1_churn": report["1"]["f1-score"],
        "precision_retained": report["0"]["precision"],
        "recall_retained": report["0"]["recall"],
        "f1_retained": report["0"]["f1-score"],
        "accuracy": report["accuracy"],
        "roc_auc": auc,
        "confusion_matrix": cm,
        "model_file": model_path.name,
    }
    meta_path = MODELS_DIR / f"churn_model_{version}.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    log.info("Saved model to %s", model_path)
    log.info("Saved metadata to %s", meta_path)
    return metadata


if __name__ == "__main__":
    train_and_save()
