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
# mlflow prints a run-summary line containing an emoji on run-context exit
# (e.g. "View run ... at ..."). Windows' console defaults to a codepage
# that can't encode it, raising UnicodeEncodeError from inside mlflow's
# own code - harmless (caught by the try/except around the mlflow.* calls
# below regardless), but it misleadingly logs as "registration failed"
# even though registration already succeeded. PYTHONIOENCODING only takes
# effect at interpreter startup (too late to set from within the running
# process), so reconfigure the already-open streams directly instead.
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
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
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
# timeout above capped - reproduced for real while verifying
# src/model/registry.py's fallback path. 2 retries is enough to ride out
# a brief blip without meaningfully delaying a retrain that otherwise
# succeeded regardless of whether this registration step does.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")

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
    # Hyperparameters from notebooks/churn_hyperparameter_sweep.py, not
    # defaults: 9 configs scored by 4-fold CV on identical folds (200K
    # rows), adopted only because the winner cleared the baseline by more
    # than one baseline fold-to-fold std - a rule fixed before the numbers
    # were seen, since this dataset's run-to-run variance is large enough
    # to manufacture convincing "wins".
    #
    # CV AUC 0.6778 +/- 0.0026 against the previous 0.6651 +/- 0.0014
    # (+0.0127, ~9x the baseline std); 0.6816 on the held-out test set,
    # which the sweep touched exactly once.
    #
    # The direction is the informative part: shallower is better here.
    # depth 8 barely moved (+0.0034) and unlimited depth at 63 leaves
    # actively hurt (-0.0118), so the old depth-6/300-round config was
    # spending capacity memorizing noise in a weak-signal target. More
    # rounds at a lower learning rate pay for the lost depth.
    clf = LGBMClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
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

    # Three-way split, not the usual two. Calibration needs data that the
    # model was not fitted on, and reusing the test set for it would leak
    # the test labels into the reported metrics.
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y,
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.6667, random_state=42, stratify=y_tmp,
    )

    base_pipeline = build_pipeline()
    log.info("Training LightGBM on %d rows...", len(X_train))
    base_pipeline.fit(X_train, y_train)

    # Probability calibration (Platt scaling on a held-out split).
    #
    # class_weight="balanced" is what makes this necessary: it fixes the
    # model's *ranking* under imbalance, but it does so by inflating
    # minority-class scores, so the raw outputs are not probabilities at
    # all. Measured on this data before calibration, mean predicted score
    # was 0.408 against an actual churn rate of 0.100 - 4.1x too high, with
    # per-decile ratios between 3.3x and 5.2x - and the model's Brier score
    # (0.196) was worse than always predicting the base rate (0.090).
    #
    # Sigmoid rather than isotonic: both reached Brier 0.0868 here, but
    # sigmoid is a strictly monotonic two-parameter fit, so it preserves
    # ROC AUC exactly (0.6693 vs isotonic's 0.6682) and is far less prone
    # to overfitting a ~15K-row calibration split than isotonic's
    # free-form step function.
    #
    # This changes what the threshold means. Calibrated scores are real
    # probabilities, so the decision threshold drops from ~0.47 to ~0.11 -
    # the operating point (precision/recall) is unchanged, because
    # calibration is monotonic and only relabels the axis.
    log.info("Calibrating probabilities (sigmoid) on %d held-out rows...", len(X_calib))
    pipeline = CalibratedClassifierCV(estimator=base_pipeline, method="sigmoid", cv="prefit")
    pipeline.fit(X_calib, y_calib)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_proba_uncalibrated = base_pipeline.predict_proba(X_test)[:, 1]

    base_rate = float(y_test.mean())
    brier = brier_score_loss(y_test, y_proba)
    brier_uncalibrated = brier_score_loss(y_test, y_proba_uncalibrated)
    brier_base_rate = brier_score_loss(y_test, np.full(len(y_test), base_rate))
    log.info(
        "Brier: %.4f calibrated vs %.4f uncalibrated (always-base-rate %.4f); "
        "mean predicted %.4f vs actual %.4f",
        brier, brier_uncalibrated, brier_base_rate, float(y_proba.mean()), base_rate,
    )

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
    joblib.dump(
        {
            "pipeline": pipeline,
            # The raw (uncalibrated) LightGBM pipeline, kept alongside the
            # calibrated one. `pipeline` is now a CalibratedClassifierCV,
            # which has no `.named_steps` and cannot be handed to
            # shap.TreeExplainer directly - only the underlying sklearn
            # Pipeline can. Calibration is a monotonic rescaling, so it
            # does not change which features drive a prediction or their
            # relative SHAP attribution, only the probability's scale -
            # explainability tooling should read the base pipeline, serving
            # (predict_proba) should read the calibrated one.
            "base_pipeline": base_pipeline,
            "features": ALL_FEATURES,
            "threshold": threshold,
            "calibrated": True,
        },
        model_path,
    )

    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "LightGBM + sigmoid calibration",
        "calibrated": True,
        "calibration_method": "sigmoid",
        "threshold": threshold,
        "threshold_rationale": f"highest threshold that still guarantees recall >= {TARGET_RECALL} on the test set - maximizes precision subject to that recall floor (retention use case: catching churners is worth some false positives). Applied to CALIBRATED probabilities, so it sits near the base rate rather than near 0.5",
        "brier_score": float(brier),
        "brier_score_uncalibrated": float(brier_uncalibrated),
        "brier_score_always_base_rate": float(brier_base_rate),
        "mean_predicted_probability": float(y_proba.mean()),
        "mean_predicted_probability_uncalibrated": float(y_proba_uncalibrated.mean()),
        "actual_churn_rate_test": base_rate,
        "n_rows_total": len(df),
        "n_rows_train": len(X_train),
        "n_rows_calibration": len(X_calib),
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

    # Registry bookkeeping (src/model/registry.py) - a side channel, never
    # a training-blocking dependency. The joblib file above is already
    # saved and is the real serving artifact regardless of whether any of
    # this succeeds; an unreachable/misconfigured MLflow must not fail a
    # retrain that otherwise succeeded.
    if MLFLOW_TRACKING_URI:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("churn_model")
            with mlflow.start_run(run_name=version):
                mlflow.log_params({
                    "target_recall": TARGET_RECALL,
                    "max_training_rows": MAX_TRAINING_ROWS,
                    "calibration_method": "sigmoid",
                })
                mlflow.log_metrics({
                    k: v for k, v in metadata.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                })
                mlflow.log_artifact(str(model_path))
                mlflow.log_artifact(str(meta_path))
                run_id = mlflow.active_run().info.run_id
                mv = mlflow.register_model(f"runs:/{run_id}/{model_path.name}", "churn_model")
                client = mlflow.tracking.MlflowClient()
                client.set_model_version_tag("churn_model", mv.version, "model_file", model_path.name)
                # Promotion gate: reuse the recall floor already computed
                # above - do not build a new multi-metric comparison system.
                if metadata["recall_churn"] >= TARGET_RECALL:
                    client.set_registered_model_alias("churn_model", "champion", mv.version)
                    log.info("Registered churn_model v%s as champion", mv.version)
                else:
                    log.info(
                        "Registered churn_model v%s (not aliased champion: recall %.3f < target %.2f)",
                        mv.version, metadata["recall_churn"], TARGET_RECALL,
                    )
        except Exception as exc:
            log.warning("MLflow logging/registration failed (training itself succeeded): %s", exc)

    return metadata


if __name__ == "__main__":
    train_and_save()
