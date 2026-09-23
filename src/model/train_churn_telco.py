"""
Trains the churn model on the REAL IBM Telco dataset, as a signal
benchmark against the synthetic 1M-row pipeline.

Why this exists: the production model tops out near ROC AUC 0.67, and a
per-feature audit (report/findings.md section 2b) showed why - 27 of the
synthetic dataset's 38 features are statistically indistinguishable from
noise, and its best single feature carries only 0.614. That is a property
of how the data was generated, not of the modelling code. This script
proves that directly by running the *same* methodology - same three-way
split, same calibration, same recall-floor threshold rule, same tuned
LightGBM hyperparameters - against a real dataset with real signal, where
it reaches ~0.835.

It is a benchmark, deliberately NOT a second production model:

  - The artifact is named `telco_churn_*.joblib`, which does not match the
    `churn_model_*.joblib` glob that src/model/api.py and
    src/ai/tools/_artifacts.py resolve against. A 7,043-row model trained
    on a different schema must never be picked up and served against
    marts.customer_360's 1,000,000 customers.
  - Nothing in the serving path, the DAG, or the frontend reads it.

The synthetic pipeline stays the platform's scale story (1M rows through
DuckDB/dbt/Airflow with streaming reads and drift monitoring); this is the
signal story. Neither substitutes for the other, and conflating their AUCs
would be comparing a 7K-row dataset to a 1M-row one.

Requires a Kaggle API token (same one README step 1 already sets up).

Usage:
    PYTHONPATH=. python -m src.model.train_churn_telco
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.model.train_churn import choose_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_churn_telco")

KAGGLE_DATASET = "blastchar/telco-customer-churn"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_telco"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models_store"

TARGET = "Churn"
ID_COL = "customerID"


def download_if_missing() -> Path:
    """Downloads the dataset on first run. Kept out of the repo the same
    way data/raw/ is - it is third-party data, not ours to redistribute."""
    existing = list(DATA_DIR.glob("*.csv"))
    if existing:
        return existing[0]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s (first run only)", KAGGLE_DATASET)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), unzip=True)
    return next(iter(DATA_DIR.glob("*.csv")))


def load_telco() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = pd.read_csv(download_if_missing())
    # TotalCharges arrives as text with blanks for customers whose tenure is
    # 0 - they have been billed nothing yet. Coerced to NaN and imputed
    # rather than dropped: "brand new customer" is a real, predictive state.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[ID_COL, TARGET])
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return X, y, numeric, categorical


def build_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    # Identical hyperparameters to the production model
    # (src/model/train_churn.py), which is the entire point: holding the
    # method fixed is what makes the AUC gap attributable to the data.
    clf = LGBMClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", clf)])


def main() -> None:
    X, y, numeric, categorical = load_telco()
    log.info("loaded %d rows, %d features, churn rate %.4f", len(X), X.shape[1], y.mean())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(build_pipeline(numeric, categorical), X, y, cv=cv, scoring="roc_auc")
    log.info("5-fold CV ROC AUC: %.4f +/- %.4f", cv_scores.mean(), cv_scores.std())

    # Same three-way split as the production trainer: calibration never
    # sees training or test data, so the reported metrics stay honest.
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y,
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.6667, random_state=42, stratify=y_tmp,
    )
    log.info("train %d / calibration %d / test %d", len(X_train), len(X_cal), len(X_test))

    base_pipeline = build_pipeline(numeric, categorical).fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(base_pipeline, method="sigmoid", cv="prefit")
    calibrated.fit(X_cal, y_cal)

    proba = calibrated.predict_proba(X_test)[:, 1]
    raw_proba = base_pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    brier_raw = brier_score_loss(y_test, raw_proba)
    base_rate_brier = brier_score_loss(y_test, np.full(len(y_test), y_train.mean()))
    log.info("Brier: %.4f calibrated vs %.4f uncalibrated (always-base-rate %.4f)",
             brier, brier_raw, base_rate_brier)

    threshold, precision, recall = choose_threshold(y_test, proba)
    log.info("Decision threshold: %.3f -> precision=%.3f recall=%.3f", threshold, precision, recall)
    preds = (proba >= threshold).astype(int)
    log.info("Classification report:\n%s", classification_report(y_test, preds))
    log.info("ROC AUC: %.4f", roc_auc)

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # telco_churn_*, NOT churn_model_* - see this module's docstring. The
    # production glob must not be able to pick this up.
    artifact_path = MODELS_DIR / f"telco_churn_{version}.joblib"
    joblib.dump({
        "pipeline": calibrated,
        "base_pipeline": base_pipeline,
        "threshold": float(threshold),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "dataset": KAGGLE_DATASET,
    }, artifact_path)

    metadata = {
        "version": version,
        "dataset": KAGGLE_DATASET,
        "purpose": "real-data signal benchmark; not served - see module docstring",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(X_train)),
        "total_rows": int(len(X)),
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "roc_auc": float(roc_auc),
        "precision_churn": float(precision),
        "recall_churn": float(recall),
        "f1_churn": float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "threshold": float(threshold),
        "brier_calibrated": float(brier),
        "brier_uncalibrated": float(brier_raw),
        "brier_always_base_rate": float(base_rate_brier),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    metadata_path = MODELS_DIR / f"telco_churn_{version}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    log.info("Saved %s", artifact_path)
    log.info("Saved %s", metadata_path)


if __name__ == "__main__":
    main()
