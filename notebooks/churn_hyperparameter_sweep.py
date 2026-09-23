"""
LightGBM hyperparameter sweep for the churn model.

The production model (src/model/train_churn.py) runs on untuned defaults -
n_estimators=300, max_depth=6, learning_rate=0.1 - chosen when the model
family was picked, never searched. notebooks/model_dev_offline.py already
compared model families, imbalance handling and training sizes and found a
0.65-0.68 AUC band it attributed to the synthetic data's generated signal
(report/findings.md section 2). Hyperparameters are the one lever that
audit did not pull, so this pulls it and reports honestly whether it
moves anything.

Two things this deliberately does NOT do:

  - Pick a winner on a single train/test split. Run-to-run variance on
    this dataset is ~0.03 AUC, which is larger than any plausible tuning
    gain, so a single split would mostly measure luck. Every candidate is
    scored by StratifiedKFold CV on identical folds, and the comparison
    that matters is mean +/- std across those folds.
  - Score candidates on the test set. The sweep selects on CV over the
    training portion only; the held-out test set is touched exactly once
    at the end, to report the chosen config's honest generalization.

Reuses build_customer_360_locally() from model_dev_offline.py, so no
Postgres or Docker is required.

Usage:
    PYTHONPATH=. python notebooks/churn_hyperparameter_sweep.py
    PYTHONPATH=. python notebooks/churn_hyperparameter_sweep.py --rows 200000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.model.train_churn import (
    ALL_FEATURES,
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hyperparameter_sweep")

# The config currently in production (src/model/train_churn.py). Every
# candidate is measured against this, on the same folds.
BASELINE = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
}

# Deliberately a small, hand-picked grid rather than a large random search:
# the expected effect size here is a few thousandths of AUC against ~0.03
# fold-to-fold noise, so a wide search would mostly generate opportunities
# to fool ourselves with a lucky draw. These vary the knobs that actually
# trade bias against variance for this shape of problem - capacity
# (leaves/depth), how fast it commits (learning rate x rounds), and
# regularization/subsampling for a weak-signal, imbalanced target.
CANDIDATES = [
    {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05},
    {"n_estimators": 1000, "max_depth": 6, "learning_rate": 0.03},
    {"n_estimators": 600, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 600, "max_depth": 8, "learning_rate": 0.05},
    {"n_estimators": 600, "max_depth": -1, "num_leaves": 63, "learning_rate": 0.05},
    {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05,
     "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8},
    {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05,
     "reg_alpha": 1.0, "reg_lambda": 5.0},
    {"n_estimators": 1000, "max_depth": 5, "learning_rate": 0.03,
     "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
     "reg_lambda": 5.0, "min_child_samples": 50},
]


def _build_preprocessor() -> ColumnTransformer:
    """Mirrors src/model/train_churn.py's build_pipeline() exactly, so the
    only thing varying across candidates is the model's hyperparameters.
    Duplicated rather than imported because build_pipeline() returns the
    preprocessor already welded to a fixed LGBMClassifier."""
    categorical_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES + BOOLEAN_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def _make_pipeline(params: dict) -> Pipeline:
    clf = LGBMClassifier(
        class_weight="balanced",  # held fixed: already compared against SMOTE
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    return Pipeline([
        ("preprocess", _build_preprocessor()),
        ("model", clf),
    ])


def _score(params: dict, X, y, cv) -> tuple[float, float, float]:
    t0 = time.monotonic()
    scores = cross_val_score(_make_pipeline(params), X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return float(scores.mean()), float(scores.std()), time.monotonic() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows", type=int, default=200_000,
        help="rows to sample for the sweep (memory-bound; see README's RAM note)",
    )
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()

    from notebooks.model_dev_offline import build_customer_360_locally

    log.info("reconstructing customer_360 from local batch CSVs (no Postgres needed)")
    df = build_customer_360_locally()
    log.info("reconstructed %d rows", len(df))

    if args.rows and len(df) > args.rows:
        df = df.sample(n=args.rows, random_state=42).reset_index(drop=True)
        log.info("sampled %d rows for the sweep", len(df))

    X = df[ALL_FEATURES]
    y = df[TARGET].astype(int)
    del df

    # The test split is held back from the entire sweep and scored once, at
    # the end, by the winner only.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )
    del X, y
    log.info("dev set %d rows, held-out test %d rows, churn rate %.4f",
             len(X_dev), len(X_test), float(y_dev.mean()))

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    results = []
    base_mean, base_std, base_secs = _score(BASELINE, X_dev, y_dev, cv)
    log.info("BASELINE (production config)  AUC %.4f +/- %.4f  [%.0fs]", base_mean, base_std, base_secs)
    results.append({"label": "baseline (production)", "params": BASELINE,
                    "auc_mean": base_mean, "auc_std": base_std, "seconds": base_secs})

    for i, params in enumerate(CANDIDATES, 1):
        mean, std, secs = _score(params, X_dev, y_dev, cv)
        delta = mean - base_mean
        log.info("candidate %d/%d  AUC %.4f +/- %.4f  (delta %+.4f)  %s  [%.0fs]",
                 i, len(CANDIDATES), mean, std, delta, params, secs)
        results.append({"label": f"candidate {i}", "params": params,
                        "auc_mean": mean, "auc_std": std, "seconds": secs})

    results.sort(key=lambda r: r["auc_mean"], reverse=True)
    best = results[0]

    # Adoption rule, decided before seeing any numbers: a candidate has to
    # clear the baseline by more than the baseline's own fold-to-fold
    # standard deviation. Anything smaller is inside the noise this dataset
    # produces run to run, and "wins" of that size are how tuning theatre
    # gets mistaken for progress.
    improvement = best["auc_mean"] - base_mean
    adopt = best["label"] != "baseline (production)" and improvement > base_std

    print("\n" + "=" * 78)
    print(f"{'config':<24} {'CV AUC':>10} {'+/-':>8} {'vs base':>10}")
    print("-" * 78)
    for r in results:
        print(f"{r['label']:<24} {r['auc_mean']:>10.4f} {r['auc_std']:>8.4f} "
              f"{r['auc_mean'] - base_mean:>+10.4f}")
    print("=" * 78)
    print(f"\nBaseline fold-to-fold std: {base_std:.4f}")
    print(f"Best improvement:          {improvement:+.4f}")
    print(f"Adoption threshold:        > {base_std:.4f} (one baseline std)")
    print(f"VERDICT: {'ADOPT ' + str(best['params']) if adopt else 'KEEP PRODUCTION CONFIG - gain is inside the noise'}")

    if adopt:
        log.info("scoring the winner once on the held-out test set")
        pipeline = _make_pipeline(best["params"]).fit(X_dev, y_dev)
        from sklearn.metrics import roc_auc_score
        test_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        print(f"Held-out test AUC for the winner: {test_auc:.4f}")
        best["held_out_test_auc"] = float(test_auc)

    out = Path("report/hyperparameter_sweep_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "rows_used": int(len(X_dev) + len(X_test)),
        "folds": args.folds,
        "baseline_auc_mean": base_mean,
        "baseline_auc_std": base_std,
        "adopted": adopt,
        "results": results,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
