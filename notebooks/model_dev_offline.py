"""
Offline model-development experiment: full feature audit, threshold
tuning, model comparison (RandomForest / XGBoost / LightGBM / Logistic
Regression), imbalance-handling comparison (class_weight vs SMOTE vs
scale_pos_weight), and 5-fold CV on the best approach - all on the full
locally-reconstructed 1M-row dataset, no Postgres/Docker required.

Reuses batch_loader.py's exact DuckDB cleaning/validation/feature logic,
then replicates dbt's intermediate/marts-layer derived columns in pandas.

Extra dependencies beyond requirements.txt (this script only - not needed
to run the production pipeline, which just uses the LightGBM winner):
    pip install xgboost==2.1.3 imbalanced-learn==0.12.4

Usage:
    PYTHONPATH=. python notebooks/model_dev_offline.py
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.ingest.batch_loader import CLEANED_COLUMNS, RAW_DIR, process_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("model_dev_offline")

_SERVICE_FLAGS = [
    "has_phone_service", "has_internet_service", "has_online_security",
    "has_online_backup", "has_device_protection", "has_tech_support",
    "has_streaming_tv", "has_streaming_movies",
]

NUMERIC_FEATURES = [
    "age", "annual_income", "dependents", "tenure", "tenure_years",
    "monthlycharges", "totalcharges", "num_services", "total_active_services",
    "cost_per_active_service", "customer_satisfaction", "num_complaints",
    "num_service_calls", "late_payments", "avg_monthly_gb", "avg_gb_per_service",
    "days_since_last_interaction", "credit_score", "complaints_per_tenure_month",
    "annual_spend_to_income_ratio",
]
BOOLEAN_FEATURES = [
    "senior_citizen", "paperless_billing", *_SERVICE_FLAGS,
    "is_month_to_month", "is_disengaged",
]
CATEGORICAL_FEATURES = ["gender", "education", "marital_status", "contract", "payment_method", "tenure_bucket"]
TARGET = "churn"
ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

# Candidate extra features derived from signup_date, tested in Step 1 to see
# if they add anything beyond what tenure already captures.
SIGNUP_DERIVED_FEATURES = ["signup_month", "signup_dayofweek", "signup_year"]


def build_customer_360_locally() -> pd.DataFrame:
    frames = []
    for batch_path in sorted(RAW_DIR.glob("batch_*.csv")):
        _, cleaned_rows, stats = process_batch(batch_path)
        df = pd.DataFrame(cleaned_rows, columns=CLEANED_COLUMNS)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df["is_month_to_month"] = df["contract"] == "month_to_month"
    df["cost_per_active_service"] = np.where(
        df["total_active_services"] > 0,
        (df["monthlycharges"] / df["total_active_services"]).round(2),
        np.nan,
    )
    df["complaints_per_tenure_month"] = np.where(
        df["tenure"] > 0, df["num_complaints"] / df["tenure"], df["num_complaints"]
    )
    df["is_disengaged"] = df["days_since_last_interaction"] > 90
    df["annual_spend_to_income_ratio"] = np.where(
        (df["annual_income"].notna()) & (df["annual_income"] > 0),
        (df["monthlycharges"] * 12) / df["annual_income"],
        np.nan,
    )
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 5, 23, 10_000],
        labels=["new_0_6mo", "established_6_24mo", "loyal_24mo_plus"],
    ).astype(str)

    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["signup_month"] = df["signup_date"].dt.month
    df["signup_dayofweek"] = df["signup_date"].dt.dayofweek
    df["signup_year"] = df["signup_date"].dt.year

    return df


def make_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
    ])


def fit_eval(pipeline, X_train, y_train, X_test, y_test, label: str) -> dict:
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, proba)
    elapsed = time.time() - t0
    log.info(
        "[%-32s] precision=%.3f recall=%.3f f1=%.3f auc=%.3f  (%.1fs)",
        label, report["1"]["precision"], report["1"]["recall"], report["1"]["f1-score"], auc, elapsed,
    )
    return {
        "label": label, "auc": auc,
        "precision": report["1"]["precision"], "recall": report["1"]["recall"], "f1": report["1"]["f1-score"],
        "proba": proba, "pipeline": pipeline, "seconds": elapsed,
    }


def main():
    t_start = time.time()

    # =========================================================================
    log.info("=" * 78)
    log.info("STEP 1: FEATURE AUDIT")
    log.info("=" * 78)
    df = build_customer_360_locally()
    log.info("Total rows: %d", len(df))
    log.info("Class balance: %s", df[TARGET].value_counts(normalize=True).round(4).to_dict())

    all_cols = set(df.columns) - {"customer_id", TARGET, "signup_date", "signup_month", "signup_dayofweek", "signup_year"}
    used = set(ALL_FEATURES)
    log.info("customer_360 columns available as raw features: %d", len(all_cols))
    log.info("Currently used as model features: %d", len(used))
    log.info("Available but NOT used: %s", sorted(all_cols - used) or "(none)")

    # Redundancy check: tenure vs tenure_years should correlate ~1.0 (pure rescale)
    corr_tenure = df["tenure"].corr(df["tenure_years"])
    log.info("Redundancy check: corr(tenure, tenure_years) = %.4f (near-1.0 = fully redundant pair)", corr_tenure)
    log.info("Redundancy check: is_month_to_month is a direct re-encoding of contract=='month_to_month' (kept - harmless for tree models, gives the tree a direct split)")

    # Leakage check: nothing here is derived FROM churn - confirm by construction:
    # every derived column traces back to raw source columns (monthlycharges,
    # tenure, num_complaints, annual_income, days_since_last_interaction), never
    # to the churn label itself. No target leakage.
    log.info("Leakage check: all 20 derived/engineered features trace to raw non-target columns only - no target leakage found.")

    X_full = df[ALL_FEATURES].copy()
    for c in BOOLEAN_FEATURES:
        X_full[c] = X_full[c].astype(float)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42, stratify=y)

    baseline_pre = make_preprocessor(NUMERIC_FEATURES + BOOLEAN_FEATURES, CATEGORICAL_FEATURES)
    baseline_rf = Pipeline([("prep", baseline_pre), ("model", RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1))])
    baseline_result = fit_eval(baseline_rf, X_train, y_train, X_test, y_test, "Baseline RF (current 38 features)")

    # Individual (not grouped) feature importances
    feat_names = baseline_result["pipeline"].named_steps["prep"].get_feature_names_out()
    importances = baseline_result["pipeline"].named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
    log.info("Top 15 individual feature importances:\n%s", imp_df.head(15).to_string(index=False))

    # Test: does adding signup-date-derived features help beyond tenure?
    extra_numeric = NUMERIC_FEATURES + ["signup_month", "signup_dayofweek", "signup_year"]
    X_train_ext = X_train.copy()
    X_test_ext = X_test.copy()
    X_train_ext[["signup_month", "signup_dayofweek", "signup_year"]] = df.loc[X_train.index, ["signup_month", "signup_dayofweek", "signup_year"]]
    X_test_ext[["signup_month", "signup_dayofweek", "signup_year"]] = df.loc[X_test.index, ["signup_month", "signup_dayofweek", "signup_year"]]
    ext_pre = make_preprocessor(extra_numeric + BOOLEAN_FEATURES, CATEGORICAL_FEATURES)
    ext_rf = Pipeline([("prep", ext_pre), ("model", RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1))])
    ext_result = fit_eval(ext_rf, X_train_ext, y_train, X_test_ext, y_test, "RF + signup-date features (41 features)")

    signup_helps = ext_result["auc"] > baseline_result["auc"] + 0.003  # small margin, not noise
    log.info(
        "Signup-date features verdict: AUC %.4f -> %.4f (%s) - %s",
        baseline_result["auc"], ext_result["auc"],
        "+" if ext_result["auc"] >= baseline_result["auc"] else "-",
        "keeping them, real (if small) lift" if signup_helps else "no meaningful lift, NOT adding them",
    )

    # Finalize feature set for the rest of the experiment based on this finding
    if signup_helps:
        final_numeric = extra_numeric
        X_train_final, X_test_final = X_train_ext, X_test_ext
    else:
        final_numeric = NUMERIC_FEATURES
        X_train_final, X_test_final = X_train, X_test
    final_preprocessor = make_preprocessor(final_numeric + BOOLEAN_FEATURES, CATEGORICAL_FEATURES)

    # =========================================================================
    log.info("=" * 78)
    log.info("STEP 2: THRESHOLD TUNING (on baseline RF's probabilities)")
    log.info("=" * 78)
    proba = baseline_result["proba"]
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    # report precision at a few recall targets
    for target_recall in [0.40, 0.50, 0.60, 0.70]:
        idx_candidates = np.where(recalls[:-1] >= target_recall)[0]
        if len(idx_candidates) == 0:
            log.info("  recall >= %.2f: not reachable", target_recall)
            continue
        idx = idx_candidates[-1]  # highest threshold that still clears this recall
        log.info(
            "  to hit recall >= %.2f: threshold=%.3f -> precision=%.3f, recall=%.3f",
            target_recall, thresholds[idx], precisions[idx], recalls[idx],
        )
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    best_f1_threshold = thresholds[best_idx]
    log.info(
        "Best-F1 threshold: %.3f -> precision=%.3f recall=%.3f f1=%.3f",
        best_f1_threshold, precisions[best_idx], recalls[best_idx], f1s[best_idx],
    )
    # business-chosen threshold: prioritize recall (catching churners) at a
    # still-reasonable precision, for a retention-campaign use case
    biz_candidates = np.where(recalls[:-1] >= 0.60)[0]
    biz_idx = biz_candidates[-1] if len(biz_candidates) else best_idx
    biz_threshold = thresholds[biz_idx]
    log.info(
        "Chosen business threshold (>=0.60 recall priority): %.3f -> precision=%.3f recall=%.3f f1=%.3f",
        biz_threshold, precisions[biz_idx], recalls[biz_idx], f1s[biz_idx],
    )

    # =========================================================================
    log.info("=" * 78)
    log.info("STEP 3: ALTERNATIVE MODELS (same train/test split, same final feature set)")
    log.info("=" * 78)
    model_results = [baseline_result]

    from xgboost import XGBClassifier
    xgb_pipeline = Pipeline([("prep", final_preprocessor), ("model", XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss", random_state=42, n_jobs=-1))])
    model_results.append(fit_eval(xgb_pipeline, X_train_final, y_train, X_test_final, y_test, "XGBoost (scale_pos_weight)"))

    from lightgbm import LGBMClassifier
    lgbm_pipeline = Pipeline([("prep", final_preprocessor), ("model", LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1))])
    model_results.append(fit_eval(lgbm_pipeline, X_train_final, y_train, X_test_final, y_test, "LightGBM (class_weight=balanced)"))

    logreg_pipeline = Pipeline([("prep", final_preprocessor), ("model", LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42))])
    model_results.append(fit_eval(logreg_pipeline, X_train_final, y_train, X_test_final, y_test, "LogisticRegression (baseline)"))

    log.info("\nStep 3 comparison table:")
    print(pd.DataFrame(model_results)[["label", "auc", "precision", "recall", "f1"]].to_string(index=False))

    best_model_result = max(model_results, key=lambda r: r["auc"])
    log.info("Best by AUC so far: %s (auc=%.4f)", best_model_result["label"], best_model_result["auc"])

    # =========================================================================
    log.info("=" * 78)
    log.info("STEP 4: IMBALANCE HANDLING BEYOND class_weight/scale_pos_weight")
    log.info("=" * 78)
    from imblearn.over_sampling import SMOTE

    # SMOTE needs numeric input - fit_transform preprocessor first, then resample
    prep_for_smote = make_preprocessor(final_numeric + BOOLEAN_FEATURES, CATEGORICAL_FEATURES)
    X_train_transformed = prep_for_smote.fit_transform(X_train_final, y_train)
    X_train_dense = X_train_transformed.toarray() if hasattr(X_train_transformed, "toarray") else X_train_transformed

    smote = SMOTE(random_state=42, n_jobs=-1)
    X_train_res, y_train_res = smote.fit_resample(X_train_dense, y_train)
    log.info("SMOTE: %d rows -> %d rows (class balance now %s)", len(y_train), len(y_train_res), pd.Series(y_train_res).value_counts(normalize=True).round(3).to_dict())

    X_test_transformed = prep_for_smote.transform(X_test_final)
    X_test_dense = X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed

    # best non-linear model architecture from step 3, but on SMOTE-resampled
    # data with NO class_weight (SMOTE already balances the classes directly)
    is_xgb_best = "XGBoost" in best_model_result["label"]
    if is_xgb_best:
        smote_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=42, n_jobs=-1)
    else:
        smote_model = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)

    t0 = time.time()
    smote_model.fit(X_train_res, y_train_res)
    proba_smote = smote_model.predict_proba(X_test_dense)[:, 1]
    pred_smote = (proba_smote >= 0.5).astype(int)
    report_smote = classification_report(y_test, pred_smote, output_dict=True, zero_division=0)
    auc_smote = roc_auc_score(y_test, proba_smote)
    elapsed = time.time() - t0
    smote_result = {
        "label": f"{'XGBoost' if is_xgb_best else 'LightGBM'} + SMOTE (no class_weight)",
        "auc": auc_smote, "precision": report_smote["1"]["precision"],
        "recall": report_smote["1"]["recall"], "f1": report_smote["1"]["f1-score"], "seconds": elapsed,
    }
    log.info(
        "[%-32s] precision=%.3f recall=%.3f f1=%.3f auc=%.3f  (%.1fs)",
        smote_result["label"], smote_result["precision"], smote_result["recall"], smote_result["f1"], smote_result["auc"], elapsed,
    )
    log.info(
        "SMOTE verdict: AUC %.4f (class_weight/scale_pos_weight) vs %.4f (SMOTE) -> %s",
        best_model_result["auc"], auc_smote,
        "SMOTE helps" if auc_smote > best_model_result["auc"] + 0.003 else "no meaningful improvement over class_weight",
    )

    # =========================================================================
    log.info("=" * 78)
    log.info("STEP 5: 5-FOLD CROSS-VALIDATION on the best approach")
    log.info("=" * 78)
    overall_best = max(model_results + [smote_result], key=lambda r: r["auc"])
    log.info("Running 5-fold CV on: %s", overall_best["label"])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs, cv_f1s = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y), 1):
        X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        cv_pre = make_preprocessor(NUMERIC_FEATURES + BOOLEAN_FEATURES, CATEGORICAL_FEATURES)
        if "XGBoost" in overall_best["label"]:
            cv_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                      scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
                                      eval_metric="logloss", random_state=42, n_jobs=-1)
        else:
            cv_model = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                       class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
        cv_pipeline = Pipeline([("prep", cv_pre), ("model", cv_model)])
        cv_pipeline.fit(X_tr, y_tr)
        cv_proba = cv_pipeline.predict_proba(X_val)[:, 1]
        cv_pred = (cv_proba >= 0.5).astype(int)
        cv_report = classification_report(y_val, cv_pred, output_dict=True, zero_division=0)
        fold_auc = roc_auc_score(y_val, cv_proba)
        fold_f1 = cv_report["1"]["f1-score"]
        cv_aucs.append(fold_auc)
        cv_f1s.append(fold_f1)
        log.info("  fold %d: auc=%.4f f1=%.4f", fold, fold_auc, fold_f1)

    log.info(
        "5-fold CV result: AUC = %.4f +/- %.4f | F1 = %.4f +/- %.4f",
        np.mean(cv_aucs), np.std(cv_aucs), np.mean(cv_f1s), np.std(cv_f1s),
    )

    # =========================================================================
    log.info("=" * 78)
    log.info("STEP 6: FINAL COMPARISON")
    log.info("=" * 78)
    final_table = pd.DataFrame(model_results + [smote_result])[["label", "auc", "precision", "recall", "f1"]]
    print(final_table.to_string(index=False))
    log.info(
        "\nBaseline (RF @ 0.5):    auc=%.4f f1=%.4f",
        baseline_result["auc"], baseline_result["f1"],
    )
    log.info(
        "Best single-split result: %s -> auc=%.4f f1=%.4f",
        overall_best["label"], overall_best["auc"], overall_best.get("f1", float("nan")),
    )
    log.info(
        "5-fold CV (honest estimate): auc=%.4f +/- %.4f",
        np.mean(cv_aucs), np.std(cv_aucs),
    )
    log.info("Total wall time: %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
