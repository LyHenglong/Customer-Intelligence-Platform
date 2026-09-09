"""
Streamlit dashboard: at-risk customers (churn probability + key risk
factors + a recommended retention action) plus aggregate churn-rate
charts. Reads customer_360 directly from Postgres and scores it live with
the latest churn/recommender model artifacts - no separate feature store.

Run locally:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import psycopg2
import streamlit as st
from dotenv import load_dotenv

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

load_dotenv()

from src.model.train_churn import ALL_FEATURES as CHURN_FEATURES  # noqa: E402
from src.model.train_recommender import recommend_for_customer  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[2] / "models_store"

st.set_page_config(page_title="Telecom Churn & Retention", layout="wide")


def get_pg_conn():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB", "warehouse"),
        user=os.environ.get("POSTGRES_USER"),
        password=os.environ.get("POSTGRES_PASSWORD"),
    )


@st.cache_data(ttl=300)
def load_customer_360() -> pd.DataFrame:
    # Raw psycopg2 connection, not a SQLAlchemy Engine - see the comment in
    # train_churn.py's load_customer_360 for why (pandas/SQLAlchemy version
    # mismatch inside the Airflow image breaks the Engine path).
    conn = get_pg_conn()
    try:
        return pd.read_sql("SELECT * FROM marts.customer_360", conn)
    finally:
        conn.close()


@st.cache_resource
def load_latest_artifact(pattern: str):
    matches = sorted(MODELS_DIR.glob(pattern))
    if not matches:
        return None, None
    path = matches[-1]
    return joblib.load(path), path.stem


@st.cache_data(ttl=300)
def score_churn(df: pd.DataFrame, _churn_artifact, churn_version: str) -> pd.Series:
    if _churn_artifact is None:
        return pd.Series([None] * len(df), index=df.index)
    X = df[CHURN_FEATURES].copy()
    for c in [f for f in CHURN_FEATURES if df[f].dtype == bool]:
        X[c] = X[c].astype(float)
    pipeline = _churn_artifact["pipeline"]
    return pd.Series(pipeline.predict_proba(X)[:, 1], index=df.index)


def top_risk_factors(row: pd.Series, importances: dict, top_k: int = 3) -> str:
    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    parts = []
    for feat, _ in ranked:
        if feat in row and pd.notna(row[feat]):
            parts.append(f"{feat}={row[feat]}")
        if len(parts) >= top_k:
            break
    return ", ".join(parts)


def main():
    st.title("Telecom Customer Churn & Retention Dashboard")
    st.caption(
        "Built on a **synthetic** ~1M-row Kaggle dataset "
        "(isandeep06/customer-churn-prediction-dataset-1m). "
        "Every number here reflects patterns in generated data, not a real telecom market."
    )

    df = load_customer_360()
    churn_artifact, churn_version = load_latest_artifact("churn_model_*.joblib")
    rec_artifact, rec_version = load_latest_artifact("recommender_*.joblib")

    st.sidebar.header("Filters")
    st.sidebar.metric("Customers loaded", f"{len(df):,}")
    if churn_version:
        st.sidebar.caption(f"Churn model: {churn_version.replace('churn_model_', '')}")
    if rec_version:
        st.sidebar.caption(f"Recommender: {rec_version.replace('recommender_', '')}")
    threshold = st.sidebar.slider("At-risk threshold (churn probability)", 0.0, 1.0, 0.5, 0.05)
    max_rows = st.sidebar.slider("Max at-risk customers to display", 10, 500, 100, 10)

    if churn_artifact is None:
        st.warning("No churn model artifact found in models_store/. Run `python -m src.model.train_churn` first.")
        return

    df = df.copy()
    df["churn_probability"] = score_churn(df, churn_artifact, churn_version)

    importances = {}
    try:
        pipeline = churn_artifact["pipeline"]
        preproc = pipeline.named_steps["preprocess"]
        clf = pipeline.named_steps["model"]
        feature_names = preproc.get_feature_names_out()
        for name, imp in zip(feature_names, clf.feature_importances_):
            base = name.split("__", 1)[-1].split("_")[0] if "__" in name else name
            importances[base] = importances.get(base, 0) + float(imp)
    except Exception:
        pass

    at_risk = df[df["churn_probability"] >= threshold].sort_values("churn_probability", ascending=False).head(max_rows)

    st.subheader(f"At-risk customers (probability ≥ {threshold:.2f})")
    st.caption(f"{len(at_risk):,} of {len(df):,} customers meet this threshold (showing top {len(at_risk.head(max_rows)):,}).")

    display_rows = []
    for _, row in at_risk.iterrows():
        risk_factors = top_risk_factors(row, importances) if importances else ""
        action = "-"
        if rec_artifact is not None:
            try:
                recs = recommend_for_customer(rec_artifact, row["customer_id"], top_n=1)
                if recs:
                    action = f"Offer: {recs[0]['service'].replace('has_', '').replace('_', ' ')}"
            except KeyError:
                action = "n/a (not in recommender training set)"
        display_rows.append({
            "customer_id": row["customer_id"],
            "churn_probability": round(row["churn_probability"], 3),
            "contract": row["contract"],
            "tenure_months": row["tenure"],
            "monthly_charges": row["monthlycharges"],
            "key_risk_factors": risk_factors,
            "recommended_action": action,
        })

    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    st.subheader("Aggregate churn patterns")
    col1, col2, col3 = st.columns(3)

    with col1:
        by_contract = df.groupby("contract")["churn"].mean().reset_index()
        by_contract.columns = ["contract", "churn_rate"]
        fig = px.bar(by_contract, x="contract", y="churn_rate", title="Churn rate by contract type")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        by_tenure = df.groupby("tenure_bucket")["churn"].mean().reset_index()
        by_tenure.columns = ["tenure_bucket", "churn_rate"]
        order = ["new_0_6mo", "established_6_24mo", "loyal_24mo_plus"]
        by_tenure["tenure_bucket"] = pd.Categorical(by_tenure["tenure_bucket"], categories=order, ordered=True)
        by_tenure = by_tenure.sort_values("tenure_bucket")
        fig = px.bar(by_tenure, x="tenure_bucket", y="churn_rate", title="Churn rate by tenure")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        by_bundle = df.groupby("total_active_services")["churn"].mean().reset_index()
        by_bundle.columns = ["total_active_services", "churn_rate"]
        fig = px.bar(by_bundle, x="total_active_services", y="churn_rate", title="Churn rate by service bundle size")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
