"""
Executive-facing churn & retention dashboard: headline KPIs, at-risk
customer list with recommended retention actions, segment breakdowns,
model performance transparency, and pipeline/data-freshness status.
Reads customer_360 directly from Postgres and scores it live with the
latest churn/recommender model artifacts - no separate feature store.

Run locally:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from dotenv import load_dotenv

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

load_dotenv()

from src.model.train_churn import ALL_FEATURES as CHURN_FEATURES  # noqa: E402
from src.model.train_churn import CATEGORICAL_FEATURES as CHURN_CATEGORICAL  # noqa: E402
from src.model.train_recommender import recommend_for_customer  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[2] / "models_store"
RETRAIN_EVERY_N_BATCHES = int(os.environ.get("RETRAIN_EVERY_N_BATCHES", "3"))
TOTAL_SIMULATED_BATCHES = 13

st.set_page_config(page_title="Retention Command Center", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Design system: a small set of CSS custom properties, applied consistently
# across every custom element rather than one-off styling per widget.
# Defined for light mode in :root, redefined under the dark media query so
# the page stays legible regardless of the viewer's OS theme.
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Public+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --rc-bg: #f5f6f8;
    --rc-surface: #ffffff;
    --rc-border: #e2e5eb;
    --rc-ink: #10192b;
    --rc-ink-soft: #5b6577;
    --rc-ink-faint: #8993a4;
    --rc-accent: #a6790a;
    --rc-accent-soft: #faf3e2;
    --rc-accent-line: #e8d3a0;
    --rc-navy: #10192b;
    --rc-navy-soft: #1b2740;
    --rc-risk-high: #b3261e;
    --rc-risk-high-soft: #fbe9e7;
    --rc-risk-med: #b06a00;
    --rc-risk-med-soft: #fdf1de;
    --rc-risk-low: #1e7a4c;
    --rc-risk-low-soft: #e5f4ec;
}
@media (prefers-color-scheme: dark) {
    :root {
        --rc-bg: #0e1320;
        --rc-surface: #161d2e;
        --rc-border: #2a3346;
        --rc-ink: #eef1f6;
        --rc-ink-soft: #a7b0c3;
        --rc-ink-faint: #78829a;
        --rc-accent: #e0ac3f;
        --rc-accent-soft: #2a2214;
        --rc-accent-line: #4a3a1a;
        --rc-navy: #0a0f1a;
        --rc-navy-soft: #141c2e;
        --rc-risk-high: #e5877e;
        --rc-risk-high-soft: #33201e;
        --rc-risk-med: #e0ac3f;
        --rc-risk-med-soft: #332510;
        --rc-risk-low: #6fce97;
        --rc-risk-low-soft: #16281f;
    }
}

html, body, [class*="css"] { font-family: 'Public Sans', -apple-system, sans-serif; }
.stApp { background: var(--rc-bg); }

h1, h2, h3 { font-family: 'Sora', sans-serif !important; color: var(--rc-ink) !important; letter-spacing: -0.01em; }

/* Sidebar: dark navy, distinct from the light main canvas */
section[data-testid="stSidebar"] {
    background: var(--rc-navy);
    border-right: 1px solid var(--rc-navy-soft);
}
section[data-testid="stSidebar"] * { color: #d7dce6 !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* KPI cards: Streamlit's own st.metric, reskinned via its stable testids */
div[data-testid="stMetric"] {
    background: var(--rc-surface);
    border: 1px solid var(--rc-border);
    border-radius: 10px;
    padding: 18px 20px 14px;
    box-shadow: 0 1px 2px rgba(16, 25, 43, 0.04);
}
div[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11.5px !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--rc-ink-faint) !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'Sora', sans-serif !important;
    color: var(--rc-ink) !important;
    font-weight: 700 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Public Sans', sans-serif;
    font-weight: 600;
    color: var(--rc-ink-soft);
}
button[data-baseweb="tab"][aria-selected="true"] { color: var(--rc-accent); }
div[data-baseweb="tab-highlight"] { background-color: var(--rc-accent) !important; }

/* View selector (st.radio, horizontal) styled as a tab bar - used instead
   of st.tabs because st.tabs runs every tab's body on every rerun, which
   OOM'd this dashboard at the current data size (see main()'s comment). */
div[role="radiogroup"] { gap: 6px; border-bottom: 1px solid var(--rc-border); padding-bottom: 0; }
div[role="radiogroup"] label {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    padding: 8px 4px;
    margin-right: 18px !important;
    font-family: 'Public Sans', sans-serif;
    font-weight: 600;
    color: var(--rc-ink-soft);
}
div[role="radiogroup"] label[data-checked="true"],
div[role="radiogroup"] label:has(input:checked) {
    color: var(--rc-accent);
    border-bottom-color: var(--rc-accent);
}
div[role="radiogroup"] label > div:first-child { display: none; }

/* Section header eyebrow */
.rc-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--rc-accent);
    background: var(--rc-accent-soft);
    border: 1px solid var(--rc-accent-line);
    display: inline-block;
    padding: 3px 9px;
    border-radius: 4px;
    margin-bottom: 6px;
}

/* Status pill (used for pipeline / model health) */
.rc-pill { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; font-weight: 600;
    padding: 3px 10px; border-radius: 20px; display: inline-block; }
.rc-pill-good { background: var(--rc-risk-low-soft); color: var(--rc-risk-low); }
.rc-pill-warn { background: var(--rc-risk-med-soft); color: var(--rc-risk-med); }
.rc-pill-bad  { background: var(--rc-risk-high-soft); color: var(--rc-risk-high); }

/* Data-disclosure banner */
.rc-disclosure {
    font-size: 13px; color: var(--rc-ink-soft); background: var(--rc-surface);
    border: 1px solid var(--rc-border); border-left: 3px solid var(--rc-accent);
    border-radius: 0 8px 8px 0; padding: 10px 14px; margin-bottom: 6px;
}
</style>
"""

# Consistent categorical palette for every chart on the page, instead of
# each px.bar() call picking its own default colors.
_CHART_COLORWAY = ["#a6790a", "#10192b", "#5b6577", "#c9a84c", "#8993a4", "#e8d3a0"]

# signup_date deliberately excluded: not used anywhere in this dashboard,
# and pulling a raw timestamp for 300K+ rows for nothing is pure memory
# waste on a machine that's already tight (see README's RAM constraint).
_DASHBOARD_COLUMNS = [c for c in ["customer_id"] + CHURN_FEATURES + [
    "churn", "contract", "tenure", "monthlycharges", "tenure_bucket", "total_active_services",
] if c != "signup_date"]
_DASHBOARD_COLUMNS = list(dict.fromkeys(_DASHBOARD_COLUMNS))  # de-dupe, keep order


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
        cols = ", ".join(_DASHBOARD_COLUMNS)
        # chunksize, not a single read_sql call: psycopg2/pandas materialize
        # the *entire* result as raw Python row-tuples (300K+ rows x 40
        # cols = ~12M individual Python objects) before converting to a
        # columnar DataFrame, creating a huge transient memory peak even
        # though the final DataFrame itself is much smaller. Profiled this
        # directly: peak RSS jumped ~880MB for a single full read vs. a
        # final DataFrame of only ~190MB. Chunking bounds how much of that
        # raw row-tuple form exists at once.
        # category, not object dtype, for low-cardinality string columns:
        # object dtype stores each cell as a separate Python string object
        # (~20MB each for these columns at 300K+ rows); category dtype
        # stores each unique value once plus integer codes per row. Applied
        # per chunk, before accumulating - converting only after the full
        # concat still briefly holds every chunk in expensive object dtype
        # at once. Profiled this exact sequence directly: dropped peak RSS
        # from ~1073MB (single unchunked read) to ~747MB.
        frames = []
        for chunk in pd.read_sql(f"SELECT {cols} FROM marts.customer_360", conn, chunksize=10_000):
            for c in CHURN_CATEGORICAL:  # gender, education, marital_status, contract, payment_method, tenure_bucket
                if c in chunk.columns:
                    chunk[c] = chunk[c].astype("category")
            frames.append(chunk)
        return pd.concat(frames, ignore_index=True)
    finally:
        conn.close()


@st.cache_data(ttl=300)
def load_ingestion_log() -> pd.DataFrame:
    conn = get_pg_conn()
    try:
        return pd.read_sql(
            "SELECT batch_file, rows_loaded, loaded_at, status FROM ingestion_log ORDER BY loaded_at", conn
        )
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
def load_all_churn_metadata() -> list[dict]:
    """All churn_model_*.json metadata files, oldest to newest - lets the
    Model Performance tab show a real retraining trend, not just the
    latest snapshot."""
    records = []
    for path in sorted(MODELS_DIR.glob("churn_model_*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return records


@st.cache_data(ttl=300)
def score_churn(df: pd.DataFrame, _churn_artifact, churn_version: str) -> pd.Series:
    if _churn_artifact is None:
        return pd.Series([None] * len(df), index=df.index)
    X = df[CHURN_FEATURES].copy()
    for c in [f for f in CHURN_FEATURES if df[f].dtype == bool]:
        X[c] = X[c].astype(float)
    pipeline = _churn_artifact["pipeline"]
    return pd.Series(pipeline.predict_proba(X)[:, 1], index=df.index)


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


def top_risk_factors(row: pd.Series, importances: dict, top_k: int = 3) -> str:
    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    parts = []
    for feat, _ in ranked:
        if feat in row and pd.notna(row[feat]):
            val = row[feat]
            if isinstance(val, float):
                val = round(val, 2)
            label = feat.replace("_", " ")
            parts.append(f"{label}: {val}")
        if len(parts) >= top_k:
            break
    return " · ".join(parts)


@st.cache_resource
def get_shap_explainer(_pipeline):
    import shap
    return shap.TreeExplainer(_pipeline.named_steps["model"])


def _shap_feature_name_to_column(name: str) -> str:
    if name.startswith("num__"):
        return name[len("num__"):]
    if name.startswith("cat__"):
        rest = name[len("cat__"):]
        return next((c for c in CHURN_CATEGORICAL if rest.startswith(c + "_")), rest)
    return name


def compute_shap_risk_factors(pipeline, customers_df: pd.DataFrame, top_k: int = 3) -> list[str]:
    """Per-customer SHAP explanations for a *bounded* subset of rows (the
    displayed at-risk table, capped at max_rows - never the full 300K+ row
    dataset, which would be far too expensive to compute and hold in
    memory on this machine). Unlike the global feature-importance ranking
    used elsewhere, this shows what actually drove *this specific*
    customer's prediction, with direction (pushing risk up vs down)."""
    X = customers_df[CHURN_FEATURES].copy()
    for c in [f for f in CHURN_FEATURES if customers_df[f].dtype == bool]:
        X[c] = X[c].astype(float)

    preproc = pipeline.named_steps["preprocess"]
    X_transformed = preproc.transform(X)
    X_transformed = X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed
    feature_names = preproc.get_feature_names_out()

    explainer = get_shap_explainer(pipeline)
    shap_values = explainer.shap_values(X_transformed)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    results = []
    for i in range(len(customers_df)):
        contributions = sv[i]
        ranked_idx = np.argsort(np.abs(contributions))[::-1][:top_k]
        parts = []
        for idx in ranked_idx:
            col = _shap_feature_name_to_column(feature_names[idx]).replace("_", " ")
            sign = "▲" if contributions[idx] > 0 else "▼"
            parts.append(f"{sign} {col}")
        results.append(" · ".join(parts))
    return results


def section_header(eyebrow: str, title: str):
    st.markdown(f'<span class="rc-eyebrow">{eyebrow}</span>', unsafe_allow_html=True)
    st.subheader(title)


def apply_chart_theme(fig, height: int = 320):
    fig.update_layout(
        colorway=_CHART_COLORWAY,
        font_family="Public Sans, sans-serif",
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font_family="Sora, sans-serif",
    )
    return fig


def main():
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Retention Command Center")
        st.caption("Telecom Customer Churn & Recommendation Platform")
        st.divider()
        st.markdown("**Filters**")
        threshold = st.slider("At-risk threshold (churn probability)", 0.0, 1.0, 0.5, 0.05)
        max_rows = st.slider("Max at-risk customers to display", 10, 500, 100, 10)
        st.divider()
        st.caption(f"Refreshed {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
        st.caption("Data cached 5 min - reruns pick up new batches/retrains automatically.")

    st.title("Retention Command Center")
    st.markdown(
        '<div class="rc-disclosure">📊 Built on a <b>synthetic</b> ~1M-row Kaggle dataset '
        '(isandeep06/customer-churn-prediction-dataset-1m). Every number on this page reflects '
        'patterns in generated data, not a real telecom market.</div>',
        unsafe_allow_html=True,
    )

    df = load_customer_360()
    churn_artifact, churn_version = load_latest_artifact("churn_model_*.joblib")
    # Recommender is NOT loaded here: its profile matrix is ~100MB+ resident
    # once loaded, and only the At-Risk Customers view actually calls
    # recommend_for_customer(). Loading it unconditionally meant every view
    # - including ones that never touch it - paid that memory cost on the
    # first script run. Loaded lazily just below, only when that view is
    # active (still @st.cache_resource, so it's still loaded at most once
    # per process lifetime, just not before it's needed).
    rec_artifact, rec_version = None, None

    if churn_artifact is None:
        st.warning("No churn model artifact found in models_store/. Run `python -m src.model.train_churn` first.")
        return

    # .assign(), not .copy() + column assignment: avoids duplicating all 40
    # existing columns' worth of data (300K+ rows) just to add one column.
    df = df.assign(churn_probability=score_churn(df, churn_artifact, churn_version))
    importances = column_importances(churn_artifact["pipeline"])

    at_risk = df[df["churn_probability"] >= threshold].sort_values("churn_probability", ascending=False)
    revenue_at_risk = at_risk["monthlycharges"].sum()
    overall_churn_rate = df["churn"].mean()
    metadata_history = load_all_churn_metadata()
    latest_meta = metadata_history[-1] if metadata_history else None
    latest_auc = latest_meta["roc_auc"] if latest_meta else None

    # ---------------------------------------------------------------- KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total customers", f"{len(df):,}")
    k2.metric("Historical churn rate", f"{overall_churn_rate:.1%}")
    k3.metric("At-risk now", f"{len(at_risk):,}", help=f"churn_probability ≥ {threshold:.2f}")
    k4.metric("Monthly revenue at risk", f"${revenue_at_risk:,.0f}", help="Sum of monthlycharges for at-risk customers")
    k5.metric("Model AUC", f"{latest_auc:.3f}" if latest_auc else "n/a", help=f"Model version {churn_version}")

    # st.radio (styled as a tab bar via CSS), not st.tabs: Streamlit's
    # st.tabs executes every tab's body on every rerun regardless of which
    # one is visible - on a 300K+ row dataset, computing all 5 sections'
    # charts/aggregations simultaneously pushed memory usage into repeated
    # OOM kills. A radio-based selector only runs the branch that's
    # actually being viewed, matching what the user asked to see.
    view = st.radio(
        "View", ["Overview", "At-Risk Customers", "Segments", "Model Performance", "Pipeline Status"],
        horizontal=True, label_visibility="collapsed",
    )
    st.markdown("&nbsp;")

    # ------------------------------------------------------------ Overview
    if view == "Overview":
        section_header("PATTERNS", "Where churn concentrates")
        c1, c2, c3 = st.columns(3)
        with c1:
            by_contract = df.groupby("contract", observed=True)["churn"].mean().reset_index()
            by_contract.columns = ["contract", "churn_rate"]
            fig = px.bar(by_contract, x="contract", y="churn_rate", title="Churn rate by contract type")
            st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
        with c2:
            by_tenure = df.groupby("tenure_bucket", observed=True)["churn"].mean().reset_index()
            by_tenure.columns = ["tenure_bucket", "churn_rate"]
            order = ["new_0_6mo", "established_6_24mo", "loyal_24mo_plus"]
            by_tenure["tenure_bucket"] = pd.Categorical(by_tenure["tenure_bucket"], categories=order, ordered=True)
            by_tenure = by_tenure.sort_values("tenure_bucket")
            fig = px.bar(by_tenure, x="tenure_bucket", y="churn_rate", title="Churn rate by tenure")
            st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
        with c3:
            by_bundle = df.groupby("total_active_services")["churn"].mean().reset_index()
            by_bundle.columns = ["total_active_services", "churn_rate"]
            fig = px.bar(by_bundle, x="total_active_services", y="churn_rate", title="Churn rate by service bundle size")
            st.plotly_chart(apply_chart_theme(fig), use_container_width=True)

        section_header("EXPOSURE", "Revenue at risk by segment")
        rev_by_contract = at_risk.groupby("contract", observed=True)["monthlycharges"].sum().reset_index()
        rev_by_contract.columns = ["contract", "revenue_at_risk"]
        fig = px.bar(
            rev_by_contract.sort_values("revenue_at_risk", ascending=True),
            x="revenue_at_risk", y="contract", orientation="h",
            title="Monthly revenue at risk by contract type", text_auto=".2s",
        )
        st.plotly_chart(apply_chart_theme(fig, height=260), use_container_width=True)

    # ------------------------------------------------------- At-risk table
    elif view == "At-Risk Customers":
        rec_artifact, _ = load_latest_artifact("recommender_*.joblib")
        section_header("ACTION LIST", f"At-risk customers (probability ≥ {threshold:.2f})")
        st.caption(
            f"{len(at_risk):,} of {len(df):,} customers meet this threshold — showing top {min(max_rows, len(at_risk)):,}, ranked by risk. "
            "Risk factors are per-customer SHAP explanations (▲ pushes risk up, ▼ pushes risk down) — what actually drove *this* prediction, not just a global average."
        )

        display_subset = at_risk.head(max_rows)
        try:
            with st.spinner("Computing per-customer SHAP explanations..."):
                shap_labels = compute_shap_risk_factors(churn_artifact["pipeline"], display_subset)
        except Exception:
            # Falls back to the global-importance heuristic if SHAP fails
            # for any reason (e.g. a future model type TreeExplainer
            # doesn't support) - degrades gracefully, doesn't break the view.
            shap_labels = [top_risk_factors(row, importances) if importances else "" for _, row in display_subset.iterrows()]

        display_rows = []
        for (_, row), risk_factors in zip(display_subset.iterrows(), shap_labels):
            action = "-"
            if rec_artifact is not None:
                try:
                    recs = recommend_for_customer(rec_artifact, row["customer_id"], top_n=1)
                    if recs:
                        action = f"Offer: {recs[0]['service'].replace('has_', '').replace('_', ' ').title()}"
                except KeyError:
                    action = "n/a (not in recommender training set)"
            display_rows.append({
                "customer_id": row["customer_id"],
                "churn_probability": row["churn_probability"],
                "contract": row["contract"],
                "tenure_months": row["tenure"],
                "monthly_charges": row["monthlycharges"],
                "key_risk_factors": risk_factors,
                "recommended_action": action,
            })

        display_df = pd.DataFrame(display_rows)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "churn_probability": st.column_config.ProgressColumn(
                    "Churn probability", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "monthly_charges": st.column_config.NumberColumn("Monthly charges", format="$%.2f"),
                "tenure_months": st.column_config.NumberColumn("Tenure (months)"),
                "key_risk_factors": st.column_config.TextColumn("Risk factors (SHAP)", width="medium"),
            },
        )

        st.download_button(
            "⬇ Export at-risk list (CSV)",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name=f"at_risk_customers_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    # ------------------------------------------------------------ Segments
    elif view == "Segments":
        section_header("DEMOGRAPHICS", "Churn rate by customer segment")
        seg_cols = [
            ("education", "Education"), ("marital_status", "Marital status"),
            ("payment_method", "Payment method"), ("gender", "Gender"),
        ]
        rows_of_cols = [st.columns(2), st.columns(2)]
        for (col_name, label), container in zip(seg_cols, [c for row in rows_of_cols for c in row]):
            with container:
                by_seg = df.groupby(col_name, observed=True)["churn"].mean().reset_index()
                by_seg.columns = [col_name, "churn_rate"]
                fig = px.bar(
                    by_seg.sort_values("churn_rate", ascending=False),
                    x=col_name, y="churn_rate", title=f"Churn rate by {label.lower()}",
                )
                st.plotly_chart(apply_chart_theme(fig, height=280), use_container_width=True)

        section_header("RISK DRIVERS", "What the model weighs most")
        if importances:
            imp_df = pd.DataFrame(sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:12],
                                   columns=["feature", "importance"])
            fig = px.bar(imp_df.sort_values("importance"), x="importance", y="feature", orientation="h",
                         title="Top 12 churn model feature importances")
            st.plotly_chart(apply_chart_theme(fig, height=420), use_container_width=True)

    # ------------------------------------------------------- Model perf
    elif view == "Model Performance":
        section_header("TRANSPARENCY", "Current production model")
        if latest_meta:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision (churn)", f"{latest_meta['precision_churn']:.2f}")
            m2.metric("Recall (churn)", f"{latest_meta['recall_churn']:.2f}")
            m3.metric("F1 (churn)", f"{latest_meta['f1_churn']:.2f}")
            m4.metric("Decision threshold", f"{latest_meta.get('threshold', 0.5):.3f}")

            st.caption(
                f"Model: {latest_meta.get('model_type', 'n/a')} · version {latest_meta['version']} · "
                f"trained on {latest_meta['n_rows_total']:,} rows · "
                f"class balance {latest_meta['class_pct'].get('1', 0):.1f}% churned"
            )
            if latest_meta.get("threshold_rationale"):
                st.caption(f"Threshold rationale: {latest_meta['threshold_rationale']}")

            cm = latest_meta.get("confusion_matrix")
            if cm:
                fig = go.Figure(data=go.Heatmap(
                    z=cm, x=["Predicted retained", "Predicted churn"], y=["Actually retained", "Actually churned"],
                    colorscale=[[0, "#faf3e2"], [1, "#a6790a"]], text=cm, texttemplate="%{text:,}",
                    showscale=False,
                ))
                fig.update_layout(title="Confusion matrix (test set)")
                st.plotly_chart(apply_chart_theme(fig, height=340), use_container_width=True)
        else:
            st.info("No model metadata found yet.")

        if len(metadata_history) > 1:
            section_header("TREND", "Model performance across retrains")
            trend_df = pd.DataFrame([
                {"version": m["version"], "roc_auc": m["roc_auc"], "f1_churn": m["f1_churn"],
                 "n_rows_total": m["n_rows_total"]}
                for m in metadata_history
            ])
            fig = px.line(trend_df, x="version", y=["roc_auc", "f1_churn"], markers=True,
                          title="AUC / F1 across model versions")
            st.plotly_chart(apply_chart_theme(fig, height=300), use_container_width=True)

    # ----------------------------------------------------------- Pipeline
    elif view == "Pipeline Status":
        section_header("OPERATIONS", "Batch ingestion & retraining status")
        log_df = load_ingestion_log()
        batches_done = len(log_df)

        p1, p2, p3 = st.columns(3)
        p1.metric("Batches ingested", f"{batches_done} / {TOTAL_SIMULATED_BATCHES}")
        batches_to_next_retrain = RETRAIN_EVERY_N_BATCHES - (batches_done % RETRAIN_EVERY_N_BATCHES or RETRAIN_EVERY_N_BATCHES)
        p2.metric("Batches to next retrain", batches_to_next_retrain if batches_to_next_retrain else 0)
        p3.metric("Model versions trained", len(metadata_history))

        st.progress(min(batches_done / TOTAL_SIMULATED_BATCHES, 1.0))

        pipeline_health = "good" if batches_done > 0 else "warn"
        st.markdown(
            f'<span class="rc-pill rc-pill-{pipeline_health}">'
            f'{"● PIPELINE ACTIVE" if pipeline_health == "good" else "○ NO BATCHES YET"}</span>',
            unsafe_allow_html=True,
        )

        st.markdown("&nbsp;")
        st.dataframe(
            log_df, use_container_width=True, hide_index=True,
            column_config={
                "rows_loaded": st.column_config.NumberColumn("Rows loaded", format="%d"),
                "loaded_at": st.column_config.DatetimeColumn("Loaded at", format="YYYY-MM-DD HH:mm"),
            },
        )


if __name__ == "__main__":
    main()
