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
import logging
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

# Must be the literal first Streamlit command in the script - Streamlit
# raises StreamlitSetPageConfigMustBeFirstCommandError otherwise. Moved
# here specifically because the st.secrets access just below also counts
# as a Streamlit command and was tripping that check when set_page_config
# came after it (caught by actually running this under `streamlit run`
# against the trimmed Streamlit Cloud requirements - a plain `import`
# doesn't reach this, so it didn't surface until then).
st.set_page_config(page_title="Retention Command Center", layout="wide", initial_sidebar_state="expanded")

load_dotenv()

# Streamlit Community Cloud: secrets are set via its own UI (a TOML blob,
# not a .env file) and land in st.secrets, not necessarily os.environ.
# Every config read in this file uses os.environ.get(...) - bridging
# st.secrets into os.environ here, once, at import time, means the exact
# same code runs unchanged locally (.env / docker-compose env vars) and on
# Streamlit Cloud, rather than needing two config-reading code paths.
#
# Gated on the file actually existing, checked with a plain path check
# rather than by just trying st.secrets and catching the failure: merely
# *accessing* st.secrets makes Streamlit render its own "No secrets found"
# warning banner at the top of the page, regardless of whether the access
# is wrapped in try/except - caught by actually running this under
# `streamlit run` locally (a plain import doesn't trigger Streamlit's UI
# layer, so it didn't surface until then). Every local and Docker run has
# no secrets.toml, so without this gate that banner would appear on every
# screenshot and every real user's session, not just this test.
_secrets_paths = [
    Path.home() / ".streamlit" / "secrets.toml",
    Path(__file__).resolve().parents[2] / ".streamlit" / "secrets.toml",
]
if any(p.exists() for p in _secrets_paths):
    try:
        for _key, _value in st.secrets.items():
            os.environ.setdefault(_key, str(_value))
    except Exception:
        pass

from src.warehouse import stream_query  # noqa: E402
from src.model.train_churn import ALL_FEATURES as CHURN_FEATURES  # noqa: E402
from src.model.train_churn import CATEGORICAL_FEATURES as CHURN_CATEGORICAL  # noqa: E402
from src.model.train_recommender import recommend_for_profile  # noqa: E402
from src.model.train_recommender import SERVICE_COLUMNS as REC_SERVICE_COLUMNS  # noqa: E402
from src.model.train_recommender import service_display_name  # noqa: E402
from src.model.explain_churn import compute_shap_details, format_risk_factors_text  # noqa: E402
from src.agents.groq_client import AgentCallFailed  # noqa: E402
from src.agents.explanation_agent import explain_churn as ai_explain_churn  # noqa: E402
from src.agents.outreach_agent import draft_outreach as ai_draft_outreach  # noqa: E402
from src.agents.cache import get_or_generate  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[2] / "models_store"
RETRAIN_EVERY_N_BATCHES = int(os.environ.get("RETRAIN_EVERY_N_BATCHES", "3"))
TOTAL_SIMULATED_BATCHES = 13

# AI Agent Layer (src/agents/): capped well below max_rows' up-to-500
# customers. Each customer the first time costs a real Groq call (or two,
# with outreach), and generation is already opt-in via a button rather
# than automatic - this cap keeps even an enthusiastic click bounded and
# fast, not a 500-call burst against a free-tier rate limit.
AI_AGENT_MAX_CUSTOMERS = 15
GROQ_CONFIGURED = bool(os.environ.get("GROQ_API_KEY"))


def log_agent_failure(agent_type: str, customer_id: str, exc: Exception) -> None:
    logging.getLogger("dashboard.agents").warning(
        "%s agent failed for %s, falling back to raw data: %s", agent_type, customer_id, exc
    )

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
        # See src/warehouse.py's get_pg_conn for why "prefer": lets this
        # same code reach both the local Docker Postgres (no SSL) and a
        # hosted provider like Neon (SSL required).
        sslmode=os.environ.get("POSTGRES_SSLMODE", "prefer"),
    )


@st.cache_data(ttl=600, show_spinner="Scoring customers...")
def score_all_customers(_churn_artifact, churn_version: str, batch_size: int = 25_000) -> pd.DataFrame:
    """One streaming pass over customer_360, scoring every customer and
    keeping only a compact per-customer result.

    The dashboard used to load the entire mart into a DataFrame and score
    it in place. That is fine at 300K rows and fatal at 1M: the frame alone
    is ~620MB, and the preprocessor materializes a dense (n x 54) float64
    matrix on top of it (412MB at 1M). Inside a 1GB container that is an
    OOM kill.

    Instead, rows are streamed from Postgres in batches, scored, and
    reduced immediately to three columns. Batches are kept small (25K) on
    purpose: each fetchmany materializes batch_rows x n_columns individual
    Python objects before pandas builds columnar arrays, so the batch size
    sets the transient peak far more than the retained result does. The retained result is ~40MB at
    1M customers regardless of how wide the mart gets, and peak memory is
    bounded by one batch rather than by the table size. Full rows for the
    handful of customers actually displayed are fetched by id later.
    """
    score_cols = ["customer_id"] + CHURN_FEATURES
    score_cols = list(dict.fromkeys(score_cols))
    pipeline = _churn_artifact["pipeline"]

    ids, scores, charges = [], [], []

    def _score_batch(batch: pd.DataFrame) -> pd.DataFrame:
        X = batch[CHURN_FEATURES].copy()
        for c in CHURN_FEATURES:
            if X[c].dtype == bool:
                X[c] = X[c].astype(float)
        ids.append(batch["customer_id"].to_numpy())
        scores.append(pipeline.predict_proba(X)[:, 1].astype(np.float32))
        charges.append(batch["monthlycharges"].to_numpy(dtype=np.float32))
        # Return an empty frame: stream_query concatenates whatever comes
        # back, and we deliberately keep none of the raw rows.
        return batch.iloc[0:0]

    stream_query(
        f"SELECT {', '.join(score_cols)} FROM marts.customer_360",
        columns=score_cols,
        batch_rows=batch_size,
        transform=_score_batch,
    )

    return pd.DataFrame({
        "customer_id": np.concatenate(ids),
        "churn_probability": np.concatenate(scores),
        "monthlycharges": np.concatenate(charges),
    })


@st.cache_data(ttl=600)
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


@st.cache_data(ttl=600)
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


@st.cache_data(ttl=600)
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


@st.cache_data(ttl=300)
def load_ingestion_log() -> pd.DataFrame:
    conn = get_pg_conn()
    try:
        return pd.read_sql(
            "SELECT batch_file, rows_loaded, loaded_at, status FROM ingestion_log ORDER BY loaded_at", conn
        )
    finally:
        conn.close()


@st.cache_data(ttl=300)
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


def compact_currency(value: float) -> str:
    """Formats a money amount to fit a KPI card ($26.1M, $845.2K, $312)."""
    for cutoff, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if abs(value) >= cutoff:
            return f"${value / cutoff:,.1f}{suffix}"
    return f"${value:,.0f}"


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


def compute_shap_risk_factors(pipeline, customers_df: pd.DataFrame, top_k: int = 3) -> list[str]:
    """Per-customer SHAP explanations for a *bounded* subset of rows (the
    displayed at-risk table, capped at max_rows - never the full 300K+ row
    dataset, which would be far too expensive to compute and hold in
    memory on this machine). Unlike the global feature-importance ranking
    used elsewhere, this shows what actually drove *this specific*
    customer's prediction, with direction (pushing risk up vs down).

    Delegates to src/model/explain_churn.py, which the FastAPI
    /explain-churn endpoint also calls - one SHAP implementation shared
    by both surfaces, rather than two that could drift apart."""
    from src.model.explain_churn import compute_shap_details, format_risk_factors_text

    details_per_row = compute_shap_details(pipeline, customers_df, top_k=top_k)
    return [format_risk_factors_text(details) for details in details_per_row]


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

    # Loaded before the sidebar because the at-risk slider defaults to this
    # model's own decision threshold (@st.cache_resource, so this costs
    # nothing after the first run).
    churn_artifact, churn_version = load_latest_artifact("churn_model_*.joblib")

    with st.sidebar:
        st.markdown("### Retention Command Center")
        st.caption("Telecom Customer Churn & Recommendation Platform")
        st.divider()
        st.markdown("**Filters**")
        # Default to the model's own trained threshold, not a hardcoded 0.5.
        # The model now emits *calibrated* probabilities, so its operating
        # point sits near the ~10% base rate (~0.11), not near 0.5 - leaving
        # the old 0.5 default here would have shown an almost-empty at-risk
        # list and made the model look broken. Step is fine-grained for the
        # same reason: at this scale 0.05 steps are far too coarse.
        model_threshold = float(churn_artifact.get("threshold", 0.5)) if churn_artifact else 0.5
        threshold = st.slider(
            "At-risk threshold (churn probability)",
            0.0, 1.0, round(model_threshold, 3), 0.005,
            help=(
                f"Defaults to the deployed model's decision threshold ({model_threshold:.3f}), "
                "chosen at training time to guarantee recall >= 0.60. Scores are calibrated, "
                "so this is a real probability."
            ),
        )
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

    # Compact per-customer scores only (id, probability, monthly spend) -
    # never the full mart. See score_all_customers for why.
    scored = score_all_customers(churn_artifact, churn_version)
    # Explainability tooling (feature importances, SHAP) needs the raw
    # sklearn Pipeline, not the CalibratedClassifierCV wrapper stored under
    # "pipeline" for serving - the wrapper has no .named_steps. Artifacts
    # trained before calibration was added have no "base_pipeline" key, in
    # which case "pipeline" already IS the raw Pipeline, so this falls back
    # to it correctly either way.
    explain_pipeline = churn_artifact.get("base_pipeline", churn_artifact["pipeline"])
    importances = column_importances(explain_pipeline)

    stats = load_overall_stats()
    at_risk_mask = scored["churn_probability"] >= threshold
    n_at_risk = int(at_risk_mask.sum())
    revenue_at_risk = float(scored.loc[at_risk_mask, "monthlycharges"].sum())
    overall_churn_rate = stats["churn_rate"]
    total_customers = stats["total_customers"]
    metadata_history = load_all_churn_metadata()
    latest_meta = metadata_history[-1] if metadata_history else None
    latest_auc = latest_meta["roc_auc"] if latest_meta else None

    # ---------------------------------------------------------------- KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total customers", f"{total_customers:,}")
    k2.metric("Historical churn rate", f"{overall_churn_rate:.1%}")
    k3.metric("At-risk now", f"{n_at_risk:,}", help=f"churn_probability ≥ {threshold:.2f}")
    k4.metric(
        "Monthly revenue at risk",
        compact_currency(revenue_at_risk),
        # Abbreviated, not full digits: st.metric renders one line inside a
        # narrow KPI card and silently truncates ("$26,051...") once the
        # number passes ~8 characters - at 1M customers it always does. The
        # exact figure stays available on hover.
        help=f"Sum of monthlycharges for at-risk customers (exact: ${revenue_at_risk:,.0f})",
    )
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
            by_contract = load_segment_rates("contract")
            fig = px.bar(by_contract, x="contract", y="churn_rate", title="Churn rate by contract type")
            st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
        with c2:
            by_tenure = load_segment_rates("tenure_bucket")
            order = ["new_0_6mo", "established_6_24mo", "loyal_24mo_plus"]
            by_tenure["tenure_bucket"] = pd.Categorical(by_tenure["tenure_bucket"], categories=order, ordered=True)
            by_tenure = by_tenure.sort_values("tenure_bucket")
            fig = px.bar(by_tenure, x="tenure_bucket", y="churn_rate", title="Churn rate by tenure")
            st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
        with c3:
            by_bundle = load_segment_rates("total_active_services")
            by_bundle["total_active_services"] = by_bundle["total_active_services"].astype(int)
            by_bundle = by_bundle.sort_values("total_active_services")
            fig = px.bar(by_bundle, x="total_active_services", y="churn_rate", title="Churn rate by service bundle size")
            st.plotly_chart(apply_chart_theme(fig), use_container_width=True)

        section_header("EXPOSURE", "Revenue at risk by segment")
        # Contract for at-risk customers comes from a targeted fetch of the
        # top slice rather than a full-table join in pandas.
        top_ids = tuple(scored.loc[at_risk_mask].nlargest(max_rows, "churn_probability")["customer_id"])
        top_rows = load_customers_by_id(top_ids)
        rev_by_contract = (
            top_rows.groupby("contract", observed=True)["monthlycharges"].sum().reset_index()
            if len(top_rows) else pd.DataFrame({"contract": [], "monthlycharges": []})
        )
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
            f"{n_at_risk:,} of {total_customers:,} customers meet this threshold — showing top {min(max_rows, n_at_risk):,}, ranked by risk. "
            "Risk factors are per-customer SHAP explanations (▲ pushes risk up, ▼ pushes risk down) — what actually drove *this* prediction, not just a global average."
        )

        # Only the displayed customers are pulled in full - the streaming
        # score pass kept just ids/probabilities/spend for everyone else.
        top_ids = tuple(scored.loc[at_risk_mask].nlargest(max_rows, "churn_probability")["customer_id"])
        display_subset = load_customers_by_id(top_ids)
        if len(display_subset):
            display_subset = display_subset.merge(
                scored[["customer_id", "churn_probability"]], on="customer_id", how="left"
            ).sort_values("churn_probability", ascending=False)
        try:
            with st.spinner("Computing per-customer SHAP explanations..."):
                # See explain_pipeline note above: TreeExplainer needs the
                # raw LightGBM pipeline, not the CalibratedClassifierCV
                # wrapper used for serving. Structured details (not just
                # formatted text) computed once here so the AI Agent Layer
                # section below can reuse them as grounding input, instead
                # of a second, redundant SHAP pass.
                shap_details_per_row = compute_shap_details(explain_pipeline, display_subset, top_k=3)
            shap_labels = [format_risk_factors_text(d) for d in shap_details_per_row]
        except Exception:
            # Falls back to the global-importance heuristic if SHAP fails
            # for any reason (e.g. a future model type TreeExplainer
            # doesn't support) - degrades gracefully, doesn't break the view.
            shap_labels = [top_risk_factors(row, importances) if importances else "" for _, row in display_subset.iterrows()]
            shap_details_per_row = [[] for _ in range(len(display_subset))]

        display_rows = []
        recommended_services_raw = []  # parallel to display_rows: raw "has_x" column name or None
        for (_, row), risk_factors in zip(display_subset.iterrows(), shap_labels):
            action = "-"
            service_raw = None
            if rec_artifact is not None:
                try:
                    # Profile-based, not id-based: the k-NN index is a
                    # reference set covering ~154K customers, while
                    # customer_360 holds 1M. Looking up by id left 84.6% of
                    # customers with no recommendation at all. Transforming
                    # this customer's profile and matching it against the
                    # index serves everyone, at no extra memory cost.
                    own = [int(row[s]) for s in REC_SERVICE_COLUMNS]
                    recs = recommend_for_profile(
                        rec_artifact, row.to_frame().T, own, top_n=1
                    )
                    if recs:
                        service_raw = recs[0]["service"]
                        action = f"Offer: {service_display_name(service_raw)}"
                except Exception:
                    action = "-"
            recommended_services_raw.append(service_raw)
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
                "key_risk_factors": st.column_config.TextColumn("Risk factors (SHAP)", width="large"),
            },
        )

        st.download_button(
            "⬇ Export at-risk list (CSV)",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name=f"at_risk_customers_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

        # --------------------------------------------------- AI Agent Layer
        st.markdown("&nbsp;")
        section_header("AI AGENT LAYER", "Plain-English explanations & drafted outreach")
        n_ai = min(AI_AGENT_MAX_CUSTOMERS, len(display_subset))
        st.caption(
            f"Explains and drafts over the SHAP values and recommendation already computed above - it "
            f"never changes the churn probability or which service is recommended, only narrates them "
            f"(see the README's AI Agent Layer section). Capped to the top {n_ai} customers by risk and "
            f"generated only on request, since each customer costs a real LLM call the first time; results "
            f"are cached per customer per model version, so re-opening this view or re-running the pipeline "
            f"without a retrain reuses the cached text instantly rather than re-calling Groq."
        )

        ai_cache_key = f"ai_layer_{churn_version}"
        if st.button(f"Generate AI explanations & outreach drafts for the top {n_ai} customers", key="ai_gen_button"):
            st.session_state[ai_cache_key] = True
        if not GROQ_CONFIGURED:
            st.info(
                "GROQ_API_KEY is not set, so this section would fall back to the raw SHAP/recommendation "
                "data shown above rather than AI prose. Set it in .env to see AI-generated output "
                "(see the README's AI Agent Layer section)."
            )
        elif st.session_state.get(ai_cache_key):
            ai_rows = list(zip(
                display_subset.head(n_ai).itertuples(index=False),
                shap_details_per_row[:n_ai],
                recommended_services_raw[:n_ai],
            ))
            progress = st.progress(0.0, text="Generating AI explanations & outreach drafts...")
            for i, (row, shap_details, service_raw) in enumerate(ai_rows):
                customer_id = row.customer_id
                churn_probability = float(row.churn_probability)
                fallback_explanation = format_risk_factors_text(shap_details) if shap_details else "No risk factors available."

                try:
                    explanation = get_or_generate(
                        customer_id, "explanation", churn_version,
                        lambda cp=churn_probability, sd=shap_details: ai_explain_churn(cp, sd),
                    )
                    explanation_source = "llm"
                except AgentCallFailed as exc:
                    explanation = fallback_explanation
                    explanation_source = "fallback"
                    log_agent_failure("explanation", customer_id, exc)

                outreach = None
                outreach_source = None
                if service_raw:
                    service_name = service_display_name(service_raw)
                    try:
                        outreach = get_or_generate(
                            customer_id, "outreach", churn_version,
                            lambda e=explanation, s=service_name: ai_draft_outreach(e, s),
                        )
                        outreach_source = "llm"
                    except AgentCallFailed as exc:
                        outreach = f"(AI outreach draft unavailable - recommended offer: {service_name})"
                        outreach_source = "fallback"
                        log_agent_failure("outreach", customer_id, exc)

                with st.expander(f"{customer_id} — {churn_probability:.0%} churn risk"):
                    badge = "🤖 AI-generated" if explanation_source == "llm" else "⚠️ fallback (raw SHAP)"
                    st.markdown(f"**Why they're at risk** _{badge}_")
                    st.write(explanation)
                    if outreach is not None:
                        badge2 = "🤖 AI-generated" if outreach_source == "llm" else "⚠️ fallback"
                        st.markdown(f"**Drafted outreach message** _{badge2}_")
                        st.text(outreach)
                    else:
                        st.caption("No service recommendation available for this customer - no outreach drafted.")

                progress.progress((i + 1) / len(ai_rows), text=f"{i + 1}/{len(ai_rows)} customers processed")
            progress.empty()

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
                by_seg = load_segment_rates(col_name)
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

        # ------------------------------------------------------ drift
        st.markdown("&nbsp;")
        section_header("MONITORING", "Feature drift (PSI)")

        drift_df = load_latest_drift()
        if drift_df.empty:
            st.info(
                "No drift scores recorded yet. They are written by the pipeline's "
                "`detect_feature_drift` task, or on demand with "
                "`python -m src.monitoring.drift <batch_file>`."
            )
        else:
            current_batch = drift_df["current_batch"].iloc[0]
            reference_batch = drift_df["reference_batch"].iloc[0]
            max_psi = float(drift_df["psi"].max())
            n_significant = int((drift_df["severity"] == "significant").sum())
            n_moderate = int((drift_df["severity"] == "moderate").sum())

            d1, d2, d3 = st.columns(3)
            d1.metric(
                "Highest PSI", f"{max_psi:.4f}",
                help="Population Stability Index: <0.10 stable, 0.10-0.25 moderate, >=0.25 significant",
            )
            d2.metric("Features drifted", f"{n_significant}", help="PSI >= 0.25 - triggers a retrain")
            d3.metric("Features to watch", f"{n_moderate}", help="PSI between 0.10 and 0.25")

            if n_significant:
                st.error(
                    f"{n_significant} feature(s) have drifted significantly in **{current_batch}** "
                    f"relative to **{reference_batch}** — the pipeline retrains on this signal."
                )
            else:
                st.success(
                    f"All {len(drift_df)} monitored features stable in **{current_batch}** "
                    f"relative to the **{reference_batch}** baseline (highest PSI {max_psi:.4f})."
                )

            st.caption(
                "Every batch here is a slice of one pre-shuffled source file, so near-zero PSI is the "
                "expected — and correct — result. The detector's ability to fire on genuinely shifted "
                "data is covered by `tests/test_drift.py` rather than by this screen."
            )
            st.dataframe(
                drift_df[["feature", "feature_type", "psi", "severity"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "feature": st.column_config.TextColumn("Feature"),
                    "feature_type": st.column_config.TextColumn("Type"),
                    "psi": st.column_config.NumberColumn("PSI", format="%.5f"),
                    "severity": st.column_config.TextColumn("Status"),
                },
            )

        # ------------------------------------------- AI retrain summary
        st.markdown("&nbsp;")
        section_header("AI AGENT LAYER", "Latest retrain summary")
        from src.agents.cache import get_latest_retrain_summary

        latest_summary = get_latest_retrain_summary()
        if latest_summary is None:
            st.info(
                "No retrain has happened yet since this feature was added - written by the DAG's "
                "`summarize_retrain` task, which only runs on the `retrain_churn_model` branch "
                "(nothing to summarize when a batch takes the `skip_retrain` path)."
            )
        else:
            st.caption(
                f"Model {latest_summary['churn_model_version']}"
                + (f" vs. previous {latest_summary['previous_version']}" if latest_summary['previous_version'] else " (first recorded version)")
                + f" · {latest_summary['created_at']:%Y-%m-%d %H:%M UTC}"
            )
            st.write(latest_summary["summary_text"])
            st.caption(
                "Plain-English narrative over the structured metrics already saved by retrain_churn_model "
                "and the drift results from detect_feature_drift - never a second modeling step, only a "
                "narration of numbers computed elsewhere (see the README's AI Agent Layer section)."
            )


if __name__ == "__main__":
    main()
