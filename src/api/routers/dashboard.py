"""Dashboard-facing analytics endpoints the Next.js frontend reads instead
of querying Postgres directly. Warehouse and filesystem reads live in
src/model/dashboard_queries.py, so every caller of those numbers agrees.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from src.agents.cache import get_latest_retrain_summary
from src.ai.schemas import CustomerLookupResult, CustomerSearchFilters, CustomerSearchResult
from src.ai.tools.customer_tool import customer_lookup as ai_customer_lookup
from src.ai.tools.customer_tool import customer_search as ai_customer_search
from src.api import state
from src.api.schemas import (
    AtRiskCustomer,
    AtRiskListResponse,
    FeatureImportance,
    OverviewStatsResponse,
    RevenueAtRiskBucket,
    RevenueAtRiskResponse,
    SegmentBucket,
    SegmentRatesResponse,
)
from src.model import dashboard_queries
from src.model.explain_churn import compute_shap_details
from src.model.train_churn import ALL_FEATURES as CHURN_FEATURES
from src.model.train_recommender import SERVICE_COLUMNS as REC_SERVICE_COLUMNS
from src.model.train_recommender import recommend_for_profile

log = logging.getLogger("api")

router = APIRouter(tags=["dashboard"])

# /at-risk runs SHAP + a recommendation per returned row, so its page size
# is clamped server-side rather than trusting the caller's max_rows.
AT_RISK_MAX_ROWS = 500

_SEGMENT_COLUMNS = ("contract", "tenure_bucket", "education", "marital_status", "payment_method", "gender")


@router.get("/overview/stats", response_model=OverviewStatsResponse)
def overview_stats(threshold: Optional[float] = None):
    artifact = state.require_churn_model()

    stats = dashboard_queries.load_overall_stats()
    effective_threshold = state.churn_threshold(threshold)

    scored = state.get_scored_customers()
    at_risk = scored[scored["churn_probability"] >= effective_threshold]

    # AUC of the model actually serving, not the newest file on disk - the
    # registry can resolve an older champion, and a wrong AUC next to the
    # serving version is worse than none.
    serving_metadata = next(
        (m for m in dashboard_queries.load_all_churn_metadata() if m.get("version") == state.models["churn_version"]),
        None,
    )
    model_auc = serving_metadata.get("roc_auc") if serving_metadata else None

    importances = dashboard_queries.column_importances(state.explain_pipeline(artifact))
    top_importances = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:12]

    return OverviewStatsResponse(
        total_customers=stats["total_customers"],
        historical_churn_rate=round(stats["churn_rate"], 4),
        at_risk_count=int(len(at_risk)),
        revenue_at_risk=round(float(at_risk["monthlycharges"].sum()), 2),
        model_auc=round(model_auc, 4) if model_auc is not None else None,
        model_version=state.models["churn_version"],
        threshold_used=round(effective_threshold, 4),
        top_feature_importances=[
            FeatureImportance(feature=f, importance=round(v, 4)) for f, v in top_importances
        ],
    )


@router.get("/overview/segment-rates", response_model=SegmentRatesResponse)
def overview_segment_rates(column: str):
    """Churn rate by segment; dashboard_queries.load_segment_rates enforces
    the column allowlist."""
    try:
        df = dashboard_queries.load_segment_rates(column)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SegmentRatesResponse(
        column=column,
        buckets=[
            SegmentBucket(key=str(row[column]), churn_rate=round(row["churn_rate"], 4), n_customers=int(row["n_customers"]))
            for _, row in df.iterrows()
        ],
    )


@router.get("/overview/revenue-at-risk-by-segment", response_model=RevenueAtRiskResponse)
def overview_revenue_at_risk_by_segment(threshold: Optional[float] = None, segment_column: str = "contract"):
    """Monthly revenue at risk by segment over the ENTIRE at-risk population,
    so the buckets sum to /overview/stats's revenue_at_risk headline. The
    segment columns ride along on the cached scored frame, so this is an
    in-memory groupby with no per-row warehouse fetch."""
    state.require_churn_model()
    if segment_column not in _SEGMENT_COLUMNS:
        raise HTTPException(status_code=422, detail=f"unexpected segment_column {segment_column!r}")

    scored = state.get_scored_customers()
    at_risk = scored[scored["churn_probability"] >= state.churn_threshold(threshold)]
    if at_risk.empty:
        return RevenueAtRiskResponse(segment_column=segment_column, buckets=[])

    grouped = at_risk.groupby(segment_column, observed=True)["monthlycharges"].sum()
    return RevenueAtRiskResponse(
        segment_column=segment_column,
        buckets=[RevenueAtRiskBucket(segment=str(k), revenue_at_risk=round(float(v), 2)) for k, v in grouped.items()],
    )


@router.get("/at-risk", response_model=AtRiskListResponse)
def at_risk(threshold: Optional[float] = None, max_rows: int = 100, offset: int = 0):
    """Customers above threshold, highest churn probability first, each with
    per-customer SHAP risk factors and a recommended retention action.
    max_rows_used in the response is the real (clamped) page size."""
    artifact = state.require_churn_model()

    effective_threshold = state.churn_threshold(threshold)
    max_rows = max(1, min(max_rows, AT_RISK_MAX_ROWS))
    offset = max(0, offset)

    scored = state.get_scored_customers()
    at_risk_all = scored[scored["churn_probability"] >= effective_threshold].sort_values(
        "churn_probability", ascending=False
    )
    page = at_risk_all.iloc[offset: offset + max_rows]

    def _response(customers: list[AtRiskCustomer]) -> AtRiskListResponse:
        return AtRiskListResponse(
            customers=customers, total_at_risk=int(len(at_risk_all)), threshold=round(effective_threshold, 4),
            max_rows_used=max_rows, offset=offset, model_version=state.models["churn_version"],
        )

    if page.empty:
        return _response([])

    full_rows = dashboard_queries.load_customers_by_id(tuple(page["customer_id"])).set_index("customer_id")
    explain_pipeline = state.explain_pipeline(artifact)
    recommender = state.models["recommender"]

    customers = []
    for _, scored_row in page.iterrows():
        cid = scored_row["customer_id"]
        if cid not in full_rows.index:
            continue
        full_row = full_rows.loc[[cid]]
        try:
            shap_details = compute_shap_details(explain_pipeline, full_row[CHURN_FEATURES], top_k=3)[0]
        except Exception as exc:
            log.warning("SHAP failed for %s, omitting risk factors: %s", cid, exc)
            shap_details = []
        recommended_action = None
        if recommender is not None:
            try:
                own = [int(full_row[s].iloc[0]) for s in REC_SERVICE_COLUMNS]
                recs = recommend_for_profile(recommender, full_row, own, top_n=1)
                if recs:
                    recommended_action = recs[0]["service"]
            except Exception as exc:
                log.warning("recommendation failed for %s, omitting: %s", cid, exc)

        customers.append(AtRiskCustomer(
            customer_id=cid,
            churn_probability=round(float(scored_row["churn_probability"]), 4),
            contract=full_row["contract"].iloc[0] if "contract" in full_row else None,
            tenure=int(full_row["tenure"].iloc[0]) if "tenure" in full_row else None,
            monthlycharges=float(full_row["monthlycharges"].iloc[0]) if "monthlycharges" in full_row else None,
            key_risk_factors=shap_details,
            recommended_action=recommended_action,
        ))

    return _response(customers)


@router.get("/customers", response_model=CustomerSearchResult)
def search_customers(
    limit: int = 25,
    offset: int = 0,
    min_churn_probability: Optional[float] = None,
    max_churn_probability: Optional[float] = None,
    min_monthly_charges: Optional[float] = None,
    max_monthly_charges: Optional[float] = None,
    min_satisfaction: Optional[float] = None,
    max_satisfaction: Optional[float] = None,
    min_complaints: Optional[float] = None,
    contract: Optional[str] = None,
    tenure_bucket: Optional[str] = None,
):
    """Thin wrapper over src/ai/tools/customer_tool.py's customer_search,
    which is already paginated, filtered and bounded."""
    filters = CustomerSearchFilters(
        min_churn_probability=min_churn_probability,
        max_churn_probability=max_churn_probability,
        min_monthly_charges=min_monthly_charges,
        max_monthly_charges=max_monthly_charges,
        min_satisfaction=min_satisfaction,
        max_satisfaction=max_satisfaction,
        min_complaints=min_complaints,
        contract=contract,
        tenure_bucket=tenure_bucket,
    )
    return ai_customer_search(filters, limit=limit, offset=offset)


@router.get("/customers/{customer_id}", response_model=CustomerLookupResult)
def get_customer(customer_id: str):
    """Returns 200 with found=false on a miss, not 404 - that is
    customer_lookup()'s own typed contract, which src/ai/graph.py also
    depends on."""
    return ai_customer_lookup(customer_id)


@router.get("/model-history")
def model_history():
    """Raw metadata list with no strict response_model: the shape of
    models_store/*.json has drifted across versions. serving_version names
    the loaded model, which is not necessarily the last entry."""
    return {
        "versions": dashboard_queries.load_all_churn_metadata(),
        "serving_version": state.models["churn_version"],
    }


@router.get("/pipeline-status")
def pipeline_status():
    ingestion_log = dashboard_queries.load_ingestion_log()
    batches_ingested = int(len(ingestion_log))
    every_n = dashboard_queries.RETRAIN_EVERY_N_BATCHES

    return {
        "ingestion_log": ingestion_log.to_dict(orient="records"),
        "batches_ingested": batches_ingested,
        "total_simulated_batches": dashboard_queries.TOTAL_SIMULATED_BATCHES,
        "batches_to_next_retrain": max(0, every_n - (batches_ingested % every_n)),
        "retrain_every_n_batches": every_n,
        "model_versions_trained": len(dashboard_queries.load_all_churn_metadata()),
        "drift": dashboard_queries.load_latest_drift().to_dict(orient="records"),
        "latest_retrain_summary": get_latest_retrain_summary(),
    }
