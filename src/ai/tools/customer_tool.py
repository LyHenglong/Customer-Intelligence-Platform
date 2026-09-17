"""customer_lookup and customer_search tools - Phase 1 of the AI tool
layer (see AI_Customer_Intelligence_Claude_Code_Plan.md section 8).

Both reuse the existing warehouse access layer (src/warehouse.py) and the
same churn/recommender artifacts the FastAPI service loads, rather than
introducing a second way to reach Postgres or a second copy of the model
artifacts. Every probability, SHAP value, and recommendation returned here
comes from the existing model code - these functions only assemble and
shape the output, they never compute anything the rest of the platform
doesn't already compute.
"""

from __future__ import annotations

import pandas as pd

from src.ai.schemas import (
    CustomerLookupResult,
    CustomerProfile,
    CustomerSearchFilters,
    CustomerSearchResult,
    Recommendation,
    ShapFactor,
)
from src.ai.tools._artifacts import load_churn_artifact, load_recommender_artifact
from src.model.explain_churn import compute_shap_details
from src.model.train_churn import ALL_FEATURES as CHURN_FEATURES
from src.model.train_churn import ID_COL as CHURN_ID_COL
from src.model.train_recommender import PROFILE_CATEGORICAL as REC_PROFILE_CATEGORICAL
from src.model.train_recommender import PROFILE_NUMERIC as REC_PROFILE_NUMERIC
from src.model.train_recommender import SERVICE_COLUMNS as REC_SERVICE_COLUMNS
from src.model.train_recommender import recommend_for_customer, recommend_for_profile
from src.warehouse import get_pg_conn

_DISPLAY_COLUMNS = [
    "customer_id", "contract", "tenure", "monthlycharges",
    "tenure_bucket", "total_active_services", "customer_satisfaction",
    "num_complaints",
]

_SEARCH_MAX_LIMIT = 200
_SEARCH_ALLOWED_FILTER_COLUMNS = {"contract", "tenure_bucket"}


def _row_to_profile(row: dict) -> CustomerProfile:
    return CustomerProfile(**{k: row.get(k) for k in _DISPLAY_COLUMNS})


def customer_lookup(customer_id: str) -> CustomerLookupResult:
    """Full profile + churn probability + risk status + SHAP factors +
    recommendation for one customer."""
    churn_artifact, churn_version = load_churn_artifact()
    fetch_columns = list(dict.fromkeys(_DISPLAY_COLUMNS + CHURN_FEATURES))

    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(fetch_columns)} FROM marts.customer_360 WHERE {CHURN_ID_COL} = %s",
                (customer_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return CustomerLookupResult(customer_id=customer_id, found=False)

    row_dict = dict(zip(fetch_columns, row))
    result = CustomerLookupResult(
        customer_id=customer_id, found=True, profile=_row_to_profile(row_dict)
    )

    if churn_artifact is not None:
        X = pd.DataFrame([row_dict], columns=CHURN_FEATURES)
        for c in [f for f in CHURN_FEATURES if X[f].dtype == bool]:
            X[c] = X[c].astype(float)
        pipeline = churn_artifact["pipeline"]
        proba = float(pipeline.predict_proba(X)[0, 1])
        threshold = float(churn_artifact.get("threshold", 0.5))

        result.churn_probability = round(proba, 4)
        result.churn_threshold = threshold
        result.risk_status = "high" if proba >= threshold else "low"
        result.model_version = churn_version

        explain_pipeline = churn_artifact.get("base_pipeline", pipeline)
        shap_details = compute_shap_details(explain_pipeline, X, top_k=5)[0]
        result.shap_factors = [ShapFactor(**d) for d in shap_details]

    rec_artifact, rec_version = load_recommender_artifact()
    if rec_artifact is not None:
        try:
            recs = recommend_for_customer(rec_artifact, customer_id, top_n=3)
        except KeyError:
            # Same projection-into-the-index fallback as the FastAPI
            # /recommend endpoint (src/model/api.py) - CHURN_FEATURES is a
            # strict superset of the recommender's profile+service columns
            # (both are drawn from customer_360), so row_dict already has
            # everything recommend_for_profile needs.
            rec_cols = REC_PROFILE_NUMERIC + REC_PROFILE_CATEGORICAL + REC_SERVICE_COLUMNS
            rec_profile = pd.DataFrame([row_dict], columns=rec_cols)
            own_services = [int(row_dict[s]) for s in REC_SERVICE_COLUMNS]
            recs = recommend_for_profile(rec_artifact, rec_profile, own_services, top_n=3)
        result.recommendation = [Recommendation(**r) for r in recs]
        result.recommender_version = rec_version

    return result


def customer_search(
    filters: CustomerSearchFilters, limit: int = 25, offset: int = 0
) -> CustomerSearchResult:
    """Filtered, paginated customer search against marts.customer_360.

    Filtering and pagination happen in SQL, never "fetch everything, filter
    in pandas" (see src/warehouse.py's memory-safety rationale).
    churn_probability isn't a stored column - filtering on it requires
    scoring, so this scores only a bounded SQL-prefiltered candidate set
    rather than the full table. When a probability filter is combined with
    a low-selectivity SQL filter, total_matched reflects that bounded
    candidate set rather than the true unbounded total - a known,
    documented limitation rather than an unbounded scoring pass.
    """
    limit = max(1, min(limit, _SEARCH_MAX_LIMIT))
    offset = max(0, offset)

    where, params = [], []
    if filters.contract:
        where.append("contract = %s")
        params.append(filters.contract)
    if filters.tenure_bucket:
        where.append("tenure_bucket = %s")
        params.append(filters.tenure_bucket)
    if filters.min_monthly_charges is not None:
        where.append("monthlycharges >= %s")
        params.append(filters.min_monthly_charges)
    if filters.max_monthly_charges is not None:
        where.append("monthlycharges <= %s")
        params.append(filters.max_monthly_charges)
    if filters.min_satisfaction is not None:
        where.append("customer_satisfaction >= %s")
        params.append(filters.min_satisfaction)
    if filters.max_satisfaction is not None:
        where.append("customer_satisfaction <= %s")
        params.append(filters.max_satisfaction)
    if filters.min_complaints is not None:
        where.append("num_complaints >= %s")
        params.append(filters.min_complaints)

    where_clause = f"WHERE {' AND '.join(where)}" if where else ""
    needs_scoring = (
        filters.min_churn_probability is not None or filters.max_churn_probability is not None
    )

    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            if not needs_scoring:
                cur.execute(f"SELECT COUNT(*) FROM marts.customer_360 {where_clause}", params)
                total_matched = int(cur.fetchone()[0])
                cur.execute(
                    f"SELECT {', '.join(_DISPLAY_COLUMNS)} FROM marts.customer_360 "
                    f"{where_clause} ORDER BY customer_id LIMIT %s OFFSET %s",
                    params + [limit, offset],
                )
                rows = cur.fetchall()
                customers = [_row_to_profile(dict(zip(_DISPLAY_COLUMNS, r))) for r in rows]
                return CustomerSearchResult(
                    customers=customers, total_matched=total_matched,
                    limit=limit, offset=offset,
                    truncated=(offset + len(customers)) < total_matched,
                )

            # needs_scoring: bounded prefilter (10x the page, capped) so
            # scoring never runs over an unbounded candidate set.
            prefilter_cap = min(limit * 10, 5000)
            fetch_columns = list(dict.fromkeys(_DISPLAY_COLUMNS + CHURN_FEATURES))
            cur.execute(
                f"SELECT {', '.join(fetch_columns)} FROM marts.customer_360 "
                f"{where_clause} ORDER BY customer_id LIMIT %s",
                params + [prefilter_cap],
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    churn_artifact, _ = load_churn_artifact()
    if churn_artifact is None or not rows:
        return CustomerSearchResult(customers=[], total_matched=0, limit=limit, offset=offset, truncated=False)

    fetch_columns = list(dict.fromkeys(_DISPLAY_COLUMNS + CHURN_FEATURES))
    df = pd.DataFrame(rows, columns=fetch_columns)
    X = df[CHURN_FEATURES].copy()
    for c in [f for f in CHURN_FEATURES if X[f].dtype == bool]:
        X[c] = X[c].astype(float)
    df["_churn_probability"] = churn_artifact["pipeline"].predict_proba(X)[:, 1]

    if filters.min_churn_probability is not None:
        df = df[df["_churn_probability"] >= filters.min_churn_probability]
    if filters.max_churn_probability is not None:
        df = df[df["_churn_probability"] <= filters.max_churn_probability]

    total_matched = len(df)
    page = df.iloc[offset: offset + limit]
    customers = [_row_to_profile(row.to_dict()) for _, row in page.iterrows()]
    return CustomerSearchResult(
        customers=customers, total_matched=total_matched,
        limit=limit, offset=offset,
        truncated=(offset + len(customers)) < total_matched or len(rows) == prefilter_cap,
    )
