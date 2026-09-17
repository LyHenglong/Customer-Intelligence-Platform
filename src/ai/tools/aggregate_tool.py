"""aggregate_analysis tool - SQL GROUP BY aggregation over marts.customer_360.

Mirrors src/dashboard/app.py's load_segment_rates: Postgres aggregates 1M
rows far more cheaply than shipping them to pandas to do the same thing,
and the result is a handful of rows.
"""

from __future__ import annotations

from src.ai.schemas import AggregateBucket, AggregateResult
from src.warehouse import get_pg_conn

# Allowlist, not a denylist: the dimension is interpolated into the SQL
# string because column identifiers can't be parameterized, so every
# permitted value here must be a known-safe column name.
_ALLOWED_DIMENSIONS = {
    "contract", "tenure_bucket", "gender", "education",
    "marital_status", "payment_method", "total_active_services",
}


def aggregate_analysis(dimension: str) -> AggregateResult:
    """Churn rate, customer count, and average monthly charges grouped by
    `dimension`."""
    if dimension not in _ALLOWED_DIMENSIONS:
        raise ValueError(
            f"unsupported aggregate dimension {dimension!r}; allowed: {sorted(_ALLOWED_DIMENSIONS)}"
        )

    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {dimension}::text, AVG(churn::int), COUNT(*), AVG(monthlycharges) "
                f"FROM marts.customer_360 GROUP BY {dimension} ORDER BY {dimension}"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    buckets = [
        AggregateBucket(
            key=str(key),
            churn_rate=round(float(rate), 4),
            n_customers=int(n),
            avg_monthly_charges=round(float(avg_charges), 2) if avg_charges is not None else None,
        )
        for key, rate, n, avg_charges in rows
    ]
    return AggregateResult(dimension=dimension, buckets=buckets)
