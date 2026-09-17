"""customer_explanation tool - SHAP-based risk factors for one customer.

Thin wrapper around src/model/explain_churn.py's compute_shap_details. The
LLM never computes SHAP values itself - this is the only place they're
produced (AI_Customer_Intelligence_Claude_Code_Plan.md section 8: "Reuse
the existing SHAP implementation. Do not ask the LLM to calculate SHAP.").
"""

from __future__ import annotations

import pandas as pd

from src.ai.schemas import ShapFactor
from src.ai.tools._artifacts import load_churn_artifact
from src.model.explain_churn import compute_shap_details
from src.model.train_churn import ALL_FEATURES as CHURN_FEATURES
from src.model.train_churn import ID_COL as CHURN_ID_COL
from src.warehouse import get_pg_conn


def customer_explanation(customer_id: str, top_k: int = 5) -> list[ShapFactor]:
    churn_artifact, _ = load_churn_artifact()
    if churn_artifact is None:
        return []

    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(CHURN_FEATURES)} FROM marts.customer_360 WHERE {CHURN_ID_COL} = %s",
                (customer_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return []

    X = pd.DataFrame([row], columns=CHURN_FEATURES)
    explain_pipeline = churn_artifact.get("base_pipeline", churn_artifact["pipeline"])
    details = compute_shap_details(explain_pipeline, X, top_k=top_k)[0]
    return [ShapFactor(**d) for d in details]
