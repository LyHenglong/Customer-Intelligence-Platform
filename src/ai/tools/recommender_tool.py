"""recommendation_analysis tool - wraps the existing content-based
recommender. The agent receives the recommendation the recommender
produced; it never invents one (AI_Customer_Intelligence_Claude_Code_Plan.md
section 8).
"""

from __future__ import annotations

import pandas as pd

from src.ai.schemas import Recommendation
from src.ai.tools._artifacts import load_recommender_artifact
from src.model.train_recommender import PROFILE_CATEGORICAL, PROFILE_NUMERIC, SERVICE_COLUMNS
from src.model.train_recommender import recommend_for_customer, recommend_for_profile
from src.warehouse import get_pg_conn


def recommendation_analysis(customer_id: str, top_n: int = 3) -> list[Recommendation]:
    artifact, _version = load_recommender_artifact()
    if artifact is None:
        return []

    try:
        recs = recommend_for_customer(artifact, customer_id, top_n=top_n)
        return [Recommendation(**r) for r in recs]
    except KeyError:
        pass

    columns = PROFILE_NUMERIC + PROFILE_CATEGORICAL + SERVICE_COLUMNS
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(columns)} FROM marts.customer_360 WHERE customer_id = %s",
                (customer_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return []

    profile = pd.DataFrame([row], columns=columns)
    own_services = [int(profile[s].iloc[0]) for s in SERVICE_COLUMNS]
    recs = recommend_for_profile(artifact, profile, own_services, top_n=top_n)
    return [Recommendation(**r) for r in recs]
