"""Environment-driven configuration for the AI assistant layer.

Follows the existing project's pattern (see MAX_TRAINING_ROWS in
src/model/train_churn.py, RETRAIN_EVERY_N_BATCHES in dags/churn_pipeline.py):
a plain os.environ.get with a sensible default, no separate settings
framework.
"""

from __future__ import annotations

import os

# --- Controlled SQL tool (src/ai/tools/sql_tool.py, src/ai/guardrails/sql_safety.py) ---

SQL_ROW_LIMIT = int(os.environ.get("AI_SQL_ROW_LIMIT", "500"))
SQL_STATEMENT_TIMEOUT_MS = int(os.environ.get("AI_SQL_STATEMENT_TIMEOUT_MS", "5000"))

# Allowlist, not a denylist: only these tables/views may appear in a
# generated query's FROM/JOIN clauses. Matches
# AI_Customer_Intelligence_Claude_Code_Plan.md section 9's recommended set -
# every table an AI-generated query is allowed to touch, never the full
# warehouse (raw_customers/customers_cleaned/llm_explanations are excluded
# on purpose: raw layers and cached LLM output aren't meant for ad hoc
# querying by a generated SQL statement).
SQL_ALLOWED_TABLES = frozenset({
    "marts.customer_360",
    "public.feature_drift",
    "public.ingestion_log",
    "public.retrain_summaries",
})
