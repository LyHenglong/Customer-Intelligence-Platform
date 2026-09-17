"""Rule-based query classification
(AI_Customer_Intelligence_Claude_Code_Plan.md section 14).

Deterministic and LLM-free on purpose: classification gates which tools
run, and a routing mistake is far easier to write a regression test for
as an explicit rule than as an LLM call this pipeline would then also
have to validate. Its accuracy against the evaluation harness's benchmark
questions (src/ai/evaluation/, Stage 6) is itself one of the measured
metrics ("tool selection accuracy" - section 21) - this is a heuristic
classifier, not a claim of perfect routing, and the eval harness is where
its real accuracy gets reported honestly rather than assumed here.
"""

from __future__ import annotations

import re

CUSTOMER_LOOKUP = "CUSTOMER_LOOKUP"
SQL_ANALYSIS = "SQL_ANALYSIS"
ML_ANALYSIS = "ML_ANALYSIS"
RAG_SEARCH = "RAG_SEARCH"
MULTI_SOURCE = "MULTI_SOURCE"
UNSUPPORTED = "UNSUPPORTED"

_CUSTOMER_ID_RE = re.compile(r"\bCUST\d{3,}\b", re.IGNORECASE)

_SQL_KEYWORDS = (
    "how many", "average", "avg", "count", "rate", "compare", "comparison",
    "which customers", "segment", "breakdown", "group by", "total", "revenue",
)
_ML_KEYWORDS = (
    "churn probability", "risk factor", "shap", "why is", "why did",
    "predicted", "at risk", "explain why", "risky", "likely to churn",
    "recommend", "recommendation",
)
_RAG_KEYWORDS = (
    "policy", "policies", "playbook", "guidance", "documentation", "document",
    "what does", "according to", "procedure", "escalation", "offer cost",
    "retention strategy", "support tier",
)


def _extract_customer_id(query: str) -> str | None:
    m = _CUSTOMER_ID_RE.search(query)
    return m.group(0).upper() if m else None


def _matches_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(kw in text for kw in keywords)


def classify_with_signals(query: str) -> tuple[str, dict]:
    """Returns (route, signals). signals carries the individual booleans
    that produced the route plus the extracted customer_id, so callers
    (src/ai/graph.py) can decide which tools to run for MULTI_SOURCE
    without re-deriving them."""
    text = (query or "").lower().strip()
    customer_id = _extract_customer_id(query or "")
    signals = {
        "has_customer_id": customer_id is not None,
        "has_sql": _matches_any(text, _SQL_KEYWORDS),
        "has_ml": _matches_any(text, _ML_KEYWORDS),
        "has_rag": _matches_any(text, _RAG_KEYWORDS),
        "customer_id": customer_id,
    }

    if not text:
        return UNSUPPORTED, signals

    # A customer id with no SQL/RAG signal is a pure lookup - customer_lookup
    # (src/ai/tools/customer_tool.py) already returns churn probability,
    # SHAP, and a recommendation together, so an ML keyword alongside a
    # customer id doesn't need its own branch.
    if signals["has_customer_id"] and not (signals["has_rag"] or signals["has_sql"]):
        return CUSTOMER_LOOKUP, signals

    active = sum([signals["has_customer_id"], signals["has_sql"], signals["has_ml"], signals["has_rag"]])
    if active >= 2:
        return MULTI_SOURCE, signals
    if signals["has_sql"]:
        return SQL_ANALYSIS, signals
    if signals["has_ml"]:
        return ML_ANALYSIS, signals
    if signals["has_rag"]:
        return RAG_SEARCH, signals
    return UNSUPPORTED, signals


def classify(query: str) -> str:
    route, _ = classify_with_signals(query)
    return route
