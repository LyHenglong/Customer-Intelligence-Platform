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
    "churn probability", "risk factor", "risk factors", "shap", "why is", "why did",
    "predicted", "at risk", "explain why", "risky", "likely to churn",
    "recommend", "recommendation",
)
_RAG_KEYWORDS = (
    "policy", "policies", "playbook", "guidance", "documentation", "document",
    "what does", "according to", "procedure", "escalation", "offer cost",
    "retention strategy", "support tier",
)

# Natural-language phrasings -> the warehouse values churn_analysis accepts
# (see src/ai/tools/churn_tool.py's _ALLOWED_FILTER_COLUMNS). Without this,
# "why is churn increasing among month-to-month customers?" scored the whole
# 1M-row population and answered with population-wide numbers, then correctly
# but uselessly reported that the evidence said nothing about month-to-month.
# Only the two filterable columns are covered on purpose - a phrase matching
# some other column would produce a filter the tool rejects outright.
_SEGMENT_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("contract", "month_to_month", r"month[\s_-]?to[\s_-]?month|monthly contract|rolling contract"),
    ("contract", "one_year", r"\bone[\s_-]?year\b|\b1[\s_-]?year\b|annual contract"),
    ("contract", "two_year", r"\btwo[\s_-]?year\b|\b2[\s_-]?year\b|biennial"),
    ("tenure_bucket", "new_0_6mo", r"\bnew customers?\b|newly signed|first six months|first 6 months"),
    ("tenure_bucket", "established_6_24mo", r"\bestablished\b"),
    ("tenure_bucket", "loyal_24mo_plus", r"\bloyal\b|long[\s_-]?tenured|long[\s_-]?standing"),
)
_SEGMENT_COMPILED = tuple((col, val, re.compile(rx)) for col, val, rx in _SEGMENT_PATTERNS)


def _keyword_pattern(keywords: tuple[str, ...]) -> re.Pattern:
    # \b-bounded, not a bare substring check: "rate" as a plain `in` test
    # matches inside "strategy" (st-RATE-gy), "count" inside "discount",
    # etc. - real false positives this project's own eval dataset caught
    # (see tests/test_ai_evaluation_datasets.py's router-agreement test).
    return re.compile(r"\b(?:" + "|".join(re.escape(kw) for kw in keywords) + r")\b")


_SQL_PATTERN = _keyword_pattern(_SQL_KEYWORDS)
_ML_PATTERN = _keyword_pattern(_ML_KEYWORDS)
_RAG_PATTERN = _keyword_pattern(_RAG_KEYWORDS)


def _extract_customer_id(query: str) -> str | None:
    m = _CUSTOMER_ID_RE.search(query)
    return m.group(0).upper() if m else None


def _matches_any(text: str, pattern: re.Pattern) -> bool:
    return bool(pattern.search(text))


def _extract_segment_filters(text: str) -> dict[str, str]:
    """First match per column wins, so "month-to-month vs two-year" narrows
    to month_to_month rather than producing a contradictory AND of both."""
    filters: dict[str, str] = {}
    for column, value, pattern in _SEGMENT_COMPILED:
        if column not in filters and pattern.search(text):
            filters[column] = value
    return filters


def classify_with_signals(query: str) -> tuple[str, dict]:
    """Returns (route, signals). signals carries the individual booleans
    that produced the route plus the extracted customer_id, so callers
    (src/ai/graph.py) can decide which tools to run for MULTI_SOURCE
    without re-deriving them."""
    text = (query or "").lower().strip()
    customer_id = _extract_customer_id(query or "")
    signals = {
        "has_customer_id": customer_id is not None,
        "has_sql": _matches_any(text, _SQL_PATTERN),
        "has_ml": _matches_any(text, _ML_PATTERN),
        "has_rag": _matches_any(text, _RAG_PATTERN),
        "customer_id": customer_id,
        # Deliberately not part of the route decision below - a segment
        # phrase narrows the evidence a route gathers, it doesn't pick the
        # route. "month-to-month customers" alone is still UNSUPPORTED.
        "segment_filters": _extract_segment_filters(text),
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
