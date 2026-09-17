"""Explicit state-machine agent workflow
(AI_Customer_Intelligence_Claude_Code_Plan.md section 13).

Implemented as plain Python functions rather than a graph-orchestration
framework (LangGraph): every node the plan's diagram describes - classify,
route, execute tools, aggregate evidence, check sufficiency, generate,
validate - is one function call in run_query() below. The project's own
"Do Not Overengineer" instruction (section 4) explicitly lists "LangChain
abstractions where plain Python is clearer" as something to avoid, and
nothing here needs a graph library's retry/checkpointing machinery: each
call is a single request/response, not a long-running multi-turn session.

Core design principle (section 3): tools produce authoritative facts
first, evidence is aggregated from their structured output, and only then
does the LLM turn that evidence into prose. If no tool produced any
evidence, the LLM is never called at all - the response is the fixed
"insufficient evidence" answer, not a guess.
"""

from __future__ import annotations

import logging
import time
import uuid

from src.agents.groq_client import AgentCallFailed
from src.ai.llm import get_llm_provider
from src.ai.rag.hybrid_search import hybrid_search
from src.ai.router import (
    CUSTOMER_LOOKUP,
    ML_ANALYSIS,
    MULTI_SOURCE,
    RAG_SEARCH,
    SQL_ANALYSIS,
    UNSUPPORTED,
    classify_with_signals,
)
from src.ai.schemas import AssistantResponse, Citation, Evidence
from src.ai.tools.churn_tool import churn_analysis
from src.ai.tools.customer_tool import customer_lookup
from src.ai.tools.sql_tool import run_sql

log = logging.getLogger("ai.graph")

UNSUPPORTED_ANSWER = (
    "This assistant can answer questions about customer records, churn/risk "
    "analysis, aggregate statistics, and this platform's own business "
    "documentation. This question doesn't appear to be one of those - try "
    "asking about a specific customer (e.g. \"CUST000123\"), a churn/segment "
    "statistic, or a retention policy."
)
INSUFFICIENT_EVIDENCE_ANSWER = (
    "I don't have enough evidence in the available data to answer that reliably."
)

_SQL_GENERATION_SYSTEM_PROMPT = """You translate a business question into a single read-only PostgreSQL SELECT query.

Allowed tables and columns (use only these):
- marts.customer_360(customer_id, contract, tenure, tenure_bucket, monthlycharges, totalcharges, customer_satisfaction, num_complaints, num_service_calls, late_payments, total_active_services, churn, gender, education, marital_status, payment_method)
- public.feature_drift(reference_batch, current_batch, feature, feature_type, psi, severity, computed_at)
- public.ingestion_log(batch_file, rows_loaded, loaded_at, status)
- public.retrain_summaries(churn_model_version, previous_version, summary_text, created_at)

Rules:
- Output ONLY the SQL query - no explanation, no markdown code fences, no leading "sql" label.
- SELECT statements only. Never write INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE.
- Include a LIMIT clause (100 or fewer) unless the query is a small aggregate.
- Reference only the tables/columns listed above."""

_GENERATION_SYSTEM_PROMPT = """You are a business intelligence assistant for a telecom customer analytics platform. Answer the user's question using ONLY the evidence listed below - never invent a number, customer fact, or claim that isn't present in it. If the evidence only partially answers the question, say what it does and does not cover. Keep the answer to 3-6 sentences, written for a business audience (no mention of "SHAP", "evidence objects", or other implementation details)."""


def _strip_sql_fences(text: str) -> str:
    text = text.strip().strip("`").strip()
    if text.lower().startswith("sql\n"):
        text = text[4:]
    return text.strip()


def _gather_customer_evidence(customer_id: str) -> tuple[list[Evidence], list[Citation], str | None]:
    evidence: list[Evidence] = []
    citations: list[Citation] = []
    result = customer_lookup(customer_id)
    if not result.found:
        return evidence, citations, None

    evidence.append(Evidence(
        type="database", source="marts.customer_360",
        claim=f"Profile for customer {customer_id}",
        value=result.profile.model_dump_json(),
    ))
    citations.append(Citation(label=f"[Customer 360: {customer_id}]", type="database", source="marts.customer_360"))

    model_version = None
    if result.churn_probability is not None:
        model_version = result.model_version
        evidence.append(Evidence(
            type="model", source=f"churn model {result.model_version}",
            claim="Predicted churn probability and risk status",
            value=(
                f"probability={result.churn_probability}, threshold={result.churn_threshold}, "
                f"status={result.risk_status}"
            ),
        ))
        citations.append(Citation(
            label=f"[Churn Model {result.model_version}]", type="model", source=result.model_version or "unknown",
        ))
    if result.shap_factors:
        evidence.append(Evidence(
            type="model", source=f"SHAP ({result.model_version})",
            claim="Top risk factors driving this prediction",
            value=", ".join(f"{f.feature} ({f.direction})" for f in result.shap_factors),
        ))
    if result.recommendation:
        evidence.append(Evidence(
            type="model", source=f"recommender {result.recommender_version}",
            claim="Recommended service",
            value=", ".join(r.service for r in result.recommendation),
        ))
    return evidence, citations, model_version


def _generate_and_run_sql(query: str, llm):
    try:
        response = llm.complete(
            system_prompt=_SQL_GENERATION_SYSTEM_PROMPT, user_prompt=query,
            max_tokens=300, temperature=0.0,
        )
    except AgentCallFailed as exc:
        log.warning("SQL generation LLM call failed: %s", exc)
        return None

    candidate_sql = _strip_sql_fences(response.text)
    try:
        return run_sql(candidate_sql)
    except Exception as exc:  # SQLSafetyError or a real Postgres error
        log.warning("LLM-generated SQL rejected/failed: %s | sql=%r", exc, candidate_sql)
        return None


def _gather_sql_evidence(query: str) -> tuple[list[Evidence], list[Citation]]:
    result = _generate_and_run_sql(query, get_llm_provider())
    if result is None or not result.rows:
        return [], []
    evidence = [Evidence(
        type="database", source="marts.customer_360 (generated query)",
        claim="Result of a generated SQL query answering this question",
        value=str(result.rows[:10]),
    )]
    citations = [Citation(label="[Customer 360: generated query]", type="database", source="marts.customer_360")]
    return evidence, citations


def _gather_ml_evidence() -> tuple[list[Evidence], list[Citation], str | None]:
    result = churn_analysis()
    if result.population_size == 0:
        return [], [], None
    evidence = [Evidence(
        type="model", source=f"churn model {result.model_version}",
        claim="Population-level churn statistics",
        value=(
            f"current_churn_rate={result.current_churn_rate}, "
            f"predicted_high_risk_count={result.predicted_high_risk_count}, "
            f"population_size={result.population_size}"
        ),
    )]
    citations = [Citation(
        label=f"[Churn Model {result.model_version}, population stats]",
        type="model", source=result.model_version or "unknown",
    )]
    return evidence, citations, result.model_version


def _gather_rag_evidence(query: str) -> tuple[list[Evidence], list[Citation]]:
    candidates = hybrid_search(query, top_k_final=3)
    evidence, citations = [], []
    for c in candidates:
        evidence.append(Evidence(
            type="document", source=c.document_id,
            claim=f"Relevant passage from {c.title}" + (f", section {c.section}" if c.section else ""),
            value=c.text,
        ))
        label = f"[{c.title}" + (f", {c.section}" if c.section else "") + "]"
        citations.append(Citation(label=label, type="document", source=c.document_id))
    return evidence, citations


def _generate_answer(query: str, evidence: list[Evidence]) -> str:
    evidence_text = "\n".join(f"- ({e.type}, {e.source}) {e.claim}: {e.value}" for e in evidence)
    try:
        response = get_llm_provider().complete(
            system_prompt=_GENERATION_SYSTEM_PROMPT,
            user_prompt=f"Question: {query}\n\nEvidence:\n{evidence_text}",
            max_tokens=400, temperature=0.2,
        )
        return response.text.strip()
    except AgentCallFailed as exc:
        log.warning("answer generation LLM call failed, falling back to a templated evidence summary: %s", exc)
        return "Based on the available evidence:\n" + evidence_text


def run_query(query: str) -> AssistantResponse:
    start = time.monotonic()
    trace_id = uuid.uuid4().hex
    route, signals = classify_with_signals(query)

    if route == UNSUPPORTED:
        return AssistantResponse(
            answer=UNSUPPORTED_ANSWER, route=route, trace_id=trace_id,
            latency_ms=round((time.monotonic() - start) * 1000, 2),
        )

    evidence: list[Evidence] = []
    citations: list[Citation] = []
    tools_used: list[str] = []
    model_version: str | None = None

    if signals["has_customer_id"]:
        tools_used.append("customer_lookup")
        e, c, mv = _gather_customer_evidence(signals["customer_id"])
        evidence += e
        citations += c
        model_version = model_version or mv

    if route == SQL_ANALYSIS or (route == MULTI_SOURCE and signals["has_sql"]):
        tools_used.append("sql_tool")
        e, c = _gather_sql_evidence(query)
        evidence += e
        citations += c

    if route == ML_ANALYSIS or (route == MULTI_SOURCE and signals["has_ml"]):
        tools_used.append("churn_analysis")
        e, c, mv = _gather_ml_evidence()
        evidence += e
        citations += c
        model_version = model_version or mv

    if route == RAG_SEARCH or (route == MULTI_SOURCE and signals["has_rag"]):
        tools_used.append("hybrid_search")
        e, c = _gather_rag_evidence(query)
        evidence += e
        citations += c

    latency_ms = round((time.monotonic() - start) * 1000, 2)

    if not evidence:
        return AssistantResponse(
            answer=INSUFFICIENT_EVIDENCE_ANSWER, tools_used=tools_used,
            model_version=model_version, route=route, trace_id=trace_id, latency_ms=latency_ms,
        )

    answer = _generate_answer(query, evidence)
    latency_ms = round((time.monotonic() - start) * 1000, 2)

    return AssistantResponse(
        answer=answer, citations=citations, evidence=evidence, tools_used=tools_used,
        model_version=model_version, route=route, trace_id=trace_id, latency_ms=latency_ms,
    )
