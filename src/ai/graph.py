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

Each tool call is wrapped so one tool's failure (a transient DB error, a
bad generated SQL string) degrades that one branch rather than the whole
request - matching this project's existing "LLM/DB outage never takes
down the response" precedent (src/agents/groq_client.py, src/model/api.py's
/explain-churn). A trace (src/ai/observability/tracing.py, section 22) is
written at the end of every call, itself wrapped so a trace-write failure
can never affect the response either.
"""

from __future__ import annotations

import logging
import time
import uuid

from src.agents.groq_client import AgentCallFailed
from src.ai.guardrails.validation import validate_response
from src.ai.llm import get_llm_provider
from src.ai.observability import tracing
from src.ai.observability.cost import estimate_cost
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
from src.ai.tools._artifacts import load_churn_artifact
from src.ai.tools.churn_tool import churn_analysis
from src.ai.tools.customer_tool import customer_lookup
from src.ai.tools.sql_tool import run_sql
from src.model.dashboard_queries import column_importances

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
  - churn is an INTEGER column (0 or 1), not boolean - write `churn = 1`, never `churn = true`.
- public.feature_drift(reference_batch, current_batch, feature, feature_type, psi, severity, computed_at)
- public.ingestion_log(batch_file, rows_loaded, loaded_at, status)
- public.retrain_summaries(churn_model_version, previous_version, summary_text, created_at)

Rules:
- Output ONLY the SQL query - no explanation, no markdown code fences, no leading "sql" label.
- SELECT statements only. Never write INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE.
- Include a LIMIT clause (100 or fewer) unless the query is a small aggregate.
- Reference only the tables/columns listed above."""

_GENERATION_SYSTEM_PROMPT = """You are a business intelligence assistant for a telecom customer analytics platform. Answer the user's question using ONLY the evidence listed below - never invent a number, customer fact, or claim that isn't present in it. Reproduce every number exactly as it appears in the evidence - do not round it, abbreviate it (e.g. "396K" for 395809), or convert its units (e.g. do not turn a decimal fraction like 0.0992 into a percentage like "9.9%"); an automated check rejects the whole answer if a number doesn't match the evidence's exact text. If the evidence only partially answers the question, say what it does and does not cover. Keep the answer to 3-6 sentences, written for a business audience (no mention of "SHAP", "evidence objects", or other implementation details)."""


def _strip_sql_fences(text: str) -> str:
    text = text.strip().strip("`").strip()
    if text.lower().startswith("sql\n"):
        text = text[4:]
    return text.strip()


def _run_timed(tool_latency: dict, errors: list, tool_name: str, fn):
    """Runs fn(), recording elapsed ms into tool_latency[tool_name]. Any
    exception is logged and appended to errors rather than propagated -
    one tool's failure degrades that one branch, not the whole request."""
    t0 = time.monotonic()
    try:
        result = fn()
    except Exception as exc:
        tool_latency[tool_name] = round((time.monotonic() - t0) * 1000, 2)
        log.exception("%s failed", tool_name)
        errors.append(f"{tool_name}: {exc}")
        return None
    tool_latency[tool_name] = round((time.monotonic() - t0) * 1000, 2)
    return result


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
    """Returns (SQLQueryResult | None, AgentResponse | None) - the second
    element lets callers add the SQL-generation call's token usage to
    the trace even when the generated query turns out empty/unsafe."""
    try:
        gen_response = llm.complete(
            system_prompt=_SQL_GENERATION_SYSTEM_PROMPT, user_prompt=query,
            max_tokens=300, temperature=0.0,
        )
    except AgentCallFailed as exc:
        log.warning("SQL generation LLM call failed: %s", exc)
        return None, None

    candidate_sql = _strip_sql_fences(gen_response.text)
    try:
        return run_sql(candidate_sql), gen_response
    except Exception as exc:  # SQLSafetyError or a real Postgres error
        log.warning("LLM-generated SQL rejected/failed: %s | sql=%r", exc, candidate_sql)
        return None, gen_response


def _gather_sql_evidence(query: str):
    sql_result, agent_response = _generate_and_run_sql(query, get_llm_provider())
    if sql_result is None or not sql_result.rows:
        return [], [], sql_result, agent_response
    evidence = [Evidence(
        type="database", source="marts.customer_360 (generated query)",
        claim="Result of a generated SQL query answering this question",
        value=str(sql_result.rows[:10]),
    )]
    citations = [Citation(label="[Customer 360: generated query]", type="database", source="marts.customer_360")]
    return evidence, citations, sql_result, agent_response


def _describe_segment(filters: dict[str, str]) -> str:
    return ", ".join(f"{col}={val}" for col, val in sorted(filters.items()))


def _gather_feature_importance_evidence(top_k: int = 6) -> list[Evidence]:
    """Which columns the churn model actually leans on, reusing the same
    column_importances() that powers the frontend's Overview page.

    Without this, "what are the biggest risk factors for churn?" gathered
    only aggregate rates and the model correctly answered that the
    evidence named no drivers - the platform knew the answer, the
    assistant just wasn't given it. Model-level importances, not SHAP:
    SHAP here is per-customer (see src/ai/tools/shap_tool.py, reached via
    customer_lookup when a question names a customer)."""
    artifact, version = load_churn_artifact()
    if artifact is None:
        return []
    pipeline = artifact.get("base_pipeline", artifact.get("pipeline"))
    importances = column_importances(pipeline) if pipeline is not None else {}
    if not importances:
        return []

    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [Evidence(
        type="model", source=f"churn model {version}",
        claim="Features the churn model weighs most heavily, most important first",
        value=", ".join(f"{col} (importance {imp:.4f})" for col, imp in ranked),
    )]


def _gather_ml_evidence(filters: dict[str, str] | None = None) -> tuple[list[Evidence], list[Citation], str | None]:
    """Scopes the churn statistics to the segment the question asked about
    (see src/ai/router.py's segment_filters). The segment is named in the
    evidence's own claim text, so the generating model can state which
    population a number describes instead of implying it covers everyone."""
    filters = filters or {}
    result = churn_analysis(filters=filters or None)
    if result.population_size == 0:
        return [], [], None

    scope = _describe_segment(filters) if filters else "all customers"
    evidence = [Evidence(
        type="model", source=f"churn model {result.model_version}",
        claim=f"Churn statistics for {scope}",
        value=(
            f"segment={scope}, "
            f"current_churn_rate={result.current_churn_rate}, "
            f"predicted_high_risk_count={result.predicted_high_risk_count}, "
            f"population_size={result.population_size}"
        ),
    )]
    citations = [Citation(
        label=f"[Churn Model {result.model_version}, {scope}]",
        type="model", source=result.model_version or "unknown",
    )]

    # A segment question is implicitly comparative ("why is churn worse for
    # X?" means worse *than the rest*), so the population baseline rides
    # along - otherwise the model has a single number and nothing to judge
    # it against, and answers "the rate is 0.28" without saying that's ~3x
    # the overall rate.
    if filters:
        overall = churn_analysis()
        if overall.population_size:
            evidence.append(Evidence(
                type="model", source=f"churn model {overall.model_version}",
                claim="Whole-population baseline, for comparison against the segment above",
                value=(
                    f"segment=all customers, "
                    f"current_churn_rate={overall.current_churn_rate}, "
                    f"predicted_high_risk_count={overall.predicted_high_risk_count}, "
                    f"population_size={overall.population_size}"
                ),
            ))

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


def _generate_answer(query: str, evidence: list[Evidence]):
    """Returns (answer_text, AgentResponse | None, generation_failed)."""
    evidence_text = "\n".join(f"- ({e.type}, {e.source}) {e.claim}: {e.value}" for e in evidence)
    try:
        response = get_llm_provider().complete(
            system_prompt=_GENERATION_SYSTEM_PROMPT,
            user_prompt=f"Question: {query}\n\nEvidence:\n{evidence_text}",
            max_tokens=400, temperature=0.2,
        )
        return response.text.strip(), response, False
    except AgentCallFailed as exc:
        log.warning("answer generation LLM call failed, falling back to a templated evidence summary: %s", exc)
        return "Based on the available evidence:\n" + evidence_text, None, True


def _write_trace_safely(**trace_fields) -> None:
    try:
        tracing.write_trace({
            "trace_id": trace_fields["trace_id"],
            "user_query": trace_fields["query"],
            "route": trace_fields["route"],
            "tools_used": trace_fields["tools_used"],
            "tool_latency_ms": trace_fields["tool_latency"],
            "sql_query_hash": trace_fields["sql_hash"],
            "retrieval_latency_ms": trace_fields["tool_latency"].get("hybrid_search"),
            "retrieved_documents": trace_fields["retrieved_documents"],
            # Not separately instrumented - see src/ai/rag/hybrid_search.py;
            # "hybrid_search" in tool_latency_ms covers retrieval end-to-end.
            "reranker_latency_ms": None,
            "llm_model": trace_fields["llm_model"],
            "input_tokens": trace_fields["input_tokens"],
            "output_tokens": trace_fields["output_tokens"],
            "estimated_cost_usd": estimate_cost(
                trace_fields["llm_model"], trace_fields["input_tokens"], trace_fields["output_tokens"]
            ),
            "total_latency_ms": trace_fields["total_latency_ms"],
            "validation_result": trace_fields["validation_result"],
            "fallback_status": trace_fields["fallback_status"],
            "error": trace_fields["error"],
        })
    except Exception:
        log.exception("failed to write trace %s (response is unaffected)", trace_fields["trace_id"])


def run_query(query: str) -> AssistantResponse:
    start = time.monotonic()
    trace_id = uuid.uuid4().hex
    route, signals = classify_with_signals(query)

    if route == UNSUPPORTED:
        latency_ms = round((time.monotonic() - start) * 1000, 2)
        _write_trace_safely(
            trace_id=trace_id, query=query, route=route, tools_used=[], tool_latency={},
            sql_hash=None, retrieved_documents=[], llm_model=None, input_tokens=0, output_tokens=0,
            validation_result="unsupported", fallback_status=False, error=None, total_latency_ms=latency_ms,
        )
        return AssistantResponse(
            answer=UNSUPPORTED_ANSWER, route=route, trace_id=trace_id, latency_ms=latency_ms,
        )

    evidence: list[Evidence] = []
    citations: list[Citation] = []
    tools_used: list[str] = []
    tool_latency: dict[str, float] = {}
    errors: list[str] = []
    llm_calls = []
    model_version: str | None = None
    sql_hash: str | None = None

    if signals["has_customer_id"]:
        tools_used.append("customer_lookup")
        result = _run_timed(
            tool_latency, errors, "customer_lookup",
            lambda: _gather_customer_evidence(signals["customer_id"]),
        )
        if result is not None:
            e, c, mv = result
            evidence += e
            citations += c
            model_version = model_version or mv

    if route == SQL_ANALYSIS or (route == MULTI_SOURCE and signals["has_sql"]):
        tools_used.append("sql_tool")
        result = _run_timed(tool_latency, errors, "sql_tool", lambda: _gather_sql_evidence(query))
        if result is not None:
            e, c, sql_result, agent_response = result
            evidence += e
            citations += c
            if sql_result is not None:
                sql_hash = tracing.hash_sql(sql_result.sql)
            if agent_response is not None:
                llm_calls.append(agent_response)

    if route == ML_ANALYSIS or (route == MULTI_SOURCE and signals["has_ml"]):
        if signals["asks_for_drivers"]:
            tools_used.append("feature_importances")
            importance_evidence = _run_timed(
                tool_latency, errors, "feature_importances", _gather_feature_importance_evidence,
            )
            if importance_evidence:
                evidence += importance_evidence

        tools_used.append("churn_analysis")
        result = _run_timed(
            tool_latency, errors, "churn_analysis",
            lambda: _gather_ml_evidence(signals["segment_filters"]),
        )
        if result is not None:
            e, c, mv = result
            evidence += e
            citations += c
            model_version = model_version or mv

    if route == RAG_SEARCH or (route == MULTI_SOURCE and signals["has_rag"]):
        tools_used.append("hybrid_search")
        result = _run_timed(tool_latency, errors, "hybrid_search", lambda: _gather_rag_evidence(query))
        if result is not None:
            e, c = result
            evidence += e
            citations += c

    retrieved_documents = [e.source for e in evidence if e.type == "document"]
    latency_ms = round((time.monotonic() - start) * 1000, 2)
    combined_error = "; ".join(errors) or None

    if not evidence:
        _write_trace_safely(
            trace_id=trace_id, query=query, route=route, tools_used=tools_used, tool_latency=tool_latency,
            sql_hash=sql_hash, retrieved_documents=retrieved_documents, llm_model=None,
            input_tokens=0, output_tokens=0, validation_result="insufficient_evidence",
            fallback_status=False, error=combined_error, total_latency_ms=latency_ms,
        )
        return AssistantResponse(
            answer=INSUFFICIENT_EVIDENCE_ANSWER, tools_used=tools_used,
            model_version=model_version, route=route, trace_id=trace_id, latency_ms=latency_ms,
        )

    answer_text, gen_response, generation_failed = _generate_answer(query, evidence)
    if gen_response is not None:
        llm_calls.append(gen_response)
    latency_ms = round((time.monotonic() - start) * 1000, 2)

    response = AssistantResponse(
        answer=answer_text, citations=citations, evidence=evidence, tools_used=tools_used,
        model_version=model_version, route=route, trace_id=trace_id, latency_ms=latency_ms,
    )
    validated = validate_response(response)

    fallback_status = generation_failed or (validated.confidence == 0.5)
    validation_result = "ungrounded_fallback" if validated.confidence == 0.5 else "grounded"
    llm_model = llm_calls[-1].model if llm_calls else None
    input_tokens = sum(r.prompt_tokens for r in llm_calls)
    output_tokens = sum(r.completion_tokens for r in llm_calls)

    _write_trace_safely(
        trace_id=trace_id, query=query, route=route, tools_used=tools_used, tool_latency=tool_latency,
        sql_hash=sql_hash, retrieved_documents=retrieved_documents, llm_model=llm_model,
        input_tokens=input_tokens, output_tokens=output_tokens, validation_result=validation_result,
        fallback_status=fallback_status, error=combined_error, total_latency_ms=validated.latency_ms,
    )
    return validated
