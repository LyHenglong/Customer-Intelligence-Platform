"""Integration tests for src/ai/graph.py's run_query() state machine.
Every tool (customer_lookup, churn_analysis, hybrid_search, run_sql) and
the LLM provider are monkeypatched - this exercises routing, evidence
aggregation, the insufficient-evidence short-circuit, and generation
fallback, not any real database/model/LLM call.
"""

from __future__ import annotations

import pytest

from src.agents.groq_client import AgentCallFailed, AgentResponse
from src.ai import graph as graph_module
from src.ai.schemas import (
    ChurnAnalysisResult,
    CustomerLookupResult,
    CustomerProfile,
    Recommendation,
    RetrievalCandidate,
    ShapFactor,
    SQLQueryResult,
)


@pytest.fixture(autouse=True)
def captured_traces(monkeypatch):
    """run_query() always attempts to write a trace (src/ai/observability/
    tracing.py) - that write is itself wrapped so a failure can't affect
    the response, but letting it actually try a real Postgres connection
    in every test here would be slow/flaky/environment-dependent. Replaced
    with an in-memory recorder instead: most tests ignore it, a few below
    inspect it to pin what run_query() actually puts in a trace (DB
    read/write behavior itself is tested separately in
    tests/test_ai_observability_tracing.py)."""
    calls = []
    monkeypatch.setattr(graph_module.tracing, "write_trace", lambda trace: calls.append(trace))
    return calls


@pytest.fixture(autouse=True)
def _no_real_artifact_load(monkeypatch):
    """Driver-style questions ("risk factors", "why...") attach the churn
    model's feature importances, which loads a real joblib artifact off
    disk/MLflow. Off by default here for the same reason every other tool
    is stubbed; the test that covers it patches the gatherer directly."""
    monkeypatch.setattr(graph_module, "load_churn_artifact", lambda: (None, None))


class _FakeProvider:
    def __init__(self, responses=None, raise_on_call=False):
        self.responses = list(responses or [])
        self.raise_on_call = raise_on_call
        self.calls = []

    def complete(self, system_prompt, user_prompt, max_tokens=500, temperature=0.3):
        self.calls.append((system_prompt, user_prompt))
        if self.raise_on_call:
            raise AgentCallFailed("simulated provider failure")
        text = self.responses.pop(0) if self.responses else "generated answer"
        return AgentResponse(text=text, prompt_tokens=5, completion_tokens=5, model="fake-model")


def _no_llm_allowed(*a, **kw):
    raise AssertionError("get_llm_provider must not be called on this path")


class TestUnsupportedRoute:
    def test_returns_fixed_answer_with_no_tool_calls(self, monkeypatch):
        monkeypatch.setattr(graph_module, "get_llm_provider", _no_llm_allowed)
        result = graph_module.run_query("tell me tomorrow's weather")

        assert result.answer == graph_module.UNSUPPORTED_ANSWER
        assert result.tools_used == []
        assert result.evidence == []
        assert result.route == "UNSUPPORTED"
        assert result.latency_ms >= 0
        assert len(result.trace_id) == 32


class TestCustomerLookupRoute:
    def test_found_customer_builds_evidence_and_generates_answer(self, monkeypatch):
        looked_up = CustomerLookupResult(
            customer_id="CUST000123", found=True,
            profile=CustomerProfile(customer_id="CUST000123", contract="month_to_month"),
            churn_probability=0.42, churn_threshold=0.3, risk_status="high",
            model_version="V1",
            shap_factors=[ShapFactor(feature="num_complaints", shap_value=0.2, direction="increases risk")],
            recommendation=[Recommendation(service="has_tech_support", score=0.9)],
            recommender_version="R1",
        )
        monkeypatch.setattr(graph_module, "customer_lookup", lambda cid: looked_up)
        fake = _FakeProvider(responses=["This customer is high risk because of complaints."])
        monkeypatch.setattr(graph_module, "get_llm_provider", lambda: fake)

        result = graph_module.run_query("Show customer CUST000123")

        assert result.route == "CUSTOMER_LOOKUP"
        assert result.tools_used == ["customer_lookup"]
        assert result.model_version == "V1"
        assert len(result.evidence) == 4  # profile, churn probability, SHAP factors, recommendation
        assert result.answer == "This customer is high risk because of complaints."
        assert len(fake.calls) == 1  # only the generation call, no SQL generation on this route

    def test_unknown_customer_short_circuits_before_any_llm_call(self, monkeypatch):
        monkeypatch.setattr(graph_module, "customer_lookup", lambda cid: CustomerLookupResult(customer_id=cid, found=False))
        monkeypatch.setattr(graph_module, "get_llm_provider", _no_llm_allowed)

        result = graph_module.run_query("Show customer CUST000999")

        assert result.answer == graph_module.INSUFFICIENT_EVIDENCE_ANSWER
        assert result.evidence == []
        assert result.tools_used == ["customer_lookup"]


class TestSQLAnalysisRoute:
    def test_generated_sql_with_rows_becomes_evidence(self, monkeypatch):
        monkeypatch.setattr(graph_module, "run_sql", lambda q: SQLQueryResult(
            sql=q, columns=["contract", "churn_rate"], rows=[["month_to_month", 30]],
            row_count=1, truncated=False, execution_time_ms=1.0,
        ))
        fake = _FakeProvider(responses=[
            "SELECT contract, AVG(churn) FROM marts.customer_360 GROUP BY contract LIMIT 10",
            # "30" (not "0.3"/"30%") so it text-matches the evidence's str(rows)
            # verbatim - the grounding guardrail (src/ai/guardrails/grounding.py)
            # does exact-text number matching, not unit-aware equivalence.
            "Month-to-month customers churn at a rate of 30.",
        ])
        monkeypatch.setattr(graph_module, "get_llm_provider", lambda: fake)

        result = graph_module.run_query("What is the average churn rate by contract?")

        assert result.route == "SQL_ANALYSIS"
        assert result.tools_used == ["sql_tool"]
        assert len(result.evidence) == 1
        assert result.evidence[0].type == "database"
        assert result.answer == "Month-to-month customers churn at a rate of 30."
        assert len(fake.calls) == 2  # SQL generation + answer generation
        assert result.confidence == 1.0  # grounding guardrail passed

    def test_unsafe_generated_sql_yields_insufficient_evidence(self, monkeypatch):
        monkeypatch.setattr(graph_module, "run_sql", lambda q: (_ for _ in ()).throw(ValueError("unsafe")))
        fake = _FakeProvider(responses=["DELETE FROM marts.customer_360"])
        monkeypatch.setattr(graph_module, "get_llm_provider", lambda: fake)

        result = graph_module.run_query("How many customers do we have on average per segment?")

        assert result.answer == graph_module.INSUFFICIENT_EVIDENCE_ANSWER
        assert result.evidence == []
        assert len(fake.calls) == 1  # only the (failed) SQL generation attempt, no answer generation


class TestMLAnalysisRoute:
    def test_population_stats_become_evidence(self, monkeypatch):
        monkeypatch.setattr(graph_module, "churn_analysis", lambda filters=None: ChurnAnalysisResult(
            population_size=1000, current_churn_rate=0.1, predicted_high_risk_count=120,
            mean_churn_probability=0.15, median_churn_probability=0.12,
            model_version="V2", threshold=0.11, filters_applied={},
        ))
        fake = _FakeProvider(responses=["Churn risk is concentrated among a subset of customers."])
        monkeypatch.setattr(graph_module, "get_llm_provider", lambda: fake)

        result = graph_module.run_query("What are the biggest risk factors for churn?")

        assert result.route == "ML_ANALYSIS"
        # "risk factors" is a driver question, so feature importances are
        # attempted too - they yield nothing here because the artifact
        # loader is stubbed out (see _no_real_artifact_load).
        assert result.tools_used == ["feature_importances", "churn_analysis"]
        assert result.model_version == "V2"
        assert len(result.evidence) == 1

    def test_driver_questions_attach_feature_importances_as_evidence(self, monkeypatch):
        """The platform already computes these for the Overview page; before
        this they were never handed to the assistant, so "what are the
        biggest risk factors" got aggregate rates and an honest "the
        evidence names no drivers"."""
        monkeypatch.setattr(graph_module, "churn_analysis", lambda filters=None: ChurnAnalysisResult(
            population_size=1000, current_churn_rate=0.1, predicted_high_risk_count=120,
            mean_churn_probability=0.15, median_churn_probability=0.12,
            model_version="V2", threshold=0.11, filters_applied={},
        ))
        monkeypatch.setattr(
            graph_module, "load_churn_artifact",
            lambda: ({"pipeline": object()}, "V2"),
        )
        monkeypatch.setattr(
            graph_module, "column_importances",
            lambda pipeline: {"contract": 0.40, "tenure": 0.25, "num_complaints": 0.10},
        )
        fake = _FakeProvider(responses=["Contract type and tenure matter most."])
        monkeypatch.setattr(graph_module, "get_llm_provider", lambda: fake)

        result = graph_module.run_query("What are the biggest risk factors for churn?")

        assert "feature_importances" in result.tools_used
        importance_evidence = [e for e in result.evidence if "weighs most heavily" in e.claim]
        assert len(importance_evidence) == 1
        # Ranked most-important-first and expressed as a share of total,
        # not the raw split counts column_importances returns: 0.40 of
        # (0.40 + 0.25 + 0.10) is 53.3%.
        assert importance_evidence[0].value.startswith(
            "contract (53.3% of total model importance)"
        )


    def test_feature_importances_are_skipped_when_the_model_reports_none(self, monkeypatch):
        monkeypatch.setattr(graph_module, "load_churn_artifact", lambda: ({"pipeline": object()}, "V2"))
        monkeypatch.setattr(graph_module, "column_importances", lambda pipeline: {})
        assert graph_module._gather_feature_importance_evidence() == []

    def test_counting_questions_skip_feature_importances(self, monkeypatch):
        monkeypatch.setattr(graph_module, "churn_analysis", lambda filters=None: ChurnAnalysisResult(
            population_size=1000, current_churn_rate=0.1, predicted_high_risk_count=120,
            mean_churn_probability=0.15, median_churn_probability=0.12,
            model_version="V2", threshold=0.11, filters_applied={},
        ))
        monkeypatch.setattr(graph_module, "get_llm_provider", lambda: _FakeProvider(responses=["120 are at risk."]))

        result = graph_module.run_query("how many customers are predicted at risk")

        assert "feature_importances" not in result.tools_used

    def test_empty_population_yields_insufficient_evidence(self, monkeypatch):
        monkeypatch.setattr(graph_module, "churn_analysis", lambda filters=None: ChurnAnalysisResult(
            population_size=0, current_churn_rate=0.0, predicted_high_risk_count=0,
            mean_churn_probability=0.0, median_churn_probability=0.0,
            model_version=None, threshold=0.0, filters_applied={},
        ))
        monkeypatch.setattr(graph_module, "get_llm_provider", _no_llm_allowed)

        result = graph_module.run_query("What are the biggest risk factors for churn?")
        assert result.answer == graph_module.INSUFFICIENT_EVIDENCE_ANSWER


class TestRagSearchRoute:
    def test_retrieved_passages_become_evidence_with_section_citations(self, monkeypatch):
        monkeypatch.setattr(graph_module, "hybrid_search", lambda query, top_k_final=3: [
            RetrievalCandidate(
                document_id="retention/retention_playbook.md", chunk_id="c1",
                text="High risk, month-to-month customers get a contract-conversion offer.",
                title="Retention Playbook", section="Risk Tiers", score=0.9, rank=1,
                retrieval_method="hybrid_reranked",
            ),
        ])
        fake = _FakeProvider(responses=["Month-to-month customers are offered a contract conversion."])
        monkeypatch.setattr(graph_module, "get_llm_provider", lambda: fake)

        result = graph_module.run_query("What does our retention policy say about month-to-month customers?")

        assert result.route == "RAG_SEARCH"
        assert result.tools_used == ["hybrid_search"]
        assert result.citations[0].label == "[Retention Playbook, Risk Tiers]"
        assert result.evidence[0].type == "document"


class TestMultiSourceRoute:
    def test_combines_evidence_from_multiple_tools(self, monkeypatch):
        monkeypatch.setattr(graph_module, "run_sql", lambda q: SQLQueryResult(
            sql=q, columns=["contract"], rows=[["month_to_month"]], row_count=1,
            truncated=False, execution_time_ms=1.0,
        ))
        monkeypatch.setattr(graph_module, "hybrid_search", lambda query, top_k_final=3: [
            RetrievalCandidate(
                document_id="policies/discount_and_offer_policy.md", chunk_id="c1",
                text="Standard offers can be approved by any team member.",
                title="Discount and Offer Policy", section=None, score=0.8, rank=1,
                retrieval_method="hybrid_reranked",
            ),
        ])
        fake = _FakeProvider(responses=["SELECT contract FROM marts.customer_360 LIMIT 10", "Combined answer."])
        monkeypatch.setattr(graph_module, "get_llm_provider", lambda: fake)

        result = graph_module.run_query(
            "Compare churn rate by contract to what our retention policy says"
        )

        assert result.route == "MULTI_SOURCE"
        assert set(result.tools_used) == {"sql_tool", "hybrid_search"}
        assert len(result.evidence) == 2


class TestGenerationFallback:
    def test_llm_failure_falls_back_to_a_templated_evidence_summary(self, monkeypatch):
        monkeypatch.setattr(graph_module, "churn_analysis", lambda filters=None: ChurnAnalysisResult(
            population_size=1000, current_churn_rate=0.1, predicted_high_risk_count=120,
            mean_churn_probability=0.15, median_churn_probability=0.12,
            model_version="V2", threshold=0.11, filters_applied={},
        ))
        monkeypatch.setattr(graph_module, "get_llm_provider", lambda: _FakeProvider(raise_on_call=True))

        result = graph_module.run_query("What are the biggest risk factors for churn?")

        assert result.answer.startswith("Based on the available evidence:")
        assert result.evidence  # evidence is still returned even though prose generation failed


class TestTracing:
    def test_unsupported_route_writes_a_trace(self, captured_traces):
        graph_module.run_query("tell me tomorrow's weather")

        assert len(captured_traces) == 1
        trace = captured_traces[0]
        assert trace["route"] == "UNSUPPORTED"
        assert trace["validation_result"] == "unsupported"
        assert trace["tools_used"] == []
        assert trace["fallback_status"] is False

    def test_insufficient_evidence_writes_a_trace_with_no_llm_usage(self, monkeypatch, captured_traces):
        monkeypatch.setattr(graph_module, "customer_lookup", lambda cid: CustomerLookupResult(customer_id=cid, found=False))

        graph_module.run_query("Show customer CUST000999")

        trace = captured_traces[0]
        assert trace["validation_result"] == "insufficient_evidence"
        assert trace["llm_model"] is None
        assert trace["input_tokens"] == 0
        assert trace["output_tokens"] == 0

    def test_grounded_answer_records_tool_latency_tokens_and_model(self, monkeypatch, captured_traces):
        monkeypatch.setattr(graph_module, "churn_analysis", lambda filters=None: ChurnAnalysisResult(
            population_size=1000, current_churn_rate=0.1, predicted_high_risk_count=120,
            mean_churn_probability=0.15, median_churn_probability=0.12,
            model_version="V2", threshold=0.11, filters_applied={},
        ))
        fake = _FakeProvider(responses=["Churn risk is concentrated among a subset of customers."])
        monkeypatch.setattr(graph_module, "get_llm_provider", lambda: fake)

        graph_module.run_query("What are the biggest risk factors for churn?")

        trace = captured_traces[0]
        assert "churn_analysis" in trace["tool_latency_ms"]
        assert trace["tool_latency_ms"]["churn_analysis"] >= 0
        assert trace["llm_model"] == "fake-model"
        assert trace["input_tokens"] == 5  # from _FakeProvider's fixed AgentResponse
        assert trace["output_tokens"] == 5
        assert trace["validation_result"] == "grounded"
        assert trace["fallback_status"] is False
        assert trace["error"] is None

    def test_ungrounded_answer_marks_fallback_in_the_trace(self, monkeypatch, captured_traces):
        monkeypatch.setattr(graph_module, "churn_analysis", lambda filters=None: ChurnAnalysisResult(
            population_size=1000, current_churn_rate=0.1, predicted_high_risk_count=120,
            mean_churn_probability=0.15, median_churn_probability=0.12,
            model_version="V2", threshold=0.11, filters_applied={},
        ))
        fake = _FakeProvider(responses=["The probability is 999999, way outside anything in evidence."])
        monkeypatch.setattr(graph_module, "get_llm_provider", lambda: fake)

        graph_module.run_query("What are the biggest risk factors for churn?")

        trace = captured_traces[0]
        assert trace["validation_result"] == "ungrounded_fallback"
        assert trace["fallback_status"] is True

    def test_llm_call_failure_marks_fallback_in_the_trace(self, monkeypatch, captured_traces):
        monkeypatch.setattr(graph_module, "churn_analysis", lambda filters=None: ChurnAnalysisResult(
            population_size=1000, current_churn_rate=0.1, predicted_high_risk_count=120,
            mean_churn_probability=0.15, median_churn_probability=0.12,
            model_version="V2", threshold=0.11, filters_applied={},
        ))
        monkeypatch.setattr(graph_module, "get_llm_provider", lambda: _FakeProvider(raise_on_call=True))

        graph_module.run_query("What are the biggest risk factors for churn?")

        trace = captured_traces[0]
        assert trace["fallback_status"] is True
        assert trace["llm_model"] is None  # the failed call produced no AgentResponse to read a model from

    def test_a_failing_tool_is_recorded_as_an_error_but_does_not_crash(self, monkeypatch, captured_traces):
        def _raises(filters=None):
            raise RuntimeError("simulated DB outage")

        monkeypatch.setattr(graph_module, "churn_analysis", _raises)
        monkeypatch.setattr(graph_module, "get_llm_provider", lambda: _FakeProvider())

        result = graph_module.run_query("What are the biggest risk factors for churn?")

        assert result.answer == graph_module.INSUFFICIENT_EVIDENCE_ANSWER
        trace = captured_traces[0]
        assert "churn_analysis: simulated DB outage" in trace["error"]

    def test_sql_result_hashes_the_generated_query_not_the_raw_text(self, monkeypatch, captured_traces):
        monkeypatch.setattr(graph_module, "run_sql", lambda q: SQLQueryResult(
            sql=q, columns=["contract"], rows=[["month_to_month"]], row_count=1,
            truncated=False, execution_time_ms=1.0,
        ))
        fake = _FakeProvider(responses=["SELECT contract FROM marts.customer_360 LIMIT 10", "Answer."])
        monkeypatch.setattr(graph_module, "get_llm_provider", lambda: fake)

        graph_module.run_query("How many customers do we have on average per segment?")

        trace = captured_traces[0]
        assert trace["sql_query_hash"] is not None
        assert trace["sql_query_hash"] != "SELECT contract FROM marts.customer_360 LIMIT 10"
