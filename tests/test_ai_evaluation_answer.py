"""Tests for src/ai/evaluation/answer.py's answer-quality metrics.
Pure functions over AssistantResponse/BenchmarkQuestion objects - no
DB/LLM."""

from __future__ import annotations

from src.ai.evaluation.answer import (
    answer_relevancy_proxy,
    citation_correctness,
    context_precision,
    context_recall,
    faithfulness_proxy,
    score_response,
    tool_selection_accuracy,
    unsupported_claim_rate,
)
from src.ai.schemas import AssistantResponse, BenchmarkQuestion, Citation, Evidence


def _response(**overrides):
    base = dict(answer="answer", citations=[], evidence=[], tools_used=[], route="ML_ANALYSIS", trace_id="t")
    base.update(overrides)
    return AssistantResponse(**base)


class TestToolSelectionAccuracy:
    def test_exact_match_is_one(self):
        assert tool_selection_accuracy(["sql_tool"], ["sql_tool"]) == 1.0

    def test_partial_overlap_is_jaccard(self):
        assert tool_selection_accuracy(["sql_tool", "churn_analysis"], ["sql_tool"]) == 1 / 2

    def test_both_empty_is_one(self):
        assert tool_selection_accuracy([], []) == 1.0

    def test_no_overlap_is_zero(self):
        assert tool_selection_accuracy(["sql_tool"], ["hybrid_search"]) == 0.0


class TestCitationCorrectness:
    def test_no_citations_is_one(self):
        assert citation_correctness(_response(citations=[])) == 1.0

    def test_all_valid_is_one(self):
        evidence = [Evidence(type="database", source="s", claim="c", value="v")]
        citations = [Citation(label="[X]", type="database", source="s")]
        assert citation_correctness(_response(evidence=evidence, citations=citations)) == 1.0

    def test_half_invalid_is_half(self):
        evidence = [Evidence(type="database", source="s", claim="c", value="v")]
        citations = [
            Citation(label="[X]", type="database", source="s"),
            Citation(label="[Y]", type="database", source="fabricated"),
        ]
        assert citation_correctness(_response(evidence=evidence, citations=citations)) == 0.5


class TestGroundingProxies:
    def test_unsupported_claim_rate_zero_with_no_numbers(self):
        assert unsupported_claim_rate(_response(answer="no numbers here")) == 0.0

    def test_unsupported_claim_rate_flags_a_fabricated_number(self):
        evidence = [Evidence(type="model", source="s", claim="c", value="42")]
        response = _response(answer="The value is 99.", evidence=evidence)
        assert unsupported_claim_rate(response) == 1.0

    def test_faithfulness_is_the_complement(self):
        response = _response(answer="no numbers here")
        assert faithfulness_proxy(response) == 1.0


class TestAnswerRelevancyProxy:
    def test_full_keyword_overlap_is_one(self):
        response = _response(answer="churn rate by contract")
        assert answer_relevancy_proxy("What is the churn rate by contract?", response) == 1.0

    def test_no_overlap_is_zero(self):
        response = _response(answer="completely unrelated text")
        assert answer_relevancy_proxy("What is the churn rate?", response) == 0.0


class TestContextPrecisionRecall:
    def test_context_precision_all_matching_sources(self):
        evidence = [Evidence(type="database", source="marts.customer_360", claim="c", value="v")]
        assert context_precision(_response(evidence=evidence), ["marts.customer_360"]) == 1.0

    def test_context_precision_no_evidence_and_none_expected(self):
        assert context_precision(_response(evidence=[]), []) == 1.0

    def test_context_precision_no_evidence_but_some_expected(self):
        assert context_precision(_response(evidence=[]), ["marts.customer_360"]) == 0.0

    def test_context_recall_all_expected_documents_present(self):
        evidence = [Evidence(type="document", source="retention/retention_playbook.md", claim="c", value="v")]
        result = context_recall(_response(evidence=evidence), ["retention/retention_playbook.md"])
        assert result == 1.0

    def test_context_recall_missing_document_is_partial(self):
        evidence = [Evidence(type="document", source="retention/retention_playbook.md", claim="c", value="v")]
        result = context_recall(
            _response(evidence=evidence),
            ["retention/retention_playbook.md", "policies/discount_and_offer_policy.md"],
        )
        assert result == 0.5

    def test_context_recall_no_documents_expected_is_one(self):
        assert context_recall(_response(evidence=[]), []) == 1.0


def test_score_response_returns_all_expected_keys():
    question = BenchmarkQuestion(id="Q1", question="What is the churn rate?", expected_route="ML_ANALYSIS")
    response = _response(answer="The churn rate is stable.", route="ML_ANALYSIS")

    scores = score_response(question, response)

    assert scores["id"] == "Q1"
    assert scores["route_correct"] is True
    assert set(scores) >= {
        "tool_selection_accuracy", "citation_correctness", "unsupported_claim_rate",
        "faithfulness", "answer_relevancy", "context_precision", "context_recall", "latency_ms",
    }
