"""Tests for src/ai/guardrails/validation.py's validate_response() - the
"Validate" step applied after generation in src/ai/graph.py."""

from __future__ import annotations

from src.ai.guardrails.validation import validate_response
from src.ai.schemas import AssistantResponse, Citation, Evidence


def _response(answer, evidence=None, citations=None):
    return AssistantResponse(
        answer=answer, evidence=evidence or [], citations=citations or [],
        tools_used=["x"], route="ML_ANALYSIS", trace_id="t1",
    )


def test_grounded_answer_passes_through_with_full_confidence():
    evidence = [Evidence(type="model", source="churn model V1", claim="stat", value="probability=42")]
    citations = [Citation(label="[Churn Model V1]", type="model", source="churn model V1")]
    response = _response("The probability is 42.", evidence, citations)

    result = validate_response(response)

    assert result.answer == "The probability is 42."
    assert result.confidence == 1.0
    assert result.citations == citations


def test_ungrounded_answer_falls_back_to_templated_summary():
    evidence = [Evidence(type="model", source="churn model V1", claim="stat", value="probability=42")]
    response = _response("The probability is 99, way higher than normal.", evidence)

    result = validate_response(response)

    assert result.answer.startswith("Based on the available evidence:")
    assert "probability=42" in result.answer
    assert result.confidence == 0.5


def test_drops_orphan_citations_even_on_a_grounded_answer():
    evidence = [Evidence(type="model", source="churn model V1", claim="stat", value="no numbers")]
    citations = [
        Citation(label="[Real]", type="model", source="churn model V1"),
        Citation(label="[Fabricated]", type="document", source="nonexistent/doc.md"),
    ]
    response = _response("No numeric claims here.", evidence, citations)

    result = validate_response(response)

    assert len(result.citations) == 1
    assert result.citations[0].source == "churn model V1"
