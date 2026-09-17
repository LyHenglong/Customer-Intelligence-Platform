"""Tests for src/ai/guardrails/citation_validation.py."""

from __future__ import annotations

from src.ai.guardrails.citation_validation import validate_citations
from src.ai.schemas import Citation, Evidence


def test_keeps_citations_backed_by_evidence():
    evidence = [Evidence(type="database", source="marts.customer_360", claim="c", value="v")]
    citations = [Citation(label="[Customer 360]", type="database", source="marts.customer_360")]

    assert validate_citations(citations, evidence) == citations


def test_drops_citations_with_no_matching_evidence():
    evidence = [Evidence(type="database", source="marts.customer_360", claim="c", value="v")]
    citations = [
        Citation(label="[Customer 360]", type="database", source="marts.customer_360"),
        Citation(label="[Fabricated Doc, p.7]", type="document", source="nonexistent/doc.md"),
    ]

    result = validate_citations(citations, evidence)

    assert len(result) == 1
    assert result[0].source == "marts.customer_360"


def test_empty_evidence_drops_all_citations():
    citations = [Citation(label="[X]", type="database", source="marts.customer_360")]
    assert validate_citations(citations, []) == []


def test_empty_citations_returns_empty():
    evidence = [Evidence(type="database", source="marts.customer_360", claim="c", value="v")]
    assert validate_citations([], evidence) == []
