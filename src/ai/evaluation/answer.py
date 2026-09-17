"""Answer-quality metrics (AI_Customer_Intelligence_Claude_Code_Plan.md
section 20).

Implemented as lightweight, inspectable proxies in plain Python rather
than via RAGAS/DeepEval: those frameworks add their own LLM-as-judge
machinery and a large dependency tree, which conflicts with this
project's consistent "avoid unnecessary infrastructure" choices
elsewhere (no LangChain, no heavy embedding dependency in the trimmed
environments - see src/ai/rag/embeddings.py). Every function below
documents exactly what it approximates and how, so a reader can judge
whether the approximation is good enough rather than trusting an opaque
score - this is explicitly a "small manually reviewed golden set" tool
(section 20), not a claim of automated, ground-truth-grade evaluation.
"""

from __future__ import annotations

from src.ai.guardrails.citation_validation import validate_citations
from src.ai.guardrails.grounding import extract_numbers, find_unsupported_numbers
from src.ai.schemas import AssistantResponse, BenchmarkQuestion

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "who",
    "why", "how", "does", "do", "did", "our", "this", "that", "for", "of",
    "in", "on", "to", "and", "or", "with", "by", "at",
}


def _keywords(text: str) -> set[str]:
    return {w.strip("?.,!'\"").lower() for w in text.split()} - _STOPWORDS


def tool_selection_accuracy(predicted_tools: list[str], expected_tools: list[str]) -> float:
    """Jaccard overlap between the tools actually used and the tools the
    benchmark question expected - 1.0 only on an exact set match, partial
    credit for a partial overlap, 1.0 when both are empty (an
    UNSUPPORTED question expecting no tools, correctly using none)."""
    predicted, expected = set(predicted_tools), set(expected_tools)
    union = predicted | expected
    if not union:
        return 1.0
    return len(predicted & expected) / len(union)


def citation_correctness(response: AssistantResponse) -> float:
    """Fraction of the response's citations that are actually backed by
    its evidence (reuses the same check the citation guardrail applies
    at request time). 1.0 when there are no citations to check."""
    if not response.citations:
        return 1.0
    valid = validate_citations(response.citations, response.evidence)
    return len(valid) / len(response.citations)


def unsupported_claim_rate(response: AssistantResponse) -> float:
    """Fraction of the numeric claims in the answer that aren't
    traceable to evidence. 0.0 when the answer has no numeric claims at
    all - there's nothing to be unsupported."""
    answer_numbers = extract_numbers(response.answer)
    if not answer_numbers:
        return 0.0
    unsupported = find_unsupported_numbers(response.answer, response.evidence)
    return len(unsupported) / len(answer_numbers)


def faithfulness_proxy(response: AssistantResponse) -> float:
    """1 - unsupported_claim_rate. A proxy, not a real NLI-based
    faithfulness score (RAGAS-style faithfulness checks each claim
    against context with an LLM judge) - this only checks numeric
    claims, so a fabricated *qualitative* claim built from real numbers
    would not be caught here. A documented gap, not assumed away."""
    return 1.0 - unsupported_claim_rate(response)


def answer_relevancy_proxy(question: str, response: AssistantResponse) -> float:
    """Lexical (keyword-overlap) proxy for whether the answer engages
    with the question's own terms - not a semantic-similarity score.
    Cheap and explainable, but under-scores a correct answer that
    paraphrases rather than reuses the question's vocabulary."""
    q_words = _keywords(question)
    if not q_words:
        return 0.0
    a_words = _keywords(response.answer)
    return len(q_words & a_words) / len(q_words)


def context_precision(response: AssistantResponse, expected_sources: list[str]) -> float:
    """Fraction of the response's evidence whose source matches one of
    the benchmark question's expected_sources. 1.0 when no evidence was
    gathered and none was expected either; 0.0 when evidence exists but
    none was expected (or vice versa)."""
    if not response.evidence:
        return 1.0 if not expected_sources else 0.0
    if not expected_sources:
        return 0.0
    matches = sum(1 for e in response.evidence if any(s in e.source for s in expected_sources))
    return matches / len(response.evidence)


def context_recall(response: AssistantResponse, expected_documents: list[str]) -> float:
    """Fraction of the benchmark question's expected_documents that
    actually appear among the response's evidence sources. 1.0 when no
    documents were expected at all (a non-RAG question, or a
    multi-source question where a specific document wasn't pinned as
    ground truth - see src/ai/evaluation/datasets.py's note on this)."""
    if not expected_documents:
        return 1.0
    retrieved_sources = {e.source for e in response.evidence}
    hits = sum(1 for doc in expected_documents if doc in retrieved_sources)
    return hits / len(expected_documents)


def score_response(question: BenchmarkQuestion, response: AssistantResponse) -> dict:
    """All per-question metrics in one dict - what
    src/ai/evaluation/benchmark.py accumulates and averages across the
    full dataset."""
    return {
        "id": question.id,
        "route_correct": response.route == question.expected_route,
        "tool_selection_accuracy": tool_selection_accuracy(response.tools_used, question.expected_tools),
        "citation_correctness": citation_correctness(response),
        "unsupported_claim_rate": unsupported_claim_rate(response),
        "faithfulness": faithfulness_proxy(response),
        "answer_relevancy": answer_relevancy_proxy(question.question, response),
        "context_precision": context_precision(response, question.expected_sources),
        "context_recall": context_recall(response, question.expected_documents),
        "latency_ms": response.latency_ms,
    }
