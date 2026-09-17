"""Tests for src/ai/evaluation/datasets.py: dataset composition, and a
genuinely useful side effect - measuring the rule-based router's
(src/ai/router.py) real agreement rate against the dataset's
expected_route labels. Both are pure/offline: no DB, no LLM, no RAG
corpus needed (the router is deterministic text logic).
"""

from __future__ import annotations

from collections import Counter

from src.ai.evaluation.datasets import load_benchmark_dataset
from src.ai.router import classify


def test_dataset_has_at_least_100_questions():
    dataset = load_benchmark_dataset()
    assert len(dataset) >= 100


def test_dataset_has_exactly_120_questions():
    assert len(load_benchmark_dataset()) == 120


def test_category_counts_match_the_plans_breakdown():
    dataset = load_benchmark_dataset()
    counts = Counter(q.expected_route for q in dataset)
    assert counts["SQL_ANALYSIS"] == 30
    assert counts["CUSTOMER_LOOKUP"] == 20
    assert counts["ML_ANALYSIS"] == 20
    assert counts["RAG_SEARCH"] == 20
    assert counts["MULTI_SOURCE"] == 20
    assert counts["UNSUPPORTED"] == 10


def test_every_question_has_a_unique_id():
    dataset = load_benchmark_dataset()
    ids = [q.id for q in dataset]
    assert len(ids) == len(set(ids))


def test_every_question_has_nonempty_text():
    assert all(q.question.strip() for q in load_benchmark_dataset())


def test_rag_questions_reference_a_real_knowledge_document():
    from src.ai.rag.ingest import discover_documents, KNOWLEDGE_DIR

    real_doc_ids = {
        str(p.relative_to(KNOWLEDGE_DIR)).replace("\\", "/") for p in discover_documents()
    }
    rag_questions = [q for q in load_benchmark_dataset() if q.expected_route == "RAG_SEARCH"]
    assert rag_questions  # sanity: the filter itself isn't accidentally empty
    for q in rag_questions:
        assert set(q.expected_documents) <= real_doc_ids


def test_router_agreement_rate_is_reasonably_high():
    """Runs the real (offline, LLM-free) router against every question's
    expected_route. This *is* one of the plan's own metrics (section 21's
    "Tool selection accuracy" / route accuracy), computed here without
    needing live infra since routing is pure text logic. Not asserting
    100%: a heuristic router mismatching some genuinely ambiguous,
    hand-authored questions is an honest, expected outcome, not a bug -
    see src/ai/router.py's own docstring. The floor here only guards
    against a wholesale routing regression.
    """
    dataset = load_benchmark_dataset()
    correct = sum(1 for q in dataset if classify(q.question) == q.expected_route)
    agreement = correct / len(dataset)
    assert agreement >= 0.85, f"router agreement dropped to {agreement:.1%} - see mismatches below"
