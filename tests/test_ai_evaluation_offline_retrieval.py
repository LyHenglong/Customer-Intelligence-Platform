"""Retrieval-quality gate over the real knowledge/ corpus
(src/ai/evaluation/offline_retrieval.py) - no database, embeddings or LLM.

The corpus has only six documents, so hit@5 is 1.0 by construction and says
nothing; the floors below are on top-1 document accuracy and MRR instead.
They sit a little under the measured values (structure_aware, the ingestion
default: hit@1 0.75, MRR 0.854), so an edit to the corpus, chunker or
tokenizer that makes BM25 surface the wrong document fails here.
"""

from __future__ import annotations

import pytest

from src.ai.evaluation.offline_retrieval import build_local_chunks, evaluate
from src.ai.rag.ingest import DEFAULT_STRATEGY


def test_every_rag_question_is_scored():
    metrics = evaluate(DEFAULT_STRATEGY, k=1)
    assert metrics["n"] == 20


@pytest.mark.parametrize(
    "strategy,min_hit_at_1,min_mrr",
    [
        ("structure_aware", 0.70, 0.80),
        ("overlapping", 0.85, 0.90),
        ("fixed", 0.95, 0.95),
    ],
)
def test_bm25_ranks_the_expected_document_first_often_enough(strategy, min_hit_at_1, min_mrr):
    metrics = evaluate(strategy, k=1)
    assert metrics["hit_at_k"] >= min_hit_at_1, metrics
    assert metrics["mrr"] >= min_mrr, metrics


def test_local_chunks_cover_every_knowledge_document():
    docs = {c["document_id"] for c in build_local_chunks(DEFAULT_STRATEGY)}
    assert len(docs) == 6
    assert all(c["text"].strip() for c in build_local_chunks(DEFAULT_STRATEGY))
