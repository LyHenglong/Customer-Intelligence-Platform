"""Retrieval-quality metrics (AI_Customer_Intelligence_Claude_Code_Plan.md
section 19): Recall@K, Precision@K, Hit@K, MRR, nDCG@K. Pure functions
over a ranked list of ids and a set of relevant ids - no dependency on
how those ids were produced, so the same functions score BM25-only,
vector-only, hybrid, and hybrid+reranked runs identically.

compare_retrieval_methods() is what actually runs each of the four
methods for a real query (needs a live Postgres + ingested RAG corpus +
embedding model - see src/ai/evaluation/benchmark.py); the metric
functions above it are pure math and need none of that.
"""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = len(set(retrieved[:k]) & relevant)
    return hit / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hit = len(set(retrieved[:k]) & relevant)
    return hit / k


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(runs: list[tuple[list[str], set[str]]]) -> float:
    if not runs:
        return 0.0
    return sum(reciprocal_rank(retrieved, relevant) for retrieved, relevant in runs) / len(runs)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k (no graded relevance in this project's
    ground truth, so gain is 0/1)."""
    dcg = sum(
        1.0 / math.log2(i + 1) for i, doc_id in enumerate(retrieved[:k], start=1) if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate_metrics(runs: list[tuple[list[str], set[str]]], k: int = 5) -> dict:
    """runs: list of (retrieved_ids, relevant_ids) pairs, one per query."""
    if not runs:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0, "hit_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0, "n": 0, "k": k}
    n = len(runs)
    return {
        "recall_at_k": sum(recall_at_k(r, rel, k) for r, rel in runs) / n,
        "precision_at_k": sum(precision_at_k(r, rel, k) for r, rel in runs) / n,
        "hit_at_k": sum(hit_at_k(r, rel, k) for r, rel in runs) / n,
        "mrr": mean_reciprocal_rank(runs),
        "ndcg_at_k": sum(ndcg_at_k(r, rel, k) for r, rel in runs) / n,
        "n": n,
        "k": k,
    }


def compare_retrieval_methods(query: str, top_k: int = 5) -> dict[str, list[str]]:
    """Runs the same query through each retrieval method
    (AI_Customer_Intelligence_Claude_Code_Plan.md section 19's
    comparison: vector only / BM25 only / hybrid / hybrid + reranker)
    and returns each method's ranked document_id list. Requires a live
    Postgres connection with an ingested RAG corpus and (for the
    embedding/reranker calls) the local models actually loaded - not
    mockable into a meaningful result, only into a wiring test (see
    tests/test_ai_evaluation_retrieval.py)."""
    from src.ai.rag.bm25 import bm25_search
    from src.ai.rag.embeddings import embed_query
    from src.ai.rag.hybrid_search import hybrid_search
    from src.ai.rag.vector_store import vector_search

    bm25_only = [c["document_id"] for c in bm25_search(query, top_k=top_k)]
    vector_only = [c["document_id"] for c in vector_search(embed_query(query), top_k=top_k)]
    hybrid = [c.document_id for c in hybrid_search(query, top_k_final=top_k, use_reranker=False)]
    hybrid_reranked = [c.document_id for c in hybrid_search(query, top_k_final=top_k, use_reranker=True)]

    return {
        "bm25_only": bm25_only,
        "vector_only": vector_only,
        "hybrid": hybrid,
        "hybrid_reranked": hybrid_reranked,
    }


def score_retrieval_methods(results: dict[str, list[str]], relevant: set[str], k: int = 5) -> dict[str, dict]:
    return {method: aggregate_metrics([(ids, relevant)], k=k) for method, ids in results.items()}
