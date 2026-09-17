"""Hybrid retrieval: fuses BM25 (sparse) and pgvector (dense) candidate
lists with Reciprocal Rank Fusion, then hands the fused pool to the
cross-encoder reranker (src/ai/rag/reranker.py).

AI_Customer_Intelligence_Claude_Code_Plan.md section 12's pipeline:
BM25 top 20 + Vector top 20 -> RRF -> top 20 -> Cross Encoder -> top 5.
"""

from __future__ import annotations

from src.ai.rag.bm25 import bm25_search
from src.ai.rag.embeddings import embed_query
from src.ai.rag.reranker import rerank
from src.ai.rag.vector_store import vector_search
from src.ai.schemas import RetrievalCandidate

RRF_K = 60  # standard smoothing constant from the original RRF paper


def rrf_fuse(*ranked_lists: list[dict], k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion: score(chunk) = sum(1 / (k + rank)) across
    every list the chunk appears in. chunk_id is the fusion key - a chunk
    retrieved by both BM25 and vector search accumulates both lists'
    contributions, ranking it above a chunk only one method found."""
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(cid, item)
    fused = sorted(by_id.values(), key=lambda item: scores[item["chunk_id"]], reverse=True)
    for item in fused:
        item["_rrf_score"] = scores[item["chunk_id"]]
    return fused


def hybrid_search(
    query: str, top_k_candidates: int = 20, top_k_final: int = 5, use_reranker: bool = True
) -> list[RetrievalCandidate]:
    """Full pipeline: BM25 top-N + vector top-N -> RRF fusion -> optional
    cross-encoder rerank -> top_k_final results with retrieval metadata
    (AI_Customer_Intelligence_Claude_Code_Plan.md section 12's citation
    shape: document_id, chunk_id, score, rank, retrieval_method)."""
    sparse = bm25_search(query, top_k=top_k_candidates)
    dense = vector_search(embed_query(query), top_k=top_k_candidates)
    fused = rrf_fuse(sparse, dense)[:top_k_candidates]

    if not fused:
        return []

    if use_reranker:
        selected = rerank(query, fused, top_k=top_k_final)
        method = "hybrid_reranked"
    else:
        selected = [{**item, "score": item["_rrf_score"]} for item in fused[:top_k_final]]
        method = "hybrid"

    return [
        RetrievalCandidate(
            document_id=item["document_id"],
            chunk_id=item["chunk_id"],
            text=item["text"],
            title=item["title"],
            section=item.get("section"),
            score=round(float(item["score"]), 4),
            rank=rank,
            retrieval_method=method,
        )
        for rank, item in enumerate(selected, start=1)
    ]
