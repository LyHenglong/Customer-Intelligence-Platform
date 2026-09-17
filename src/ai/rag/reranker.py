"""Cross-encoder reranking of the RRF-fused candidate pool
(src/ai/rag/hybrid_search.py).

A cross-encoder scores (query, passage) pairs jointly rather than
comparing independently-computed embeddings, so it is slower but more
accurate than the bi-encoder/BM25 retrieval that produced the candidate
pool - used only on the already-small fused shortlist (~20 candidates),
never the full corpus.
"""

from __future__ import annotations

import os
from functools import lru_cache

RERANKER_MODEL_NAME = os.environ.get("AI_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL_NAME)


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """candidates: dicts with at least a "text" key. Returns the top_k
    candidates, each with "score" replaced by the cross-encoder's score
    (overwriting whatever fusion score it carried in)."""
    if not candidates:
        return []
    model = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)[:top_k]
    return [{**c, "score": float(s)} for c, s in ranked]
