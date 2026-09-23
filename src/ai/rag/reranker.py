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
    """Loads the cross-encoder and proves it can actually score before
    letting lru_cache keep it.

    CrossEncoder's constructor succeeds even when the weights never
    materialise - under memory pressure recent transformers versions leave
    them on the "meta" device, and the failure only surfaces later inside
    predict(), as "Cannot copy out of meta tensor; no data!". Without the
    probe below, lru_cache stored that half-built object and every
    subsequent rerank() reused it, so one unlucky first load disabled RAG
    retrieval for the entire life of the process - the assistant kept
    answering, just with the retrieval evidence silently missing.

    Raising instead means nothing is cached (lru_cache does not memoize an
    exception), so the next call gets a clean attempt once memory frees
    up."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(RERANKER_MODEL_NAME)
    model.predict([("warm-up query", "warm-up passage")])
    return model


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
