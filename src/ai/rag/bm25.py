"""In-memory BM25 sparse retrieval over the same chunks stored in
pgvector (src/ai/rag/vector_store.py). BM25 has no persistent index of
its own in this project - rebuilding it from the (corpus-sized, not
warehouse-sized) chunk table is cheap and keeps one source of truth
rather than a second store that could drift from public.rag_chunks.
"""

from __future__ import annotations

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from src.ai.rag.vector_store import fetch_all_chunks_for_bm25

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def _load_index():
    chunks = fetch_all_chunks_for_bm25()
    corpus_tokens = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None
    return bm25, chunks


def reload_bm25_index() -> None:
    """Forces the next bm25_search() call to rebuild from the table's
    current contents. Call after ingestion (src/ai/rag/ingest.py) and in
    tests that swap the underlying chunk source between cases - lru_cache
    would otherwise pin whichever chunks were indexed on the first call
    for the rest of the process."""
    _load_index.cache_clear()


def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    bm25, chunks = _load_index()
    if bm25 is None or not chunks:
        return []
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in ranked]
