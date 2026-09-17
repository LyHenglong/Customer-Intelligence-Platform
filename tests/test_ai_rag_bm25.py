"""Tests for src/ai/rag/bm25.py. fetch_all_chunks_for_bm25 (the DB read)
is monkeypatched - this exercises BM25Okapi wiring and the lru_cache
reload behavior, not the database.
"""

from __future__ import annotations

from src.ai.rag import bm25 as bm25_module


def _chunk(chunk_id, text):
    return {
        "chunk_id": chunk_id, "document_id": "doc1", "title": "T", "source": "doc1",
        "section": None, "page": None, "text": text, "metadata": {},
    }


def test_bm25_search_ranks_the_more_relevant_chunk_first(monkeypatch):
    # 4 distractor-shaped docs, not 2: with rank_bm25's default (non-smoothed)
    # IDF, a term appearing in exactly half of a 2-document corpus scores an
    # exact 0.0 IDF (log(1.5) - log(1.5) == 0), which masks the ranking
    # signal this test wants to observe rather than exercising a bug.
    chunks = [
        _chunk("A", "retention offers for month to month contract customers"),
        _chunk("B", "streaming movies and streaming tv service catalog"),
        _chunk("C", "device protection and online backup add-on pricing"),
        _chunk("D", "support escalation tiers and complaint handling"),
    ]
    monkeypatch.setattr(bm25_module, "fetch_all_chunks_for_bm25", lambda: chunks)
    bm25_module.reload_bm25_index()

    results = bm25_module.bm25_search("retention offer contract", top_k=4)

    assert results[0]["chunk_id"] == "A"
    assert results[0]["score"] > results[1]["score"]


def test_bm25_search_returns_empty_list_when_no_chunks_indexed(monkeypatch):
    monkeypatch.setattr(bm25_module, "fetch_all_chunks_for_bm25", lambda: [])
    bm25_module.reload_bm25_index()

    assert bm25_module.bm25_search("anything") == []


def test_reload_bm25_index_picks_up_new_chunks(monkeypatch):
    monkeypatch.setattr(bm25_module, "fetch_all_chunks_for_bm25", lambda: [_chunk("A", "alpha content")])
    bm25_module.reload_bm25_index()
    first = bm25_module.bm25_search("alpha")
    assert [r["chunk_id"] for r in first] == ["A"]

    monkeypatch.setattr(bm25_module, "fetch_all_chunks_for_bm25", lambda: [_chunk("B", "beta content")])
    bm25_module.reload_bm25_index()
    second = bm25_module.bm25_search("beta")
    assert [r["chunk_id"] for r in second] == ["B"]


def test_top_k_limits_result_count(monkeypatch):
    chunks = [_chunk(f"C{i}", f"content number {i}") for i in range(10)]
    monkeypatch.setattr(bm25_module, "fetch_all_chunks_for_bm25", lambda: chunks)
    bm25_module.reload_bm25_index()

    results = bm25_module.bm25_search("content", top_k=3)
    assert len(results) == 3
