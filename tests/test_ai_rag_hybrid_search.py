"""Tests for src/ai/rag/hybrid_search.py: Reciprocal Rank Fusion and the
full hybrid_search() wiring. bm25_search/vector_search/embed_query/rerank
are all monkeypatched - this file tests fusion and orchestration logic,
not the embedding model or the database (see test_ai_rag_chunking.py and
test_ai_rag_vector_store.py for those)."""

from __future__ import annotations

from src.ai.rag.hybrid_search import RRF_K, hybrid_search, rrf_fuse


def _cand(chunk_id, **kw):
    base = {
        "chunk_id": chunk_id, "document_id": "doc1", "title": "Doc One",
        "section": None, "text": f"text for {chunk_id}",
    }
    base.update(kw)
    return base


class TestRRFFuse:
    def test_chunk_found_by_both_lists_outranks_one_found_by_a_single_list(self):
        sparse = [_cand("A"), _cand("B")]
        dense = [_cand("B"), _cand("C")]

        fused = rrf_fuse(sparse, dense)

        assert fused[0]["chunk_id"] == "B"  # appears rank 2 in sparse, rank 1 in dense

    def test_score_matches_the_rrf_formula(self):
        sparse = [_cand("A")]
        dense = [_cand("A")]
        fused = rrf_fuse(sparse, dense)

        expected = 1 / (RRF_K + 1) + 1 / (RRF_K + 1)
        assert fused[0]["_rrf_score"] == expected

    def test_empty_lists_produce_empty_fusion(self):
        assert rrf_fuse([], []) == []

    def test_single_list_is_just_reordered_by_its_own_rank(self):
        sparse = [_cand("A"), _cand("B"), _cand("C")]
        fused = rrf_fuse(sparse)
        assert [c["chunk_id"] for c in fused] == ["A", "B", "C"]


class TestHybridSearch:
    def test_wires_bm25_vector_fusion_and_reranker_together(self, monkeypatch):
        from src.ai.rag import hybrid_search as hs

        monkeypatch.setattr(hs, "bm25_search", lambda query, top_k: [_cand("A"), _cand("B")])
        monkeypatch.setattr(hs, "vector_search", lambda emb, top_k: [_cand("B"), _cand("C")])
        monkeypatch.setattr(hs, "embed_query", lambda query: [0.0] * 384)
        monkeypatch.setattr(
            hs, "rerank",
            lambda query, candidates, top_k: [
                {**c, "score": 1.0} for c in sorted(candidates, key=lambda c: c["chunk_id"])[:top_k]
            ],
        )

        results = hs.hybrid_search("some question", top_k_final=2)

        assert len(results) == 2
        assert results[0].retrieval_method == "hybrid_reranked"
        assert [r.rank for r in results] == [1, 2]
        assert {r.chunk_id for r in results} <= {"A", "B", "C"}

    def test_skips_reranker_when_disabled(self, monkeypatch):
        from src.ai.rag import hybrid_search as hs

        monkeypatch.setattr(hs, "bm25_search", lambda query, top_k: [_cand("A")])
        monkeypatch.setattr(hs, "vector_search", lambda emb, top_k: [])
        monkeypatch.setattr(hs, "embed_query", lambda query: [0.0] * 384)

        def _rerank_should_not_be_called(*a, **kw):
            raise AssertionError("rerank must not be called when use_reranker=False")

        monkeypatch.setattr(hs, "rerank", _rerank_should_not_be_called)

        results = hs.hybrid_search("q", use_reranker=False)

        assert len(results) == 1
        assert results[0].retrieval_method == "hybrid"

    def test_no_candidates_returns_empty_list(self, monkeypatch):
        from src.ai.rag import hybrid_search as hs

        monkeypatch.setattr(hs, "bm25_search", lambda query, top_k: [])
        monkeypatch.setattr(hs, "vector_search", lambda emb, top_k: [])
        monkeypatch.setattr(hs, "embed_query", lambda query: [0.0] * 384)

        assert hs.hybrid_search("q") == []
