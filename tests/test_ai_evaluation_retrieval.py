"""Tests for src/ai/evaluation/retrieval.py. The metric functions are
pure math, tested directly. compare_retrieval_methods() is a wiring test
only (bm25_search/vector_search/hybrid_search/embed_query monkeypatched
at their source modules) - it proves the four methods are called and
their results shaped correctly, not real retrieval quality."""

from __future__ import annotations

from src.ai.evaluation.retrieval import (
    aggregate_metrics,
    compare_retrieval_methods,
    hit_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval_methods,
)


class TestPerQueryMetrics:
    def test_recall_at_k(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b", "x"}, k=3) == 2 / 3

    def test_recall_at_k_empty_relevant_set(self):
        assert recall_at_k(["a"], set(), k=3) == 0.0

    def test_precision_at_k(self):
        assert precision_at_k(["a", "b", "c"], {"a"}, k=3) == 1 / 3

    def test_hit_at_k_true_and_false(self):
        assert hit_at_k(["a", "b"], {"b"}, k=2) == 1.0
        assert hit_at_k(["a", "b"], {"z"}, k=2) == 0.0

    def test_reciprocal_rank_finds_first_match(self):
        assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3

    def test_reciprocal_rank_no_match(self):
        assert reciprocal_rank(["a", "b"], {"z"}) == 0.0

    def test_ndcg_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_ndcg_no_relevant_at_all_is_zero(self):
        assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0

    def test_ndcg_worse_ranking_scores_lower_than_ideal(self):
        ideal = ndcg_at_k(["a", "b"], {"a"}, k=2)
        worse = ndcg_at_k(["b", "a"], {"a"}, k=2)
        assert worse < ideal


class TestAggregateMetrics:
    def test_mean_reciprocal_rank_averages_across_queries(self):
        runs = [(["a"], {"a"}), (["x", "a"], {"a"})]
        assert mean_reciprocal_rank(runs) == (1.0 + 0.5) / 2

    def test_aggregate_metrics_empty_runs(self):
        result = aggregate_metrics([], k=5)
        assert result["n"] == 0

    def test_aggregate_metrics_shape(self):
        runs = [(["a", "b"], {"a"})]
        result = aggregate_metrics(runs, k=2)
        assert result["n"] == 1
        assert result["k"] == 2
        assert set(result) >= {"recall_at_k", "precision_at_k", "hit_at_k", "mrr", "ndcg_at_k"}


class TestCompareRetrievalMethods:
    def test_calls_all_four_methods_and_extracts_document_ids(self, monkeypatch):
        import src.ai.rag.bm25 as bm25_module
        import src.ai.rag.embeddings as embeddings_module
        import src.ai.rag.hybrid_search as hybrid_module
        import src.ai.rag.vector_store as vector_store_module
        from src.ai.schemas import RetrievalCandidate

        monkeypatch.setattr(bm25_module, "bm25_search", lambda q, top_k: [{"document_id": "docA"}])
        monkeypatch.setattr(vector_store_module, "vector_search", lambda emb, top_k: [{"document_id": "docB"}])
        monkeypatch.setattr(embeddings_module, "embed_query", lambda q: [0.0] * 384)

        def _fake_hybrid_search(query, top_k_final, use_reranker):
            doc = "docC" if use_reranker else "docD"
            return [RetrievalCandidate(
                document_id=doc, chunk_id="c1", text="t", title="T",
                score=1.0, rank=1, retrieval_method="hybrid",
            )]

        monkeypatch.setattr(hybrid_module, "hybrid_search", _fake_hybrid_search)

        results = compare_retrieval_methods("some question", top_k=5)

        assert results == {
            "bm25_only": ["docA"], "vector_only": ["docB"],
            "hybrid": ["docD"], "hybrid_reranked": ["docC"],
        }

    def test_score_retrieval_methods_scores_each_method_independently(self):
        results = {"bm25_only": ["a", "b"], "vector_only": ["b", "a"]}
        scored = score_retrieval_methods(results, relevant={"a"}, k=2)
        assert scored["bm25_only"]["mrr"] == 1.0  # "a" ranked first
        assert scored["vector_only"]["mrr"] == 0.5  # "a" ranked second
