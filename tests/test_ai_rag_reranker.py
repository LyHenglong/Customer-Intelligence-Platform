"""Tests for src/ai/rag/reranker.py's cached cross-encoder loader.

No real model is ever loaded here - CrossEncoder is monkeypatched. What
these pin is the caching contract, which is where a real outage came
from: a half-initialised model got memoized and disabled RAG retrieval
for the whole process lifetime.
"""

from __future__ import annotations

import pytest

from src.ai.rag import reranker as reranker_module


@pytest.fixture(autouse=True)
def _clear_loader_cache():
    # getattr guard: some tests below replace _get_reranker with a plain
    # callable, which has no cache_clear.
    def _clear():
        clear = getattr(reranker_module._get_reranker, "cache_clear", None)
        if clear:
            clear()

    _clear()
    yield
    _clear()


class _FakeCrossEncoder:
    """Mimics the real failure: constructing is fine, predicting is not.

    That asymmetry is the whole bug - CrossEncoder's constructor returns
    successfully even when the weights were left on the meta device, and
    only predict() raises.
    """

    instances = 0
    # Shared across instances on purpose: the real failure is a property of
    # the process (memory pressure at load time), not of one object, so a
    # retry must be able to succeed where the first attempt failed.
    remaining_failures = 0

    def __init__(self, name):
        type(self).instances += 1
        self.name = name

    def predict(self, pairs):
        if type(self).remaining_failures > 0:
            type(self).remaining_failures -= 1
            raise NotImplementedError("Cannot copy out of meta tensor; no data!")
        return [0.5] * len(pairs)


def _install(monkeypatch, *, fail_predict_times):
    _FakeCrossEncoder.instances = 0
    _FakeCrossEncoder.remaining_failures = fail_predict_times
    import sentence_transformers

    monkeypatch.setattr(
        sentence_transformers, "CrossEncoder", lambda name, **kw: _FakeCrossEncoder(name),
    )


def test_a_model_that_cannot_predict_is_not_cached(monkeypatch):
    """Regression: the loader used to return straight from the constructor,
    so a model whose weights never materialised was memoized and every
    later rerank() reused it. RAG retrieval stayed broken until the process
    restarted, while the assistant kept answering without any retrieved
    evidence."""
    _install(monkeypatch, fail_predict_times=99)

    with pytest.raises(NotImplementedError):
        reranker_module._get_reranker()

    assert reranker_module._get_reranker.cache_info().currsize == 0


def test_a_failed_load_can_be_retried_successfully(monkeypatch):
    """Because nothing was cached, a later call gets a clean attempt - which
    matters because the original trigger was transient memory pressure."""
    _install(monkeypatch, fail_predict_times=1)

    with pytest.raises(NotImplementedError):
        reranker_module._get_reranker()

    model = reranker_module._get_reranker()
    assert model.predict([("q", "d")]) == [0.5]
    assert _FakeCrossEncoder.instances == 2  # a fresh object, not the poisoned one


def test_a_working_model_is_loaded_once_and_reused(monkeypatch):
    _install(monkeypatch, fail_predict_times=0)

    first = reranker_module._get_reranker()
    second = reranker_module._get_reranker()

    assert first is second
    assert _FakeCrossEncoder.instances == 1


def test_rerank_orders_by_score_and_truncates(monkeypatch):
    class _Scorer:
        def predict(self, pairs):
            # Score by passage length so the ordering is deterministic.
            return [len(text) for _, text in pairs]

    monkeypatch.setattr(reranker_module, "_get_reranker", lambda: _Scorer())

    candidates = [{"text": "short"}, {"text": "much longer passage"}, {"text": "mid size"}]
    out = reranker_module.rerank("q", candidates, top_k=2)

    assert [c["text"] for c in out] == ["much longer passage", "mid size"]
    assert out[0]["score"] == float(len("much longer passage"))


def test_rerank_short_circuits_on_no_candidates(monkeypatch):
    def _must_not_load():
        raise AssertionError("model loaded for an empty candidate list")

    monkeypatch.setattr(reranker_module, "_get_reranker", _must_not_load)

    assert reranker_module.rerank("q", []) == []
