"""Tests for src/ai/evaluation/regression.py's stratified sampling."""

from __future__ import annotations

from src.ai.evaluation import regression as regression_module
from src.ai.evaluation.datasets import load_benchmark_dataset
from src.ai.schemas import AssistantResponse


def test_sample_questions_picks_per_route_sample_size():
    sampled = regression_module.sample_questions(per_route=2)
    routes = {q.expected_route for q in load_benchmark_dataset()}

    assert len(sampled) == 2 * len(routes)
    for route in routes:
        assert sum(1 for q in sampled if q.expected_route == route) == 2


def test_sample_questions_is_a_subset_of_the_full_dataset():
    full_ids = {q.id for q in load_benchmark_dataset()}
    sampled_ids = {q.id for q in regression_module.sample_questions()}
    assert sampled_ids <= full_ids


def test_run_regression_calls_run_benchmark_with_sampled_questions(monkeypatch):
    captured = {}

    def _fake_run_benchmark(questions=None, limit=None):
        captured["n"] = len(questions)
        return {"queries": len(questions)}

    monkeypatch.setattr(regression_module, "run_benchmark", _fake_run_benchmark)

    result = regression_module.run_regression()

    assert result["queries"] == captured["n"]
    assert captured["n"] == len(regression_module.sample_questions())
