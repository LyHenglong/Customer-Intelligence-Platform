"""Tests for src/ai/evaluation/benchmark.py's orchestration (run_query is
monkeypatched throughout - this proves the loop/aggregation/failure
handling, not real model quality, which only a live run can produce)."""

from __future__ import annotations

from src.ai.evaluation import benchmark as benchmark_module
from src.ai.schemas import AssistantResponse, BenchmarkQuestion


def _question(id_, route="ML_ANALYSIS"):
    return BenchmarkQuestion(id=id_, question=f"question {id_}", expected_route=route, expected_tools=["churn_analysis"])


def test_run_benchmark_scores_every_question(monkeypatch):
    questions = [_question("Q1"), _question("Q2")]

    def _fake_run_query(query):
        return AssistantResponse(
            answer="ok", route="ML_ANALYSIS", trace_id="t", tools_used=["churn_analysis"], latency_ms=10.0,
        )

    monkeypatch.setattr(benchmark_module, "run_query", _fake_run_query)

    summary = benchmark_module.run_benchmark(questions=questions)

    assert summary["queries"] == 2
    assert summary["route_accuracy"] == 1.0
    assert summary["tool_selection_accuracy"] == 1.0


def test_run_benchmark_scores_a_raised_exception_as_a_full_failure(monkeypatch):
    questions = [_question("Q1")]

    def _raises(query):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(benchmark_module, "run_query", _raises)

    summary = benchmark_module.run_benchmark(questions=questions)

    assert summary["queries"] == 1
    assert summary["route_accuracy"] == 0.0
    assert summary["faithfulness"] == 0.0


def test_run_benchmark_respects_limit(monkeypatch):
    questions = [_question(f"Q{i}") for i in range(10)]
    monkeypatch.setattr(
        benchmark_module, "run_query",
        lambda query: AssistantResponse(answer="ok", route="ML_ANALYSIS", trace_id="t", latency_ms=1.0),
    )

    summary = benchmark_module.run_benchmark(limit=3, questions=questions)

    assert summary["queries"] == 3


def test_summarize_empty_scores():
    assert benchmark_module.summarize([], 0.0) == {"queries": 0}


def test_summarize_computes_median_and_p95_latency():
    scores = [
        {"id": f"Q{i}", "route_correct": True, "tool_selection_accuracy": 1.0,
         "citation_correctness": 1.0, "unsupported_claim_rate": 0.0, "faithfulness": 1.0,
         "answer_relevancy": 1.0, "context_precision": 1.0, "context_recall": 1.0, "latency_ms": float(i)}
        for i in range(1, 21)  # latencies 1..20
    ]
    summary = benchmark_module.summarize(scores, total_elapsed_s=5.0)
    assert summary["median_latency_ms"] == 10.5
    assert summary["p95_latency_ms"] == 19.0


def test_print_report_handles_zero_queries(capsys):
    benchmark_module.print_report({"queries": 0})
    captured = capsys.readouterr()
    assert "No questions were run" in captured.out


def test_print_report_prints_key_metrics(capsys):
    summary = benchmark_module.summarize(
        [{"id": "Q1", "route_correct": True, "tool_selection_accuracy": 0.9,
          "citation_correctness": 1.0, "unsupported_claim_rate": 0.1, "faithfulness": 0.9,
          "answer_relevancy": 0.8, "context_precision": 1.0, "context_recall": 1.0, "latency_ms": 50.0}],
        total_elapsed_s=1.0,
    )
    benchmark_module.print_report(summary)
    captured = capsys.readouterr()
    assert "AI ASSISTANT EVALUATION" in captured.out
    assert "Queries:" in captured.out
    assert "Median latency:" in captured.out
