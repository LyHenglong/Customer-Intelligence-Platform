"""Tests for src/ai/observability/metrics.py's pure aggregation over a
synthetic list of trace dicts - no DB."""

from __future__ import annotations

from src.ai.observability.metrics import compute_metrics


def _trace(**overrides):
    base = {
        "route": "ML_ANALYSIS", "tools_used": ["churn_analysis"], "total_latency_ms": 100.0,
        "error": None, "fallback_status": False, "retrieved_documents": [],
        "input_tokens": 10, "output_tokens": 5, "estimated_cost_usd": 0.001,
    }
    base.update(overrides)
    return base


def test_empty_trace_list():
    assert compute_metrics([]) == {"requests": 0}


def test_success_and_error_rate():
    traces = [_trace(), _trace(error="boom")]
    metrics = compute_metrics(traces)
    assert metrics["requests"] == 2
    assert metrics["success_rate"] == 0.5
    assert metrics["error_rate"] == 0.5


def test_latency_percentiles():
    traces = [_trace(total_latency_ms=float(i)) for i in range(1, 21)]
    metrics = compute_metrics(traces)
    assert metrics["average_latency_ms"] == 10.5
    assert metrics["p50_latency_ms"] == 10.5
    assert metrics["p95_latency_ms"] == 19.0


def test_tool_usage_counts_across_traces():
    traces = [
        _trace(tools_used=["churn_analysis", "hybrid_search"]),
        _trace(tools_used=["churn_analysis"]),
    ]
    metrics = compute_metrics(traces)
    assert metrics["tool_usage"] == {"churn_analysis": 2, "hybrid_search": 1}


def test_retrieval_hit_rate_only_counts_rag_traces():
    traces = [
        _trace(tools_used=["hybrid_search"], retrieved_documents=["doc1"]),
        _trace(tools_used=["hybrid_search"], retrieved_documents=[]),
        _trace(tools_used=["churn_analysis"]),  # not a RAG trace, excluded from the denominator
    ]
    metrics = compute_metrics(traces)
    assert metrics["retrieval_hit_rate"] == 0.5


def test_retrieval_hit_rate_is_none_with_no_rag_traces():
    metrics = compute_metrics([_trace(tools_used=["churn_analysis"])])
    assert metrics["retrieval_hit_rate"] is None


def test_fallback_and_unsupported_rate():
    traces = [
        _trace(fallback_status=True),
        _trace(fallback_status=False),
        _trace(route="UNSUPPORTED", tools_used=[]),
    ]
    metrics = compute_metrics(traces)
    assert metrics["fallback_rate"] == round(1 / 3, 4)
    assert metrics["unsupported_query_rate"] == round(1 / 3, 4)


def test_token_and_cost_totals():
    traces = [_trace(input_tokens=10, output_tokens=5, estimated_cost_usd=0.001) for _ in range(3)]
    metrics = compute_metrics(traces)
    assert metrics["input_tokens"] == 30
    assert metrics["output_tokens"] == 15
    assert metrics["estimated_cost_usd"] == 0.003
