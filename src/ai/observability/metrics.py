"""Observability metrics computed from stored traces
(AI_Customer_Intelligence_Claude_Code_Plan.md section 22's "minimum
dashboard metrics"). Operates on trace dicts already fetched from
Postgres (src/ai/observability/tracing.py's list_recent_traces) - pure
aggregation with no database access of its own, so it's testable against
a synthetic trace list without a live warehouse.
"""

from __future__ import annotations

import statistics
from collections import Counter


def compute_metrics(traces: list[dict]) -> dict:
    n = len(traces)
    if n == 0:
        return {"requests": 0}

    errored = [t for t in traces if t.get("error")]
    fallback = [t for t in traces if t.get("fallback_status")]
    unsupported = [t for t in traces if t.get("route") == "UNSUPPORTED"]
    rag_traces = [t for t in traces if "hybrid_search" in (t.get("tools_used") or [])]
    rag_hits = [t for t in rag_traces if t.get("retrieved_documents")]

    latencies = sorted(
        t["total_latency_ms"] for t in traces if t.get("total_latency_ms") is not None
    )
    tool_usage = Counter(tool for t in traces for tool in (t.get("tools_used") or []))

    input_tokens = sum(t.get("input_tokens") or 0 for t in traces)
    output_tokens = sum(t.get("output_tokens") or 0 for t in traces)
    total_cost = sum(t.get("estimated_cost_usd") or 0.0 for t in traces)

    return {
        "requests": n,
        "success_rate": round(1 - len(errored) / n, 4),
        "error_rate": round(len(errored) / n, 4),
        "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p50_latency_ms": round(statistics.median(latencies), 2) if latencies else None,
        "p95_latency_ms": (
            round(latencies[max(0, int(len(latencies) * 0.95) - 1)], 2) if latencies else None
        ),
        "tool_usage": dict(tool_usage),
        "retrieval_hit_rate": round(len(rag_hits) / len(rag_traces), 4) if rag_traces else None,
        "fallback_rate": round(len(fallback) / n, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(total_cost, 6),
        "unsupported_query_rate": round(len(unsupported) / n, 4),
    }
