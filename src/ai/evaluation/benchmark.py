"""Regression evaluation command
(AI_Customer_Intelligence_Claude_Code_Plan.md section 21).

Usage:
    python -m src.ai.evaluation.benchmark [--limit N]

Runs every question in the benchmark dataset (src/ai/evaluation/datasets.py)
through the real agent graph (src/ai/graph.run_query) - a live Postgres
connection, an ingested RAG corpus, and a configured GROQ_API_KEY are all
required for this to produce real numbers. There is no mocked or
simulated mode: a report from this command is either a real measurement
or it doesn't run at all, per the plan's "never invent benchmark
numbers" instruction (section 19/21). Running this and pasting its real
output into report/ai_assistant_evaluation.md is a separate, explicit
next step - the code here has been unit-tested (see
tests/test_ai_evaluation_benchmark.py) with every tool/LLM call mocked,
which proves the wiring, not the model's actual quality.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time

from src.ai.evaluation.answer import score_response
from src.ai.evaluation.datasets import load_benchmark_dataset
from src.ai.graph import run_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ai.evaluation.benchmark")

_FAILURE_SCORE = {
    "route_correct": False, "tool_selection_accuracy": 0.0, "citation_correctness": 0.0,
    "unsupported_claim_rate": 1.0, "faithfulness": 0.0, "answer_relevancy": 0.0,
    "context_precision": 0.0, "context_recall": 0.0, "latency_ms": None,
}


def run_benchmark(limit: int | None = None, questions=None) -> dict:
    if questions is None:
        questions = load_benchmark_dataset()
    if limit is not None:
        questions = questions[:limit]

    scores = []
    start = time.monotonic()
    for q in questions:
        try:
            response = run_query(q.question)
            scores.append(score_response(q, response))
        except Exception:
            log.exception("question %s raised, scoring as a full failure", q.id)
            scores.append({"id": q.id, **_FAILURE_SCORE})
    total_elapsed_s = time.monotonic() - start

    return summarize(scores, total_elapsed_s)


def summarize(scores: list[dict], total_elapsed_s: float) -> dict:
    n = len(scores)
    if n == 0:
        return {"queries": 0}

    def _avg(key):
        values = [s[key] for s in scores if isinstance(s.get(key), (int, float))]
        return sum(values) / len(values) if values else 0.0

    latencies = sorted(s["latency_ms"] for s in scores if s.get("latency_ms") is not None)
    median_latency = statistics.median(latencies) if latencies else None
    p95_latency = latencies[max(0, int(len(latencies) * 0.95) - 1)] if latencies else None

    return {
        "queries": n,
        "route_accuracy": round(sum(1 for s in scores if s["route_correct"]) / n, 4),
        "tool_selection_accuracy": round(_avg("tool_selection_accuracy"), 4),
        "faithfulness": round(_avg("faithfulness"), 4),
        "answer_relevancy": round(_avg("answer_relevancy"), 4),
        "citation_accuracy": round(_avg("citation_correctness"), 4),
        "context_precision": round(_avg("context_precision"), 4),
        "context_recall": round(_avg("context_recall"), 4),
        "unsupported_claims": round(_avg("unsupported_claim_rate"), 4),
        "median_latency_ms": median_latency,
        "p95_latency_ms": p95_latency,
        "total_elapsed_s": round(total_elapsed_s, 2),
    }


def print_report(summary: dict) -> None:
    print("=" * 37)
    print("AI ASSISTANT EVALUATION")
    print("=" * 37)
    print()
    if summary.get("queries", 0) == 0:
        print("No questions were run.")
        return
    print(f"Queries:                 {summary['queries']}")
    print(f"Route accuracy:          {summary['route_accuracy']:.1%}")
    print(f"Tool selection accuracy: {summary['tool_selection_accuracy']:.1%}")
    print(f"Faithfulness:            {summary['faithfulness']:.2f}")
    print(f"Answer relevancy:        {summary['answer_relevancy']:.2f}")
    print(f"Citation accuracy:       {summary['citation_accuracy']:.1%}")
    print(f"Context precision:       {summary['context_precision']:.2f}")
    print(f"Context recall:          {summary['context_recall']:.2f}")
    print(f"Unsupported claims:      {summary['unsupported_claims']:.1%}")
    if summary["median_latency_ms"] is not None:
        print(f"Median latency:          {summary['median_latency_ms']:.0f}ms")
        print(f"P95 latency:             {summary['p95_latency_ms']:.0f}ms")
    print(f"Total elapsed:           {summary['total_elapsed_s']:.1f}s")
    print("=" * 37)


def main():
    parser = argparse.ArgumentParser(description="Run the AI assistant evaluation benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions")
    args = parser.parse_args()
    print_report(run_benchmark(limit=args.limit))


if __name__ == "__main__":
    main()
