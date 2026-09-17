"""Lightweight regression sample of the full benchmark
(AI_Customer_Intelligence_Claude_Code_Plan.md sections 21 and 30): a
small, stratified subset (a fixed number of questions per route) for a
fast sanity check after a change, without waiting on the full
120-question run.

Like src/ai/evaluation/benchmark.py, this needs live infra (Postgres, an
ingested RAG corpus, GROQ_API_KEY) - there is no mocked substitute, since
a regression check against fake data would not actually catch a
regression. It is deliberately NOT wired into CI (see .github/workflows/ci.yml
and AI_Customer_Intelligence_Claude_Code_Plan.md section 30: "CI must not
require... live vector database... live external website") - this is a
local/manual pre-release check, not an automated gate.

Usage:
    python -m src.ai.evaluation.regression
"""

from __future__ import annotations

from src.ai.evaluation.benchmark import print_report, run_benchmark
from src.ai.evaluation.datasets import load_benchmark_dataset

PER_ROUTE_SAMPLE = 2


def sample_questions(per_route: int = PER_ROUTE_SAMPLE) -> list:
    dataset = load_benchmark_dataset()
    by_route: dict[str, list] = {}
    for q in dataset:
        by_route.setdefault(q.expected_route, []).append(q)
    sampled = []
    for route in sorted(by_route):
        sampled.extend(by_route[route][:per_route])
    return sampled


def run_regression() -> dict:
    return run_benchmark(questions=sample_questions())


def main():
    print_report(run_regression())


if __name__ == "__main__":
    main()
