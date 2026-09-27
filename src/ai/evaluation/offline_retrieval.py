"""Offline retrieval evaluation: BM25 over the real knowledge/ corpus,
scored against the benchmark's RAG questions and their expected documents.

The full benchmark (src/ai/evaluation/benchmark.py) needs Postgres, the
embedded corpus and a Groq key, so it can only run by hand. This covers the
part of retrieval that needs none of those - chunking plus the BM25 leg of
hybrid search - so CI can catch a change to the corpus, the chunker or the
tokenizer that makes the right document stop surfacing. It measures BM25
alone, not the fused + reranked pipeline, so its numbers are a floor for
the live system rather than a report of it.

Usage:
    python -m src.ai.evaluation.offline_retrieval [--k 5]
"""

from __future__ import annotations

import argparse

from rank_bm25 import BM25Okapi

from src.ai.evaluation.datasets import load_benchmark_dataset
from src.ai.evaluation.retrieval import aggregate_metrics
from src.ai.rag.bm25 import _tokenize
from src.ai.rag.chunking import CHUNKING_STRATEGIES, clean_text
from src.ai.rag.ingest import _document_id, discover_documents


def build_local_chunks(strategy: str) -> list[dict]:
    """The chunks ingest() would store, built in memory with no database
    or embedding model."""
    chunk_fn = CHUNKING_STRATEGIES[strategy]
    chunks = []
    for path in discover_documents():
        document_id = _document_id(path)
        for raw in chunk_fn(clean_text(path.read_text(encoding="utf-8"))):
            chunks.append({"document_id": document_id, "text": raw.text})
    return chunks


def rank_documents(bm25: BM25Okapi, chunks: list[dict], query: str) -> list[str]:
    """Document ids in the order their best-scoring chunk ranks, deduplicated -
    the unit the benchmark's expected_documents are labelled in."""
    scores = bm25.get_scores(_tokenize(query))
    ranked_docs: list[str] = []
    for i in sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True):
        doc = chunks[i]["document_id"]
        if doc not in ranked_docs:
            ranked_docs.append(doc)
    return ranked_docs


def evaluate(strategy: str = "structure_aware", k: int = 5) -> dict:
    chunks = build_local_chunks(strategy)
    bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
    questions = [q for q in load_benchmark_dataset() if q.expected_route == "RAG_SEARCH" and q.expected_documents]
    runs = [(rank_documents(bm25, chunks, q.question), set(q.expected_documents)) for q in questions]
    return {"strategy": strategy, "n_chunks": len(chunks), **aggregate_metrics(runs, k=k)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    print(f"{'strategy':<17} {'chunks':>6} {'hit@k':>6} {'mrr':>6} {'ndcg@k':>7}   (k={args.k}, BM25 only)")
    for strategy in CHUNKING_STRATEGIES:
        m = evaluate(strategy, k=args.k)
        print(f"{strategy:<17} {m['n_chunks']:>6} {m['hit_at_k']:>6.3f} {m['mrr']:>6.3f} {m['ndcg_at_k']:>7.3f}")


if __name__ == "__main__":
    main()
