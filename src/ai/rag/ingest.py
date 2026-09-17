"""RAG ingestion command: discovers knowledge/ documents, parses, cleans,
chunks, embeds, and indexes them into pgvector (public.rag_chunks).

Idempotent: chunk_id is a stable hash of (document_id, strategy, chunk
index), and each document's existing chunks are deleted before its new
ones are inserted, so running this command twice does not duplicate
chunks (AI_Customer_Intelligence_Claude_Code_Plan.md section 33).

Usage:
    python -m src.ai.rag.ingest [--strategy fixed|overlapping|structure_aware]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from src.ai.rag.bm25 import reload_bm25_index
from src.ai.rag.chunking import CHUNKING_STRATEGIES, clean_text
from src.ai.rag.embeddings import embed_texts
from src.ai.rag.vector_store import delete_document, ensure_table, upsert_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag.ingest")

KNOWLEDGE_DIR = Path(__file__).resolve().parents[3] / "knowledge"
DEFAULT_STRATEGY = "structure_aware"


def discover_documents() -> list[Path]:
    return sorted(KNOWLEDGE_DIR.rglob("*.md"))


def _document_id(path: Path) -> str:
    return str(path.relative_to(KNOWLEDGE_DIR)).replace("\\", "/")


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _chunk_id(document_id: str, strategy: str, index: int) -> str:
    digest = hashlib.sha1(f"{document_id}:{strategy}:{index}".encode("utf-8")).hexdigest()[:16]
    return f"{document_id}::{strategy}::{index}::{digest}"


def ingest(strategy: str = DEFAULT_STRATEGY) -> dict:
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(
            f"unknown chunking strategy {strategy!r}; choose from {sorted(CHUNKING_STRATEGIES)}"
        )
    chunk_fn = CHUNKING_STRATEGIES[strategy]

    ensure_table()
    documents = discover_documents()
    total_chunks = 0
    failed = 0

    for path in documents:
        try:
            raw_text = path.read_text(encoding="utf-8")
            document_id = _document_id(path)
            title = _title_from_text(raw_text, fallback=path.stem)
            cleaned = clean_text(raw_text)
            raw_chunks = chunk_fn(cleaned)

            if not raw_chunks:
                log.warning("no chunks produced for %s, skipping", document_id)
                continue

            texts = [c.text for c in raw_chunks]
            embeddings = embed_texts(texts)

            delete_document(document_id)  # idempotent re-ingest: drop stale chunks first
            rows = [
                {
                    "chunk_id": _chunk_id(document_id, strategy, i),
                    "document_id": document_id,
                    "title": title,
                    "source": document_id,
                    "section": rc.section,
                    "page": None,
                    "text": rc.text,
                    "metadata": {"strategy": strategy, "chunk_index": i},
                    "embedding": embeddings[i],
                }
                for i, rc in enumerate(raw_chunks)
            ]
            upsert_chunks(rows)
            total_chunks += len(rows)
            log.info("ingested %s: %d chunk(s)", document_id, len(rows))
        except Exception:
            log.exception("failed to ingest %s", path)
            failed += 1

    reload_bm25_index()

    report = {
        "documents": len(documents),
        "chunks": total_chunks,
        "embedded": total_chunks,
        "indexed": total_chunks,
        "failed": failed,
        "strategy": strategy,
    }
    log.info(
        "Documents: %d\nChunks: %d\nEmbedded: %d\nIndexed: %d\nFailed: %d",
        report["documents"], report["chunks"], report["embedded"], report["indexed"], report["failed"],
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Ingest knowledge/ documents into the RAG vector store")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, choices=sorted(CHUNKING_STRATEGIES))
    args = parser.parse_args()
    ingest(strategy=args.strategy)


if __name__ == "__main__":
    main()
