"""pgvector-backed storage and similarity search for RAG chunks.

Adds one table, public.rag_chunks, to the same Postgres warehouse every
other part of this platform already uses (src/warehouse.py) - no
separate vector database service, per
AI_Customer_Intelligence_Claude_Code_Plan.md section 4's instruction to
avoid unnecessary infrastructure. Requires the pgvector extension
(CREATE EXTENSION IF NOT EXISTS vector, applied here and in
db/schema.sql) and a pgvector-enabled Postgres image
(pgvector/pgvector:pg16 in docker-compose.yml; Neon supports the
extension directly).
"""

from __future__ import annotations

import json

from pgvector.psycopg2 import register_vector

from src.ai.rag.embeddings import EMBEDDING_DIM
from src.warehouse import get_pg_conn


def ensure_table() -> None:
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS public.rag_chunks (
                    chunk_id      TEXT PRIMARY KEY,
                    document_id   TEXT NOT NULL,
                    title         TEXT NOT NULL,
                    source        TEXT NOT NULL,
                    section       TEXT,
                    page          INTEGER,
                    text          TEXT NOT NULL,
                    metadata      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding     VECTOR({EMBEDDING_DIM}),
                    created_at    TIMESTAMP NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id ON public.rag_chunks (document_id)"
            )
        conn.commit()
    finally:
        conn.close()


def upsert_chunks(chunks: list[dict]) -> None:
    """chunks: dicts with chunk_id, document_id, title, source, section,
    page, text, metadata (dict), embedding (list[float]/ndarray).
    Idempotent: re-running ingestion with the same chunk_ids overwrites
    rather than duplicates (see src/ai/rag/ingest.py)."""
    if not chunks:
        return
    ensure_table()
    conn = get_pg_conn()
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            for c in chunks:
                cur.execute(
                    """
                    INSERT INTO public.rag_chunks
                        (chunk_id, document_id, title, source, section, page, text, metadata, embedding, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (chunk_id) DO UPDATE
                       SET document_id = EXCLUDED.document_id,
                           title = EXCLUDED.title,
                           source = EXCLUDED.source,
                           section = EXCLUDED.section,
                           page = EXCLUDED.page,
                           text = EXCLUDED.text,
                           metadata = EXCLUDED.metadata,
                           embedding = EXCLUDED.embedding,
                           created_at = EXCLUDED.created_at
                    """,
                    (
                        c["chunk_id"], c["document_id"], c["title"], c["source"],
                        c.get("section"), c.get("page"), c["text"],
                        json.dumps(c.get("metadata", {})), c["embedding"],
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def delete_document(document_id: str) -> None:
    """Removes every chunk belonging to document_id - called before
    re-ingesting a document, so chunks from a section that no longer
    exists don't linger (see src/ai/rag/ingest.py's idempotency)."""
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.rag_chunks')")
            if cur.fetchone()[0] is None:
                return
            cur.execute("DELETE FROM public.rag_chunks WHERE document_id = %s", (document_id,))
        conn.commit()
    finally:
        conn.close()


def vector_search(query_embedding, top_k: int = 20) -> list[dict]:
    """Cosine-distance nearest neighbors via pgvector's <=> operator.
    Returns an empty list if the table doesn't exist yet (no ingestion
    has run), mirroring src/monitoring/drift.py's to_regclass pattern for
    a warehouse that predates a feature, rather than raising."""
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.rag_chunks')")
            if cur.fetchone()[0] is None:
                return []
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, document_id, title, source, section, page, text, metadata,
                       1 - (embedding <=> %s) AS score
                  FROM public.rag_chunks
                 ORDER BY embedding <=> %s
                 LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()


def fetch_all_chunks_for_bm25() -> list[dict]:
    """All chunks' text + identifying metadata, for building the
    in-memory BM25 index (src/ai/rag/bm25.py) - BM25 has no persistent
    index of its own, so this is its source of truth on every (re)build."""
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.rag_chunks')")
            if cur.fetchone()[0] is None:
                return []
            cur.execute(
                "SELECT chunk_id, document_id, title, source, section, page, text, metadata "
                "FROM public.rag_chunks"
            )
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()
