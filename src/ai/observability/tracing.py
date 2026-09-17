"""Request tracing (AI_Customer_Intelligence_Claude_Code_Plan.md
section 22).

Every /assistant/query request writes one row to public.ai_traces (see
src/ai/graph.py's run_query, at the very end, after the response is
already built) - a trace-write failure must never break the actual
response, so write_trace() is always called from inside a caller-side
try/except and never raises upward itself in normal operation (see its
own try/finally shape below, which mirrors every other Postgres-backed
module in this project - src/agents/cache.py, src/monitoring/drift.py).

Deliberately excludes the generated answer text and full evidence values
from the stored trace (section 22: "Do not store sensitive customer data
unnecessarily") - only routing/timing/token/cost metadata, a short SQL
hash, and retrieved document ids, never the customer-specific content a
response's evidence carried.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from src.warehouse import get_pg_conn

log = logging.getLogger("ai.observability.tracing")

_COLUMNS = [
    "trace_id", "request_timestamp", "user_query", "route", "tools_used", "tool_latency_ms",
    "sql_query_hash", "retrieval_latency_ms", "retrieved_documents", "reranker_latency_ms",
    "llm_model", "input_tokens", "output_tokens", "estimated_cost_usd", "total_latency_ms",
    "validation_result", "fallback_status", "error",
]


def ensure_table() -> None:
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS public.ai_traces (
                    trace_id             TEXT PRIMARY KEY,
                    request_timestamp    TIMESTAMP NOT NULL DEFAULT now(),
                    user_query           TEXT NOT NULL,
                    route                TEXT,
                    tools_used           JSONB NOT NULL DEFAULT '[]'::jsonb,
                    tool_latency_ms      JSONB NOT NULL DEFAULT '{}'::jsonb,
                    sql_query_hash       TEXT,
                    retrieval_latency_ms DOUBLE PRECISION,
                    retrieved_documents  JSONB NOT NULL DEFAULT '[]'::jsonb,
                    reranker_latency_ms  DOUBLE PRECISION,
                    llm_model            TEXT,
                    input_tokens         INTEGER,
                    output_tokens        INTEGER,
                    estimated_cost_usd   DOUBLE PRECISION,
                    total_latency_ms     DOUBLE PRECISION,
                    validation_result    TEXT,
                    fallback_status      BOOLEAN NOT NULL DEFAULT false,
                    error                TEXT
                )
                """
            )
        conn.commit()
    finally:
        conn.close()


def hash_sql(sql: str) -> str:
    return hashlib.sha1(sql.encode("utf-8")).hexdigest()[:16]


def write_trace(trace: dict) -> None:
    ensure_table()
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.ai_traces (
                    trace_id, user_query, route, tools_used, tool_latency_ms,
                    sql_query_hash, retrieval_latency_ms, retrieved_documents,
                    reranker_latency_ms, llm_model, input_tokens, output_tokens,
                    estimated_cost_usd, total_latency_ms, validation_result,
                    fallback_status, error
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (trace_id) DO NOTHING
                """,
                (
                    trace["trace_id"], trace["user_query"], trace.get("route"),
                    json.dumps(trace.get("tools_used", [])), json.dumps(trace.get("tool_latency_ms", {})),
                    trace.get("sql_query_hash"), trace.get("retrieval_latency_ms"),
                    json.dumps(trace.get("retrieved_documents", [])), trace.get("reranker_latency_ms"),
                    trace.get("llm_model"), trace.get("input_tokens"), trace.get("output_tokens"),
                    trace.get("estimated_cost_usd"), trace.get("total_latency_ms"),
                    trace.get("validation_result"), trace.get("fallback_status", False), trace.get("error"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def get_trace(trace_id: str) -> Optional[dict]:
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.ai_traces')")
            if cur.fetchone()[0] is None:
                return None
            cur.execute(f"SELECT {', '.join(_COLUMNS)} FROM public.ai_traces WHERE trace_id = %s", (trace_id,))
            row = cur.fetchone()
            if row is None:
                return None
        return dict(zip(_COLUMNS, row))
    finally:
        conn.close()


def list_recent_traces(limit: int = 500) -> list[dict]:
    """All trace fields needed by src/ai/observability/metrics.py's
    aggregation, most recent first. Used by an internal observability
    view (section 22), not exposed as a public API endpoint."""
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.ai_traces')")
            if cur.fetchone()[0] is None:
                return []
            cur.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM public.ai_traces "
                f"ORDER BY request_timestamp DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [dict(zip(_COLUMNS, r)) for r in rows]
    finally:
        conn.close()
