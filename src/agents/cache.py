"""
Postgres-backed cache for AI agent output.

Keyed so a customer's explanation/outreach draft is generated once per
churn model version, not once per dashboard refresh - LLM calls are the
slowest and only rate-limited part of this stack (Groq's free tier), so
reading a cached row instead of re-calling the API is what makes it
practical to show AI content for a table of at-risk customers rather than
one customer at a time.

The churn model version is part of the cache key, not just an audit
column: an explanation is grounded in one specific model's SHAP output,
and reusing it after a retrain would describe a different model's
reasoning as if it were the current one.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.warehouse import get_pg_conn

log = logging.getLogger("agents.cache")


def ensure_tables() -> None:
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS public.llm_explanations (
                    customer_id        TEXT NOT NULL,
                    agent_type         TEXT NOT NULL,
                    model_version      TEXT NOT NULL,
                    content            TEXT NOT NULL,
                    prompt_tokens      INTEGER,
                    completion_tokens  INTEGER,
                    created_at         TIMESTAMP NOT NULL DEFAULT now(),
                    PRIMARY KEY (customer_id, agent_type, model_version)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS public.retrain_summaries (
                    churn_model_version  TEXT PRIMARY KEY,
                    previous_version     TEXT,
                    summary_text         TEXT NOT NULL,
                    prompt_tokens        INTEGER,
                    completion_tokens    INTEGER,
                    created_at           TIMESTAMP NOT NULL DEFAULT now()
                );
                """
            )
        conn.commit()
    finally:
        conn.close()


# ------------------------------------------------------ per-customer cache


def get_cached_content(customer_id: str, agent_type: str, model_version: str) -> Optional[str]:
    """Returns None on a cache miss - including when the table doesn't
    exist yet at all (a fresh warehouse that has never cached anything).
    That second case matters: the very first call this project ever makes
    is a read, not a write, so if only put_cached_content ensured the
    table existed, that first read would crash outright rather than
    correctly report "no cache entry yet"."""
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.llm_explanations')")
            if cur.fetchone()[0] is None:
                return None
            cur.execute(
                "SELECT content FROM public.llm_explanations "
                "WHERE customer_id = %s AND agent_type = %s AND model_version = %s",
                (customer_id, agent_type, model_version),
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def put_cached_content(
    customer_id: str,
    agent_type: str,
    model_version: str,
    content: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    ensure_tables()
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.llm_explanations
                    (customer_id, agent_type, model_version, content, prompt_tokens, completion_tokens, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (customer_id, agent_type, model_version) DO UPDATE
                   SET content = EXCLUDED.content,
                       prompt_tokens = EXCLUDED.prompt_tokens,
                       completion_tokens = EXCLUDED.completion_tokens,
                       created_at = EXCLUDED.created_at;
                """,
                (customer_id, agent_type, model_version, content, prompt_tokens, completion_tokens),
            )
        conn.commit()
    finally:
        conn.close()


def get_or_generate(customer_id: str, agent_type: str, model_version: str, generate_fn):
    """Cache-through helper: returns the cached text on a hit, otherwise
    calls `generate_fn()` (expected to return an AgentResponse - see
    src/agents/groq_client.py), caches its .text, and returns that.

    Exceptions from generate_fn (including AgentCallFailed) propagate
    uncaught - this function only owns the cache-or-generate decision,
    not the fallback-on-failure behavior, which is caller-specific (the
    dashboard shows raw SHAP/recommender output, the API returns a 200
    with a fallback flag - see their respective call sites)."""
    cached = get_cached_content(customer_id, agent_type, model_version)
    if cached is not None:
        return cached
    response = generate_fn()
    put_cached_content(
        customer_id, agent_type, model_version,
        response.text, response.prompt_tokens, response.completion_tokens,
    )
    return response.text


# ---------------------------------------------------------- retrain summaries


def put_retrain_summary(
    churn_model_version: str,
    summary_text: str,
    previous_version: Optional[str] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    ensure_tables()
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.retrain_summaries
                    (churn_model_version, previous_version, summary_text, prompt_tokens, completion_tokens, created_at)
                VALUES (%s, %s, %s, %s, %s, now())
                ON CONFLICT (churn_model_version) DO UPDATE
                   SET previous_version = EXCLUDED.previous_version,
                       summary_text = EXCLUDED.summary_text,
                       prompt_tokens = EXCLUDED.prompt_tokens,
                       completion_tokens = EXCLUDED.completion_tokens,
                       created_at = EXCLUDED.created_at;
                """,
                (churn_model_version, previous_version, summary_text, prompt_tokens, completion_tokens),
            )
        conn.commit()
    finally:
        conn.close()


def get_latest_retrain_summary() -> Optional[dict]:
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.retrain_summaries')"
            )
            if cur.fetchone()[0] is None:
                return None
            cur.execute(
                """
                SELECT churn_model_version, previous_version, summary_text, created_at
                  FROM public.retrain_summaries
                 ORDER BY created_at DESC LIMIT 1
                """
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "churn_model_version": row[0],
                "previous_version": row[1],
                "summary_text": row[2],
                "created_at": row[3],
            }
    finally:
        conn.close()
