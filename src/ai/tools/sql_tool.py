"""Controlled, read-only SQL tool - the only way the AI assistant layer
runs a caller-supplied query against Postgres
(AI_Customer_Intelligence_Claude_Code_Plan.md section 9).

Every query is validated by src/ai/guardrails/sql_safety.py before it
reaches the database. On top of that text-level check, the connection
itself is set read-only at the session level (defense in depth - a write
would be rejected by Postgres even if a future validation gap let one
through), a statement timeout bounds runaway queries, and results are
capped at SQL_ROW_LIMIT rows - this tool never returns an unbounded
result set.
"""

from __future__ import annotations

import logging
import time

from src.ai.config import SQL_ROW_LIMIT, SQL_STATEMENT_TIMEOUT_MS
from src.ai.guardrails.sql_safety import validate_sql
from src.ai.schemas import SQLQueryResult
from src.warehouse import get_pg_conn

log = logging.getLogger("ai.sql_tool")


def run_sql(query: str) -> SQLQueryResult:
    validated = validate_sql(query)

    conn = get_pg_conn()
    try:
        # Not autocommit: SET LOCAL only applies for the current
        # transaction, so it and the query itself must run in the same
        # (implicitly started) transaction rather than each autocommitting
        # separately.
        conn.set_session(readonly=True)
        start = time.monotonic()
        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL statement_timeout = {SQL_STATEMENT_TIMEOUT_MS}")
            cur.execute(validated)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchmany(SQL_ROW_LIMIT + 1)
        elapsed_ms = (time.monotonic() - start) * 1000
    finally:
        conn.close()

    truncated = len(rows) > SQL_ROW_LIMIT
    rows = rows[:SQL_ROW_LIMIT]

    log.info(
        "AI SQL tool: %.1fms, %d row(s)%s | %s",
        elapsed_ms, len(rows), " (truncated)" if truncated else "", validated,
    )

    return SQLQueryResult(
        sql=validated,
        columns=columns,
        rows=[list(r) for r in rows],
        row_count=len(rows),
        truncated=truncated,
        execution_time_ms=round(elapsed_ms, 2),
    )
