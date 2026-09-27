"""End-to-end checks against a real Postgres, run by CI's integration job
after it has loaded db/schema.sql, ingested tests/fixtures/sample_batch.csv,
run `dbt build`, and applied db/ai_readonly_role.sql.

Skipped unless RUN_INTEGRATION=1, so `pytest` on a laptop with no
warehouse stays green and fast.
"""

from __future__ import annotations

import os

import psycopg2
import pytest

from src.ai.guardrails.sql_safety import SQLSafetyError
from src.ai.tools import sql_tool
from src.warehouse import get_pg_conn

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1", reason="needs the CI warehouse (RUN_INTEGRATION=1)"
)

READER_USER = "ai_sql_reader"
READER_PASSWORD = os.environ.get("AI_SQL_READER_PASSWORD", "reader-pw")


def _scalar(query: str, **conn_kwargs):
    conn = get_pg_conn(**conn_kwargs)
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchone()[0]
    finally:
        conn.close()


def test_ingestion_dropped_invalid_rows_and_dbt_built_the_mart():
    # The fixture has 5 rows, 2 of which fail the ingestion quality gate.
    assert _scalar("SELECT COUNT(*) FROM public.customers_cleaned") == 3
    assert _scalar("SELECT COUNT(*) FROM marts.customer_360") == 3


def test_customer_360_has_one_row_per_customer():
    assert _scalar("SELECT COUNT(*) - COUNT(DISTINCT customer_id) FROM marts.customer_360") == 0


def test_sql_tool_answers_through_the_least_privilege_role(monkeypatch):
    monkeypatch.setenv("AI_SQL_POSTGRES_USER", READER_USER)
    monkeypatch.setenv("AI_SQL_POSTGRES_PASSWORD", READER_PASSWORD)

    result = sql_tool.run_sql("SELECT COUNT(*) AS n FROM marts.customer_360")

    assert result.columns == ["n"]
    assert result.rows == [[3]]


def test_the_validator_rejects_a_disallowed_table_before_the_database_sees_it(monkeypatch):
    monkeypatch.setenv("AI_SQL_POSTGRES_USER", READER_USER)
    monkeypatch.setenv("AI_SQL_POSTGRES_PASSWORD", READER_PASSWORD)

    with pytest.raises(SQLSafetyError):
        sql_tool.run_sql("SELECT * FROM marts.customer_360, public.raw_customers")


@pytest.mark.parametrize(
    "query",
    [
        "SELECT COUNT(*) FROM public.raw_customers",
        "SELECT passwd FROM pg_shadow",
    ],
)
def test_the_role_itself_cannot_read_outside_the_allowlist(query):
    """The database layer on its own - no validator involved - so a future
    parsing gap in sql_safety.py still cannot reach these tables."""
    with pytest.raises(psycopg2.errors.InsufficientPrivilege):
        _scalar(query, user=READER_USER, password=READER_PASSWORD)


def test_the_role_cannot_write():
    with pytest.raises(psycopg2.errors.ReadOnlySqlTransaction):
        _scalar("CREATE TABLE public.should_not_exist (a int) ; SELECT 1", user=READER_USER, password=READER_PASSWORD)
