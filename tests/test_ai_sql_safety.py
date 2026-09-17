"""Unit tests for src/ai/guardrails/sql_safety.py - the validation layer
every AI-generated SQL query must pass before src/ai/tools/sql_tool.py
executes it. No database needed: validate_sql is pure text validation.
"""

from __future__ import annotations

import pytest

from src.ai.guardrails.sql_safety import SQLSafetyError, validate_sql


def test_accepts_a_plain_select_against_an_allowed_table():
    q = "SELECT customer_id, churn FROM marts.customer_360 LIMIT 10"
    assert validate_sql(q) == q


def test_strips_a_single_trailing_semicolon():
    assert validate_sql("SELECT 1 FROM marts.customer_360;") == "SELECT 1 FROM marts.customer_360"


def test_is_case_insensitive_on_select():
    assert validate_sql("select customer_id from marts.customer_360")


@pytest.mark.parametrize("q", ["", "   ", None])
def test_rejects_empty_query(q):
    with pytest.raises(SQLSafetyError):
        validate_sql(q)


def test_rejects_non_select_statement():
    with pytest.raises(SQLSafetyError, match="only SELECT"):
        validate_sql("DELETE FROM marts.customer_360")


def test_rejects_multiple_statements():
    with pytest.raises(SQLSafetyError, match="multiple"):
        validate_sql("SELECT 1 FROM marts.customer_360; DROP TABLE marts.customer_360")


def test_rejects_sql_line_comment():
    with pytest.raises(SQLSafetyError, match="comment"):
        validate_sql("SELECT 1 FROM marts.customer_360 -- sneaky")


def test_rejects_sql_block_comment():
    with pytest.raises(SQLSafetyError, match="comment"):
        validate_sql("SELECT 1 /* sneaky */ FROM marts.customer_360")


@pytest.mark.parametrize("keyword", [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE", "GRANT",
])
def test_rejects_forbidden_keyword_inside_an_otherwise_select_shaped_string(keyword):
    with pytest.raises(SQLSafetyError, match="forbidden keyword"):
        validate_sql(f"SELECT * FROM marts.customer_360 WHERE 1=1 OR {keyword} ")


def test_does_not_false_positive_on_a_column_alias_containing_a_keyword_substring():
    """'updated_field' contains the substring 'UPDATE' but is not the
    keyword UPDATE - word-boundary matching must not flag it."""
    q = "SELECT monthlycharges AS updated_field FROM marts.customer_360"
    assert validate_sql(q) == q


def test_rejects_table_not_on_the_allowlist():
    with pytest.raises(SQLSafetyError, match="allowlist"):
        validate_sql("SELECT * FROM public.raw_customers")


def test_rejects_query_with_no_identifiable_table():
    with pytest.raises(SQLSafetyError, match="FROM/JOIN"):
        validate_sql("SELECT 1")


@pytest.mark.parametrize("table", [
    "marts.customer_360", "public.feature_drift", "public.ingestion_log", "public.retrain_summaries",
])
def test_accepts_every_allowlisted_table(table):
    assert validate_sql(f"SELECT * FROM {table}")


def test_accepts_a_join_between_two_allowed_tables():
    q = (
        "SELECT c.customer_id, f.psi FROM marts.customer_360 c "
        "JOIN public.feature_drift f ON c.customer_id = f.feature"
    )
    assert validate_sql(q) == q


def test_rejects_join_that_pulls_in_a_disallowed_table():
    q = "SELECT * FROM marts.customer_360 JOIN public.raw_customers ON true"
    with pytest.raises(SQLSafetyError, match="allowlist"):
        validate_sql(q)
