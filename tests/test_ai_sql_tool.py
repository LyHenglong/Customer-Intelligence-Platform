"""Tests for src/ai/tools/sql_tool.py - the controlled SQL execution path.

Uses a minimal fake psycopg2-shaped connection/cursor rather than a live
database, consistent with the rest of the suite (see tests/test_ai_tools.py).
Focus here is the tool's own contract: validation runs before any DB
connection is opened, the session is set read-only, results are capped
and marked truncated, and execution time/row count are reported.
"""

from __future__ import annotations

import pytest

from src.ai.guardrails.sql_safety import SQLSafetyError


class _FakeSQLCursor:
    def __init__(self, rows, columns):
        self._table_rows = rows
        self._table_columns = columns
        self.description = None
        self._pending = []
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append(query)
        q = " ".join(query.split())
        if q.upper().startswith("SET LOCAL"):
            self.description = None
            self._pending = []
            return
        # The tool never rewrites the validated query - whatever reaches
        # here is executed as-is against the fixed fake table.
        self.description = [(c,) for c in self._table_columns]
        self._pending = list(self._table_rows)

    def fetchmany(self, n):
        return self._pending[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeSQLConn:
    def __init__(self, rows, columns):
        self._rows = rows
        self._columns = columns
        self.readonly_set = None
        self.closed = False
        self.last_cursor = None

    def set_session(self, readonly=False, **kw):
        self.readonly_set = readonly

    def cursor(self):
        self.last_cursor = _FakeSQLCursor(self._rows, self._columns)
        return self.last_cursor

    def close(self):
        self.closed = True


def test_run_sql_returns_structured_result(monkeypatch):
    from src.ai.tools import sql_tool

    rows = [("CUST0001", 0), ("CUST0002", 1)]
    conn = _FakeSQLConn(rows, ["customer_id", "churn"])
    monkeypatch.setattr(sql_tool, "get_pg_conn", lambda: conn)

    result = sql_tool.run_sql("SELECT customer_id, churn FROM marts.customer_360 LIMIT 10")

    assert result.columns == ["customer_id", "churn"]
    assert result.rows == [["CUST0001", 0], ["CUST0002", 1]]
    assert result.row_count == 2
    assert result.truncated is False
    assert result.sql == "SELECT customer_id, churn FROM marts.customer_360 LIMIT 10"
    assert result.execution_time_ms >= 0
    assert conn.readonly_set is True
    assert conn.closed is True


def test_run_sql_sets_a_statement_timeout_before_the_query(monkeypatch):
    from src.ai.tools import sql_tool

    conn = _FakeSQLConn([("CUST0001", 0)], ["customer_id", "churn"])
    monkeypatch.setattr(sql_tool, "get_pg_conn", lambda: conn)

    sql_tool.run_sql("SELECT customer_id, churn FROM marts.customer_360")

    assert any("SET LOCAL statement_timeout" in q for q in conn.last_cursor.executed)


def test_run_sql_truncates_at_the_configured_row_limit(monkeypatch):
    from src.ai.tools import sql_tool

    rows = [(f"CUST{i:04d}", i % 2) for i in range(10)]
    conn = _FakeSQLConn(rows, ["customer_id", "churn"])
    monkeypatch.setattr(sql_tool, "get_pg_conn", lambda: conn)
    monkeypatch.setattr(sql_tool, "SQL_ROW_LIMIT", 3)

    result = sql_tool.run_sql("SELECT customer_id, churn FROM marts.customer_360")

    assert result.row_count == 3
    assert result.truncated is True


def test_run_sql_rejects_unsafe_query_without_touching_the_database(monkeypatch):
    from src.ai.tools import sql_tool

    calls = {"n": 0}

    def _should_not_be_called():
        calls["n"] += 1
        raise AssertionError("get_pg_conn must not be called for an unsafe query")

    monkeypatch.setattr(sql_tool, "get_pg_conn", _should_not_be_called)

    with pytest.raises(SQLSafetyError):
        sql_tool.run_sql("DELETE FROM marts.customer_360")

    assert calls["n"] == 0
