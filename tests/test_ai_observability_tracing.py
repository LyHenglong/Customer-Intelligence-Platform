"""Tests for src/ai/observability/tracing.py using a fake psycopg2-shaped
connection - no live Postgres."""

from __future__ import annotations

from src.ai.observability import tracing


class _RecordingCursor:
    def __init__(self, responses):
        self._responses = responses
        self.executed = []
        self.description = None

    def execute(self, query, params=None):
        q = " ".join(query.split())
        self.executed.append((q, params))
        if "SELECT trace_id" in q or "SELECT " + ", ".join(tracing._COLUMNS) in q:
            self.description = [(c,) for c in tracing._COLUMNS]

    def fetchone(self):
        return self._responses.pop(0) if self._responses else None

    def fetchall(self):
        return self._responses.pop(0) if self._responses else []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _RecordingConn:
    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.cursors = []
        self.committed = False
        self.closed = False

    def cursor(self, *a, **kw):
        cur = _RecordingCursor(self._responses)
        self.cursors.append(cur)
        return cur

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True

    @property
    def executed(self):
        return [stmt for cur in self.cursors for stmt in cur.executed]


def test_hash_sql_is_deterministic_and_short():
    h1 = tracing.hash_sql("SELECT 1")
    h2 = tracing.hash_sql("SELECT 1")
    h3 = tracing.hash_sql("SELECT 2")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16


def test_ensure_table_creates_table_and_commits(monkeypatch):
    conn = _RecordingConn()
    monkeypatch.setattr(tracing, "get_pg_conn", lambda: conn)

    tracing.ensure_table()

    joined = " ".join(q for q, _ in conn.executed)
    assert "CREATE TABLE IF NOT EXISTS public.ai_traces" in joined
    assert conn.committed is True


def test_write_trace_inserts_with_on_conflict_do_nothing(monkeypatch):
    conn = _RecordingConn()
    monkeypatch.setattr(tracing, "get_pg_conn", lambda: conn)

    tracing.write_trace({
        "trace_id": "t1", "user_query": "hello", "route": "ML_ANALYSIS",
        "tools_used": ["churn_analysis"], "tool_latency_ms": {"churn_analysis": 12.0},
        "sql_query_hash": None, "retrieval_latency_ms": None, "retrieved_documents": [],
        "reranker_latency_ms": None, "llm_model": "fake-model", "input_tokens": 5,
        "output_tokens": 5, "estimated_cost_usd": 0.0, "total_latency_ms": 42.0,
        "validation_result": "grounded", "fallback_status": False, "error": None,
    })

    insert_statements = [q for q, p in conn.executed if "INSERT INTO public.ai_traces" in q]
    assert len(insert_statements) == 1
    assert "ON CONFLICT (trace_id) DO NOTHING" in insert_statements[0]
    assert conn.committed is True


def test_get_trace_returns_none_when_table_missing(monkeypatch):
    conn = _RecordingConn(responses=[(None,)])
    monkeypatch.setattr(tracing, "get_pg_conn", lambda: conn)

    assert tracing.get_trace("t1") is None


def test_get_trace_returns_none_when_row_missing(monkeypatch):
    conn = _RecordingConn(responses=[(12345,), None])
    monkeypatch.setattr(tracing, "get_pg_conn", lambda: conn)

    assert tracing.get_trace("t1") is None


def test_get_trace_returns_a_dict_when_found(monkeypatch):
    row = tuple(f"value_{c}" for c in tracing._COLUMNS)
    conn = _RecordingConn(responses=[(12345,), row])
    monkeypatch.setattr(tracing, "get_pg_conn", lambda: conn)

    result = tracing.get_trace("t1")

    assert result["trace_id"] == "value_trace_id"
    assert set(result) == set(tracing._COLUMNS)


def test_list_recent_traces_returns_empty_when_table_missing(monkeypatch):
    conn = _RecordingConn(responses=[(None,)])
    monkeypatch.setattr(tracing, "get_pg_conn", lambda: conn)

    assert tracing.list_recent_traces() == []


def test_list_recent_traces_returns_rows(monkeypatch):
    row = tuple(f"value_{c}" for c in tracing._COLUMNS)
    conn = _RecordingConn(responses=[(12345,), [row, row]])
    monkeypatch.setattr(tracing, "get_pg_conn", lambda: conn)

    results = tracing.list_recent_traces(limit=10)

    assert len(results) == 2
    assert results[0]["trace_id"] == "value_trace_id"
