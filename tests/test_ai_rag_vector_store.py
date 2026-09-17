"""Tests for src/ai/rag/vector_store.py using a fake psycopg2-shaped
connection - no live Postgres, no pgvector extension needed to run these.
register_vector (the real pgvector adapter, which expects a genuine
psycopg2 connection) is monkeypatched to a no-op for the same reason.
"""

from __future__ import annotations

from src.ai.rag import vector_store


class _RecordingCursor:
    def __init__(self, responses):
        self._responses = responses  # shared, mutable list across cursors on the same conn
        self.executed = []
        self.description = None

    def execute(self, query, params=None):
        q = " ".join(query.split())
        self.executed.append((q, params))
        if "AS score" in q:
            self.description = [
                (c,) for c in
                ["chunk_id", "document_id", "title", "source", "section", "page", "text", "metadata", "score"]
            ]
        elif "FROM public.rag_chunks" in q and "SELECT chunk_id" in q:
            self.description = [
                (c,) for c in
                ["chunk_id", "document_id", "title", "source", "section", "page", "text", "metadata"]
            ]
        else:
            self.description = None

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


def test_ensure_table_creates_extension_and_table(monkeypatch):
    conn = _RecordingConn()
    monkeypatch.setattr(vector_store, "get_pg_conn", lambda: conn)

    vector_store.ensure_table()

    joined = " ".join(q for q, _ in conn.executed)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in joined
    assert "CREATE TABLE IF NOT EXISTS public.rag_chunks" in joined
    assert conn.committed is True
    assert conn.closed is True


def test_upsert_chunks_with_empty_list_touches_no_connection(monkeypatch):
    def _fail():
        raise AssertionError("get_pg_conn must not be called for an empty chunk list")

    monkeypatch.setattr(vector_store, "get_pg_conn", _fail)
    vector_store.upsert_chunks([])  # must not raise


def test_upsert_chunks_issues_one_upsert_per_chunk(monkeypatch):
    conn = _RecordingConn()
    monkeypatch.setattr(vector_store, "get_pg_conn", lambda: conn)
    monkeypatch.setattr(vector_store, "register_vector", lambda c: None)

    chunks = [
        {"chunk_id": "c1", "document_id": "d1", "title": "T", "source": "d1",
         "section": "S", "page": None, "text": "hello", "metadata": {}, "embedding": [0.1, 0.2]},
        {"chunk_id": "c2", "document_id": "d1", "title": "T", "source": "d1",
         "section": "S", "page": None, "text": "world", "metadata": {}, "embedding": [0.3, 0.4]},
    ]
    vector_store.upsert_chunks(chunks)

    insert_statements = [q for q, _ in conn.executed if "INSERT INTO public.rag_chunks" in q]
    assert len(insert_statements) == 2
    assert all("ON CONFLICT (chunk_id) DO UPDATE" in q for q in insert_statements)
    assert conn.committed is True


def test_delete_document_no_ops_when_table_does_not_exist(monkeypatch):
    conn = _RecordingConn(responses=[(None,)])  # to_regclass -> table missing
    monkeypatch.setattr(vector_store, "get_pg_conn", lambda: conn)

    vector_store.delete_document("doc1")

    assert not any("DELETE FROM" in q for q, _ in conn.executed)
    assert conn.committed is False
    assert conn.closed is True


def test_delete_document_deletes_when_table_exists(monkeypatch):
    conn = _RecordingConn(responses=[(12345,)])  # to_regclass -> table exists
    monkeypatch.setattr(vector_store, "get_pg_conn", lambda: conn)

    vector_store.delete_document("doc1")

    delete_calls = [(q, p) for q, p in conn.executed if "DELETE FROM public.rag_chunks" in q]
    assert delete_calls == [(delete_calls[0][0], ("doc1",))]
    assert conn.committed is True


def test_vector_search_returns_empty_list_when_table_missing(monkeypatch):
    conn = _RecordingConn(responses=[(None,)])
    monkeypatch.setattr(vector_store, "get_pg_conn", lambda: conn)

    assert vector_store.vector_search([0.1, 0.2], top_k=5) == []


def test_vector_search_returns_scored_rows_when_table_exists(monkeypatch):
    row = ("c1", "d1", "T", "d1", "S", None, "hello", {}, 0.87)
    conn = _RecordingConn(responses=[(12345,), [row]])
    monkeypatch.setattr(vector_store, "get_pg_conn", lambda: conn)
    monkeypatch.setattr(vector_store, "register_vector", lambda c: None)

    results = vector_store.vector_search([0.1, 0.2], top_k=5)

    assert results == [{
        "chunk_id": "c1", "document_id": "d1", "title": "T", "source": "d1",
        "section": "S", "page": None, "text": "hello", "metadata": {}, "score": 0.87,
    }]


def test_fetch_all_chunks_for_bm25_returns_empty_when_table_missing(monkeypatch):
    conn = _RecordingConn(responses=[(None,)])
    monkeypatch.setattr(vector_store, "get_pg_conn", lambda: conn)

    assert vector_store.fetch_all_chunks_for_bm25() == []


def test_fetch_all_chunks_for_bm25_returns_rows(monkeypatch):
    row = ("c1", "d1", "T", "d1", "S", None, "hello", {})
    conn = _RecordingConn(responses=[(12345,), [row]])
    monkeypatch.setattr(vector_store, "get_pg_conn", lambda: conn)

    results = vector_store.fetch_all_chunks_for_bm25()

    assert results == [{
        "chunk_id": "c1", "document_id": "d1", "title": "T", "source": "d1",
        "section": "S", "page": None, "text": "hello", "metadata": {},
    }]
