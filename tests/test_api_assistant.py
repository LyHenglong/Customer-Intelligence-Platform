"""Tests for POST /assistant/query (src/model/api.py). The underlying
agent graph (src/ai/graph.run_query) is monkeypatched - this pins the
HTTP contract (request validation, response shape), not the graph's own
routing/evidence logic (see tests/test_ai_graph.py for that).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.ai.schemas import AssistantResponse
from src.model import api as api_module


def test_assistant_query_returns_the_graph_result(monkeypatch):
    canned = AssistantResponse(
        answer="Month-to-month customers churn more than long-term customers.",
        citations=[], evidence=[], tools_used=["churn_analysis"],
        model_version="V1", route="ML_ANALYSIS", trace_id="abc123", latency_ms=12.3,
    )
    monkeypatch.setattr(api_module, "ai_run_query", lambda query: canned)

    with TestClient(api_module.app) as client:
        response = client.post("/assistant/query", json={"query": "Why is churn increasing?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == canned.answer
    assert body["route"] == "ML_ANALYSIS"
    assert body["trace_id"] == "abc123"


def test_assistant_query_rejects_empty_query(monkeypatch):
    def _should_not_be_called(query):
        raise AssertionError("run_query must not be called for an empty query")

    monkeypatch.setattr(api_module, "ai_run_query", _should_not_be_called)

    with TestClient(api_module.app) as client:
        response = client.post("/assistant/query", json={"query": "   "})

    assert response.status_code == 422


def test_assistant_query_rejects_missing_query_field():
    with TestClient(api_module.app) as client:
        response = client.post("/assistant/query", json={})

    assert response.status_code == 422  # pydantic: "query" is a required field


def test_assistant_query_accepts_optional_conversation_id(monkeypatch):
    canned = AssistantResponse(answer="ok", route="UNSUPPORTED", trace_id="x")
    captured = {}

    def _fake_run_query(query):
        captured["query"] = query
        return canned

    monkeypatch.setattr(api_module, "ai_run_query", _fake_run_query)

    with TestClient(api_module.app) as client:
        response = client.post(
            "/assistant/query", json={"query": "hello", "conversation_id": "conv-1"}
        )

    assert response.status_code == 200
    assert captured["query"] == "hello"


def test_assistant_trace_returns_the_stored_trace(monkeypatch):
    canned_trace = {"trace_id": "t1", "route": "ML_ANALYSIS", "total_latency_ms": 42.0}
    monkeypatch.setattr(api_module, "ai_get_trace", lambda trace_id: canned_trace)

    with TestClient(api_module.app) as client:
        response = client.get("/assistant/trace/t1")

    assert response.status_code == 200
    assert response.json() == canned_trace


def test_assistant_trace_404s_when_not_found(monkeypatch):
    monkeypatch.setattr(api_module, "ai_get_trace", lambda trace_id: None)

    with TestClient(api_module.app) as client:
        response = client.get("/assistant/trace/does-not-exist")

    assert response.status_code == 404
