"""Tests for src/dashboard/app.py's call_assistant_api() - the dashboard's
HTTP client for the API service's /assistant/query (the AI Decision
Assistant tab lives in the API service, not this dashboard process - see
that function's own docstring). Network calls are mocked; no live API
service or Streamlit runtime needed, matching tests/test_dashboard.py's
existing "import the module, call the function directly" approach.
"""

from __future__ import annotations

import json

from src.dashboard import app as dashboard_app


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_returns_not_configured_when_ai_api_url_is_empty(monkeypatch):
    monkeypatch.setattr(dashboard_app, "AI_API_URL", "")

    result = dashboard_app.call_assistant_api("Why is churn increasing?")

    assert result == {"error": "not_configured"}


def test_successful_call_returns_parsed_json(monkeypatch):
    monkeypatch.setattr(dashboard_app, "AI_API_URL", "http://api:8000")
    canned = {"answer": "Churn is concentrated in month-to-month contracts.", "route": "ML_ANALYSIS"}

    def _fake_urlopen(req, timeout=None):
        assert req.full_url == "http://api:8000/assistant/query"
        assert req.get_method() == "POST"
        assert json.loads(req.data) == {"query": "Why is churn increasing?"}
        return _FakeResponse(canned)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    result = dashboard_app.call_assistant_api("Why is churn increasing?")

    assert result == canned


def test_connection_failure_returns_an_error_dict_not_an_exception(monkeypatch):
    monkeypatch.setattr(dashboard_app, "AI_API_URL", "http://api:8000")

    def _fake_urlopen(req, timeout=None):
        raise ConnectionRefusedError("nobody home")

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    result = dashboard_app.call_assistant_api("hello")

    assert "error" in result
    assert "nobody home" in result["error"]
