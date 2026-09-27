"""API-key auth and LLM rate limiting (src/api/security.py)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.ai.schemas import AssistantResponse
from src.api import app as app_module
from src.api import security
from src.api.routers import assistant as assistant_router


@pytest.fixture(autouse=True)
def _no_real_startup_work(monkeypatch):
    monkeypatch.setattr(app_module, "load_models", lambda: None)
    monkeypatch.setattr(app_module, "_warm_all", lambda: None)


@pytest.fixture
def canned_assistant(monkeypatch):
    canned = AssistantResponse(
        answer="ok", citations=[], evidence=[], tools_used=[],
        model_version="V1", route="GENERAL", trace_id="t", latency_ms=1.0,
    )
    monkeypatch.setattr(assistant_router, "ai_run_query", lambda query: canned)


# ------------------------------------------------------------------ auth


def test_auth_is_off_when_no_keys_are_configured():
    with TestClient(app_module.app) as client:
        assert client.get("/model-info").status_code == 200


def test_missing_key_is_rejected_when_keys_are_configured(monkeypatch):
    monkeypatch.setenv("API_KEYS", "k1,k2")
    with TestClient(app_module.app) as client:
        assert client.get("/model-info").status_code == 401


def test_wrong_key_is_rejected(monkeypatch):
    monkeypatch.setenv("API_KEYS", "k1")
    with TestClient(app_module.app) as client:
        assert client.get("/model-info", headers={"X-API-Key": "nope"}).status_code == 401


@pytest.mark.parametrize("key", ["k1", "k2"])
def test_any_configured_key_is_accepted(monkeypatch, key):
    monkeypatch.setenv("API_KEYS", "k1, k2")
    with TestClient(app_module.app) as client:
        assert client.get("/model-info", headers={"X-API-Key": key}).status_code == 200


def test_health_stays_open_so_platform_healthchecks_work(monkeypatch):
    monkeypatch.setenv("API_KEYS", "k1")
    with TestClient(app_module.app) as client:
        assert client.get("/health").status_code == 200


# ------------------------------------------------------------ rate limit


def test_assistant_is_rate_limited_per_client(monkeypatch, canned_assistant):
    monkeypatch.setattr(security.llm_limiter, "limit", 2)
    with TestClient(app_module.app) as client:
        codes = [client.post("/assistant/query", json={"query": "hi"}).status_code for _ in range(3)]
        blocked = client.post("/assistant/query", json={"query": "hi"})

    assert codes == [200, 200, 429]
    assert int(blocked.headers["Retry-After"]) >= 1


def test_non_llm_endpoints_are_not_rate_limited(monkeypatch):
    monkeypatch.setattr(security.llm_limiter, "limit", 1)
    with TestClient(app_module.app) as client:
        assert all(client.get("/model-info").status_code == 200 for _ in range(5))


def test_overlong_assistant_query_is_rejected_before_reaching_the_llm(monkeypatch):
    def _must_not_run(query):
        raise AssertionError("LLM called for an oversized query")

    monkeypatch.setattr(assistant_router, "ai_run_query", _must_not_run)
    with TestClient(app_module.app) as client:
        assert client.post("/assistant/query", json={"query": "x" * 2001}).status_code == 422


def test_limiter_window_expires(monkeypatch):
    limiter = security.RateLimiter(limit=1, window_s=60)
    clock = iter([0.0, 1.0, 61.0])
    monkeypatch.setattr(security.time, "monotonic", lambda: next(clock))

    assert limiter.hit("a") is None
    assert limiter.hit("a") == pytest.approx(59.0)
    assert limiter.hit("a") is None


def test_limits_are_tracked_per_client():
    limiter = security.RateLimiter(limit=1)
    assert limiter.hit("a") is None
    assert limiter.hit("b") is None
    assert limiter.hit("a") is not None


class _Req:
    def __init__(self, host, headers=None):
        self.client = type("C", (), {"host": host})()
        self.headers = headers or {}


def test_forwarded_for_is_ignored_unless_proxy_headers_are_trusted():
    req = _Req("10.0.0.1", {"x-forwarded-for": "1.2.3.4"})
    assert security.client_id(req) == "10.0.0.1"


def test_forwarded_for_is_used_when_trusted(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
    req = _Req("10.0.0.1", {"x-forwarded-for": "1.2.3.4, 10.0.0.1"})
    assert security.client_id(req) == "1.2.3.4"
