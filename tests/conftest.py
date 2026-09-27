"""Shared test-session setup.

Nothing in tests/ should depend on a live external service being
reachable from whatever the developer's ambient shell environment happens
to be (e.g. after `source .env` for local dev work) - that's exactly the
kind of hidden coupling that makes a test suite flaky. src/model/
registry.py's resolve_model_path() makes a real network call to MLflow
when MLFLOW_TRACKING_URI is set, which none of the existing tests (e.g.
tests/test_ai_tools.py) were written to expect or mock - a real hang was
reproduced this way (a live MLflow lookup repeated across many tests,
compounding into multi-minute stalls even with a short per-call timeout).
"""

import pytest


@pytest.fixture(autouse=True)
def _no_live_mlflow_calls(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)


@pytest.fixture(autouse=True)
def _isolated_api_security(monkeypatch):
    """The LLM rate limiter is process-global, so without a reset one test's
    calls count against the next. API_KEYS from a developer's shell would
    likewise 401 every endpoint test that doesn't send a key."""
    from src.api.security import llm_limiter

    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    llm_limiter.reset()
    yield
    llm_limiter.reset()
