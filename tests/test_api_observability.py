"""Request IDs, Prometheus metrics and structured logs (src/api/observability.py)."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api import observability


@pytest.fixture(autouse=True)
def _no_real_startup_work(monkeypatch):
    monkeypatch.setattr(app_module, "load_models", lambda: None)
    monkeypatch.setattr(app_module, "_warm_all", lambda: None)


def _sample(route: str, status: str) -> float:
    value = observability.REQUESTS.labels(route, "GET", status)._value.get()
    return value


def test_every_response_carries_a_request_id():
    with TestClient(app_module.app) as client:
        response = client.get("/health")
    assert len(response.headers["X-Request-ID"]) == 32


def test_a_caller_supplied_request_id_is_propagated():
    with TestClient(app_module.app) as client:
        response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"


def test_metrics_are_labelled_by_route_template_not_raw_path(monkeypatch):
    from src.ai.schemas import CustomerLookupResult
    from src.api.routers import dashboard as dashboard_router

    monkeypatch.setattr(
        dashboard_router, "ai_customer_lookup", lambda cid: CustomerLookupResult(customer_id=cid, found=False)
    )
    before = _sample("/customers/{customer_id}", "200")

    with TestClient(app_module.app) as client:
        client.get("/customers/CUST0001")
        client.get("/customers/CUST0002")
        body = client.get("/metrics").text

    assert _sample("/customers/{customer_id}", "200") == before + 2
    assert "CUST0001" not in body
    assert 'api_request_duration_seconds_bucket{le="0.01",method="GET",route="/customers/{customer_id}"}' in body


def test_error_statuses_are_counted():
    before = _sample("/assistant/trace/{trace_id}", "404")
    from src.api.routers import assistant as assistant_router

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(assistant_router, "ai_get_trace", lambda trace_id: None)
        with TestClient(app_module.app) as client:
            client.get("/assistant/trace/nope")

    assert _sample("/assistant/trace/{trace_id}", "404") == before + 1


def test_json_formatter_emits_the_structured_fields():
    record = logging.LogRecord("api.access", logging.INFO, __file__, 1, "GET /health 200", None, None)
    record.fields = {"request_id": "r1", "status": 200}

    payload = json.loads(observability.JsonFormatter().format(record))

    assert payload["message"] == "GET /health 200"
    assert payload["request_id"] == "r1"
    assert payload["status"] == 200
    assert payload["level"] == "INFO"
