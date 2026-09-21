"""Tests for the dashboard-facing FastAPI endpoints added to
src/model/api.py (§B of the frontend build plan) - /overview/*, /at-risk,
/customers, /customers/{id}, /model-history, /pipeline-status,
/outreach-draft/{id}, and CORS configuration.

Same philosophy as tests/test_api_assistant.py: monkeypatch the
underlying functions (dashboard_queries.*, the customer_tool functions,
_get_explanation/_recommend_with_fallback) rather than hitting a live
database or a real model artifact - this pins the HTTP contract, not the
query/model logic already covered by tests/test_dashboard_queries.py and
tests/test_ai_tools.py.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.ai.schemas import CustomerLookupResult, CustomerProfile, CustomerSearchResult
from src.model import api as api_module
from src.model import dashboard_queries


@pytest.fixture(autouse=True)
def _isolate_api_state(monkeypatch):
    """TestClient(api_module.app) as a context manager fires the real
    @app.on_event("startup") handler (load_models()) on every __enter__ -
    a genuine call to resolve_model_path()/MLflow/disk that would
    silently overwrite whatever _state this test set up, and secretly
    make these "unit" tests depend on live infra (and take real seconds
    doing it). No-op it for this whole file - every test here manages
    _state itself instead. Also resets the module-level scored-customers
    cache, which would otherwise leak a DataFrame between tests."""
    monkeypatch.setattr(api_module, "load_models", lambda: None)
    api_module._scored_cache.update(data=None, expires_at=0.0, version=None)
    yield
    api_module._scored_cache.update(data=None, expires_at=0.0, version=None)


@pytest.fixture
def fake_churn_state(monkeypatch):
    """A minimal in-memory churn artifact - not a real fitted model, since
    every test here monkeypatches the functions that would actually touch
    it (dashboard_queries.score_all_customers, compute_shap_details,
    etc.) rather than exercising real ML math."""
    monkeypatch.setitem(api_module._state, "churn", {"pipeline": object(), "threshold": 0.3})
    monkeypatch.setitem(api_module._state, "churn_version", "TESTV1")


def _scored_df():
    return pd.DataFrame({
        "customer_id": ["CUST0001", "CUST0002", "CUST0003"],
        "churn_probability": [0.9, 0.5, 0.1],
        "monthlycharges": [80.0, 60.0, 40.0],
    })


# --------------------------------------------------------------- overview


def test_overview_stats_combines_the_expected_sources(monkeypatch, fake_churn_state):
    monkeypatch.setattr(dashboard_queries, "load_overall_stats", lambda: {"total_customers": 1000, "churn_rate": 0.1})
    monkeypatch.setattr(dashboard_queries, "score_all_customers", lambda *a, **kw: _scored_df())
    monkeypatch.setattr(dashboard_queries, "load_all_churn_metadata", lambda: [{"roc_auc": 0.65}])
    monkeypatch.setattr(dashboard_queries, "column_importances", lambda pipeline: {"tenure": 0.4, "contract": 0.6})

    with TestClient(api_module.app) as client:
        response = client.get("/overview/stats", params={"threshold": 0.3})

    assert response.status_code == 200
    body = response.json()
    assert body["total_customers"] == 1000
    assert body["at_risk_count"] == 2  # 0.9 and 0.5 are >= 0.3
    assert body["revenue_at_risk"] == pytest.approx(140.0)  # 80 + 60
    assert body["model_auc"] == 0.65
    assert body["top_feature_importances"][0]["feature"] == "contract"  # sorted descending


def test_overview_stats_503s_when_churn_model_not_loaded(monkeypatch):
    # _state["churn"] must be cleared *after* TestClient's startup event
    # runs (real load_models(), which would otherwise repopulate it from
    # whatever's actually on disk/MLflow) - setting it before entering the
    # `with` block gets silently overwritten.
    with TestClient(api_module.app) as client:
        monkeypatch.setitem(api_module._state, "churn", None)
        response = client.get("/overview/stats")

    assert response.status_code == 503


def test_overview_segment_rates_rejects_bad_column(monkeypatch):
    def _raise(column):
        raise ValueError(f"unexpected segment column {column!r}")

    monkeypatch.setattr(dashboard_queries, "load_segment_rates", _raise)

    with TestClient(api_module.app) as client:
        response = client.get("/overview/segment-rates", params={"column": "customer_id; DROP TABLE x"})

    assert response.status_code == 422


def test_overview_segment_rates_returns_buckets(monkeypatch):
    df = pd.DataFrame({"contract": ["month-to-month", "two_year"], "churn_rate": [0.4, 0.05], "n_customers": [500, 300]})
    monkeypatch.setattr(dashboard_queries, "load_segment_rates", lambda column: df)

    with TestClient(api_module.app) as client:
        response = client.get("/overview/segment-rates", params={"column": "contract"})

    assert response.status_code == 200
    body = response.json()
    assert body["column"] == "contract"
    assert len(body["buckets"]) == 2
    assert body["buckets"][0] == {"key": "month-to-month", "churn_rate": 0.4, "n_customers": 500}


def test_revenue_at_risk_by_segment_groups_correctly(monkeypatch, fake_churn_state):
    monkeypatch.setattr(dashboard_queries, "score_all_customers", lambda *a, **kw: _scored_df())
    full_rows = pd.DataFrame({
        "customer_id": ["CUST0001", "CUST0002"],
        "contract": ["month-to-month", "month-to-month"],
    })
    monkeypatch.setattr(dashboard_queries, "load_customers_by_id", lambda ids: full_rows)

    with TestClient(api_module.app) as client:
        response = client.get("/overview/revenue-at-risk-by-segment", params={"threshold": 0.3})

    assert response.status_code == 200
    body = response.json()
    assert body["buckets"] == [{"segment": "month-to-month", "revenue_at_risk": 140.0}]


def test_revenue_at_risk_by_segment_rejects_bad_segment_column(fake_churn_state):
    with TestClient(api_module.app) as client:
        response = client.get("/overview/revenue-at-risk-by-segment", params={"segment_column": "not_a_real_column"})

    assert response.status_code == 422


# --------------------------------------------------------------- at-risk


def test_at_risk_clamps_max_rows_server_side(monkeypatch, fake_churn_state):
    monkeypatch.setattr(dashboard_queries, "score_all_customers", lambda *a, **kw: _scored_df())
    monkeypatch.setattr(dashboard_queries, "load_customers_by_id", lambda ids: pd.DataFrame({"customer_id": list(ids)}))
    monkeypatch.setattr(api_module, "compute_shap_details", lambda pipeline, X, top_k: [[]])

    with TestClient(api_module.app) as client:
        response = client.get("/at-risk", params={"threshold": 0.0, "max_rows": 10_000_000})

    assert response.status_code == 200
    assert response.json()["max_rows_used"] == api_module._AT_RISK_MAX_ROWS


def test_at_risk_returns_customers_sorted_by_probability(monkeypatch, fake_churn_state):
    monkeypatch.setattr(dashboard_queries, "score_all_customers", lambda *a, **kw: _scored_df())
    monkeypatch.setattr(
        dashboard_queries, "load_customers_by_id",
        lambda ids: pd.DataFrame({"customer_id": list(ids), "contract": ["month-to-month"] * len(ids)}),
    )
    monkeypatch.setattr(api_module, "compute_shap_details", lambda pipeline, X, top_k: [[{"feature": "tenure", "shap_value": 0.1, "direction": "increases risk"}]])
    monkeypatch.setitem(api_module._state, "recommender", None)  # skip recommendation path

    with TestClient(api_module.app) as client:
        response = client.get("/at-risk", params={"threshold": 0.0})

    assert response.status_code == 200
    body = response.json()
    probs = [c["churn_probability"] for c in body["customers"]]
    assert probs == sorted(probs, reverse=True)
    assert body["total_at_risk"] == 3


# --------------------------------------------------------------- outreach-draft


def test_outreach_draft_soft_fails_when_no_recommendation(monkeypatch, fake_churn_state):
    monkeypatch.setattr(api_module, "_get_explanation", lambda customer_id: {
        "churn_probability": 0.8, "risk_factors": [], "explanation": "at risk", "source": "llm",
    })
    monkeypatch.setitem(api_module._state, "recommender", object())
    monkeypatch.setattr(api_module, "_recommend_with_fallback", lambda customer_id, top_n=1: [])

    with TestClient(api_module.app) as client:
        response = client.post("/outreach-draft/CUST0001")

    assert response.status_code == 200
    body = response.json()
    assert body["recommended_service"] is None
    assert body["draft"] is None
    assert body["explanation"] == "at risk"


def test_outreach_draft_404s_when_customer_not_found(monkeypatch, fake_churn_state):
    monkeypatch.setattr(api_module, "_get_explanation", lambda customer_id: None)

    with TestClient(api_module.app) as client:
        response = client.post("/outreach-draft/does-not-exist")

    assert response.status_code == 404


def test_outreach_draft_generates_when_a_service_is_recommended(monkeypatch, fake_churn_state):
    monkeypatch.setattr(api_module, "_get_explanation", lambda customer_id: {
        "churn_probability": 0.8, "risk_factors": [], "explanation": "at risk", "source": "llm",
    })
    monkeypatch.setitem(api_module._state, "recommender", object())
    monkeypatch.setattr(api_module, "_recommend_with_fallback", lambda customer_id, top_n=1: [{"service": "has_tech_support", "score": 0.9}])
    monkeypatch.setattr(api_module, "get_or_generate", lambda customer_id, agent_type, version, generate_fn: "We'd love to offer you tech support.")

    with TestClient(api_module.app) as client:
        response = client.post("/outreach-draft/CUST0001")

    assert response.status_code == 200
    body = response.json()
    assert body["recommended_service"] == "has_tech_support"
    assert body["draft"] == "We'd love to offer you tech support."
    assert body["draft_source"] == "llm"


# --------------------------------------------------------------- customers


def test_search_customers_wraps_customer_search(monkeypatch):
    canned = CustomerSearchResult(customers=[CustomerProfile(customer_id="CUST0001")], total_matched=1, limit=25, offset=0, truncated=False)
    captured = {}

    def _fake_search(filters, limit, offset):
        captured["filters"] = filters
        captured["limit"] = limit
        captured["offset"] = offset
        return canned

    monkeypatch.setattr(api_module, "ai_customer_search", _fake_search)

    with TestClient(api_module.app) as client:
        response = client.get("/customers", params={"contract": "month-to-month", "limit": 10})

    assert response.status_code == 200
    assert response.json()["total_matched"] == 1
    assert captured["filters"].contract == "month-to-month"
    assert captured["limit"] == 10


def test_get_customer_returns_200_with_found_false_on_a_miss(monkeypatch):
    monkeypatch.setattr(api_module, "ai_customer_lookup", lambda customer_id: CustomerLookupResult(customer_id=customer_id, found=False))

    with TestClient(api_module.app) as client:
        response = client.get("/customers/does-not-exist")

    assert response.status_code == 200  # deliberate - not a 404, see the endpoint's docstring
    assert response.json()["found"] is False


# --------------------------------------------------------------- model-history / pipeline-status


def test_model_history_returns_raw_metadata_list(monkeypatch):
    monkeypatch.setattr(dashboard_queries, "load_all_churn_metadata", lambda: [{"version": "V1"}, {"version": "V2"}])

    with TestClient(api_module.app) as client:
        response = client.get("/model-history")

    assert response.status_code == 200
    assert response.json() == {"versions": [{"version": "V1"}, {"version": "V2"}]}


def test_pipeline_status_combines_all_sources(monkeypatch):
    monkeypatch.setattr(dashboard_queries, "load_ingestion_log", lambda: pd.DataFrame({"batch_file": ["batch_001.csv"], "rows_loaded": [76924], "loaded_at": ["2026-01-01"], "status": ["success"]}))
    monkeypatch.setattr(dashboard_queries, "load_latest_drift", lambda: pd.DataFrame())
    monkeypatch.setattr(dashboard_queries, "load_all_churn_metadata", lambda: [{"version": "V1"}])
    monkeypatch.setattr(api_module, "get_latest_retrain_summary", lambda: {"summary_text": "improved"})

    with TestClient(api_module.app) as client:
        response = client.get("/pipeline-status")

    assert response.status_code == 200
    body = response.json()
    assert body["batches_ingested"] == 1
    assert body["total_simulated_batches"] == dashboard_queries.TOTAL_SIMULATED_BATCHES
    assert body["model_versions_trained"] == 1
    assert body["latest_retrain_summary"] == {"summary_text": "improved"}


# --------------------------------------------------------------- CORS


def test_cors_allows_configured_origin(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000")
    reloaded = importlib.reload(api_module)
    try:
        with TestClient(reloaded.app) as client:
            response = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    finally:
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
        importlib.reload(api_module)  # restore the module other tests in the suite import


def test_cors_blocks_unconfigured_origin_by_default(monkeypatch):
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    reloaded = importlib.reload(api_module)

    with TestClient(reloaded.app) as client:
        response = client.get("/health", headers={"Origin": "http://evil.example.com"})

    assert "access-control-allow-origin" not in response.headers
