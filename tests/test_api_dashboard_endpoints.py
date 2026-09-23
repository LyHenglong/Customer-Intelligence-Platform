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

# Captured before the autouse fixture below replaces the module attribute
# with a no-op, so the few tests that exercise the warm itself can still
# reach the real implementation.
_REAL_WARM_SCORED_CACHE = api_module._warm_scored_cache


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
    # The startup handler also kicks off a background thread that scores the
    # whole population to warm _scored_cache. Left alone it hits the real
    # warehouse from every TestClient context, races the fake data these
    # tests set up, and takes the suite from seconds to minutes. Patching
    # the target works where patching the handler does not: FastAPI captured
    # start_cache_warm at decoration time, but it resolves
    # _warm_scored_cache from module globals when it runs.
    monkeypatch.setattr(api_module, "_warm_scored_cache", lambda: None)
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
    """Mirrors what dashboard_queries.score_all_customers actually returns,
    including the `category`-dtype segment columns that ride along with the
    scores so /overview/revenue-at-risk-by-segment can group the whole
    at-risk population without a per-row Postgres fetch."""
    return pd.DataFrame({
        "customer_id": ["CUST0001", "CUST0002", "CUST0003"],
        "churn_probability": [0.9, 0.5, 0.1],
        "monthlycharges": [80.0, 60.0, 40.0],
        "contract": pd.Categorical(["month-to-month", "month-to-month", "two_year"]),
        "tenure_bucket": pd.Categorical(["new_0_6mo", "loyal_24mo_plus", "loyal_24mo_plus"]),
    })


# --------------------------------------------------------------- overview


def test_overview_stats_combines_the_expected_sources(monkeypatch, fake_churn_state):
    monkeypatch.setattr(dashboard_queries, "load_overall_stats", lambda: {"total_customers": 1000, "churn_rate": 0.1})
    monkeypatch.setattr(dashboard_queries, "score_all_customers", lambda *a, **kw: _scored_df())
    # Carries a version now: model_auc is looked up by the serving version
    # rather than taken from the newest file, so unversioned metadata
    # correctly matches nothing.
    monkeypatch.setattr(dashboard_queries, "load_all_churn_metadata",
                        lambda: [{"version": "TESTV1", "roc_auc": 0.65}])
    monkeypatch.setattr(dashboard_queries, "column_importances", lambda pipeline: {"tenure": 0.4, "contract": 0.6})

    with TestClient(api_module.app) as client:
        monkeypatch.setitem(api_module._state, "churn_version", "TESTV1")
        response = client.get("/overview/stats", params={"threshold": 0.3})

    assert response.status_code == 200
    body = response.json()
    assert body["total_customers"] == 1000
    assert body["at_risk_count"] == 2  # 0.9 and 0.5 are >= 0.3
    assert body["revenue_at_risk"] == pytest.approx(140.0)  # 80 + 60
    assert body["model_auc"] == 0.65
    assert body["top_feature_importances"][0]["feature"] == "contract"  # sorted descending


def test_cache_warm_populates_the_scored_cache(monkeypatch, fake_churn_state):
    """The warm exists so the first visitor doesn't pay a ~35-110s
    full-population scoring pass (which also saturates the single uvicorn
    worker while it runs)."""
    monkeypatch.setattr(dashboard_queries, "score_all_customers", lambda *a, **kw: _scored_df())
    api_module._scored_cache.update(data=None, expires_at=0.0, version=None)

    _REAL_WARM_SCORED_CACHE()

    assert api_module._scored_cache["data"] is not None
    assert api_module._scored_cache["version"] == "TESTV1"


def test_cache_warm_is_a_no_op_without_a_model(monkeypatch):
    def _must_not_run(*a, **kw):
        raise AssertionError("scoring attempted with no churn model loaded")

    monkeypatch.setattr(dashboard_queries, "score_all_customers", _must_not_run)
    monkeypatch.setitem(api_module._state, "churn", None)

    _REAL_WARM_SCORED_CACHE()  # must not raise


def test_cache_warm_failure_never_propagates(monkeypatch, fake_churn_state):
    """It is an optimisation - if it fails the request path just pays the
    cost itself, exactly as before. A raising warm thread must not be able
    to take the API down with it."""
    def _boom(*a, **kw):
        raise RuntimeError("warehouse unreachable")

    monkeypatch.setattr(dashboard_queries, "score_all_customers", _boom)

    _REAL_WARM_SCORED_CACHE()  # must not raise

    assert api_module._scored_cache["data"] is None


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

    with TestClient(api_module.app) as client:
        response = client.get("/overview/revenue-at-risk-by-segment", params={"threshold": 0.3})

    assert response.status_code == 200
    body = response.json()
    # CUST0003 (0.1) is below the threshold, so only the two
    # month-to-month customers count: 80 + 60.
    assert body["buckets"] == [{"segment": "month-to-month", "revenue_at_risk": 140.0}]


def test_revenue_at_risk_buckets_sum_to_the_overview_stats_headline(monkeypatch, fake_churn_state):
    """Regression: these buckets used to be computed over only the top
    `max_rows` (default 100) at-risk customers while /overview/stats
    reported the full population, so the chart silently disagreed with
    the KPI beside it by orders of magnitude at real data volumes."""
    monkeypatch.setattr(dashboard_queries, "load_overall_stats", lambda: {"total_customers": 3, "churn_rate": 0.1})
    monkeypatch.setattr(dashboard_queries, "score_all_customers", lambda *a, **kw: _scored_df())
    monkeypatch.setattr(dashboard_queries, "load_all_churn_metadata", lambda: [{"roc_auc": 0.65}])
    monkeypatch.setattr(dashboard_queries, "column_importances", lambda pipeline: {})

    with TestClient(api_module.app) as client:
        stats = client.get("/overview/stats", params={"threshold": 0.3}).json()
        buckets = client.get("/overview/revenue-at-risk-by-segment", params={"threshold": 0.3}).json()["buckets"]

    assert sum(b["revenue_at_risk"] for b in buckets) == pytest.approx(stats["revenue_at_risk"])


def test_revenue_at_risk_by_segment_supports_other_segment_columns(monkeypatch, fake_churn_state):
    monkeypatch.setattr(dashboard_queries, "score_all_customers", lambda *a, **kw: _scored_df())

    with TestClient(api_module.app) as client:
        response = client.get(
            "/overview/revenue-at-risk-by-segment",
            params={"threshold": 0.3, "segment_column": "tenure_bucket"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["segment_column"] == "tenure_bucket"
    assert {b["segment"]: b["revenue_at_risk"] for b in body["buckets"]} == {
        "new_0_6mo": 80.0, "loyal_24mo_plus": 60.0,
    }


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

    # churn_version is set *inside* the context on purpose: TestClient's
    # __enter__ fires the real startup handler, which FastAPI captured at
    # decoration time, so patching api_module.load_models beforehand does
    # not stop it repopulating _state. Same workaround as
    # test_overview_stats_503s_when_churn_model_not_loaded above.
    with TestClient(api_module.app) as client:
        monkeypatch.setitem(api_module._state, "churn_version", "V2")
        response = client.get("/model-history")

    assert response.status_code == 200
    assert response.json() == {
        "versions": [{"version": "V1"}, {"version": "V2"}],
        "serving_version": "V2",
    }


def test_model_history_reports_the_served_version_not_the_newest(monkeypatch):
    """Regression: the registry can resolve an older artifact than the
    newest on disk - an MLflow "champion" alias pinned to a previous
    version, or a retrain that saved but failed to register. The frontend
    labelled the last entry "current production model", so it showed a
    non-serving model's threshold and confusion matrix as live."""
    monkeypatch.setattr(dashboard_queries, "load_all_churn_metadata",
                        lambda: [{"version": "V1"}, {"version": "V2_NEWEST"}])

    with TestClient(api_module.app) as client:
        monkeypatch.setitem(api_module._state, "churn_version", "V1")
        body = client.get("/model-history").json()

    assert body["serving_version"] == "V1"
    assert body["versions"][-1]["version"] == "V2_NEWEST"


def test_overview_stats_auc_describes_the_served_model(monkeypatch, fake_churn_state):
    """Regression: model_auc came from the newest metadata file while
    model_version came from the loaded model, so a single response paired
    one model's version with another model's score."""
    monkeypatch.setattr(dashboard_queries, "load_overall_stats", lambda: {"total_customers": 3, "churn_rate": 0.1})
    monkeypatch.setattr(dashboard_queries, "score_all_customers", lambda *a, **kw: _scored_df())
    monkeypatch.setattr(dashboard_queries, "column_importances", lambda pipeline: {})
    monkeypatch.setattr(dashboard_queries, "load_all_churn_metadata", lambda: [
        {"version": "TESTV1", "roc_auc": 0.61},
        {"version": "NEWER_BUT_NOT_SERVED", "roc_auc": 0.99},
    ])

    with TestClient(api_module.app) as client:
        monkeypatch.setitem(api_module._state, "churn_version", "TESTV1")
        body = client.get("/overview/stats", params={"threshold": 0.3}).json()

    assert body["model_version"] == "TESTV1"
    assert body["model_auc"] == 0.61


def test_overview_stats_auc_is_null_when_served_version_has_no_metadata(monkeypatch, fake_churn_state):
    """Absent is better than wrong: rather than falling back to some other
    model's AUC, report nothing."""
    monkeypatch.setattr(dashboard_queries, "load_overall_stats", lambda: {"total_customers": 3, "churn_rate": 0.1})
    monkeypatch.setattr(dashboard_queries, "score_all_customers", lambda *a, **kw: _scored_df())
    monkeypatch.setattr(dashboard_queries, "column_importances", lambda pipeline: {})
    monkeypatch.setattr(dashboard_queries, "load_all_churn_metadata",
                        lambda: [{"version": "SOMETHING_ELSE", "roc_auc": 0.99}])

    with TestClient(api_module.app) as client:
        monkeypatch.setitem(api_module._state, "churn_version", "TESTV1")
        body = client.get("/overview/stats", params={"threshold": 0.3}).json()

    assert body["model_auc"] is None


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
