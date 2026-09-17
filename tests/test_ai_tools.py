"""Tests for the AI tool layer (src/ai/tools/) - Phase 1 of the AI
assistant architecture (see AI_Customer_Intelligence_Claude_Code_Plan.md).

Deliberately built on synthetic in-memory data and fake Postgres
connections, like test_model.py and test_dashboard.py - no live database,
no saved artifact, no external API. Every tool under test is a thin
wrapper around already-tested modules (train_churn, train_recommender,
explain_churn, agents/cache); these tests pin the *wrapping* contract -
correct SQL shape, correct artifact routing, graceful handling of missing
artifacts/customers - not the underlying model math, which is covered
elsewhere.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import NearestNeighbors

from src.ai.schemas import CustomerSearchFilters
from src.ai.tools import _artifacts
from src.model.train_churn import (
    ALL_FEATURES,
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_pipeline,
)
from src.model.train_recommender import (
    PROFILE_CATEGORICAL,
    PROFILE_NUMERIC,
    SERVICE_COLUMNS,
    build_preprocessor,
)

N_ROWS = 60


def _synthetic_customers(n=N_ROWS, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC_FEATURES})
    for c in BOOLEAN_FEATURES:
        df[c] = rng.integers(0, 2, size=n).astype(float)
    for c in CATEGORICAL_FEATURES:
        df[c] = rng.choice(["a", "b", "c"], size=n)
    # tenure/total_active_services are INTEGER columns in db/schema.sql (see
    # marts.customer_360) - CustomerProfile (src/ai/schemas.py) types them
    # as int accordingly, so the fixture must match that real shape rather
    # than the continuous floats used for the other numeric features.
    df["tenure"] = rng.integers(1, 72, size=n)
    df["total_active_services"] = rng.integers(0, 8, size=n)
    df[TARGET] = (df[NUMERIC_FEATURES[0]] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    df["customer_id"] = [f"CUST{i:04d}" for i in range(n)]
    return df


# --------------------------------------------------------------- fake DB


_WHERE_COND_RE = re.compile(r"(\w+)\s*(=|>=|<=)\s*%s")


def _apply_where(df: pd.DataFrame, query: str, params: list):
    head = query.split("GROUP BY")[0].split("ORDER BY")[0].split("LIMIT")[0]
    conds = _WHERE_COND_RE.findall(head)
    n = len(conds)
    where_params, rest_params = params[:n], params[n:]
    result = df
    for (col, op), val in zip(conds, where_params):
        if op == "=":
            result = result[result[col] == val]
        elif op == ">=":
            result = result[result[col] >= val]
        elif op == "<=":
            result = result[result[col] <= val]
    return result, rest_params


class _FakeCursor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.description = None
        self._rows = []

    def execute(self, query, params=None):
        params = list(params) if params else []
        q = " ".join(query.split())
        upper = q.upper()

        if upper.startswith("SELECT COUNT(*)"):
            filtered, _ = _apply_where(self.df, q, params)
            self.description = [("count",)]
            self._rows = [(len(filtered),)]
            return

        if "TO_REGCLASS" in upper:
            self.description = [("to_regclass",)]
            self._rows = [(None,)]
            return

        if "GROUP BY" in upper:
            m = re.match(r"SELECT\s+(\w+)::text", q)
            dim = m.group(1)
            rows = []
            for key, g in self.df.groupby(dim):
                rows.append((
                    str(key), float(g["churn"].astype(int).mean()),
                    int(len(g)), float(g["monthlycharges"].mean()),
                ))
            rows.sort(key=lambda r: r[0])
            self.description = [(dim,), ("churn_rate",), ("n",), ("avg_charges",)]
            self._rows = rows
            return

        cols_match = re.match(r"SELECT\s+(.*?)\s+FROM", q)
        cols = [c.strip() for c in cols_match.group(1).split(",")]
        filtered, rest_params = _apply_where(self.df, q, params)
        if "customer_id" in filtered.columns:
            filtered = filtered.sort_values("customer_id")

        limit = offset = None
        if "OFFSET" in upper:
            limit, offset = rest_params[0], rest_params[1]
        elif "LIMIT" in upper:
            limit = rest_params[0]
        if limit is not None:
            filtered = filtered.iloc[(offset or 0): (offset or 0) + limit]

        self.description = [(c,) for c in cols]
        self._rows = [tuple(row[c] for c in cols) for _, row in filtered.iterrows()]

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def cursor(self, *a, **kw):
        return _FakeCursor(self.df)

    def close(self):
        pass


def _fake_stream_query(df: pd.DataFrame):
    def _run(query, columns, batch_rows=50_000, transform=None, params=None):
        filtered, _ = _apply_where(df, " ".join(query.split()), list(params) if params else [])
        rows = filtered[columns]
        frames = []
        for start in range(0, len(rows), batch_rows):
            batch = rows.iloc[start:start + batch_rows].reset_index(drop=True)
            if transform is not None:
                batch = transform(batch)
            frames.append(batch)
        if not frames:
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True)
    return _run


# --------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def customers_df():
    return _synthetic_customers()


@pytest.fixture(scope="module")
def churn_artifact(customers_df):
    X, y = customers_df[ALL_FEATURES], customers_df[TARGET]
    base = build_pipeline()
    base.fit(X.iloc[:40], y.iloc[:40])
    calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid", cv="prefit")
    calibrated.fit(X.iloc[40:], y.iloc[40:])
    proba = calibrated.predict_proba(X)[:, 1]
    return {
        "pipeline": calibrated,
        "base_pipeline": base,
        "features": ALL_FEATURES,
        "threshold": float(np.median(proba)),
        "calibrated": True,
    }


@pytest.fixture(scope="module")
def recommender_artifact(customers_df):
    preprocessor = build_preprocessor()
    X_profile = preprocessor.fit_transform(customers_df[PROFILE_NUMERIC + PROFILE_CATEGORICAL])
    X_profile = np.asarray(X_profile.todense()) if hasattr(X_profile, "todense") else np.asarray(X_profile)
    X_profile = X_profile.astype(np.float32)
    nn_model = NearestNeighbors(n_neighbors=min(6, len(customers_df)), metric="euclidean")
    nn_model.fit(X_profile)
    return {
        "preprocessor": preprocessor,
        "nn_model": nn_model,
        "X_profile": X_profile,
        "customer_ids": customers_df["customer_id"].to_numpy(),
        "service_matrix": customers_df[SERVICE_COLUMNS].to_numpy(),
        "service_columns": SERVICE_COLUMNS,
        "profile_numeric": PROFILE_NUMERIC,
        "profile_categorical": PROFILE_CATEGORICAL,
    }


@pytest.fixture(scope="module")
def artifacts_dir(tmp_path_factory, churn_artifact, recommender_artifact):
    import joblib

    d = tmp_path_factory.mktemp("models_store")
    joblib.dump(churn_artifact, d / "churn_model_TESTV1.joblib")
    joblib.dump(recommender_artifact, d / "recommender_TESTV1.joblib")
    return d


@pytest.fixture(autouse=True)
def _patch_models_dir(monkeypatch, artifacts_dir):
    monkeypatch.setattr(_artifacts, "MODELS_DIR", artifacts_dir)
    _artifacts.clear_artifact_cache()
    yield
    _artifacts.clear_artifact_cache()


# ----------------------------------------------------------- customer_tool


class TestCustomerLookup:
    def test_found_customer_has_prediction_and_shap_and_recommendation(self, monkeypatch, customers_df):
        from src.ai.tools import customer_tool

        monkeypatch.setattr(customer_tool, "get_pg_conn", lambda: _FakeConn(customers_df))
        result = customer_tool.customer_lookup("CUST0005")

        assert result.found is True
        assert result.profile.customer_id == "CUST0005"
        assert 0.0 <= result.churn_probability <= 1.0
        assert result.risk_status == (
            "high" if result.churn_probability >= result.churn_threshold else "low"
        )
        assert len(result.shap_factors) == 5
        assert all(f.feature in ALL_FEATURES for f in result.shap_factors)
        assert result.recommendation, "recommender fast path should return at least one service"
        assert all(r.service in SERVICE_COLUMNS for r in result.recommendation)

    def test_unknown_customer_returns_not_found(self, monkeypatch, customers_df):
        from src.ai.tools import customer_tool

        monkeypatch.setattr(customer_tool, "get_pg_conn", lambda: _FakeConn(customers_df))
        result = customer_tool.customer_lookup("CUST9999")

        assert result.found is False
        assert result.profile is None
        assert result.churn_probability is None
        assert result.shap_factors == []


class TestCustomerSearch:
    def test_filters_and_paginates_without_scoring(self, monkeypatch, customers_df):
        from src.ai.tools import customer_tool

        monkeypatch.setattr(customer_tool, "get_pg_conn", lambda: _FakeConn(customers_df))
        expected_total = int((customers_df["contract"] == "a").sum())

        result = customer_tool.customer_search(
            CustomerSearchFilters(contract="a"), limit=3, offset=0
        )

        assert result.total_matched == expected_total
        assert len(result.customers) == min(3, expected_total)
        assert all(c.customer_id for c in result.customers)

    def test_limit_is_capped(self, monkeypatch, customers_df):
        from src.ai.tools import customer_tool

        monkeypatch.setattr(customer_tool, "get_pg_conn", lambda: _FakeConn(customers_df))
        result = customer_tool.customer_search(CustomerSearchFilters(), limit=10_000)
        assert result.limit == customer_tool._SEARCH_MAX_LIMIT

    def test_churn_probability_filter_triggers_scoring_path(self, monkeypatch, customers_df):
        from src.ai.tools import customer_tool

        monkeypatch.setattr(customer_tool, "get_pg_conn", lambda: _FakeConn(customers_df))
        result = customer_tool.customer_search(
            CustomerSearchFilters(min_churn_probability=0.0), limit=5
        )
        assert len(result.customers) <= 5
        assert result.total_matched >= len(result.customers)


# ---------------------------------------------------------- aggregate_tool


class TestAggregateAnalysis:
    def test_valid_dimension_returns_one_bucket_per_group(self, monkeypatch, customers_df):
        from src.ai.tools import aggregate_tool

        monkeypatch.setattr(aggregate_tool, "get_pg_conn", lambda: _FakeConn(customers_df))
        result = aggregate_tool.aggregate_analysis("contract")

        assert result.dimension == "contract"
        assert {b.key for b in result.buckets} == set(customers_df["contract"].unique())
        assert sum(b.n_customers for b in result.buckets) == len(customers_df)
        for b in result.buckets:
            assert 0.0 <= b.churn_rate <= 1.0

    def test_unknown_dimension_rejected(self):
        from src.ai.tools import aggregate_tool

        with pytest.raises(ValueError):
            aggregate_tool.aggregate_analysis("customer_id")  # not an allowed grouping column


# -------------------------------------------------------------- churn_tool


class TestChurnAnalysis:
    def test_unfiltered_population_stats(self, monkeypatch, customers_df):
        from src.ai.tools import churn_tool

        monkeypatch.setattr(churn_tool, "stream_query", _fake_stream_query(customers_df))
        result = churn_tool.churn_analysis()

        assert result.population_size == len(customers_df)
        assert result.current_churn_rate == pytest.approx(customers_df[TARGET].mean(), abs=1e-4)
        assert 0 <= result.predicted_high_risk_count <= len(customers_df)
        assert result.threshold > 0

    def test_unknown_filter_key_rejected(self):
        from src.ai.tools import churn_tool

        with pytest.raises(ValueError):
            churn_tool.churn_analysis(filters={"not_a_real_column": "x"})


# --------------------------------------------------------------- shap_tool


class TestCustomerExplanation:
    def test_known_customer_returns_top_k_factors(self, monkeypatch, customers_df):
        from src.ai.tools import shap_tool

        monkeypatch.setattr(shap_tool, "get_pg_conn", lambda: _FakeConn(customers_df))
        factors = shap_tool.customer_explanation("CUST0010", top_k=3)

        assert len(factors) == 3
        assert all(f.direction in ("increases risk", "decreases risk") for f in factors)

    def test_unknown_customer_returns_empty_list(self, monkeypatch, customers_df):
        from src.ai.tools import shap_tool

        monkeypatch.setattr(shap_tool, "get_pg_conn", lambda: _FakeConn(customers_df))
        assert shap_tool.customer_explanation("CUST9999") == []


# ---------------------------------------------------------- recommender_tool


class TestRecommendationAnalysis:
    def test_indexed_customer_uses_fast_path(self, monkeypatch, customers_df):
        from src.ai.tools import recommender_tool

        monkeypatch.setattr(recommender_tool, "get_pg_conn", lambda: _FakeConn(customers_df))
        recs = recommender_tool.recommendation_analysis("CUST0003", top_n=2)

        assert len(recs) <= 2
        assert all(r.service in SERVICE_COLUMNS for r in recs)

    def test_no_artifact_returns_empty_list(self, monkeypatch, customers_df, tmp_path):
        from src.ai.tools import recommender_tool

        monkeypatch.setattr(_artifacts, "MODELS_DIR", tmp_path)  # empty dir, no artifact
        _artifacts.clear_artifact_cache()
        monkeypatch.setattr(recommender_tool, "get_pg_conn", lambda: _FakeConn(customers_df))

        assert recommender_tool.recommendation_analysis("CUST0003") == []


# ----------------------------------------------------------- retraining_tool


class TestRetrainingAnalysis:
    def test_computes_metric_changes_between_two_versions(self, monkeypatch, tmp_path):
        from src.ai.tools import retraining_tool

        older = {"version": "V1", "roc_auc": 0.60, "accuracy": 0.80}
        newer = {"version": "V2", "roc_auc": 0.65, "accuracy": 0.82}
        (tmp_path / "churn_model_V1.json").write_text(json.dumps(older))
        (tmp_path / "churn_model_V2.json").write_text(json.dumps(newer))

        monkeypatch.setattr(retraining_tool, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(retraining_tool, "get_pg_conn", lambda: _FakeConn(pd.DataFrame()))
        monkeypatch.setattr(
            retraining_tool, "get_latest_retrain_summary",
            lambda: {"summary_text": "retrained because of drift"},
        )

        result = retraining_tool.retraining_analysis()

        assert result.current_model_version == "V2"
        assert result.previous_model_version == "V1"
        assert result.metric_changes["roc_auc"] == pytest.approx(0.05)
        assert result.metric_changes["accuracy"] == pytest.approx(0.02)
        assert result.retrain_reason == "retrained because of drift"
        assert result.latest_drift == []  # to_regclass fake reports no table

    def test_no_metadata_files_returns_none_versions(self, monkeypatch, tmp_path):
        from src.ai.tools import retraining_tool

        monkeypatch.setattr(retraining_tool, "MODELS_DIR", tmp_path)
        monkeypatch.setattr(retraining_tool, "get_pg_conn", lambda: _FakeConn(pd.DataFrame()))
        monkeypatch.setattr(retraining_tool, "get_latest_retrain_summary", lambda: None)

        result = retraining_tool.retraining_analysis()

        assert result.current_model_version is None
        assert result.metric_changes == {}
        assert result.retrain_reason is None
