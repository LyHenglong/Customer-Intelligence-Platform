"""
Tests for src/model/dashboard_queries.py - the warehouse/filesystem read
functions behind the dashboard-facing FastAPI endpoints (originally
extracted out of the now-retired Streamlit dashboard, which shared this
same code).

Deliberately built on a fake Postgres connection/cursor, like the rest of
this suite (see tests/test_ai_tools.py) - no live database.
"""

from __future__ import annotations

import pytest

from src.model import dashboard_queries


class _FakeCursor:
    def __init__(self, result):
        self._result = result
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((" ".join(query.split()), params))

    def fetchone(self):
        return self._result

    def fetchall(self):
        return self._result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, result):
        self._result = result
        self.last_cursor = None
        self.closed = False

    def cursor(self, *a, **kw):
        self.last_cursor = _FakeCursor(self._result)
        return self.last_cursor

    def close(self):
        self.closed = True


def test_load_overall_stats_returns_typed_dict(monkeypatch):
    monkeypatch.setattr(dashboard_queries, "get_pg_conn", lambda: _FakeConn((1_000_000, 0.0992)))

    result = dashboard_queries.load_overall_stats()

    assert result == {"total_customers": 1_000_000, "churn_rate": pytest.approx(0.0992)}


def test_load_segment_rates_rejects_unknown_column(monkeypatch):
    monkeypatch.setattr(dashboard_queries, "get_pg_conn", lambda: (_ for _ in ()).throw(AssertionError("must not connect")))

    with pytest.raises(ValueError):
        dashboard_queries.load_segment_rates("customer_id; DROP TABLE marts.customer_360")


def test_load_segment_rates_allows_a_real_categorical_column(monkeypatch):
    rows = [("month-to-month", 0.4, 500), ("two_year", 0.05, 300)]
    conn = _FakeConn(rows)
    monkeypatch.setattr(dashboard_queries, "get_pg_conn", lambda: conn)

    result = dashboard_queries.load_segment_rates("contract")

    assert list(result.columns) == ["contract", "churn_rate", "n_customers"]
    assert len(result) == 2
    assert conn.closed is True


def test_load_customers_by_id_empty_tuple_skips_the_database(monkeypatch):
    monkeypatch.setattr(dashboard_queries, "get_pg_conn", lambda: (_ for _ in ()).throw(AssertionError("must not connect")))

    result = dashboard_queries.load_customers_by_id(())

    assert list(result.columns) == dashboard_queries._DASHBOARD_COLUMNS
    assert len(result) == 0


def test_load_customers_by_id_queries_with_the_given_ids(monkeypatch):
    conn = _FakeConn([])
    monkeypatch.setattr(dashboard_queries, "get_pg_conn", lambda: conn)

    dashboard_queries.load_customers_by_id(("CUST0001", "CUST0002"))

    query, params = conn.last_cursor.executed[0]
    assert "WHERE customer_id = ANY(%s)" in query
    assert params == (["CUST0001", "CUST0002"],)


def test_load_latest_drift_returns_empty_frame_when_table_missing(monkeypatch):
    conn = _FakeConn((None,))  # to_regclass(...) returns NULL - table doesn't exist
    monkeypatch.setattr(dashboard_queries, "get_pg_conn", lambda: conn)

    result = dashboard_queries.load_latest_drift()

    assert result.empty


def test_load_all_churn_metadata_skips_unreadable_files(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_queries, "MODELS_DIR", tmp_path)
    (tmp_path / "churn_model_20260101T000000Z.json").write_text('{"version": "20260101T000000Z"}')
    (tmp_path / "churn_model_20260102T000000Z.json").write_text("not valid json")

    result = dashboard_queries.load_all_churn_metadata()

    assert len(result) == 1
    assert result[0]["version"] == "20260101T000000Z"


def test_column_importances_maps_expanded_names_back_to_columns():
    class _FakePreprocessor:
        def get_feature_names_out(self):
            return ["num__num_complaints", "cat__contract_two_year", "cat__contract_month-to-month"]

    class _FakeModel:
        feature_importances_ = [0.5, 0.3, 0.2]

    class _FakePipeline:
        named_steps = {"preprocess": _FakePreprocessor(), "model": _FakeModel()}

    result = dashboard_queries.column_importances(_FakePipeline())

    assert result["num_complaints"] == pytest.approx(0.5)
    assert result["contract"] == pytest.approx(0.5)  # both cat__contract_* buckets summed


def test_column_importances_prefers_gain_over_lightgbms_split_counts():
    """Regression: LightGBM's feature_importances_ defaults to split counts,
    which are biased toward high-cardinality continuous columns. That put
    credit_score (univariate AUC 0.51 - noise) top of the dashboard and the
    AI assistant, while contract (4.95x churn separation) fell outside the
    top six. Gain ranks them correctly."""
    class _FakeBooster:
        def feature_importance(self, importance_type):
            assert importance_type == "gain"
            return [10.0, 900.0]

    class _FakePreprocessor:
        def get_feature_names_out(self):
            return ["num__credit_score", "cat__contract_two_year"]

    class _FakeModel:
        booster_ = _FakeBooster()
        # What the old code read: split counts telling the opposite story.
        feature_importances_ = [900.0, 10.0]

    class _FakePipeline:
        named_steps = {"preprocess": _FakePreprocessor(), "model": _FakeModel()}

    result = dashboard_queries.column_importances(_FakePipeline())

    assert result["contract"] > result["credit_score"]
    assert result["contract"] == pytest.approx(900.0)


def test_column_importances_falls_back_when_there_is_no_booster():
    """Non-LightGBM estimators (this project shipped RandomForest first)
    expose mean impurity decrease via feature_importances_, which is already
    a gain measure - no booster to read."""
    class _FakePreprocessor:
        def get_feature_names_out(self):
            return ["num__tenure"]

    class _FakeModel:
        feature_importances_ = [0.75]

    class _FakePipeline:
        named_steps = {"preprocess": _FakePreprocessor(), "model": _FakeModel()}

    assert dashboard_queries.column_importances(_FakePipeline())["tenure"] == pytest.approx(0.75)


def test_column_importances_survives_a_booster_that_raises():
    class _AngryBooster:
        def feature_importance(self, importance_type):
            raise RuntimeError("booster not available")

    class _FakePreprocessor:
        def get_feature_names_out(self):
            return ["num__tenure"]

    class _FakeModel:
        booster_ = _AngryBooster()
        feature_importances_ = [0.42]

    class _FakePipeline:
        named_steps = {"preprocess": _FakePreprocessor(), "model": _FakeModel()}

    assert dashboard_queries.column_importances(_FakePipeline())["tenure"] == pytest.approx(0.42)


def test_column_importances_returns_empty_dict_on_unexpected_pipeline_shape():
    class _NotAPipeline:
        named_steps = {}

    assert dashboard_queries.column_importances(_NotAPipeline()) == {}
