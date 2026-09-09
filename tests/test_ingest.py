"""
Unit tests for the DuckDB cleaning/validation/feature-engineering logic in
src/ingest/batch_loader.py. Deliberately tests process_batch() only (not
load_batch()/get_pg_conn()), so these run without a live Postgres
connection - safe for CI.
"""

from pathlib import Path

import pytest

from src.ingest.batch_loader import CLEANED_COLUMNS, RAW_COLUMNS, process_batch

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_batch.csv"


@pytest.fixture(scope="module")
def batch_result():
    raw_rows, cleaned_rows, stats = process_batch(FIXTURE_PATH)
    return raw_rows, cleaned_rows, stats


def _as_dicts(rows, columns):
    return [dict(zip(columns, row)) for row in rows]


def test_raw_layer_keeps_all_rows(batch_result):
    raw_rows, _, stats = batch_result
    # 5 rows in the fixture: all pass type-cast (no malformed values), so
    # all 5 land in the raw layer even though 2 get dropped from cleaned.
    assert stats["total_rows"] == 5
    assert len(raw_rows) == 5


def test_cleaned_layer_drops_invalid_rows(batch_result):
    """CUST_TEST_0004 has churn=2 (outside {0,1}) and CUST_TEST_0005 has
    tenure=-3 (negative) - both should be dropped from the cleaned layer
    as a data-quality gate, while everything else survives."""
    _, cleaned_rows, stats = batch_result
    cleaned = _as_dicts(cleaned_rows, CLEANED_COLUMNS)
    ids = {row["customer_id"] for row in cleaned}

    assert "CUST_TEST_0004" not in ids
    assert "CUST_TEST_0005" not in ids
    assert ids == {"CUST_TEST_0001", "CUST_TEST_0002", "CUST_TEST_0003"}
    assert stats["invalid_rows_dropped"] == 2
    assert stats["loaded_rows"] == 3


def test_type_casting(batch_result):
    raw_rows, _, _ = batch_result
    raw = _as_dicts(raw_rows, RAW_COLUMNS)
    row = next(r for r in raw if r["customer_id"] == "CUST_TEST_0001")

    assert isinstance(row["age"], int)
    assert row["age"] == 35
    assert isinstance(row["monthlycharges"], float)
    assert row["monthlycharges"] == pytest.approx(80.00)
    assert row["churn"] == 0


def test_null_values_preserved_not_coerced(batch_result):
    """CUST_TEST_0003 has a blank annual_income and blank credit_score in
    the source CSV - these must stay NULL (None), not become 0 or empty
    string, through both the raw and cleaned layers."""
    _, cleaned_rows, _ = batch_result
    cleaned = _as_dicts(cleaned_rows, CLEANED_COLUMNS)
    row = next(r for r in cleaned if r["customer_id"] == "CUST_TEST_0003")

    assert row["annual_income"] is None
    assert row["credit_score"] is None


def test_boolean_flags_converted(batch_result):
    _, cleaned_rows, _ = batch_result
    cleaned = _as_dicts(cleaned_rows, CLEANED_COLUMNS)
    row = next(r for r in cleaned if r["customer_id"] == "CUST_TEST_0001")

    assert row["paperless_billing"] is True
    assert row["senior_citizen"] is False
    assert row["has_phone_service"] is True
    assert row["has_online_backup"] is False


def test_engineered_features(batch_result):
    """CUST_TEST_0001: tenure=24 -> tenure_years=2.0; 3 active services
    (phone, internet, online_security) with avg_monthly_gb=30.00 ->
    avg_gb_per_service=10.0."""
    _, cleaned_rows, _ = batch_result
    cleaned = _as_dicts(cleaned_rows, CLEANED_COLUMNS)
    row = next(r for r in cleaned if r["customer_id"] == "CUST_TEST_0001")

    assert row["tenure_years"] == pytest.approx(2.0)
    assert row["total_active_services"] == 3
    assert row["avg_gb_per_service"] == pytest.approx(10.0)


def test_engineered_features_zero_active_services():
    """A customer with zero active services should get a NULL (not a
    divide-by-zero error) for avg_gb_per_service."""
    import csv
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        with open(FIXTURE_PATH, encoding="utf-8") as src:
            reader = csv.reader(src)
            header = next(reader)
            writer = csv.writer(f)
            writer.writerow(header)
            row = next(r for r in csv.DictReader(open(FIXTURE_PATH, encoding="utf-8")) if r["customer_id"] == "CUST_TEST_0001")
            row = dict(row)
            row["customer_id"] = "CUST_TEST_ZERO"
            for svc in [
                "has_phone_service", "has_internet_service", "has_online_security",
                "has_online_backup", "has_device_protection", "has_tech_support",
                "has_streaming_tv", "has_streaming_movies",
            ]:
                row[svc] = "0"
            writer.writerow([row[c] for c in header])
        temp_path = Path(f.name)

    try:
        _, cleaned_rows, _ = process_batch(temp_path)
        cleaned = _as_dicts(cleaned_rows, CLEANED_COLUMNS)
        target = next(r for r in cleaned if r["customer_id"] == "CUST_TEST_ZERO")
        assert target["total_active_services"] == 0
        assert target["avg_gb_per_service"] is None
    finally:
        temp_path.unlink(missing_ok=True)
