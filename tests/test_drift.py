"""
Tests for PSI drift detection.

All synthetic - no database, no model artifacts - so these run in CI on a
clean checkout. The important tests here are the *positive* ones: on this
project's actual data PSI is near zero everywhere (the batches are slices
of one pre-shuffled file), so without tests that inject real shifts there
would be no evidence the detector can fire at all.
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    MODERATE_THRESHOLD,
    SIGNIFICANT_THRESHOLD,
    categorical_psi,
    compute_drift,
    numeric_psi,
    severity,
)

RNG = np.random.default_rng(42)


# --------------------------------------------------------------- severity


def test_severity_bands():
    assert severity(0.0) == "stable"
    assert severity(0.09) == "stable"
    assert severity(MODERATE_THRESHOLD) == "moderate"
    assert severity(0.2) == "moderate"
    assert severity(SIGNIFICANT_THRESHOLD) == "significant"
    assert severity(1.5) == "significant"


# ---------------------------------------------------------------- numeric


def test_identical_distributions_have_near_zero_psi():
    values = pd.Series(RNG.normal(50, 10, 20_000))
    assert numeric_psi(values, values) == pytest.approx(0.0, abs=1e-9)


def test_same_distribution_different_samples_stays_stable():
    """Two independent draws from one distribution must not look like drift,
    or every batch would trigger a spurious retrain."""
    a = pd.Series(RNG.normal(50, 10, 20_000))
    b = pd.Series(RNG.normal(50, 10, 20_000))
    assert numeric_psi(a, b) < MODERATE_THRESHOLD


def test_mean_shift_is_detected():
    a = pd.Series(RNG.normal(50, 10, 20_000))
    b = pd.Series(RNG.normal(70, 10, 20_000))   # +2 standard deviations
    assert numeric_psi(a, b) >= SIGNIFICANT_THRESHOLD


def test_variance_shift_is_detected():
    """A distribution can drift without its mean moving at all."""
    a = pd.Series(RNG.normal(50, 5, 20_000))
    b = pd.Series(RNG.normal(50, 20, 20_000))
    assert numeric_psi(a, b) >= SIGNIFICANT_THRESHOLD


def test_psi_grows_monotonically_with_shift_size():
    base = pd.Series(RNG.normal(50, 10, 20_000))
    scores = [
        numeric_psi(base, pd.Series(RNG.normal(50 + delta, 10, 20_000)))
        for delta in (0, 2, 5, 10, 20)
    ]
    assert scores == sorted(scores)


def test_missingness_shift_is_detected():
    """The NULL bucket is the point: values that stop arriving are drift."""
    a = pd.Series(RNG.normal(50, 10, 20_000))
    b = a.copy()
    b.iloc[:8_000] = np.nan          # 0% -> 40% missing
    assert numeric_psi(a, b) >= SIGNIFICANT_THRESHOLD


def test_out_of_range_values_are_not_silently_dropped():
    """Values beyond the reference range must land in the open end bins."""
    a = pd.Series(RNG.uniform(0, 100, 20_000))
    b = pd.Series(RNG.uniform(200, 300, 20_000))   # entirely outside a
    assert numeric_psi(a, b) >= SIGNIFICANT_THRESHOLD


def test_constant_reference_feature_does_not_crash():
    a = pd.Series([5.0] * 1_000)
    assert numeric_psi(a, pd.Series([5.0] * 1_000)) == pytest.approx(0.0, abs=1e-6)
    assert numeric_psi(a, pd.Series([9.0] * 1_000)) > 0


def test_all_null_actual_does_not_crash():
    a = pd.Series(RNG.normal(50, 10, 1_000))
    b = pd.Series([np.nan] * 1_000)
    assert numeric_psi(a, b) > 0


# ------------------------------------------------------------ categorical


def test_categorical_identical_is_zero():
    s = pd.Series(["a"] * 600 + ["b"] * 300 + ["c"] * 100)
    assert categorical_psi(s, s) == pytest.approx(0.0, abs=1e-9)


def test_categorical_mix_shift_is_detected():
    a = pd.Series(["month_to_month"] * 500 + ["one_year"] * 300 + ["two_year"] * 200)
    b = pd.Series(["month_to_month"] * 900 + ["one_year"] * 60 + ["two_year"] * 40)
    assert categorical_psi(a, b) >= SIGNIFICANT_THRESHOLD


def test_new_unseen_category_is_detected():
    """A category absent from the reference must not produce inf/NaN."""
    a = pd.Series(["a"] * 500 + ["b"] * 500)
    b = pd.Series(["a"] * 400 + ["b"] * 300 + ["brand_new"] * 300)
    value = categorical_psi(a, b)
    assert np.isfinite(value)
    assert value > MODERATE_THRESHOLD


def test_categorical_nulls_are_a_category():
    a = pd.Series(["a"] * 900 + ["b"] * 100)
    b = pd.Series(["a"] * 500 + ["b"] * 100 + [None] * 400)
    assert categorical_psi(a, b) >= SIGNIFICANT_THRESHOLD


# ------------------------------------------------------------ compute_drift


def _frame(n, income_mean=60_000, contract_weights=(0.5, 0.3, 0.2)):
    return pd.DataFrame(
        {
            "age": RNG.normal(45, 12, n),
            "annual_income": RNG.normal(income_mean, 15_000, n),
            "contract": RNG.choice(
                ["month_to_month", "one_year", "two_year"], n, p=list(contract_weights)
            ),
        }
    )


def test_compute_drift_flags_only_the_drifted_feature():
    reference = _frame(20_000)
    current = _frame(20_000, income_mean=95_000)     # income moved, age did not

    result = compute_drift(
        reference,
        current,
        numeric_features=["age", "annual_income"],
        categorical_features=["contract"],
    )

    assert set(result["feature"]) == {"age", "annual_income", "contract"}
    by_feature = result.set_index("feature")
    assert by_feature.loc["annual_income", "severity"] == "significant"
    assert by_feature.loc["age", "severity"] == "stable"
    assert by_feature.loc["contract", "severity"] == "stable"


def test_compute_drift_is_sorted_worst_first():
    result = compute_drift(
        _frame(20_000),
        _frame(20_000, income_mean=95_000),
        numeric_features=["age", "annual_income"],
        categorical_features=["contract"],
    )
    assert result.iloc[0]["feature"] == "annual_income"
    assert list(result["psi"]) == sorted(result["psi"], reverse=True)


def test_compute_drift_ignores_features_missing_from_either_side():
    reference = _frame(1_000)
    current = _frame(1_000).drop(columns=["annual_income"])
    result = compute_drift(
        reference,
        current,
        numeric_features=["age", "annual_income"],
        categorical_features=["contract"],
    )
    assert "annual_income" not in set(result["feature"])
    assert "age" in set(result["feature"])


def test_compute_drift_on_no_shared_features_returns_empty_frame():
    result = compute_drift(
        _frame(100),
        _frame(100),
        numeric_features=["nonexistent"],
        categorical_features=[],
    )
    assert result.empty
    assert list(result.columns) == ["feature", "feature_type", "psi", "severity"]
