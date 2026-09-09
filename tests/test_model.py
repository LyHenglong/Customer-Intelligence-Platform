"""
Unit tests for the modeling layer: expected-value threshold selection,
recall-target threshold selection, and the recommender's core invariant.

Deliberately built on synthetic in-memory inputs rather than the live
warehouse or a saved model artifact, so these run in CI with no database,
no Postgres credentials, and no multi-hundred-MB artifacts checked out.
"""

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from src.model.threshold_analysis import (
    customer_value,
    expected_value_at_threshold,
    optimal_threshold,
    profit_curve,
    sensitivity_grid,
)
from src.model.train_churn import choose_threshold
from src.model.train_recommender import recommend_for_customer


# --------------------------------------------------------------------------
# Expected-value threshold analysis
# --------------------------------------------------------------------------

def test_customer_value_scales_with_horizon():
    charges = np.array([10.0, 100.0])
    assert list(customer_value(charges, horizon_months=12)) == [120.0, 1200.0]
    assert list(customer_value(charges, horizon_months=1)) == [10.0, 100.0]


def test_expected_value_math_is_exact():
    """Hand-computable case: 2 flagged customers, one a true churner.

    value saved = 1 churner * $1000 value * 0.5 success = $500
    campaign cost = 2 offers * $10 = $20
    net = $480
    """
    y_true = np.array([1, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.1])
    values = np.array([1000.0, 1000.0, 1000.0])

    r = expected_value_at_threshold(y_true, y_proba, values, threshold=0.5,
                                     offer_cost=10.0, p_offer_success=0.5)

    assert r["n_flagged"] == 2
    assert r["true_positives"] == 1
    assert r["false_positives"] == 1
    assert r["false_negatives"] == 0
    assert r["expected_value_saved"] == pytest.approx(500.0)
    assert r["campaign_cost"] == pytest.approx(20.0)
    assert r["net_value"] == pytest.approx(480.0)


def test_expected_value_flags_nobody_at_high_threshold():
    y_true = np.array([1, 0, 1])
    y_proba = np.array([0.4, 0.2, 0.3])
    values = np.array([500.0, 500.0, 500.0])

    r = expected_value_at_threshold(y_true, y_proba, values, threshold=0.99)

    assert r["n_flagged"] == 0
    assert r["net_value"] == pytest.approx(0.0)  # no campaign, no incremental value
    assert r["recall"] == 0.0
    assert r["precision"] == 0.0  # guarded against divide-by-zero


def test_optimal_threshold_finds_the_profit_maximum():
    """A model that ranks perfectly should be thresholded to catch exactly
    the churners and nobody else."""
    rng = np.random.RandomState(0)
    y_true = np.array([1] * 50 + [0] * 450)
    # perfect separation: churners score high, non-churners score low
    y_proba = np.concatenate([rng.uniform(0.8, 0.99, 50), rng.uniform(0.01, 0.2, 450)])
    values = np.full(500, 1000.0)

    best_t, best = optimal_threshold(y_true, y_proba, values,
                                      offer_cost=10.0, p_offer_success=0.5)

    assert 0.2 <= best_t <= 0.8, "threshold should land in the separation gap"
    assert best["true_positives"] == 50
    assert best["false_positives"] == 0
    # 50 * 1000 * 0.5 saved, minus 50 * 10 cost
    assert best["net_value"] == pytest.approx(50 * 1000 * 0.5 - 50 * 10)


def test_profit_curve_covers_threshold_range_and_is_finite():
    y_true = np.array([1, 0] * 50)
    y_proba = np.linspace(0.01, 0.99, 100)
    values = np.full(100, 800.0)

    curve = profit_curve(y_true, y_proba, values)

    assert len(curve) == 99
    assert curve["threshold"].is_monotonic_increasing
    assert np.isfinite(curve["net_value"]).all()
    # flagging strictly decreases as the threshold rises
    assert curve["n_flagged"].is_monotonic_decreasing


def test_cheaper_offers_justify_a_lower_threshold():
    """Core economic intuition: if outreach is nearly free, contact more
    people; if it's expensive, be selective."""
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.1, 2000)
    # noisy but informative scores
    y_proba = np.clip(0.1 + 0.5 * y_true + rng.normal(0, 0.2, 2000), 0.01, 0.99)
    values = np.full(2000, 1000.0)

    cheap_t, _ = optimal_threshold(y_true, y_proba, values, offer_cost=1.0, p_offer_success=0.3)
    pricey_t, _ = optimal_threshold(y_true, y_proba, values, offer_cost=200.0, p_offer_success=0.3)

    assert cheap_t < pricey_t


def test_sensitivity_grid_shape_and_columns():
    y_true = np.array([1, 0] * 100)
    y_proba = np.linspace(0.01, 0.99, 200)
    values = np.full(200, 900.0)

    grid = sensitivity_grid(y_true, y_proba, values,
                            offer_costs=[10.0, 50.0], success_rates=[0.2, 0.4, 0.6])

    assert len(grid) == 2 * 3
    assert {"offer_cost", "p_offer_success", "optimal_threshold", "net_value"} <= set(grid.columns)


# --------------------------------------------------------------------------
# Recall-target threshold selection (train_churn.choose_threshold)
# --------------------------------------------------------------------------

def test_choose_threshold_hits_the_recall_target():
    y_true = np.array([1] * 100 + [0] * 900)
    rng = np.random.RandomState(7)
    y_proba = np.concatenate([rng.uniform(0.3, 0.95, 100), rng.uniform(0.05, 0.6, 900)])

    threshold, precision, recall = choose_threshold(y_true, y_proba, target_recall=0.60)

    assert recall >= 0.60, "must clear the recall floor it was asked for"
    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= precision <= 1.0


def test_choose_threshold_stricter_target_lowers_threshold():
    """Demanding more recall means casting a wider net."""
    y_true = np.array([1] * 100 + [0] * 900)
    rng = np.random.RandomState(11)
    y_proba = np.concatenate([rng.uniform(0.3, 0.95, 100), rng.uniform(0.05, 0.6, 900)])

    t_low, _, _ = choose_threshold(y_true, y_proba, target_recall=0.50)
    t_high, _, _ = choose_threshold(y_true, y_proba, target_recall=0.90)

    assert t_high <= t_low


# --------------------------------------------------------------------------
# Recommender invariants
# --------------------------------------------------------------------------

def _toy_recommender_artifact():
    """Three customers in a tiny 2-D profile space, four services.

    Customer 0 already has services 0 and 1; its neighbors (1 and 2) both
    have service 2, and only one has service 3.
    """
    X_profile = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
    service_matrix = np.array([
        [1, 1, 0, 0],   # customer 0: has A and B
        [1, 0, 1, 1],   # customer 1
        [0, 1, 1, 0],   # customer 2
    ])
    nn = NearestNeighbors(n_neighbors=3).fit(X_profile)
    return {
        "customer_ids": np.array(["C0", "C1", "C2"]),
        "X_profile": X_profile,
        "service_matrix": service_matrix,
        "service_columns": ["svc_a", "svc_b", "svc_c", "svc_d"],
        "nn_model": nn,
    }


def test_recommender_never_recommends_an_already_subscribed_service():
    """The core business invariant: don't try to upsell something the
    customer already pays for."""
    artifact = _toy_recommender_artifact()

    recs = recommend_for_customer(artifact, "C0", top_n=4)

    recommended = {r["service"] for r in recs}
    assert "svc_a" not in recommended, "customer 0 already has svc_a"
    assert "svc_b" not in recommended, "customer 0 already has svc_b"
    assert recommended <= {"svc_c", "svc_d"}


def test_recommender_ranks_by_neighbour_popularity():
    """svc_c is held by both neighbours, svc_d by only one, so svc_c ranks first."""
    artifact = _toy_recommender_artifact()

    recs = recommend_for_customer(artifact, "C0", top_n=2)

    assert recs[0]["service"] == "svc_c"
    assert recs[0]["score"] >= recs[1]["score"]


def test_recommender_respects_top_n():
    artifact = _toy_recommender_artifact()
    assert len(recommend_for_customer(artifact, "C0", top_n=1)) == 1


def test_recommender_raises_for_unknown_customer():
    artifact = _toy_recommender_artifact()
    with pytest.raises(KeyError):
        recommend_for_customer(artifact, "does-not-exist")
