"""
Expected-value analysis for the churn decision threshold.

A churn model outputs a probability; turning that into an *action* requires
a threshold, and the statistically convenient choices (0.5, or "whatever
hits recall >= 0.60") are not business decisions. This module derives the
threshold that maximizes expected profit from a retention campaign, given
an explicit cost/benefit model.

The decision framing, per customer, relative to doing nothing at all:

    - True positive  (flagged, would have churned):
        we send an offer. It works with probability P_OFFER_SUCCESS, saving
        that customer's value. We pay the offer cost either way.
    - False positive (flagged, would have stayed):
        we pay the offer cost on someone who was never going to leave.
    - False negative (not flagged, churns):
        no action, customer lost - same as the do-nothing baseline, so it
        contributes no *incremental* loss in this formulation.
    - True negative: no action, no cost.

    net_value(t) = sum over flagged-and-would-churn of
                       (P_OFFER_SUCCESS * customer_value)
                 - sum over all flagged of OFFER_COST

Customer value is computed per customer from their own monthlycharges over
a retention horizon, not a flat average - a high-spend customer is worth
more to save, and the threshold should reflect that.

IMPORTANT: the cost parameters below are illustrative assumptions, not
measured facts. This dataset is synthetic and carries no offer-cost or
campaign-response data. In a real deployment these would come from finance
and from a historical retention-campaign holdout, and the whole point of
this module is that they are explicit and easy to change - see
notebooks/threshold_and_business_value.ipynb for a sensitivity analysis
showing how the recommended threshold moves as they vary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Illustrative business assumptions (see module docstring) -------------
OFFER_COST = 30.0          # cost of extending one retention offer
P_OFFER_SUCCESS = 0.30     # probability an offer actually retains a would-be churner
RETENTION_HORIZON_MONTHS = 12  # months of revenue credited to a saved customer


def customer_value(monthly_charges: pd.Series | np.ndarray,
                   horizon_months: int = RETENTION_HORIZON_MONTHS) -> np.ndarray:
    """Value of saving a customer: their own monthly spend over the horizon."""
    return np.asarray(monthly_charges, dtype=float) * horizon_months


def expected_value_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    values: np.ndarray,
    threshold: float,
    offer_cost: float = OFFER_COST,
    p_offer_success: float = P_OFFER_SUCCESS,
) -> dict:
    """Expected incremental profit of running a retention campaign at
    `threshold`, versus doing nothing."""
    y_true = np.asarray(y_true)
    flagged = y_proba >= threshold

    n_flagged = int(flagged.sum())
    saved_value = float((values[flagged & (y_true == 1)] * p_offer_success).sum())
    campaign_cost = float(n_flagged * offer_cost)

    tp = int((flagged & (y_true == 1)).sum())
    fp = int((flagged & (y_true == 0)).sum())
    fn = int((~flagged & (y_true == 1)).sum())

    return {
        "threshold": float(threshold),
        "n_flagged": n_flagged,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / n_flagged if n_flagged else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "expected_value_saved": saved_value,
        "campaign_cost": campaign_cost,
        "net_value": saved_value - campaign_cost,
    }


def profit_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    values: np.ndarray,
    thresholds: np.ndarray | None = None,
    offer_cost: float = OFFER_COST,
    p_offer_success: float = P_OFFER_SUCCESS,
) -> pd.DataFrame:
    """Expected profit across a sweep of thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = [
        expected_value_at_threshold(y_true, y_proba, values, t, offer_cost, p_offer_success)
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    values: np.ndarray,
    offer_cost: float = OFFER_COST,
    p_offer_success: float = P_OFFER_SUCCESS,
) -> tuple[float, dict]:
    """The profit-maximizing threshold and its full breakdown.

    The break-even intuition: it is worth contacting a customer when
    p_churn * p_offer_success * value > offer_cost, i.e. a *true* churn
    probability above offer_cost / (p_offer_success * value). With the
    default assumptions and a ~$86/mo customer that is
    30 / (0.30 * 1035) ~= 0.0966.

    Since the churn model is now calibrated (sigmoid, on a held-out split -
    see train_churn.py), that break-even applies *directly* to its output.
    Measured on the test set: theory says 0.0966, the empirical profit peak
    is 0.1000, a gap of 0.0034 that is grid resolution rather than
    disagreement.

    This was not always true, and the history is worth keeping. Before
    calibration, class_weight="balanced" inflated the model's scores to a
    mean of ~0.41 against an actual churn rate of ~0.10, so the empirical
    peak sat near 0.47 - the same 0.0966 break-even, seen through the
    model's miscalibration. The empirical search this function performs was
    correct either way, which is precisely why it is done empirically; what
    calibration bought is that the theoretical number is now directly
    usable and the two agree.
    """
    curve = profit_curve(y_true, y_proba, values, offer_cost=offer_cost, p_offer_success=p_offer_success)
    best_idx = int(curve["net_value"].idxmax())
    best = curve.loc[best_idx].to_dict()
    return float(best["threshold"]), best


def sensitivity_grid(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    values: np.ndarray,
    offer_costs: list[float],
    success_rates: list[float],
) -> pd.DataFrame:
    """How the profit-maximizing threshold moves as the two most uncertain
    assumptions vary. A recommendation that swings wildly here is fragile
    and should be reported as such."""
    rows = []
    for cost in offer_costs:
        for p_success in success_rates:
            t, best = optimal_threshold(y_true, y_proba, values, offer_cost=cost, p_offer_success=p_success)
            rows.append({
                "offer_cost": cost,
                "p_offer_success": p_success,
                "optimal_threshold": t,
                "net_value": best["net_value"],
                "n_flagged": best["n_flagged"],
                "recall": best["recall"],
            })
    return pd.DataFrame(rows)
