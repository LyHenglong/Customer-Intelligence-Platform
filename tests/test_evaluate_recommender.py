"""
Tests for the recommender's offline evaluation.

Evaluation code is where a quiet bug is most dangerous: a broken metric
doesn't crash, it just reports a number that nobody can tell is wrong, and
that number then goes into the README. These tests pin the metrics against
hand-computed values and check the properties that would catch a silently
inverted or mis-scaled result.

All synthetic - no database, no artifact.
"""

import numpy as np
import pytest

from src.model.evaluate_recommender import (
    hit_at_k,
    leave_one_out,
    mcnemar_test,
    paired_bootstrap_ci,
    rank_by_neighbors,
    rank_by_popularity,
    rank_randomly,
    reciprocal_rank,
    summarize,
)

SERVICES = ["a", "b", "c", "d"]


# ------------------------------------------------------------ metrics


def test_reciprocal_rank_by_position():
    ranked = ["a", "b", "c"]
    assert reciprocal_rank(ranked, "a") == 1.0
    assert reciprocal_rank(ranked, "b") == 0.5
    assert reciprocal_rank(ranked, "c") == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_absent():
    assert reciprocal_rank(["a", "b"], "z") == 0.0


def test_hit_at_k_boundary():
    ranked = ["a", "b", "c"]
    assert hit_at_k(ranked, "b", 2) is True
    assert hit_at_k(ranked, "c", 2) is False   # just outside the cutoff
    assert hit_at_k(ranked, "c", 3) is True


def test_summarize_matches_hand_computed_values():
    records = [
        {"hit@1": True,  "hit@2": True,  "hit@3": True,  "reciprocal_rank": 1.0, "n_candidates": 4},
        {"hit@1": False, "hit@2": True,  "hit@3": True,  "reciprocal_rank": 0.5, "n_candidates": 4},
        {"hit@1": False, "hit@2": False, "hit@3": True,  "reciprocal_rank": 0.25, "n_candidates": 4},
        {"hit@1": False, "hit@2": False, "hit@3": False, "reciprocal_rank": 0.0, "n_candidates": 4},
    ]
    s = summarize(records)
    assert s["n_evaluated"] == 4
    assert s["hit_rate@1"] == 0.25
    assert s["hit_rate@2"] == 0.50
    assert s["hit_rate@3"] == 0.75
    assert s["mrr"] == pytest.approx(0.4375)
    # one relevant item per customer, so precision@k = hit_rate@k / k
    assert s["precision@2"] == pytest.approx(0.25)


def test_summarize_handles_no_records():
    assert summarize([])["n_evaluated"] == 0


# ------------------------------------------------------------ rankers


def test_popularity_ranks_most_common_first():
    popularity = {"a": 0.1, "b": 0.9, "c": 0.5}
    assert rank_by_popularity(["a", "b", "c"], popularity) == ["b", "c", "a"]


def test_popularity_only_ranks_given_candidates():
    """A service the customer already owns must never be recommended."""
    popularity = {"a": 0.1, "b": 0.9, "c": 0.5}
    assert rank_by_popularity(["a", "c"], popularity) == ["c", "a"]


def test_random_ranker_is_a_permutation():
    rng = np.random.default_rng(0)
    out = rank_randomly(["a", "b", "c"], rng)
    assert sorted(out) == ["a", "b", "c"]


def test_neighbors_rank_service_held_by_nearest_neighbours():
    """Service 'c' is held by every neighbour, so it must rank first."""
    #                      a  b  c  d
    neighbor_services = np.array([
        [1, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 1, 1, 0],
    ])
    distances = np.array([0.1, 0.2, 0.3])
    ranked = rank_by_neighbors(["b", "c", "d"], neighbor_services, distances, SERVICES)
    assert ranked[0] == "c"
    assert ranked[-1] == "d"      # held by nobody


def test_neighbors_rank_weights_closer_neighbours_more():
    """The nearest neighbour holds 'b'; a distant one holds 'd'."""
    neighbor_services = np.array([
        [0, 1, 0, 0],   # distance 0.01 - very close
        [0, 0, 0, 1],   # distance 5.0  - far away
    ])
    distances = np.array([0.01, 5.0])
    ranked = rank_by_neighbors(["b", "d"], neighbor_services, distances, SERVICES)
    assert ranked[0] == "b"


# ------------------------------------------------------------ splitting


def test_leave_one_out_hides_an_owned_service():
    rng = np.random.default_rng(0)
    owned = np.array([1, 1, 0, 0])
    masked, held_out, candidates = leave_one_out(owned, rng)

    assert owned[held_out] == 1          # something they actually had
    assert masked[held_out] == 0         # now hidden
    assert held_out in candidates        # and therefore rankable
    assert masked.sum() == owned.sum() - 1


def test_leave_one_out_returns_none_when_nothing_owned():
    assert leave_one_out(np.array([0, 0, 0, 0]), np.random.default_rng(0)) is None


def test_leave_one_out_skips_trivial_single_candidate_case():
    """Owning 4 of 4 services leaves one candidate - a guaranteed hit for
    every ranker, which would inflate all of them equally."""
    assert leave_one_out(np.array([1, 1, 1, 1]), np.random.default_rng(0)) is None


def test_leave_one_out_does_not_mutate_input():
    owned = np.array([1, 1, 0, 0])
    original = owned.copy()
    leave_one_out(owned, np.random.default_rng(0))
    assert np.array_equal(owned, original)


# -------------------------------------------------------- significance


def test_bootstrap_detects_a_consistent_advantage():
    a = [0.9] * 200
    b = [0.5] * 200
    result = paired_bootstrap_ci(a, b, n_boot=500)
    assert result["mean_difference"] == pytest.approx(0.4)
    assert result["ci_lower"] > 0
    assert result["significant"] is True


def test_bootstrap_reports_no_effect_for_identical_rankers():
    values = list(np.random.default_rng(0).random(300))
    result = paired_bootstrap_ci(values, values, n_boot=500)
    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["significant"] is False


def test_bootstrap_is_direction_aware():
    """A worse ranker must produce a negative difference, not just a
    'significant' flag - a sign error here would invert the conclusion."""
    result = paired_bootstrap_ci([0.2] * 200, [0.8] * 200, n_boot=500)
    assert result["mean_difference"] < 0
    assert result["ci_upper"] < 0


def test_mcnemar_uses_only_discordant_pairs():
    """Agreements carry no information about which ranker is better."""
    a = [True] * 50 + [False] * 50 + [True] * 30 + [False] * 5
    b = [True] * 50 + [False] * 50 + [False] * 30 + [True] * 5
    result = mcnemar_test(a, b)
    assert result["a_only"] == 30
    assert result["b_only"] == 5
    assert result["significant"] is True


def test_mcnemar_on_identical_rankers_is_not_significant():
    a = [True, False, True, False] * 25
    result = mcnemar_test(a, a)
    assert result["a_only"] == 0 and result["b_only"] == 0
    assert result["p_value"] == 1.0
    assert result["significant"] is False


def test_mcnemar_even_split_is_not_significant():
    a = [True] * 20 + [False] * 20
    b = [False] * 20 + [True] * 20
    result = mcnemar_test(a, b)
    assert result["a_only"] == result["b_only"] == 20
    assert result["significant"] is False
