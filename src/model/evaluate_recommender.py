"""
Offline evaluation for the content-based recommender.

Why this exists: the churn model is cross-validated, threshold-tuned and
reported with per-class metrics, while the recommender shipped with no
measured quality at all - its metadata recorded only structural facts
(how many profiles, how many neighbours). "The recommendations look
plausible" is not an evaluation, and a recommender nobody has scored is
indistinguishable from one that doesn't work.

Protocol: leave-one-out. For a customer who subscribes to services S, hide
one service s (chosen at random, seeded), present the customer to the
recommender as if they owned only S \\ {s}, and ask where s appears in the
ranking of everything they don't own. The hidden service is the one item we
know they genuinely wanted, so its rank is a direct quality signal.

    HitRate@k   fraction of customers whose held-out service made the top k
    MRR         mean of 1/rank over all evaluated customers
    Precision@k HitRate@k / k (only one relevant item exists per customer,
                so this is fully determined by HitRate - reported because
                it is the conventional number, not because it adds signal)

Three rankers are scored on identical splits, because a recommender's
metrics mean nothing without something to compare them to:

    knn         the production model
    popularity  rank by how common each service is overall - the standard
                non-personalized baseline, and a genuinely hard one to beat
    random      shuffle the candidates - the floor

Two deliberate choices:

1. Evaluation customers are drawn from *outside* the k-NN reference set.
   Scoring the model on profiles it was fitted on would measure
   memorization, and since a customer is their own nearest neighbour it
   would be close to meaningless.
2. Customers with fewer than 2 candidate services are excluded. If someone
   already owns 7 of 8 services, masking one leaves a single candidate and
   every ranker scores a guaranteed hit - that inflates all three rankers
   equally and hides real differences between them.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.model.train_recommender import (
    ID_COL,
    PROFILE_CATEGORICAL,
    PROFILE_NUMERIC,
    SERVICE_COLUMNS,
)
from src.warehouse import stream_query

log = logging.getLogger("evaluate_recommender")

MODELS_DIR = Path(__file__).resolve().parents[2] / "models_store"

DEFAULT_SAMPLE_SIZE = int(os.environ.get("REC_EVAL_SAMPLE", "5000"))
DEFAULT_K_VALUES = (1, 2, 3)
RANDOM_SEED = 42

# A customer needs at least this many rankable candidates for the task to
# be non-trivial (see module docstring, choice 2).
MIN_CANDIDATES = 2


# --------------------------------------------------------------- metrics


def reciprocal_rank(ranked_services: list[str], held_out: str) -> float:
    """1/rank of the held-out service, or 0.0 if it never appears."""
    for position, service in enumerate(ranked_services, start=1):
        if service == held_out:
            return 1.0 / position
    return 0.0


def hit_at_k(ranked_services: list[str], held_out: str, k: int) -> bool:
    return held_out in ranked_services[:k]


def summarize(records: list[dict], k_values=DEFAULT_K_VALUES) -> dict:
    """Aggregates per-customer outcomes into the reported metrics."""
    if not records:
        return {"n_evaluated": 0}

    summary: dict = {"n_evaluated": len(records)}
    for k in k_values:
        hits = sum(1 for r in records if r[f"hit@{k}"])
        summary[f"hit_rate@{k}"] = hits / len(records)
        summary[f"precision@{k}"] = hits / (len(records) * k)
    summary["mrr"] = float(np.mean([r["reciprocal_rank"] for r in records]))
    summary["mean_candidates"] = float(np.mean([r["n_candidates"] for r in records]))
    return summary


# --------------------------------------------------------------- rankers


def rank_by_popularity(candidates: list[str], popularity: dict[str, float]) -> list[str]:
    """Non-personalized baseline: most commonly held services first."""
    return sorted(candidates, key=lambda s: popularity.get(s, 0.0), reverse=True)


def rank_randomly(candidates: list[str], rng: np.random.Generator) -> list[str]:
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return shuffled


def rank_by_neighbors(
    candidates: list[str],
    neighbor_services: np.ndarray,
    distances: np.ndarray,
    service_columns: list[str],
) -> list[str]:
    """The production ranking rule: inverse-distance-weighted subscription
    rate among the customer's nearest profile-neighbours.

    Mirrors `_rank_unsubscribed` in train_recommender.py, but returns the
    full ordering rather than a truncated top-N, because MRR needs to know
    where the held-out service landed even when it ranks last.
    """
    weights = 1.0 / (distances + 1e-6)
    weights = weights / weights.sum()
    weighted_rate = (neighbor_services * weights[:, None]).sum(axis=0)

    scores = {service_columns[i]: float(weighted_rate[i]) for i in range(len(service_columns))}
    return sorted(candidates, key=lambda s: scores.get(s, 0.0), reverse=True)


# ------------------------------------------------------------- splitting


def leave_one_out(owned_flags: np.ndarray, rng: np.random.Generator) -> Optional[tuple]:
    """Hides one owned service.

    Returns (masked_flags, held_out_index, candidate_indices), or None when
    the customer owns nothing to hide or the task would be trivial.
    """
    owned_idx = np.flatnonzero(owned_flags == 1)
    if len(owned_idx) == 0:
        return None

    held_out = int(rng.choice(owned_idx))
    masked = owned_flags.copy()
    masked[held_out] = 0

    candidates = np.flatnonzero(masked == 0)
    if len(candidates) < MIN_CANDIDATES:
        return None
    return masked, held_out, candidates


# ------------------------------------------------------------------ I/O


def latest_artifact_path() -> Optional[Path]:
    matches = sorted(MODELS_DIR.glob("recommender_*.joblib"))
    return matches[-1] if matches else None


def load_eval_sample(exclude_ids: set, sample_size: int) -> pd.DataFrame:
    """Draws evaluation customers, then drops any that are in the reference
    set so the model is scored on profiles it has never seen.

    Over-draws before filtering because the excluded ids are a meaningful
    share of the table and an exact-size draw would come back short.
    """
    columns = [ID_COL] + PROFILE_NUMERIC + PROFILE_CATEGORICAL + SERVICE_COLUMNS
    over_draw = min(sample_size * 4, 200_000)
    df = stream_query(
        f"SELECT {', '.join(columns)} FROM marts.customer_360 "
        f"ORDER BY md5(customer_id) LIMIT {int(over_draw)}",
        columns=columns,
    )
    df = df[~df[ID_COL].isin(exclude_ids)]
    return df.head(sample_size).reset_index(drop=True)


# ------------------------------------------------------------ evaluation


def evaluate(artifact: dict, eval_df: pd.DataFrame, seed: int = RANDOM_SEED) -> dict:
    """Scores all three rankers on identical leave-one-out splits."""
    service_columns = artifact["service_columns"]
    preprocessor = artifact["preprocessor"]
    nn_model = artifact["nn_model"]
    reference_services = artifact["service_matrix"]

    # Baseline popularity comes from the reference set, not the evaluation
    # sample: the baseline must only use information the model also had.
    popularity = {
        service_columns[i]: float(reference_services[:, i].mean())
        for i in range(len(service_columns))
    }

    profile_cols = artifact["profile_numeric"] + artifact["profile_categorical"]
    X = preprocessor.transform(eval_df[profile_cols])
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X = X.astype(np.float32)

    # One batched query rather than one per customer.
    log.info("Querying neighbours for %d evaluation customers...", len(eval_df))
    distances, neighbor_idxs = nn_model.kneighbors(X)

    rng = np.random.default_rng(seed)
    owned_matrix = eval_df[service_columns].to_numpy()

    records = {"knn": [], "popularity": [], "random": []}
    skipped = 0

    for row in range(len(eval_df)):
        split = leave_one_out(owned_matrix[row].astype(int), rng)
        if split is None:
            skipped += 1
            continue
        _, held_out_idx, candidate_idxs = split
        held_out = service_columns[held_out_idx]
        candidates = [service_columns[i] for i in candidate_idxs]

        rankings = {
            "knn": rank_by_neighbors(
                candidates,
                reference_services[neighbor_idxs[row]],
                distances[row],
                service_columns,
            ),
            "popularity": rank_by_popularity(candidates, popularity),
            "random": rank_randomly(candidates, rng),
        }

        for name, ranked in rankings.items():
            record = {
                "reciprocal_rank": reciprocal_rank(ranked, held_out),
                "n_candidates": len(candidates),
            }
            for k in DEFAULT_K_VALUES:
                record[f"hit@{k}"] = hit_at_k(ranked, held_out, k)
            records[name].append(record)

    results = {name: summarize(rows) for name, rows in records.items()}
    results["significance"] = compare_rankers(records)
    results["_meta"] = {
        "n_sampled": int(len(eval_df)),
        "n_skipped_trivial": int(skipped),
        "seed": seed,
        "service_popularity": {k: round(v, 4) for k, v in popularity.items()},
    }
    return results


def paired_bootstrap_ci(
    a_values: list[float], b_values: list[float], n_boot: int = 2000, seed: int = RANDOM_SEED
) -> dict:
    """95% CI for the mean difference (a - b) on paired observations.

    Paired, and resampling *customers* rather than the two score lists
    independently: both rankers are scored on the same customers with the
    same held-out service, so the per-customer difference is the unit of
    evidence. Treating the two columns as independent samples would
    overstate the uncertainty and could hide a real effect.
    """
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_means = diff[idx].mean(axis=1)
    return {
        "mean_difference": float(diff.mean()),
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
        "significant": bool(np.percentile(boot_means, 2.5) > 0 or np.percentile(boot_means, 97.5) < 0),
    }


def mcnemar_test(a_hits: list[bool], b_hits: list[bool]) -> dict:
    """Exact McNemar test for two rankers' paired hit/miss outcomes.

    Only the discordant pairs carry information - customers both rankers
    got right, or both got wrong, say nothing about which is better - so
    the test reduces to asking whether the discordant pairs split evenly.
    """
    from scipy.stats import binomtest

    a = np.asarray(a_hits, dtype=bool)
    b = np.asarray(b_hits, dtype=bool)
    a_only = int(np.sum(a & ~b))   # a hit, b missed
    b_only = int(np.sum(~a & b))   # b hit, a missed

    n_discordant = a_only + b_only
    if n_discordant == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0, "significant": False}

    p = binomtest(a_only, n_discordant, 0.5).pvalue
    return {
        "a_only": a_only,
        "b_only": b_only,
        "p_value": float(p),
        "significant": bool(p < 0.05),
    }


def compare_rankers(records: dict, a: str = "knn", b: str = "popularity") -> dict:
    """Is the difference between two rankers real, or sampling noise?"""
    return {
        "comparison": f"{a} vs {b}",
        "mrr": paired_bootstrap_ci(
            [r["reciprocal_rank"] for r in records[a]],
            [r["reciprocal_rank"] for r in records[b]],
        ),
        "hit@1_mcnemar": mcnemar_test(
            [r["hit@1"] for r in records[a]], [r["hit@1"] for r in records[b]]
        ),
        "hit@3_mcnemar": mcnemar_test(
            [r["hit@3"] for r in records[a]], [r["hit@3"] for r in records[b]]
        ),
    }


def format_report(results: dict) -> str:
    meta = results["_meta"]
    lines = [
        "",
        "Recommender offline evaluation (leave-one-out)",
        "=" * 62,
        f"evaluated: {results['knn']['n_evaluated']} customers "
        f"({meta['n_skipped_trivial']} skipped as trivial, of {meta['n_sampled']} sampled)",
        f"mean candidate services per customer: {results['knn']['mean_candidates']:.2f}",
        "",
        f"{'ranker':<12} {'hit@1':>8} {'hit@2':>8} {'hit@3':>8} {'MRR':>8}",
        "-" * 62,
    ]
    for name in ("knn", "popularity", "random"):
        r = results[name]
        lines.append(
            f"{name:<12} {r['hit_rate@1']:>8.4f} {r['hit_rate@2']:>8.4f} "
            f"{r['hit_rate@3']:>8.4f} {r['mrr']:>8.4f}"
        )
    lines.append("-" * 62)

    knn_mrr, pop_mrr = results["knn"]["mrr"], results["popularity"]["mrr"]
    lift = (knn_mrr - pop_mrr) / pop_mrr * 100 if pop_mrr else 0.0
    lines.append(f"knn vs popularity: {lift:+.1f}% MRR")

    sig = results.get("significance")
    if sig:
        mrr = sig["mrr"]
        lines += [
            "",
            "Is that difference real? (knn - popularity, paired)",
            f"  MRR   diff {mrr['mean_difference']:+.4f}  "
            f"95% CI [{mrr['ci_lower']:+.4f}, {mrr['ci_upper']:+.4f}]  "
            f"{'significant' if mrr['significant'] else 'NOT significant'}",
        ]
        for label in ("hit@1_mcnemar", "hit@3_mcnemar"):
            m = sig[label]
            name = label.split("_")[0]
            lines.append(
                f"  {name:<5} McNemar p={m['p_value']:.2e}  "
                f"(knn-only {m['a_only']}, popularity-only {m['b_only']})  "
                f"{'significant' if m['significant'] else 'NOT significant'}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> dict:
    import joblib

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    path = latest_artifact_path()
    if path is None:
        raise SystemExit(f"No recommender artifact found in {MODELS_DIR}")
    log.info("Evaluating %s", path.name)
    artifact = joblib.load(path)

    exclude = set(artifact["customer_ids"].tolist())
    eval_df = load_eval_sample(exclude, DEFAULT_SAMPLE_SIZE)
    log.info("Evaluation sample: %d customers held out of the reference set", len(eval_df))

    results = evaluate(artifact, eval_df)
    print(format_report(results))

    out = MODELS_DIR / f"{path.stem}_evaluation.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out.name)
    return results


if __name__ == "__main__":
    main()
