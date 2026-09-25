"""One-off / periodic script: scores every customer once and persists the
result to public.churn_scores (src/model/dashboard_queries.store_scored_customers),
so a memory-constrained deployment can read a cheap SQL SELECT instead of
running the ~1M-row scoring pass in-process on every cache miss.

Why this exists: Render's free-tier web service memory ceiling is far
below the ~3GB this project's local Docker container budgets for the
churn model + RAG models + the population-scoring pass together (see
src/model/api.py's WARM_MODELS_ON_STARTUP comment, and
dashboard_queries.score_all_customers's own docstring on why that pass
is expensive). Turning off eager startup warming fixed the first crash
loop; requests that actually trigger the scoring pass (/overview/stats,
/at-risk, etc.) still OOM the free-tier container on their own. Moving
the scoring pass out of the request path entirely - run it here, once,
somewhere with enough memory, then read the result back with a plain
SELECT - fixes that without paying for a bigger plan and without
sampling/estimating the numbers the dashboard shows.

Run locally, pointed at whichever Postgres the deployment actually
serves from (set POSTGRES_* in .env, e.g. the same Neon connection
string used for the frontend/API):

    python -m scripts.precompute_churn_scores

Re-run this after every retrain that promotes a new churn model champion
- public.churn_scores is keyed by model_version, and
dashboard_queries.get_scored_customers only reads rows matching the
version currently loaded in-process; stale rows for other versions are
deleted automatically on each run, but a version with zero rows falls
back to live in-process scoring on Render until this is re-run for it.
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

load_dotenv()

# See scripts/migrate_models_to_mlflow.py's identical fix for the
# Windows-console-encoding UnicodeEncodeError this guards against.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

import joblib  # noqa: E402

from src.model.dashboard_queries import score_all_customers, store_scored_customers  # noqa: E402
from src.model.registry import MODELS_DIR, resolve_model_path  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("precompute_churn_scores")


def main() -> None:
    churn_path = resolve_model_path("churn_model", "churn_model_*.joblib")
    if churn_path is None:
        raise SystemExit(f"No churn model artifact found in {MODELS_DIR}")

    churn_version = churn_path.stem.replace("churn_model_", "")
    log.info("Loading churn model %s...", churn_path.name)
    churn_artifact = joblib.load(churn_path)

    log.info("Scoring all customers (this streams the full warehouse, expect a couple of minutes)...")
    frame = score_all_customers(churn_artifact, churn_version)
    log.info("Scored %d customers, persisting to public.churn_scores...", len(frame))

    store_scored_customers(frame, churn_version)
    log.info("Done. get_scored_customers(version=%s) will now read this table when USE_PRECOMPUTED_SCORES=true.", churn_version)


if __name__ == "__main__":
    main()
