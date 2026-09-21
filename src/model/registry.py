"""Resolves the current Production/champion model's local models_store path.

Tries MLflow's registry first (alias "champion" -> a registered model
version -> that version's `model_file` tag, matched back to a file under
models_store/ on local disk - MLflow is bookkeeping/promotion here, not
the serving path itself: the actual bytes a process loads always come
from models_store/*.joblib, never a network fetch from MLflow, so
inference stays free of any MLflow dependency on the hot path).

Falls back to the glob-latest-by-timestamp behavior - previously
duplicated independently in src/model/api.py, src/ai/tools/_artifacts.py,
and src/dashboard/app.py - whenever MLflow is unreachable, misconfigured
(MLFLOW_TRACKING_URI unset), or has no aliased version yet for the given
registered model name. Mirrors the same fallback convention this codebase
already uses elsewhere for optional external calls (e.g.
src/agents/groq_client.py's AgentCallFailed -> raw-data degrade,
/explain-churn's fallback to raw SHAP text).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

MODELS_DIR = Path(__file__).resolve().parents[2] / "models_store"
log = logging.getLogger("model.registry")

# mlflow's own defaults - MLFLOW_HTTP_REQUEST_TIMEOUT=120s and
# MLFLOW_HTTP_REQUEST_MAX_RETRIES=5 with exponential backoff - compound
# into a multi-minute stall when mlflow is unreachable, on a call that's
# meant to fail fast and fall back (this is on the read path: api/
# dashboard startup, dashboard model-loading). Both reproduced for real:
# the timeout alone during a test run, the retry/backoff compounding when
# verifying the fallback path with the mlflow service stopped. setdefault
# everywhere, not overwrite: respects an operator's own explicit value.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
# 0, not mlflow's default of 5: even 1 retry adds several seconds of
# backoff (reproduced for real - a single retry alone added ~8s before
# falling back), and this read path only ever wants a single fast
# probe-or-fallback, never a retry loop.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")


def _glob_latest(models_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(models_dir.glob(pattern))
    return matches[-1] if matches else None


def resolve_model_path(
    registered_name: str, glob_pattern: str, models_dir: Path = MODELS_DIR
) -> Optional[Path]:
    """Returns the local models_store path to load for `registered_name`
    (e.g. "churn_model" or "recommender"), preferring MLflow's "champion"
    alias and falling back to `glob_pattern` (e.g. "churn_model_*.joblib")
    matched against `models_dir` when MLflow can't resolve one.

    `models_dir` defaults to this module's own MODELS_DIR but is
    overridable per-call so callers that already have their own
    test-patchable MODELS_DIR constant (src/ai/tools/_artifacts.py) can
    pass it straight through, rather than this module silently reading
    the real models_store/ regardless of what a test patched elsewhere."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        try:
            import mlflow

            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            mv = client.get_model_version_by_alias(registered_name, "champion")
            model_file = mv.tags.get("model_file", "")
            candidate = models_dir / Path(model_file).name if model_file else None
            if candidate and candidate.exists():
                return candidate
            log.warning(
                "MLflow champion for %s points at %r, not found locally; falling back to glob",
                registered_name, model_file,
            )
        except Exception as exc:
            log.warning(
                "MLflow unreachable or no champion alias for %s (%s); falling back to glob-latest",
                registered_name, exc,
            )
    return _glob_latest(models_dir, glob_pattern)
