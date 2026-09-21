"""One-off, manually-run script: backdates the model artifacts that
already existed in models_store/ before the MLflow registry existed
(src/model/registry.py) into MLflow's tracking/registry history, so the
MLflow UI's run history doesn't start artificially empty on first real
use.

Purely cosmetic/optional - src/model/registry.py's fallback to
glob-latest-by-timestamp means nothing breaks if this is never run, or
is run more than once against a database that no longer has these
artifacts (already-migrated files are skipped). Never sets the
"champion" alias on any of these - that's left exactly as training
already set it (the two real training runs already done set it
correctly), so this script only adds history, never changes what's
actually served.

Version numbering caveat: MLflow assigns registered-model version
numbers sequentially at registration time, not from an artifact's real
age - since two real training runs already registered "churn_model" v1
and "recommender" v1 today, the historical artifacts backfilled here
will land at v2, v3, ... even though they actually predate v1
chronologically. Each backfilled run's own start/end time IS set to the
artifact's real trained_at timestamp (visible in the MLflow UI's run
list), so the true chronology is still recoverable there - it's only
the integer version *number* that won't match age.

Usage:
    python -m scripts.migrate_models_to_mlflow
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# mlflow prints a run-summary line containing an emoji on run termination.
# Windows' console defaults to a codepage that can't encode it - see
# src/model/train_churn.py's identical fix for the real traceback this
# was found from (a plain UnicodeEncodeError crashing the whole script).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("migrate_models_to_mlflow")

MODELS_DIR = Path(__file__).resolve().parents[1] / "models_store"

# (registered_model_name, glob_pattern, metadata_key_for_artifact_filename)
_TARGETS = [
    ("churn_model", "churn_model_*.json", "model_file"),
    ("recommender", "recommender_*.json", "artifact_file"),
]

_SCALAR_TYPES = (int, float, str, bool)


def _to_epoch_ms(iso_ts: str) -> int:
    return int(datetime.fromisoformat(iso_ts).timestamp() * 1000)


def _already_registered(client: MlflowClient, registered_name: str, artifact_filename: str) -> bool:
    try:
        versions = client.search_model_versions(f"name='{registered_name}'")
    except Exception:
        return False
    return any(v.tags.get("model_file") == artifact_filename for v in versions)


def _migrate_one(client: MlflowClient, experiment_id: str, registered_name: str, meta_path: Path, artifact_key: str) -> None:
    metadata = json.loads(meta_path.read_text())
    artifact_filename = metadata[artifact_key]
    artifact_path = MODELS_DIR / artifact_filename

    if not artifact_path.exists():
        log.warning("Skipping %s: artifact %s not found on disk", meta_path.name, artifact_filename)
        return
    if _already_registered(client, registered_name, artifact_filename):
        log.info("Skipping %s: already registered", artifact_filename)
        return

    start_ms = _to_epoch_ms(metadata["trained_at"])
    run = client.create_run(
        experiment_id,
        start_time=start_ms,
        tags={"mlflow.runName": metadata["version"]},
    )
    run_id = run.info.run_id

    for key, value in metadata.items():
        if key in ("version", "trained_at") or not isinstance(value, _SCALAR_TYPES):
            continue
        if isinstance(value, bool) or isinstance(value, str):
            client.log_param(run_id, key, value)
        else:
            client.log_metric(run_id, key, value, timestamp=start_ms)

    client.log_artifact(run_id, str(artifact_path))
    client.log_artifact(run_id, str(meta_path))
    client.set_terminated(run_id, status="FINISHED", end_time=start_ms)

    mv = client.create_model_version(
        registered_name, source=f"runs:/{run_id}/{artifact_filename}", run_id=run_id,
    )
    client.set_model_version_tag(registered_name, mv.version, "model_file", artifact_filename)
    log.info("Backfilled %s as %s v%s (run %s)", artifact_filename, registered_name, mv.version, run_id)


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("MLFLOW_TRACKING_URI is not set - point it at a running mlflow service first.")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    for registered_name, glob_pattern, artifact_key in _TARGETS:
        experiment = mlflow.set_experiment(registered_name)
        try:
            client.get_registered_model(registered_name)
        except Exception:
            client.create_registered_model(registered_name)

        for meta_path in sorted(MODELS_DIR.glob(glob_pattern)):
            if meta_path.stem.endswith("_evaluation"):
                continue  # recommender_*_evaluation.json is an eval report, not a version's own metadata
            _migrate_one(client, experiment.experiment_id, registered_name, meta_path, artifact_key)


if __name__ == "__main__":
    main()
