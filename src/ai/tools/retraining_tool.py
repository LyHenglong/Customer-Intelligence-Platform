"""retraining_analysis tool - current/previous model version, metrics,
drift, and the latest AI Agent Layer retrain narrative.

Reads the same models_store/*.json metadata files the dashboard's Model
Performance tab reads (src/dashboard/app.py's load_all_churn_metadata),
public.feature_drift (src/monitoring/drift.py), and public.retrain_summaries
(src/agents/cache.py) - no new state, just a typed read of what the
existing pipeline already produces.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.agents.cache import get_latest_retrain_summary
from src.ai.schemas import RetrainingAnalysisResult
from src.warehouse import get_pg_conn

MODELS_DIR = Path(__file__).resolve().parents[3] / "models_store"

_COMPARABLE_METRICS = ("roc_auc", "accuracy", "precision_churn", "recall_churn", "f1_churn")


def _load_metadata_versions() -> list[dict]:
    records = []
    for path in sorted(MODELS_DIR.glob("churn_model_*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return records


def _load_latest_drift() -> list[dict]:
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.feature_drift')")
            if cur.fetchone()[0] is None:
                return []
            cur.execute(
                """
                SELECT feature, feature_type, psi, severity, reference_batch, current_batch
                  FROM public.feature_drift
                 WHERE current_batch = (
                       SELECT current_batch FROM public.feature_drift ORDER BY computed_at DESC LIMIT 1
                 )
                 ORDER BY psi DESC
                """
            )
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()


def retraining_analysis() -> RetrainingAnalysisResult:
    versions = _load_metadata_versions()
    current = versions[-1] if versions else None
    previous = versions[-2] if len(versions) > 1 else None

    metric_changes = {}
    if current and previous:
        for key in _COMPARABLE_METRICS:
            if key in current and key in previous:
                metric_changes[key] = round(float(current[key]) - float(previous[key]), 4)

    summary = get_latest_retrain_summary()

    return RetrainingAnalysisResult(
        current_model_version=current.get("version") if current else None,
        previous_model_version=previous.get("version") if previous else None,
        current_metrics=current,
        previous_metrics=previous,
        metric_changes=metric_changes,
        latest_drift=_load_latest_drift(),
        retrain_reason=summary["summary_text"] if summary else None,
    )
