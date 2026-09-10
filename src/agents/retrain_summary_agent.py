"""
Retrain-Summary Agent.

Takes the current and previous churn model artifact's metrics (F1,
precision, recall, AUC) and the batch's PSI drift results, and writes a
short paragraph summarizing whether the model improved, stayed stable, or
regressed, and whether drift played a role in triggering the retrain.

Runs on MODEL_FAST rather than MODEL_QUALITY: this is a more templated
task (summarize a small metrics table) than the per-customer explanation
and outreach agents, and it runs at most a few times per DAG execution
rather than once per displayed at-risk customer, so the last mile of
prose quality matters less than for the other two agents.
"""

from __future__ import annotations

from typing import Optional

from src.agents.groq_client import MODEL_FAST, AgentCallFailed, AgentResponse, complete

SYSTEM_PROMPT = """You are an MLOps analyst writing a short retrain report for a churn
model pipeline. You are given the new model's metrics, the previous
model's metrics (if one exists), and the feature-drift results that were
computed for the batch that triggered this retrain.

Rules:
- Write one short paragraph (3-5 sentences), plain English, no headers,
  no bullet points.
- State plainly whether the model improved, stayed about the same, or
  regressed - compare the actual numbers given, do not hedge.
- A change smaller than about 0.01 in a metric should be described as
  "essentially unchanged", not as an improvement or regression - do not
  over-read noise as a trend.
- State whether feature drift was detected in the triggering batch, and
  if so, name the drifted feature(s) given to you.
- If no previous model is given, say this is the first recorded model
  version rather than comparing to nothing.
- Do not invent any number not given to you."""


def _format_metrics(label: str, metrics: Optional[dict]) -> str:
    if metrics is None:
        return f"{label}: none (first recorded model version)"
    return (
        f"{label}: version={metrics.get('version', '?')} "
        f"precision={metrics.get('precision_churn', float('nan')):.4f} "
        f"recall={metrics.get('recall_churn', float('nan')):.4f} "
        f"f1={metrics.get('f1_churn', float('nan')):.4f} "
        f"roc_auc={metrics.get('roc_auc', float('nan')):.4f}"
    )


def _format_drift(drift_summary: Optional[dict]) -> str:
    if not drift_summary:
        return "Drift results: not available for this run."
    if drift_summary.get("drift_detected"):
        features = ", ".join(drift_summary.get("drifted_features", [])) or "unspecified"
        return (
            f"Drift results: significant drift detected (max PSI "
            f"{drift_summary.get('max_psi', 0):.4f}) in: {features}"
        )
    return f"Drift results: no significant drift detected (max PSI {drift_summary.get('max_psi', 0):.4f})"


def build_prompt(
    current_metrics: dict,
    previous_metrics: Optional[dict],
    drift_summary: Optional[dict],
) -> str:
    """Separated from summarize_retrain() so tests can assert the real
    metric values and drift feature names are present in the constructed
    prompt."""
    return (
        f"{_format_metrics('New model', current_metrics)}\n"
        f"{_format_metrics('Previous model', previous_metrics)}\n"
        f"{_format_drift(drift_summary)}\n\n"
        "Write the retrain summary now."
    )


def summarize_retrain(
    current_metrics: dict,
    previous_metrics: Optional[dict] = None,
    drift_summary: Optional[dict] = None,
) -> AgentResponse:
    """Raises AgentCallFailed on any failure. The caller (the DAG task)
    must catch this and fall back to writing the raw metrics/drift dicts
    to the summary file instead of a prose paragraph - a retrain must
    never fail, or even lose its summary artifact, just because Groq is
    unreachable."""
    if not current_metrics:
        raise AgentCallFailed("No current model metrics provided - nothing to summarize")
    user_prompt = build_prompt(current_metrics, previous_metrics, drift_summary)
    # See explanation_agent's note on max_tokens: reasoning-model overhead
    # applies here too, even on the "fast" tier.
    return complete(SYSTEM_PROMPT, user_prompt, model=MODEL_FAST, max_tokens=400, temperature=0.3)
