"""
Churn Explanation Agent.

Takes a customer's SHAP values (the model's own, already-computed
attribution - see src/model/explain_churn.py) and turns them into a
short, plain-English explanation of why that customer is flagged as
at-risk.

This agent explains an existing prediction; it never makes one. The churn
probability and the SHAP attribution it reads are both already final by
the time this runs - see the README's "AI Agent Layer" section for the
boundary between this presentation layer and the ML layer underneath it.

The system prompt constrains the model to the factors it was actually
given, specifically to stop it inventing plausible-sounding reasons that
never appeared in the customer's real SHAP output - a real risk with an
open-ended "explain this customer" prompt, and the reason the prompt
below enumerates hard rules rather than just describing the task.
"""

from __future__ import annotations

from src.agents.groq_client import MODEL_QUALITY, AgentCallFailed, AgentResponse, complete

SYSTEM_PROMPT = """You are a churn-risk analyst writing a short note for a retention team.
You are given a customer's churn probability and the specific factors the
model's SHAP explanation identified as driving that prediction, each with
a direction (increases or decreases risk) and a signed numeric SHAP value.

Rules:
- Write exactly 2-3 sentences, plain English, no bullet points, no headers.
- Base your explanation ONLY on the factors given. Do not invent reasons,
  and do not mention any factor that was not provided to you.
- Reference the actual factor names and their direction naturally in the
  explanation (e.g. "frequent service calls" rather than "feature 3").
- State the churn probability as a percentage.
- Do not use the word "SHAP" or explain the underlying methodology -
  write for a retention agent who wants the plain-English reason, not a
  data scientist who wants the mechanism.
- Do not recommend any action - drafting outreach is a separate agent's job."""


def _format_factors(shap_details: list[dict]) -> str:
    return "\n".join(
        f"- {d['feature']}: {d['direction']} (SHAP value {d['shap_value']:+.4f})"
        for d in shap_details
    )


def build_prompt(churn_probability: float, shap_details: list[dict]) -> str:
    """Separated from explain_churn() so tests can assert the real SHAP
    feature names and values are actually present in the constructed
    prompt - the concrete mechanism that keeps the explanation grounded
    in this customer's real data rather than the model guessing."""
    return (
        f"Churn probability: {churn_probability:.1%}\n\n"
        f"Contributing factors, ranked by importance:\n{_format_factors(shap_details)}\n\n"
        "Write the explanation now."
    )


def explain_churn(churn_probability: float, shap_details: list[dict]) -> AgentResponse:
    """Raises AgentCallFailed on any failure. Every call site must catch
    this and fall back to src.model.explain_churn.format_risk_factors_text
    (the raw SHAP display string) rather than let the exception propagate -
    an LLM outage must never break the churn explanation feature outright,
    only degrade it to its pre-agent form."""
    if not shap_details:
        raise AgentCallFailed("No SHAP details provided - nothing to explain")
    user_prompt = build_prompt(churn_probability, shap_details)
    # max_tokens higher than the 2-3 sentence output alone needs: both
    # Groq models here are reasoning models that spend part of the budget
    # on hidden chain-of-thought before the visible answer (see
    # groq_client.DEFAULT_REASONING_EFFORT) - too tight a budget produces
    # an empty completion, not a short one.
    return complete(SYSTEM_PROMPT, user_prompt, model=MODEL_QUALITY, max_tokens=400, temperature=0.3)
