"""
Retention Outreach Agent.

Takes the churn explanation agent's output plus the recommender's top
suggested service for that customer, and drafts a short, personalized
retention message. Output only - this agent never sends anything; it
produces text for a human retention agent to review and send, the same
way the dashboard's other AI output is a draft, not an autonomous action.
"""

from __future__ import annotations

import re

from src.agents.groq_client import MODEL_QUALITY, AgentCallFailed, AgentResponse, complete

SYSTEM_PROMPT = """You are a customer retention specialist drafting a short outreach
message for a telecom customer flagged as at-risk of churning.

You are given: (1) a plain-English explanation of why this customer is at
risk, and (2) one specific service the company's recommender system
suggests offering them.

Rules:
- Write a short email-length message (4-6 sentences), warm but professional.
- You MUST explicitly mention the specific recommended service by name -
  this is the concrete offer the message exists to make.
- Do not invent a discount, price, or promotion that was not given to you.
- Do not reference "SHAP", "the model", "churn probability", or any
  internal system - write as if a human retention agent wrote this.
- Sign off generically (e.g. "The Retention Team") - do not invent a
  person's name.
- Output only the message text: no subject line, no preamble, no
  markdown formatting."""


def build_prompt(explanation: str, recommended_service: str) -> str:
    """Separated from draft_outreach() so tests can assert the real
    recommended service name is present in the constructed prompt, and so
    a test can independently check the same name later appears in the
    completion (see mentions_service below) - the concrete evidence that
    the draft is grounded in the actual recommendation, not a hallucinated
    one."""
    return (
        f"Why this customer is at risk: {explanation}\n\n"
        f"Recommended service to offer: {recommended_service}\n\n"
        "Draft the outreach message now."
    )


def mentions_service(text: str, recommended_service: str) -> bool:
    """Whether the drafted message actually names the recommended
    service. A case-insensitive substring check on the service's
    significant words (not stopwords like 'the'), tolerant of minor
    rewording ('Streaming TV' vs 'streaming tv service') but still a real
    check - not just trusting the prompt's instruction was followed.

    Used as a soft validation signal (logged, not enforced) rather than a
    hard failure: the LLM sometimes rephrases a two-word service name
    (e.g. "Online Security" -> "our security features") in a way a human
    would recognize as compliant but a strict substring match would miss.
    """
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", recommended_service) if len(w) > 2]
    if not words:
        return False
    lowered = text.lower()
    return any(w in lowered for w in words)


def draft_outreach(explanation: str, recommended_service: str) -> AgentResponse:
    """Raises AgentCallFailed on any failure. Every call site must catch
    this and fall back to showing the raw recommendation (service name +
    score) instead of a drafted message."""
    if not recommended_service:
        raise AgentCallFailed("No recommended service provided - nothing to offer")
    user_prompt = build_prompt(explanation, recommended_service)
    # See explanation_agent's note on max_tokens: both Groq models here are
    # reasoning models with hidden chain-of-thought overhead, so the
    # budget must cover that plus the visible 4-6 sentence email.
    response = complete(SYSTEM_PROMPT, user_prompt, model=MODEL_QUALITY, max_tokens=500, temperature=0.4)
    if not mentions_service(response.text, recommended_service):
        # Not fatal - the message is still shown - but worth knowing about,
        # since it means the guardrail in the prompt didn't hold.
        import logging
        logging.getLogger("agents.outreach_agent").warning(
            "drafted message may not mention the recommended service %r: %r",
            recommended_service, response.text[:200],
        )
    return response
