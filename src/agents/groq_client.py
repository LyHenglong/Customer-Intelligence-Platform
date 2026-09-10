"""
Shared Groq client wrapper for the AI agent layer.

Three agents sit on top of this (explanation_agent, outreach_agent,
retrain_summary_agent). All three funnel through the single `complete()`
function here rather than each constructing its own `groq.Groq()` client
and reimplementing retry/timeout/logging - one place to get rate-limit
handling right, one place tests mock.

This layer is presentation, not prediction: it explains and drafts text
around outputs the churn model and recommender already produced (SHAP
values, service recommendations, retrain metrics). It never influences a
churn probability or a recommendation ranking - see the README's
"AI Agent Layer" section.

Two model tiers, matched to task complexity and call volume:
  MODEL_QUALITY - explanation + outreach agents. Tone and grounding in the
                  real SHAP numbers matter, and these run once per
                  displayed at-risk customer, so quality is worth the cost.
  MODEL_FAST    - retrain-summary agent. A more templated task (summarize
                  a metrics table) that runs at most a few times per DAG
                  run, so speed/cost matter more than the last mile of
                  quality.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APITimeoutError,
    Groq,
    RateLimitError,
)

load_dotenv()

log = logging.getLogger("agents.groq_client")

# Groq's model catalog moves - by the time this ran end to end,
# llama-3.3-70b-versatile and llama-3.1-8b-instant (the two models
# originally chosen here) had both been fully retired from Groq's
# offering (confirmed via client.models.list() - a 404 model_not_found,
# not an access issue). Replaced with the two nearest equivalents Groq
# actually serves now: gpt-oss-120b/20b, OpenAI's open-weight reasoning
# models, in the same quality/speed pairing as originally intended. See
# the README's "AI Agent Layer" section and Deviations for the full story
# - this is exactly the kind of pinned-model drift a real production
# system has to handle, not papered over here.
MODEL_QUALITY = "openai/gpt-oss-120b"
MODEL_FAST = "openai/gpt-oss-20b"

# Both are reasoning models: they spend some of the completion token
# budget on hidden chain-of-thought before the visible answer, even for a
# short reply (measured: "connection ok" cost 50-86 total tokens at
# default effort). "low" cut that by roughly 3x in testing (50 -> 17
# tokens for the same trivial reply) with no visible quality loss for
# these short, constrained-output tasks - none of the three agents need
# deep multi-step reasoning, just faithful phrasing of data already
# provided in the prompt.
DEFAULT_REASONING_EFFORT = "low"

# Free-tier rate limits are the expected failure mode here, not an edge
# case - retried with exponential backoff rather than surfaced as an error.
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 15.0


class AgentCallFailed(Exception):
    """Raised once retries are exhausted. Every call site in this project
    catches this and falls back to showing the raw underlying data (SHAP
    values, recommender output, metrics) instead of crashing - an LLM
    outage must never take down the dashboard or the API."""


@dataclass
class AgentResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str


_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise AgentCallFailed(
                "GROQ_API_KEY is not set. Add it to .env (see .env.example) - "
                "a free key is available at https://console.groq.com/keys"
            )
        _client = Groq(api_key=api_key)
    return _client


def complete(
    system_prompt: str,
    user_prompt: str,
    model: str = MODEL_QUALITY,
    max_tokens: int = 500,
    temperature: float = 0.3,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> AgentResponse:
    """One chat completion, with retry/backoff on rate limits and
    transient network errors.

    Deliberately low temperature (0.3, not the API default of ~0.7-1.0):
    these agents explain and draft over fixed, already-computed numbers
    (SHAP values, recommender scores, model metrics) - the point is
    faithful, consistent phrasing of real data, not creative variation.

    max_tokens defaults higher than the visible output alone would need
    (500, not ~150) because both models here are reasoning models that
    spend part of the budget on hidden chain-of-thought before the answer
    - see DEFAULT_REASONING_EFFORT above. A budget sized only for the
    visible text risks the empty-completion failure mode this project hit
    during verification (reasoning consumed the entire budget, leaving
    nothing for the answer).

    Raises AgentCallFailed after MAX_RETRIES. Every caller in this project
    must catch it and degrade to showing raw data rather than propagating
    the exception - see the module docstring.
    """
    client = _get_client()
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            choice = response.choices[0]
            usage = response.usage
            log.info(
                "groq call ok: model=%s prompt_tokens=%s completion_tokens=%s attempt=%d/%d",
                model,
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
                attempt,
                MAX_RETRIES,
            )
            text = (choice.message.content or "").strip()
            if not text:
                raise AgentCallFailed(f"Groq returned an empty completion (model={model})")
            return AgentResponse(
                text=text,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                model=model,
            )
        except RateLimitError as e:
            last_error = e
            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            log.warning(
                "groq rate limited (429), backing off %.1fs before attempt %d/%d",
                wait, attempt + 1, MAX_RETRIES,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)
            continue
        except (APITimeoutError, APIConnectionError) as e:
            last_error = e
            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            log.warning(
                "groq transient error (%s), retrying in %.1fs (attempt %d/%d)",
                type(e).__name__, wait, attempt + 1, MAX_RETRIES,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)
            continue
        except AgentCallFailed:
            raise
        except Exception as e:
            # Anything else (auth failure, malformed request, SDK bug) is
            # not worth retrying - fail fast rather than burn the retry
            # budget on an error that won't resolve itself.
            last_error = e
            log.error("groq call failed non-retryably: %s: %s", type(e).__name__, e)
            break

    raise AgentCallFailed(f"Groq call failed after {MAX_RETRIES} attempt(s): {last_error}")
