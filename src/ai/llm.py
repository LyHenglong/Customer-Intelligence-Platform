"""LLM provider abstraction (AI_Customer_Intelligence_Claude_Code_Plan.md
section 23 / Phase 7).

GroqProvider wraps the existing src/agents/groq_client.py rather than
duplicating its retry/backoff/rate-limit handling - that module's
complete() is already the one place in this project that gets Groq call
failures right, and every existing agent (explanation, outreach, retrain
summary) already depends on that behavior staying centralized.

OpenAICompatibleProvider exists so the application depends on this
module's interface rather than importing groq_client directly, without
forcing a second paid API dependency into requirements.txt: the `openai`
package is imported lazily, only if AI_LLM_PROVIDER=openai is actually
selected (same lazy-import pattern as src/ai/rag/embeddings.py's
sentence-transformers).
"""

from __future__ import annotations

import os
from typing import Protocol

from src.agents.groq_client import AgentResponse
from src.agents.groq_client import complete as groq_complete


class LLMProvider(Protocol):
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> AgentResponse: ...


class GroqProvider:
    """Default provider - the AI Agent Layer's existing Groq client."""

    def complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 500, temperature: float = 0.3
    ) -> AgentResponse:
        return groq_complete(
            system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature
        )


class OpenAICompatibleProvider:
    """Any OpenAI-Chat-Completions-compatible endpoint, configured via
    AI_OPENAI_BASE_URL / AI_OPENAI_API_KEY / AI_OPENAI_MODEL. Not wired
    into this project's default configuration or requirements.txt - it
    exists so the interface has more than one real implementation, per
    the plan's explicit instruction not to couple the application to one
    provider, without adding a dependency nothing currently uses."""

    def __init__(self):
        self._model = os.environ.get("AI_OPENAI_MODEL", "gpt-4o-mini")

    def complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 500, temperature: float = 0.3
    ) -> AgentResponse:
        from openai import OpenAI  # optional dependency - only needed for this provider

        client = OpenAI(
            api_key=os.environ.get("AI_OPENAI_API_KEY"),
            base_url=os.environ.get("AI_OPENAI_BASE_URL") or None,
        )
        response = client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = response.choices[0].message.content or ""
        usage = response.usage
        return AgentResponse(
            text=choice,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=self._model,
        )


_PROVIDERS = {"groq": GroqProvider, "openai": OpenAICompatibleProvider}


def get_llm_provider(name: str | None = None) -> LLMProvider:
    name = name or os.environ.get("AI_LLM_PROVIDER", "groq")
    if name not in _PROVIDERS:
        raise ValueError(f"unknown LLM provider {name!r}; available: {sorted(_PROVIDERS)}")
    return _PROVIDERS[name]()
