"""Tests for src/ai/llm.py's provider abstraction. GroqProvider is tested
against a mocked groq_client.complete (same mocking pattern as
tests/test_agents.py), never a real Groq API call. OpenAICompatibleProvider
is tested only for its selection/config wiring - it lazily imports
`openai`, which isn't installed in this project's dependencies, and isn't
exercised for real here."""

from __future__ import annotations

import pytest

from src.agents.groq_client import AgentResponse
from src.ai.llm import GroqProvider, OpenAICompatibleProvider, get_llm_provider


def test_get_llm_provider_defaults_to_groq(monkeypatch):
    monkeypatch.delenv("AI_LLM_PROVIDER", raising=False)
    assert isinstance(get_llm_provider(), GroqProvider)


def test_get_llm_provider_honors_env_var(monkeypatch):
    monkeypatch.setenv("AI_LLM_PROVIDER", "openai")
    assert isinstance(get_llm_provider(), OpenAICompatibleProvider)


def test_get_llm_provider_explicit_name_overrides_env(monkeypatch):
    monkeypatch.setenv("AI_LLM_PROVIDER", "openai")
    assert isinstance(get_llm_provider("groq"), GroqProvider)


def test_get_llm_provider_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        get_llm_provider("not_a_real_provider")


def test_groq_provider_delegates_to_groq_client(monkeypatch):
    from src.ai import llm as llm_module

    captured = {}

    def _fake_complete(system_prompt, user_prompt, max_tokens=500, temperature=0.3):
        captured.update(system_prompt=system_prompt, user_prompt=user_prompt,
                         max_tokens=max_tokens, temperature=temperature)
        return AgentResponse(text="answer", prompt_tokens=10, completion_tokens=5, model="test-model")

    monkeypatch.setattr(llm_module, "groq_complete", _fake_complete)

    provider = GroqProvider()
    response = provider.complete("sys", "user", max_tokens=200, temperature=0.1)

    assert response.text == "answer"
    assert captured == {"system_prompt": "sys", "user_prompt": "user", "max_tokens": 200, "temperature": 0.1}
