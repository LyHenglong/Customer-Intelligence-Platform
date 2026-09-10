"""
Tests for the AI Agent Layer (src/agents/).

All Groq API calls are mocked - nothing here hits the real network, so
these run in CI with no GROQ_API_KEY and no rate-limit risk, the same
CI-safety discipline as the rest of this project's tests (synthetic
in-memory fixtures, no external service, no database).

Two mocking levels are used deliberately:
  - groq_client.complete() tests mock the underlying groq.Groq client
    directly, to exercise the real retry/backoff/error-mapping logic.
  - The three agent modules mock groq_client.complete() itself (one level
    up), since their own job is prompt construction and response
    handling, not re-testing the retry logic underneath them.
"""

from __future__ import annotations

import httpx
import pytest
from groq import APIConnectionError, APITimeoutError, RateLimitError

from src.agents import groq_client
from src.agents.groq_client import AgentCallFailed, AgentResponse
from src.agents import explanation_agent, outreach_agent, retrain_summary_agent


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------


class _FakeChoice:
    def __init__(self, content: str):
        self.message = type("M", (), {"content": content})()


class _FakeUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=20):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeCompletion:
    def __init__(self, content: str, prompt_tokens=10, completion_tokens=20):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


def _timeout_error() -> APITimeoutError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return APITimeoutError(request=request)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Every retry test below would otherwise sleep for real
    (BASE_BACKOFF_SECONDS * 2**n, up to several seconds per test) -
    patched to a no-op so the suite stays fast."""
    monkeypatch.setattr(groq_client.time, "sleep", lambda seconds: None)


@pytest.fixture
def fake_client(monkeypatch):
    """Replaces the module-level Groq client with a Mock whose
    .chat.completions.create is what each test configures."""
    from unittest.mock import MagicMock

    client = MagicMock()
    monkeypatch.setattr(groq_client, "_client", client)
    monkeypatch.setattr(groq_client, "_get_client", lambda: client)
    return client


SAMPLE_SHAP_DETAILS = [
    {"feature": "num_complaints", "shap_value": 0.42, "direction": "increases risk"},
    {"feature": "is_month_to_month", "shap_value": 0.31, "direction": "increases risk"},
    {"feature": "tenure", "shap_value": -0.18, "direction": "decreases risk"},
]


# --------------------------------------------------------------------------
# groq_client.complete() - retry, backoff, error mapping
# --------------------------------------------------------------------------


def test_complete_returns_response_on_first_success(fake_client):
    fake_client.chat.completions.create.return_value = _FakeCompletion("Hello world")
    result = groq_client.complete("system", "user")
    assert isinstance(result, AgentResponse)
    assert result.text == "Hello world"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 20
    assert fake_client.chat.completions.create.call_count == 1


def test_complete_retries_on_rate_limit_then_succeeds(fake_client):
    fake_client.chat.completions.create.side_effect = [
        _rate_limit_error(),
        _rate_limit_error(),
        _FakeCompletion("recovered"),
    ]
    result = groq_client.complete("system", "user")
    assert result.text == "recovered"
    assert fake_client.chat.completions.create.call_count == 3


def test_complete_retries_on_transient_timeout(fake_client):
    fake_client.chat.completions.create.side_effect = [
        _timeout_error(),
        _FakeCompletion("ok after timeout"),
    ]
    result = groq_client.complete("system", "user")
    assert result.text == "ok after timeout"


def test_complete_raises_agent_call_failed_after_max_retries(fake_client):
    fake_client.chat.completions.create.side_effect = _rate_limit_error()
    with pytest.raises(AgentCallFailed):
        groq_client.complete("system", "user")
    assert fake_client.chat.completions.create.call_count == groq_client.MAX_RETRIES


def test_complete_does_not_retry_non_retryable_errors(fake_client):
    """A malformed-request-style error should fail fast, not burn the
    retry budget on something that will never succeed on retry."""
    fake_client.chat.completions.create.side_effect = ValueError("bad request")
    with pytest.raises(AgentCallFailed):
        groq_client.complete("system", "user")
    assert fake_client.chat.completions.create.call_count == 1


def test_complete_raises_on_empty_completion(fake_client):
    fake_client.chat.completions.create.return_value = _FakeCompletion("   ")
    with pytest.raises(AgentCallFailed):
        groq_client.complete("system", "user")


def test_missing_api_key_raises_without_calling_client(monkeypatch):
    monkeypatch.setattr(groq_client, "_client", None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(AgentCallFailed, match="GROQ_API_KEY"):
        groq_client.complete("system", "user")


# --------------------------------------------------------------------------
# explanation_agent
# --------------------------------------------------------------------------


def test_explanation_prompt_contains_real_shap_values():
    prompt = explanation_agent.build_prompt(0.42, SAMPLE_SHAP_DETAILS)
    assert "42.0%" in prompt
    assert "num_complaints" in prompt
    assert "increases risk" in prompt
    assert "0.4200" in prompt  # the real signed SHAP value, not a placeholder
    assert "tenure" in prompt
    assert "-0.1800" in prompt or "-0.1800".lstrip("-") in prompt


def test_explanation_prompt_does_not_invent_factors():
    """Only the given features may appear as factors - guards against a
    prompt that accidentally includes unrelated data the model could
    then latch onto."""
    prompt = explanation_agent.build_prompt(0.42, SAMPLE_SHAP_DETAILS)
    given_features = {d["feature"] for d in SAMPLE_SHAP_DETAILS}
    for other_feature in ("annual_income", "credit_score", "gender"):
        assert other_feature not in given_features  # sanity on the fixture itself
        assert other_feature not in prompt


def test_explain_churn_calls_complete_with_grounded_prompt(monkeypatch):
    captured = {}

    def fake_complete(system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return AgentResponse(text="Flagged due to complaints and short tenure.", prompt_tokens=5, completion_tokens=8, model="x")

    monkeypatch.setattr(explanation_agent, "complete", fake_complete)
    result = explanation_agent.explain_churn(0.65, SAMPLE_SHAP_DETAILS)

    assert result.text == "Flagged due to complaints and short tenure."
    assert "num_complaints" in captured["user"]
    assert "65.0%" in captured["user"]


def test_explain_churn_raises_on_empty_shap_details():
    with pytest.raises(AgentCallFailed):
        explanation_agent.explain_churn(0.5, [])


def test_explain_churn_propagates_agent_call_failed(monkeypatch):
    """Callers (dashboard, API) are documented to catch AgentCallFailed
    and fall back to raw SHAP text - this only works if the agent
    actually lets the exception through rather than swallowing it."""
    def failing_complete(*args, **kwargs):
        raise AgentCallFailed("groq down")

    monkeypatch.setattr(explanation_agent, "complete", failing_complete)
    with pytest.raises(AgentCallFailed):
        explanation_agent.explain_churn(0.5, SAMPLE_SHAP_DETAILS)


# --------------------------------------------------------------------------
# outreach_agent
# --------------------------------------------------------------------------


def test_outreach_prompt_contains_real_recommended_service():
    prompt = outreach_agent.build_prompt("They are at risk due to complaints.", "Streaming Tv")
    assert "Streaming Tv" in prompt
    assert "They are at risk due to complaints." in prompt


@pytest.mark.parametrize(
    "text,service,expected",
    [
        ("We'd love to offer you our Streaming TV package!", "Streaming Tv", True),
        ("we think streaming tv would suit you", "Streaming Tv", True),
        ("Here's a great deal on Online Security for your account.", "Online Security", True),
        ("Thanks for being a loyal customer, here's 10% off.", "Streaming Tv", False),
        ("We have a great new Tech Support offer.", "Online Security", False),
    ],
)
def test_mentions_service(text, service, expected):
    assert outreach_agent.mentions_service(text, service) == expected


def test_draft_outreach_returns_response_that_mentions_service(monkeypatch):
    def fake_complete(system_prompt, user_prompt, **kwargs):
        return AgentResponse(
            text="We noticed you might enjoy our Streaming Tv service - take a look!",
            prompt_tokens=5, completion_tokens=12, model="x",
        )

    monkeypatch.setattr(outreach_agent, "complete", fake_complete)
    result = outreach_agent.draft_outreach("At risk due to short tenure.", "Streaming Tv")
    assert "Streaming Tv" in result.text


def test_draft_outreach_warns_when_service_not_mentioned(monkeypatch, caplog):
    """The hallucination guard: if the model ignores the instruction to
    name the service, this must be detectable (logged), not silently
    accepted as if the draft were grounded."""
    def fake_complete(system_prompt, user_prompt, **kwargs):
        return AgentResponse(text="Thanks for being a great customer!", prompt_tokens=5, completion_tokens=8, model="x")

    monkeypatch.setattr(outreach_agent, "complete", fake_complete)
    with caplog.at_level("WARNING"):
        result = outreach_agent.draft_outreach("At risk.", "Streaming Tv")

    assert result.text == "Thanks for being a great customer!"  # still returned
    assert any("may not mention" in rec.message for rec in caplog.records)


def test_draft_outreach_raises_on_no_service():
    with pytest.raises(AgentCallFailed):
        outreach_agent.draft_outreach("some explanation", "")


def test_draft_outreach_propagates_agent_call_failed(monkeypatch):
    def failing_complete(*args, **kwargs):
        raise AgentCallFailed("groq down")

    monkeypatch.setattr(outreach_agent, "complete", failing_complete)
    with pytest.raises(AgentCallFailed):
        outreach_agent.draft_outreach("explanation", "Streaming Tv")


# --------------------------------------------------------------------------
# retrain_summary_agent
# --------------------------------------------------------------------------

CURRENT_METRICS = {
    "version": "20260910T124230Z", "precision_churn": 0.16, "recall_churn": 0.60,
    "f1_churn": 0.2528, "roc_auc": 0.6693,
}
PREVIOUS_METRICS = {
    "version": "20260909T163041Z", "precision_churn": 0.146, "recall_churn": 0.649,
    "f1_churn": 0.2350, "roc_auc": 0.6519,
}
DRIFT_SUMMARY_NO_DRIFT = {"max_psi": 0.0006, "drift_detected": False, "drifted_features": []}
DRIFT_SUMMARY_WITH_DRIFT = {"max_psi": 0.31, "drift_detected": True, "drifted_features": ["monthlycharges"]}


def test_retrain_prompt_contains_real_metrics():
    prompt = retrain_summary_agent.build_prompt(CURRENT_METRICS, PREVIOUS_METRICS, DRIFT_SUMMARY_NO_DRIFT)
    assert "20260910T124230Z" in prompt
    assert "0.6693" in prompt
    assert "20260909T163041Z" in prompt
    assert "0.6519" in prompt


def test_retrain_prompt_handles_no_previous_model():
    prompt = retrain_summary_agent.build_prompt(CURRENT_METRICS, None, DRIFT_SUMMARY_NO_DRIFT)
    assert "first recorded model version" in prompt


def test_retrain_prompt_names_drifted_features():
    prompt = retrain_summary_agent.build_prompt(CURRENT_METRICS, PREVIOUS_METRICS, DRIFT_SUMMARY_WITH_DRIFT)
    assert "monthlycharges" in prompt
    assert "0.3100" in prompt


def test_summarize_retrain_calls_complete_with_grounded_prompt(monkeypatch):
    captured = {}

    def fake_complete(system_prompt, user_prompt, **kwargs):
        captured["user"] = user_prompt
        return AgentResponse(text="The model improved slightly.", prompt_tokens=15, completion_tokens=30, model="x")

    monkeypatch.setattr(retrain_summary_agent, "complete", fake_complete)
    result = retrain_summary_agent.summarize_retrain(CURRENT_METRICS, PREVIOUS_METRICS, DRIFT_SUMMARY_NO_DRIFT)

    assert result.text == "The model improved slightly."
    assert "20260910T124230Z" in captured["user"]


def test_summarize_retrain_raises_on_no_current_metrics():
    with pytest.raises(AgentCallFailed):
        retrain_summary_agent.summarize_retrain({}, None, None)


def test_summarize_retrain_propagates_agent_call_failed(monkeypatch):
    """The DAG task is documented to catch this and fall back to writing
    the raw metrics dict - only works if this doesn't swallow it first."""
    def failing_complete(*args, **kwargs):
        raise AgentCallFailed("groq down")

    monkeypatch.setattr(retrain_summary_agent, "complete", failing_complete)
    with pytest.raises(AgentCallFailed):
        retrain_summary_agent.summarize_retrain(CURRENT_METRICS, PREVIOUS_METRICS, DRIFT_SUMMARY_NO_DRIFT)
