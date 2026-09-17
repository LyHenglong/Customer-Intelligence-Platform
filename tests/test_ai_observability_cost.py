"""Tests for src/ai/observability/cost.py's estimate_cost()."""

from __future__ import annotations

from src.ai.observability.cost import estimate_cost


def test_known_model_uses_its_own_price():
    cost = estimate_cost("openai/gpt-oss-120b", input_tokens=1_000_000, output_tokens=0)
    assert cost == 0.15


def test_output_tokens_priced_separately_from_input():
    cost = estimate_cost("openai/gpt-oss-120b", input_tokens=0, output_tokens=1_000_000)
    assert cost == 0.75


def test_unknown_model_falls_back_to_default_price_without_raising():
    cost = estimate_cost("some-future-model", input_tokens=1_000_000, output_tokens=0)
    assert cost == 0.20


def test_none_model_does_not_raise():
    assert estimate_cost(None, input_tokens=100, output_tokens=100) >= 0.0


def test_zero_tokens_is_zero_cost():
    assert estimate_cost("openai/gpt-oss-120b", 0, 0) == 0.0
