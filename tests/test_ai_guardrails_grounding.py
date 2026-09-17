"""Tests for src/ai/guardrails/grounding.py's numeric-grounding check.
Pure text logic, no DB/LLM."""

from __future__ import annotations

from src.ai.guardrails.grounding import extract_numbers, find_unsupported_numbers
from src.ai.schemas import Evidence


def test_extract_numbers_normalizes_currency_commas_and_percent():
    assert extract_numbers("Revenue was $1,234.50 and churn was 12%") == {"1234.50", "12"}


def test_extract_numbers_ignores_single_digit_incidental_numbers():
    assert extract_numbers("This is 1 of the factors") == set()


def test_extract_numbers_keeps_two_or_more_digit_numbers():
    assert extract_numbers("42 customers were flagged") == {"42"}


def test_find_unsupported_numbers_empty_when_every_number_is_in_evidence():
    evidence = [Evidence(type="model", source="churn model V1", claim="stat", value="probability=42, threshold=30")]
    unsupported = find_unsupported_numbers("The probability is 42 against a threshold of 30.", evidence)
    assert unsupported == []


def test_find_unsupported_numbers_flags_a_fabricated_number():
    evidence = [Evidence(type="model", source="churn model V1", claim="stat", value="probability=42")]
    unsupported = find_unsupported_numbers("The probability is 99, far above normal.", evidence)
    assert unsupported == ["99"]


def test_find_unsupported_numbers_does_not_recognize_unit_equivalence():
    """Documents the known limitation: exact-text matching cannot see
    that "30%" and "0.3" are the same value - see the module docstring."""
    evidence = [Evidence(type="database", source="marts.customer_360", claim="rate", value="0.3")]
    unsupported = find_unsupported_numbers("The rate is 30%.", evidence)
    assert unsupported == ["30"]


def test_find_unsupported_numbers_with_no_evidence_flags_every_number():
    unsupported = find_unsupported_numbers("There are 500 customers.", [])
    assert unsupported == ["500"]


def test_find_unsupported_numbers_empty_answer_has_nothing_to_flag():
    evidence = [Evidence(type="database", source="s", claim="c", value="42")]
    assert find_unsupported_numbers("No numbers here at all.", evidence) == []
