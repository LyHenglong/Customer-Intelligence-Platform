"""Hallucination-prevention guardrail: flags numeric claims in a
generated answer that don't appear anywhere in the evidence that
grounded it (AI_Customer_Intelligence_Claude_Code_Plan.md section 17).

Deliberately a post-hoc text check, not constrained decoding or a
retry-until-grounded loop: the generation prompt (src/ai/graph.py)
already instructs the model to answer only from evidence, and every
other LLM-failure path in this project degrades to a template rather
than retries a content-quality issue (see src/agents/groq_client.py's
retry policy, which only retries transient API errors, never a disliked
response) - this guardrail follows that same precedent rather than
inventing a new one.

This is a heuristic, not a proof of correctness: exact-text number
matching cannot recognize that "30%" and "0.3" are the same value, so it
will flag a correctly-derived percentage as unsupported if the evidence
only states the raw fraction. That's a known, accepted false-positive
mode - it fails toward caution (falling back to the literal evidence
text) rather than toward silently trusting an unverified transformation.
"""

from __future__ import annotations

import re

from src.ai.schemas import Evidence

_NUMBER_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")

# Some models group thousands with a space rather than a comma - "17 801"
# for 17801, often with a Unicode space (NBSP, narrow NBSP, thin space).
# Those are the same digits in the same order with no arithmetic applied,
# so they normalize to the same token instead of being reported as two
# invented numbers ("17" and "801"). This is representation, not the unit
# equivalence the module docstring deliberately refuses to infer.
#
# Only the thousands shape collapses - one separator between a digit and
# exactly three more digits - so unrelated adjacent numbers ("in 2024 300
# customers") aren't fused. If that guard ever misfires it produces a
# number that matches no evidence, i.e. it fails toward the same cautious
# fallback as before, never toward accepting an unverified claim.
_THOUSANDS_SEP_RE = re.compile(r"(?<=\d)[      ](?=\d{3}(?!\d))")

# Below this many digits, a token is treated as incidental (a lone digit
# in prose, an ordinal-style reference) rather than a checkable numeric
# claim.
_MIN_DIGITS_TO_CHECK = 2


def _normalize(number_token: str) -> str:
    return number_token.replace(",", "").replace("$", "").replace("%", "").strip()


def extract_numbers(text: str) -> set[str]:
    numbers = set()
    for match in _NUMBER_RE.findall(_THOUSANDS_SEP_RE.sub("", text or "")):
        normalized = _normalize(match)
        digit_count = sum(c.isdigit() for c in normalized)
        if digit_count >= _MIN_DIGITS_TO_CHECK:
            numbers.add(normalized)
    return numbers


def find_unsupported_numbers(answer: str, evidence: list[Evidence]) -> list[str]:
    """Returns every number in `answer` that does not appear (after the
    same normalization) anywhere in the evidence's claim/value text. An
    empty list means every numeric claim in the answer is traceable to at
    least one evidence item - not proof the answer is otherwise correct,
    only that it didn't invent a number."""
    evidence_text = " ".join(f"{e.claim} {e.value or ''}" for e in evidence)
    evidence_numbers = extract_numbers(evidence_text)
    answer_numbers = extract_numbers(answer)
    return sorted(answer_numbers - evidence_numbers)
