"""Approximate LLM cost estimation
(AI_Customer_Intelligence_Claude_Code_Plan.md section 22's "estimated
cost" trace field).

The prices below are illustrative placeholders, not verified against
Groq's current pricing page - provider rate cards change, and this
project has no billing integration to cross-check against. Treat
estimated_cost_usd as useful for *relative* cost tracking across
requests/models, never as a billing-accurate figure. Re-check
https://groq.com/pricing before trusting this for anything real. An
unknown model falls back to a conservative default rather than raising,
since a trace write should never fail over a missing price entry.
"""

from __future__ import annotations

# USD per 1,000,000 tokens, (input_price, output_price).
_PRICE_PER_MILLION_TOKENS = {
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
}
_DEFAULT_PRICE = (0.20, 0.80)


def estimate_cost(model: str | None, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = _PRICE_PER_MILLION_TOKENS.get(model, _DEFAULT_PRICE)
    cost = (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price
    return round(cost, 8)
