"""Output validation - the "Validate" step in
AI_Customer_Intelligence_Claude_Code_Plan.md section 13's workflow
diagram. Applied only to a generated answer (never to the fixed
UNSUPPORTED / insufficient-evidence templates in src/ai/graph.py, which
aren't model output and need no validation).

On a failed grounding check, the answer is replaced with the same
templated evidence summary src/ai/graph.py falls back to on an outright
LLM call failure, rather than retried - see
src/ai/guardrails/grounding.py's docstring for why this project doesn't
retry on a content-quality signal.
"""

from __future__ import annotations

import logging

from src.ai.guardrails.citation_validation import validate_citations
from src.ai.guardrails.grounding import find_unsupported_numbers
from src.ai.schemas import AssistantResponse

log = logging.getLogger("ai.guardrails.validation")


def templated_evidence_summary(evidence) -> str:
    return "Based on the available evidence:\n" + "\n".join(
        f"- ({e.type}, {e.source}) {e.claim}: {e.value}" for e in evidence
    )


def validate_response(response: AssistantResponse) -> AssistantResponse:
    citations = validate_citations(response.citations, response.evidence)
    dropped = len(response.citations) - len(citations)
    if dropped:
        log.warning(
            "dropped %d citation(s) with no matching evidence (trace_id=%s)",
            dropped, response.trace_id,
        )

    unsupported = find_unsupported_numbers(response.answer, response.evidence)
    if unsupported:
        # The rejected answer is logged alongside the flagged numbers
        # because the numbers alone aren't diagnosable: a run that flagged
        # ['17', '19', '801', '992'] turned out to be 17801/19992 written
        # with a Unicode space as a thousands separator, which took a
        # separate repro script to work out.
        log.warning(
            "answer contained number(s) not present in its evidence, falling back to a "
            "templated summary (trace_id=%s): %s | rejected answer: %r",
            response.trace_id, unsupported, response.answer[:500],
        )
        return response.model_copy(update={
            "answer": templated_evidence_summary(response.evidence),
            "citations": citations,
            "confidence": 0.5,
        })

    return response.model_copy(update={"citations": citations, "confidence": 1.0})
