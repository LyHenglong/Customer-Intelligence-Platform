"""Citation guardrail: every citation returned to the caller must
correspond to a real piece of evidence gathered for this answer - never
a fabricated source, page, or document
(AI_Customer_Intelligence_Claude_Code_Plan.md section 17: "Do not
fabricate page numbers or document metadata").
"""

from __future__ import annotations

from src.ai.schemas import Citation, Evidence


def validate_citations(citations: list[Citation], evidence: list[Evidence]) -> list[Citation]:
    """Returns only the citations whose `source` matches some evidence
    item's `source` - drops anything else rather than passing it
    through. In this project's current pipeline (src/ai/graph.py),
    citations are always built directly from the same evidence they
    cite, so this is a safety net against a future integration bug more
    than an expected runtime filter today."""
    evidence_sources = {e.source for e in evidence}
    return [c for c in citations if c.source in evidence_sources]
