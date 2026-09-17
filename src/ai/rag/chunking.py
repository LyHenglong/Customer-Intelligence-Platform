"""Configurable chunking strategies for the RAG ingestion pipeline
(src/ai/rag/ingest.py).

Three strategies are implemented, per
AI_Customer_Intelligence_Claude_Code_Plan.md section 11's instruction not
to blindly pick one chunk size: "fixed", "overlapping", and
"structure_aware". Which one performs best is a retrieval-quality
question the evaluation harness (src/ai/evaluation/) answers by running
real benchmark queries against each, not something assumed here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


@dataclass
class RawChunk:
    text: str
    section: str | None = None
    metadata: dict = field(default_factory=dict)


def clean_text(text: str) -> str:
    """Normalizes whitespace without touching markdown structure -
    chunking strategies below still need headers/paragraph breaks intact."""
    text = text.replace("\r\n", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _words(text: str) -> list[str]:
    return text.split()


def chunk_fixed(text: str, chunk_size_words: int = 150) -> list[RawChunk]:
    """Non-overlapping fixed-size windows, by word count. The simplest
    baseline - cheap, but can split a sentence or an idea's supporting
    evidence across two chunks with no shared context between them."""
    words = _words(text)
    chunks = []
    for start in range(0, len(words), chunk_size_words):
        window = words[start:start + chunk_size_words]
        if window:
            chunks.append(RawChunk(text=" ".join(window)))
    return chunks


def chunk_overlapping(
    text: str, chunk_size_words: int = 150, overlap_words: int = 40
) -> list[RawChunk]:
    """Fixed-size windows with overlap, so an idea split across a chunk
    boundary is still fully present in at least one chunk."""
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")
    words = _words(text)
    if not words:
        return []
    chunks = []
    step = chunk_size_words - overlap_words
    start = 0
    while True:
        window = words[start:start + chunk_size_words]
        if not window:
            break
        chunks.append(RawChunk(text=" ".join(window)))
        if start + chunk_size_words >= len(words):
            break
        start += step
    return chunks


def chunk_structure_aware(
    text: str, max_chunk_size_words: int = 200, overlap_words: int = 30
) -> list[RawChunk]:
    """Splits on markdown headers first, so each chunk stays within one
    section's context - a section title never gets separated from its own
    content. A section still too long for one chunk is further split with
    chunk_overlapping, keeping the section name attached to every piece.
    Falls back to chunk_overlapping whole-document if the text has no
    markdown headers at all.
    """
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return chunk_overlapping(text, max_chunk_size_words, overlap_words)

    chunks: list[RawChunk] = []
    for i, m in enumerate(matches):
        section_title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        words = _words(body)
        if len(words) <= max_chunk_size_words:
            chunks.append(RawChunk(text=body, section=section_title))
        else:
            for sub in chunk_overlapping(body, max_chunk_size_words, overlap_words):
                chunks.append(RawChunk(text=sub.text, section=section_title))
    return chunks


CHUNKING_STRATEGIES = {
    "fixed": chunk_fixed,
    "overlapping": chunk_overlapping,
    "structure_aware": chunk_structure_aware,
}
