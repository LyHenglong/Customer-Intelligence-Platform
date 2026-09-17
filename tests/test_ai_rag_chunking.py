"""Unit tests for src/ai/rag/chunking.py's three chunking strategies.
Pure text processing - no database, no embedding model."""

from __future__ import annotations

from src.ai.rag.chunking import (
    chunk_fixed,
    chunk_overlapping,
    chunk_structure_aware,
    clean_text,
)


def test_clean_text_collapses_whitespace_and_blank_lines():
    raw = "Line one.  \t\r\n\r\n\r\n\r\nLine two."
    cleaned = clean_text(raw)
    assert "\n\n\n" not in cleaned
    assert "Line one." in cleaned and "Line two." in cleaned


def test_chunk_fixed_splits_into_non_overlapping_windows():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_fixed(text, chunk_size_words=100)

    assert len(chunks) == 3
    assert all(len(c.text.split()) == 100 for c in chunks)
    # No overlap: concatenating all chunks reproduces every word exactly once.
    reconstructed = " ".join(c.text for c in chunks).split()
    assert reconstructed == text.split()


def test_chunk_overlapping_shares_words_between_adjacent_chunks():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_overlapping(text, chunk_size_words=100, overlap_words=20)

    assert len(chunks) >= 2
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    shared = set(first_words[-20:]) & set(second_words[:20])
    assert len(shared) == 20


def test_chunk_overlapping_rejects_overlap_not_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_overlapping("a b c", chunk_size_words=10, overlap_words=10)


def test_chunk_overlapping_handles_short_text_as_one_chunk():
    chunks = chunk_overlapping("just a few words here", chunk_size_words=100, overlap_words=20)
    assert len(chunks) == 1
    assert chunks[0].text == "just a few words here"


def test_chunk_structure_aware_splits_on_markdown_headers():
    text = (
        "# Title\n\nIntro text.\n\n"
        "## Section One\n\nContent for section one.\n\n"
        "## Section Two\n\nContent for section two.\n"
    )
    chunks = chunk_structure_aware(text, max_chunk_size_words=200)

    sections = [c.section for c in chunks]
    assert "Section One" in sections
    assert "Section Two" in sections
    for c in chunks:
        if c.section == "Section One":
            assert "section one" in c.text.lower()
        if c.section == "Section Two":
            assert "section two" in c.text.lower()


def test_chunk_structure_aware_falls_back_to_overlapping_with_no_headers():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_structure_aware(text, max_chunk_size_words=100)
    assert all(c.section is None for c in chunks)
    assert len(chunks) >= 2


def test_chunk_structure_aware_further_splits_an_oversized_section():
    long_body = " ".join(f"word{i}" for i in range(500))
    text = f"# Title\n\n## Big Section\n\n{long_body}\n"
    chunks = chunk_structure_aware(text, max_chunk_size_words=100, overlap_words=20)

    big_section_chunks = [c for c in chunks if c.section == "Big Section"]
    assert len(big_section_chunks) > 1
    assert all(len(c.text.split()) <= 100 for c in big_section_chunks)
