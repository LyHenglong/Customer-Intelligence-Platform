"""Tests for src/ai/rag/ingest.py's orchestration: document discovery,
id/title derivation, chunk_id determinism, and the idempotent
delete-then-upsert wiring. embed_texts and every vector_store/bm25 write
are monkeypatched - no embedding model, no database.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ai.rag import ingest as ingest_module


def test_discover_documents_finds_the_real_knowledge_corpus():
    docs = ingest_module.discover_documents()
    doc_ids = {str(p.relative_to(ingest_module.KNOWLEDGE_DIR)).replace("\\", "/") for p in docs}

    assert "retention/retention_playbook.md" in doc_ids
    assert "telecom_services/service_catalog.md" in doc_ids
    assert "customer_support/support_escalation_policy.md" in doc_ids
    assert "churn_strategy/churn_risk_segments.md" in doc_ids
    assert "product_catalog/plans_and_addons.md" in doc_ids
    assert "policies/discount_and_offer_policy.md" in doc_ids


def test_document_id_uses_forward_slash_relative_path():
    path = ingest_module.KNOWLEDGE_DIR / "retention" / "retention_playbook.md"
    assert ingest_module._document_id(path) == "retention/retention_playbook.md"


def test_title_from_text_uses_the_first_h1():
    text = "# Retention Playbook\n\nSome intro.\n\n## Section\n"
    assert ingest_module._title_from_text(text, fallback="fallback") == "Retention Playbook"


def test_title_from_text_falls_back_when_no_h1():
    assert ingest_module._title_from_text("no headers here", fallback="fallback_title") == "fallback_title"


def test_chunk_id_is_deterministic_and_index_sensitive():
    a = ingest_module._chunk_id("doc1", "fixed", 0)
    b = ingest_module._chunk_id("doc1", "fixed", 0)
    c = ingest_module._chunk_id("doc1", "fixed", 1)

    assert a == b  # deterministic for the same (document_id, strategy, index)
    assert a != c  # different index -> different id


def test_ingest_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        ingest_module.ingest(strategy="not_a_real_strategy")


def test_ingest_deletes_before_upserting_each_document(monkeypatch):
    calls = {"ensure_table": 0, "delete_document": [], "upsert_chunks": [], "reload_bm25": 0}

    monkeypatch.setattr(ingest_module, "ensure_table", lambda: calls.__setitem__("ensure_table", calls["ensure_table"] + 1))
    monkeypatch.setattr(ingest_module, "delete_document", lambda doc_id: calls["delete_document"].append(doc_id))
    monkeypatch.setattr(ingest_module, "upsert_chunks", lambda rows: calls["upsert_chunks"].append(rows))
    monkeypatch.setattr(ingest_module, "reload_bm25_index", lambda: calls.__setitem__("reload_bm25", calls["reload_bm25"] + 1))
    monkeypatch.setattr(ingest_module, "embed_texts", lambda texts: [[0.0, 0.0] for _ in texts])

    report = ingest_module.ingest(strategy="fixed")

    n_docs = len(ingest_module.discover_documents())
    assert calls["ensure_table"] == 1
    assert len(calls["delete_document"]) == n_docs
    assert len(calls["upsert_chunks"]) == n_docs
    assert calls["reload_bm25"] == 1
    assert report["documents"] == n_docs
    assert report["failed"] == 0
    assert report["chunks"] > 0
    assert report["chunks"] == sum(len(rows) for rows in calls["upsert_chunks"])

    # Every upserted row carries the fields the schema/vector_store expect.
    for rows in calls["upsert_chunks"]:
        for row in rows:
            assert set(row) >= {
                "chunk_id", "document_id", "title", "source", "section",
                "page", "text", "metadata", "embedding",
            }
            assert row["embedding"] == [0.0, 0.0]
