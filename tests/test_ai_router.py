"""Tests for src/ai/router.py's rule-based query classification. Pure
text logic, no DB, no LLM - see AI_Customer_Intelligence_Claude_Code_Plan.md
section 14 for the routing categories and example table this pins."""

from __future__ import annotations

import pytest

from src.ai.router import (
    CUSTOMER_LOOKUP,
    ML_ANALYSIS,
    MULTI_SOURCE,
    RAG_SEARCH,
    SQL_ANALYSIS,
    UNSUPPORTED,
    classify,
    classify_with_signals,
)


@pytest.mark.parametrize("query,expected", [
    ("Show customer CUST000123", CUSTOMER_LOOKUP),
    ("Why is CUST0005 at risk?", CUSTOMER_LOOKUP),  # customer id + ML keyword -> still a lookup
    ("What is the average monthly charge by contract type?", SQL_ANALYSIS),
    ("Compare churn rate between month-to-month and long-term customers.", SQL_ANALYSIS),
    ("What are the biggest risk factors for churn?", ML_ANALYSIS),
    ("What does our retention policy say about month-to-month customers?", RAG_SEARCH),
    ("What retention actions are supported by our business documentation?", RAG_SEARCH),
    ("Tell me tomorrow's weather", UNSUPPORTED),
    ("", UNSUPPORTED),
])
def test_classify_matches_expected_route(query, expected):
    assert classify(query) == expected


def test_customer_id_plus_rag_keyword_is_multi_source():
    route = classify("What policy supports a retention offer for CUST000123?")
    assert route == MULTI_SOURCE


def test_customer_id_plus_sql_keyword_is_multi_source():
    route = classify("Compare CUST000123's monthly charges against the average")
    assert route == MULTI_SOURCE


def test_sql_and_rag_keywords_together_is_multi_source():
    route = classify("Compare the churn rate by contract to what our retention policy recommends")
    assert route == MULTI_SOURCE


def test_signals_extract_the_customer_id_uppercased():
    _, signals = classify_with_signals("what about cust000123")
    assert signals["customer_id"] == "CUST000123"


def test_signals_report_false_when_nothing_matches():
    _, signals = classify_with_signals("hello there")
    assert signals == {
        "has_customer_id": False, "has_sql": False, "has_ml": False,
        "has_rag": False, "customer_id": None,
    }


def test_classify_is_case_insensitive():
    assert classify("WHAT DOES OUR POLICY SAY") == RAG_SEARCH
