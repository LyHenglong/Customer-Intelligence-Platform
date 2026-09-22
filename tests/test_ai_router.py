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
        "has_rag": False, "customer_id": None, "segment_filters": {},
    }


def test_classify_is_case_insensitive():
    assert classify("WHAT DOES OUR POLICY SAY") == RAG_SEARCH


# ------------------------------------------------- segment filter extraction


def test_month_to_month_phrasing_extracts_a_contract_filter():
    _, signals = classify_with_signals("Why is churn increasing among month-to-month customers?")
    assert signals["segment_filters"] == {"contract": "month_to_month"}


def test_segment_phrases_map_to_warehouse_values():
    cases = {
        "why are two year contracts churning": {"contract": "two_year"},
        "risk factors for our one-year customers": {"contract": "one_year"},
        "why do new customers churn": {"tenure_bucket": "new_0_6mo"},
        "churn probability for loyal customers": {"tenure_bucket": "loyal_24mo_plus"},
    }
    for query, expected in cases.items():
        _, signals = classify_with_signals(query)
        assert signals["segment_filters"] == expected, query


def test_contract_and_tenure_phrases_combine():
    _, signals = classify_with_signals("churn probability for new customers on month-to-month")
    assert signals["segment_filters"] == {
        "contract": "month_to_month", "tenure_bucket": "new_0_6mo",
    }


def test_tenure_phrases_require_adjacency_and_degrade_safely():
    """"new ... customers" with words in between matches only the contract.
    Deliberate: the alternative is a looser pattern that would read "new"
    in unrelated phrasings and silently answer about the wrong population.
    Missing a filter yields a broader, correctly-labelled answer; inventing
    one yields a confidently wrong one."""
    _, signals = classify_with_signals("churn probability for new month-to-month customers")
    assert signals["segment_filters"] == {"contract": "month_to_month"}


def test_first_match_per_column_wins_so_comparisons_do_not_contradict():
    _, signals = classify_with_signals("compare month-to-month against two-year risk factors")
    assert signals["segment_filters"]["contract"] == "month_to_month"


def test_a_segment_phrase_alone_does_not_create_a_route():
    """segment_filters narrows evidence within a route; it must not be
    what promotes an otherwise-unsupported question into one."""
    route, signals = classify_with_signals("month-to-month")
    assert signals["segment_filters"] == {"contract": "month_to_month"}
    assert route == UNSUPPORTED
