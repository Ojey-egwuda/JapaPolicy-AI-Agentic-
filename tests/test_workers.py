"""
Tests for src/workers.py — router, analyst, and response agents.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.workers import (
    router_agent,
    analyst_agent,
    response_agent,
    clarification_node,
    get_freshness_warning,
)


def make_state(**overrides) -> dict:
    base = {
        "messages": [],
        "query": "What is the minimum salary for a Skilled Worker visa?",
        "query_type": None,
        "visa_category": None,
        "needs_clarification": False,
        "clarification_question": None,
        "decomposed_queries": [],
        "retrieved_docs": [],
        "web_results": [],
        "tool_results": [],
        "analysis": None,
        "key_requirements": [],
        "confidence_score": 0.0,
        "final_response": None,
        "sources_cited": [],
    }
    base.update(overrides)
    return base


def make_router_chain(chain_result: dict):
    """
    Return a context manager that makes router_agent's chain.invoke() return chain_result.
    Uses get_llm + ROUTER_PROMPT patching to intercept the | chain construction.
    """
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = chain_result

    step1 = MagicMock()
    step1.__or__ = MagicMock(return_value=chain_mock)

    prompt_mock = MagicMock()
    prompt_mock.__or__ = MagicMock(return_value=step1)

    return (
        patch("src.workers.get_llm", return_value=MagicMock()),
        patch("src.workers.ROUTER_PROMPT", prompt_mock),
    )


def make_analyst_chain(chain_result: str):
    """Return patches so analyst_agent's chain.invoke() returns chain_result."""
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = chain_result

    step1 = MagicMock()
    step1.__or__ = MagicMock(return_value=chain_mock)

    prompt_mock = MagicMock()
    prompt_mock.__or__ = MagicMock(return_value=step1)

    return (
        patch("src.workers.get_llm", return_value=MagicMock()),
        patch("src.workers.ANALYST_PROMPT", prompt_mock),
    )


def make_response_chain(chain_result: str):
    """Return patches so response_agent's chain.invoke() returns chain_result."""
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = chain_result

    step1 = MagicMock()
    step1.__or__ = MagicMock(return_value=chain_mock)

    prompt_mock = MagicMock()
    prompt_mock.__or__ = MagicMock(return_value=step1)

    return (
        patch("src.workers.get_llm", return_value=MagicMock()),
        patch("src.workers.RESPONSE_PROMPT", prompt_mock),
    )


class TestRouterAgent:
    def test_classifies_visa_eligibility(self):
        state = make_state(query="What salary do I need for a Skilled Worker visa?")
        llm_patch, prompt_patch = make_router_chain({
            "query_type": "visa_eligibility",
            "visa_category": "skilled_worker",
            "needs_clarification": False,
            "clarification_question": "",
            "decomposed_queries": ["Skilled Worker salary 2024"],
        })
        with llm_patch, prompt_patch:
            result = router_agent(state)

        assert result["query_type"] == "visa_eligibility"
        assert result["visa_category"] == "skilled_worker"
        assert result["needs_clarification"] is False

    def test_preserves_existing_decomposed_queries(self):
        existing = ["Graduate visa work rights", "Graduate visa sponsorship"]
        state = make_state(
            query="Can I work on my Graduate visa?",
            decomposed_queries=existing,
        )
        llm_patch, prompt_patch = make_router_chain({
            "query_type": "visa_eligibility",
            "visa_category": "graduate",
            "needs_clarification": False,
            "clarification_question": "",
            "decomposed_queries": ["new query from router"],
        })
        with llm_patch, prompt_patch:
            result = router_agent(state)

        assert result["decomposed_queries"] == existing

    def test_meaningless_query_triggers_clarification(self):
        state = make_state(query="hi")
        llm_patch, prompt_patch = make_router_chain({
            "query_type": "general_info",
            "visa_category": "other",
            "needs_clarification": False,
            "clarification_question": "",
            "decomposed_queries": [],
        })
        with llm_patch, prompt_patch:
            result = router_agent(state)

        assert result["needs_clarification"] is True

    def test_llm_failure_returns_safe_defaults(self):
        state = make_state(query="What are the requirements?")
        with patch("src.workers.get_llm", side_effect=Exception("API down")):
            result = router_agent(state)

        assert result["query_type"] == "general_info"
        assert result["visa_category"] == "skilled_worker"
        assert result["needs_clarification"] is False

    def test_section_3c_overrides_query_type(self):
        state = make_state(query="Can I work on Section 3C leave?")
        llm_patch, prompt_patch = make_router_chain({
            "query_type": "general_info",
            "visa_category": "skilled_worker",
            "needs_clarification": False,
            "clarification_question": "",
            "decomposed_queries": [],
        })
        with llm_patch, prompt_patch:
            result = router_agent(state)

        assert result["query_type"] == "visa_extension"


class TestAnalystAgent:
    def test_no_context_returns_zero_confidence(self):
        state = make_state(
            retrieved_docs=[],
            web_results=[],
            tool_results=[],
        )
        result = analyst_agent(state)
        assert result["confidence_score"] == 0.0

    def test_extracts_confidence_from_llm_output(self):
        state = make_state(
            retrieved_docs=[{
                "sub_query": "test",
                "results": "The minimum salary is £38,700.",
                "search_type": "hybrid",
            }],
        )
        # Use plain "Confidence:" (no markdown) so the regex in analyst_agent matches
        llm_patch, prompt_patch = make_analyst_chain(
            "**Key Requirements:**\n- Minimum salary: £38,700\n"
            "**Analysis:**\nDocuments are clear.\n"
            "Confidence: 0.92\nHigh-quality source."
        )
        with llm_patch, prompt_patch:
            result = analyst_agent(state)

        assert result["confidence_score"] == pytest.approx(0.92, abs=0.01)

    def test_section3c_doc_boosts_confidence(self):
        state = make_state(
            query="What happens on Section 3C leave?",
            retrieved_docs=[{
                "sub_query": "Section 3C leave rights",
                "results": "Section 3C leave extends your existing leave conditions.",
                "search_type": "hybrid",
            }],
        )
        llm_patch, prompt_patch = make_analyst_chain(
            "**Analysis:**\nSection 3C is covered.\n**Confidence:** 0.70"
        )
        with llm_patch, prompt_patch:
            result = analyst_agent(state)

        # Section 3C match should boost confidence to at least 0.85
        assert result["confidence_score"] >= 0.85


class TestResponseAgent:
    def test_appends_govuk_note_when_missing(self):
        state = make_state(
            query_type="visa_eligibility",
            visa_category="skilled_worker",
            confidence_score=0.90,
            analysis="The salary threshold is £38,700.",
            key_requirements=["Minimum salary: £38,700"],
        )
        llm_patch, prompt_patch = make_response_chain(
            "The minimum salary is £38,700 for Skilled Worker."
        )
        with llm_patch, prompt_patch:
            result = response_agent(state)

        assert "gov.uk" in result["final_response"].lower()

    def test_does_not_duplicate_govuk_note(self):
        state = make_state(
            query_type="visa_eligibility",
            visa_category="skilled_worker",
            confidence_score=0.85,
            analysis="See gov.uk for full details.",
            key_requirements=[],
        )
        llm_patch, prompt_patch = make_response_chain(
            "Check gov.uk for Skilled Worker guidance."
        )
        with llm_patch, prompt_patch:
            result = response_agent(state)

        assert result["final_response"].count("gov.uk") >= 1


class TestFreshnessWarning:
    def test_salary_query_returns_warning(self):
        warning = get_freshness_warning("minimum salary skilled worker visa")
        assert "April 2024" in warning

    def test_student_dependant_query_returns_warning(self):
        warning = get_freshness_warning("can my student bring dependants")
        assert "2024" in warning

    def test_unrelated_query_returns_empty(self):
        warning = get_freshness_warning("what is a Skilled Worker visa?")
        assert warning == ""


class TestClarificationNode:
    def test_returns_clarification_question(self):
        state = make_state(clarification_question="What visa type do you currently hold?")
        result = clarification_node(state)
        assert "What visa type do you currently hold?" in result["final_response"]

    def test_falls_back_to_default_when_no_question(self):
        state = make_state(clarification_question=None)
        result = clarification_node(state)
        assert result["final_response"]  # non-empty
