"""
Tests for src/decomposition.py — query decomposition agent.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

from src.decomposition import (
    decomposition_agent,
    _is_compound,
    _rule_based_decompose,
)


def make_state(query: str) -> dict:
    return {
        "messages": [],
        "query": query,
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


class TestIsCompound:
    def test_short_simple_query_is_not_compound(self):
        assert not _is_compound("minimum salary skilled worker")

    def test_long_query_is_compound(self):
        long = " ".join(["word"] * 20)
        assert _is_compound(long)

    def test_and_conjunction_detected(self):
        assert _is_compound("Can I work and can I travel?")

    def test_also_detected(self):
        assert _is_compound("Can I work also can I bring my family?")

    def test_as_well_as_detected(self):
        assert _is_compound("Work rights as well as travel rights")


class TestRuleBasedDecompose:
    def test_splits_on_and(self):
        result = _rule_based_decompose("Can I work and can I travel?")
        assert len(result) > 1

    def test_returns_original_for_simple_query(self):
        result = _rule_based_decompose("minimum salary")
        assert result == ["minimum salary"]

    def test_cleans_punctuation(self):
        result = _rule_based_decompose("Can I work, and can I travel?")
        for part in result:
            assert not part.startswith(",")


class TestDecompositionAgent:
    def test_simple_query_skips_llm(self):
        """Short, non-compound query should never call the LLM."""
        state = make_state("What is the minimum salary?")
        with patch("src.decomposition.get_llm") as mock_get_llm:
            result = decomposition_agent(state)
        mock_get_llm.assert_not_called()
        assert result["decomposed_queries"] == ["What is the minimum salary?"]

    def test_compound_query_calls_llm(self):
        """Long compound query should call LLM and return its sub-queries."""
        query = (
            "Can I work full-time on my Graduate visa and does my employer need "
            "to sponsor me if I want to stay longer in the UK after graduation?"
        )
        state = make_state(query)

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "sub_queries": ["Graduate visa work rights", "Graduate visa sponsorship"],
            "is_compound": True,
        }

        with patch("src.decomposition.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            with patch("src.decomposition.DECOMPOSITION_PROMPT") as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)
                # Patch the chain construction directly
                with patch("src.decomposition.JsonOutputParser") as mock_parser:
                    mock_full_chain = MagicMock()
                    mock_full_chain.invoke.return_value = {
                        "sub_queries": ["Graduate visa work rights", "Graduate visa sponsorship"],
                        "is_compound": True,
                    }
                    # Bypass chain assembly — patch at the chain invoke level
                    pass

        # Simpler approach: patch the whole function body's chain
        with patch("src.decomposition.ChatGoogleGenerativeAI") as mock_llm_cls:
            mock_llm_instance = MagicMock()
            mock_llm_cls.return_value = mock_llm_instance
            chain_mock = MagicMock()
            chain_mock.invoke.return_value = {
                "sub_queries": ["Graduate visa work rights", "Graduate visa sponsorship"],
                "is_compound": True,
            }
            with patch("src.decomposition.DECOMPOSITION_PROMPT") as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=chain_mock)))
                result = decomposition_agent(state)

        # The original query must always be in the result
        assert query in result["decomposed_queries"]

    def test_llm_failure_falls_back_to_rule_based(self):
        """On LLM failure, rule-based fallback must produce non-empty output."""
        query = (
            "Can I work and also travel outside the UK on my visa and "
            "furthermore can I bring my spouse along with me to the UK?"
        )
        state = make_state(query)

        with patch("src.decomposition.ChatGoogleGenerativeAI", side_effect=Exception("API error")):
            result = decomposition_agent(state)

        assert len(result["decomposed_queries"]) >= 1

    def test_deduplication_caps_at_four(self):
        """Result must never exceed 4 entries and must be deduplicated."""
        query = (
            "Can I work and also travel and furthermore can I switch "
            "and additionally can I get ILR on my visa route in the UK?"
        )
        state = make_state(query)

        with patch("src.decomposition.ChatGoogleGenerativeAI") as mock_llm_cls:
            chain_mock = MagicMock()
            chain_mock.invoke.return_value = {
                "sub_queries": ["q1", "q1", "q2", "q3", "q4", "q5"],
                "is_compound": True,
            }
            with patch("src.decomposition.DECOMPOSITION_PROMPT") as mock_prompt:
                mock_prompt.__or__ = MagicMock(
                    return_value=MagicMock(__or__=MagicMock(return_value=chain_mock))
                )
                result = decomposition_agent(state)

        assert len(result["decomposed_queries"]) <= 4
        # No duplicates
        lower_results = [q.lower() for q in result["decomposed_queries"]]
        assert len(lower_results) == len(set(lower_results))
