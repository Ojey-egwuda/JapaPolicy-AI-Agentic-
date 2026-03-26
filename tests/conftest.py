"""
Shared pytest fixtures for JapaPolicy AI tests.
"""

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub out heavy / optional dependencies that may not be installed in the
# test environment.  These must be registered BEFORE any src.* imports so
# that module-level import statements don't fail at collection time.
# ---------------------------------------------------------------------------
_STUB_MODULES = [
    "chromadb",
    "sentence_transformers",
    "rank_bm25",
    "langchain_google_genai",
    "langchain_community",
    "langchain_community.tools",
    "langchain_community.tools.tavily_search",
    "langchain_community.document_loaders",
    "langchain_core",
    "langchain_core.messages",
    "langchain_core.prompts",
    "langchain_core.output_parsers",
    "langchain_core.tools",
    "langchain_core.documents",
    "langgraph",
    "langgraph.graph",
    "langgraph.graph.message",
    "langgraph.graph.state",
    "langgraph.checkpoint",
    "langgraph.checkpoint.memory",
]

import importlib.util as _ilu

for _mod in _STUB_MODULES:
    if _mod not in sys.modules:
        # Only stub if the real top-level package is NOT installed in the env.
        # This lets the venv's real packages (langchain_core, chromadb, etc.)
        # be used when available, while still allowing offline test runs.
        root_pkg = _mod.split(".")[0]
        if _ilu.find_spec(root_pkg) is None:
            mock = MagicMock()
            mock.__name__ = _mod
            mock.__spec__ = None
            sys.modules[_mod] = mock


# Make @tool behave as an identity decorator that adds .invoke(args_dict)
# Only apply if langchain_core was stubbed (i.e. not installed in this env).
def _tool_decorator(fn):
    """Minimal @tool stub: preserves the function and adds .invoke(kwargs_dict)."""
    def invoke(args: dict):
        return fn(**args)
    fn.invoke = invoke
    return fn

if isinstance(sys.modules.get("langchain_core.tools"), MagicMock):
    sys.modules["langchain_core.tools"].tool = _tool_decorator


# Ensure AIMessage is available from the stub
from unittest.mock import MagicMock as _MM
_ai_msg_cls = type("AIMessage", (), {"__init__": lambda self, content="": setattr(self, "content", content)})
if isinstance(sys.modules.get("langchain_core.messages"), MagicMock):
    sys.modules["langchain_core.messages"].AIMessage = _ai_msg_cls
else:
    # Use the real AIMessage from installed langchain_core
    try:
        from langchain_core.messages import AIMessage as _ai_msg_cls  # noqa: F811
    except ImportError:
        pass

import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def set_google_api_key(monkeypatch):
    """Ensure GOOGLE_API_KEY is always set so Settings() never raises."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-fake-api-key")


@pytest.fixture
def mock_llm():
    """A mock LLM that returns a canned AIMessage."""
    llm = MagicMock()
    llm.invoke.return_value = _ai_msg_cls(content="Mocked LLM response")
    return llm


@pytest.fixture
def mock_vector_db():
    """A mock VectorDB that returns synthetic search results."""
    db = MagicMock()
    db.search.return_value = {
        "documents": ["Sample immigration policy text about Skilled Worker requirements."],
        "metadatas": [{"source": "skilled_worker_guidance.pdf", "page": 5}],
        "similarities": [0.88],
        "search_type": "hybrid",
        "top_semantic_sim": 0.88,
    }
    return db


@pytest.fixture
def sample_state():
    """A fully-populated AgentState dict for testing agent functions."""
    return {
        "messages": [],
        "query": "What is the minimum salary for a Skilled Worker visa?",
        "query_type": "visa_eligibility",
        "visa_category": "skilled_worker",
        "needs_clarification": False,
        "clarification_question": None,
        "decomposed_queries": ["Skilled Worker visa minimum salary threshold 2024"],
        "retrieved_docs": [
            {
                "sub_query": "Skilled Worker visa minimum salary threshold 2024",
                "results": "Source: skilled_worker_guidance.pdf, Page 5\nThe minimum salary for a Skilled Worker visa is £38,700.",
                "search_type": "hybrid",
            }
        ],
        "web_results": [],
        "tool_results": [],
        "analysis": "The minimum salary threshold is £38,700 for most roles.\n\n**Confidence:** 0.90",
        "key_requirements": ["Minimum salary: £38,700", "Valid job offer required"],
        "confidence_score": 0.90,
        "final_response": None,
        "sources_cited": [],
    }


@pytest.fixture
def persistence_store():
    """An in-memory ConversationStore for isolated persistence tests."""
    from src.persistence import ConversationStore
    return ConversationStore(db_path=":memory:")


@pytest.fixture(autouse=True)
def set_google_api_key(monkeypatch):
    """Ensure GOOGLE_API_KEY is always set so Settings() never raises."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-fake-api-key")


@pytest.fixture
def mock_llm():
    """A mock LLM that returns a canned AIMessage."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="Mocked LLM response")
    return llm


@pytest.fixture
def mock_vector_db():
    """A mock VectorDB that returns synthetic search results."""
    db = MagicMock()
    db.search.return_value = {
        "documents": ["Sample immigration policy text about Skilled Worker requirements."],
        "metadatas": [{"source": "skilled_worker_guidance.pdf", "page": 5}],
        "similarities": [0.88],
        "search_type": "hybrid",
        "top_semantic_sim": 0.88,
    }
    return db


@pytest.fixture
def sample_state():
    """A fully-populated AgentState dict for testing agent functions."""
    return {
        "messages": [],
        "query": "What is the minimum salary for a Skilled Worker visa?",
        "query_type": "visa_eligibility",
        "visa_category": "skilled_worker",
        "needs_clarification": False,
        "clarification_question": None,
        "decomposed_queries": ["Skilled Worker visa minimum salary threshold 2024"],
        "retrieved_docs": [
            {
                "sub_query": "Skilled Worker visa minimum salary threshold 2024",
                "results": "Source: skilled_worker_guidance.pdf, Page 5\nThe minimum salary for a Skilled Worker visa is £38,700.",
                "search_type": "hybrid",
            }
        ],
        "web_results": [],
        "tool_results": [],
        "analysis": "The minimum salary threshold is £38,700 for most roles.\n\n**Confidence:** 0.90",
        "key_requirements": ["Minimum salary: £38,700", "Valid job offer required"],
        "confidence_score": 0.90,
        "final_response": None,
        "sources_cited": [],
    }


@pytest.fixture
def persistence_store():
    """An in-memory ConversationStore for isolated persistence tests."""
    from src.persistence import ConversationStore
    return ConversationStore(db_path=":memory:")
