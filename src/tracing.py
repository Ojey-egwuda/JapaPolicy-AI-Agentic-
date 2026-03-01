"""
JapaPolicy AI — LangSmith Tracing
Centralised tracing setup. Import configure_tracing() before anything else.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def configure_tracing(project_name: str = "japapolicy-ai") -> bool:
    """
    Enable LangSmith tracing for the entire LangGraph pipeline.
    Call this ONCE at app startup — before building the graph.
    Returns True if tracing is active, False if LANGSMITH_API_KEY is missing.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")

    if not api_key:
        print("⚠️  LangSmith: LANGSMITH_API_KEY not set — tracing disabled.")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]     = api_key
    os.environ["LANGCHAIN_PROJECT"]     = project_name
    os.environ["LANGCHAIN_ENDPOINT"]    = "https://eu.api.smith.langchain.com"

    print(f"✅ LangSmith tracing active  →  project: '{project_name}'")
    return True


def get_run_metadata(query_type: str = "", visa_category: str = "") -> dict:
    """
    Return metadata tags to attach to a LangSmith run.
    """
    return {
        "app":           "japapolicy-ai",
        "query_type":    query_type    or "unknown",
        "visa_category": visa_category or "unknown",
    }