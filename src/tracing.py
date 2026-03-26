"""
JapaPolicy AI — LangSmith Tracing
Centralised tracing setup. Import configure_tracing() before anything else.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from .config import settings


def configure_tracing(project_name: Optional[str] = None) -> bool:
    """
    Enable LangSmith tracing for the entire LangGraph pipeline.
    Call this ONCE at app startup — before building the graph.
    Returns True if tracing is active, False if LANGSMITH_API_KEY is missing.
    """
    api_key = settings.langsmith_api_key

    if not api_key:
        print("⚠️  LangSmith: LANGSMITH_API_KEY not set — tracing disabled.")
        return False

    project = project_name or settings.langsmith_project

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]     = api_key
    os.environ["LANGCHAIN_PROJECT"]     = project
    os.environ["LANGCHAIN_ENDPOINT"]    = settings.langsmith_endpoint

    print(f"✅ LangSmith tracing active  →  project: '{project}'")
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