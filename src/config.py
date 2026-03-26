"""
JapaPolicy AI — Centralised Configuration

Single source of truth for all configurable values.
All other modules import from here instead of calling os.getenv() directly.

Usage:
    from .config import settings

    model = settings.google_model
    llm = ChatGoogleGenerativeAI(model=settings.google_model, api_key=settings.google_api_key)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Required (validated at LLM call time, not import time) ───────────────
    google_api_key: str = ""

    # ── LLM / Google Gemini ───────────────────────────────────────────────────
    google_model: str = "gemini-2.5-flash"
    google_model_temperature_default: float = 0.0
    google_model_temperature_hyde: float = 0.2
    google_model_temperature_response: float = 0.3

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"

    # ── Vector Database ───────────────────────────────────────────────────────
    chroma_collection_name: str = "uk_immigration_docs"
    chroma_db_path: str = "./chroma_db"

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir: str = "data"

    # ── Web Search ────────────────────────────────────────────────────────────
    tavily_api_key: str = ""

    # ── LangSmith Tracing (optional) ──────────────────────────────────────────
    langsmith_api_key: str = ""
    langsmith_project: str = "japapolicy-ai"
    # Default is the global (non-EU) endpoint. Override for EU: https://eu.api.smith.langchain.com
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # ── Conversation Persistence ──────────────────────────────────────────────
    conversation_db_path: str = "conversations.db"
    conversation_max_turns: int = 20

    # ── Incremental Updater ───────────────────────────────────────────────────
    updater_db_path: str = "updater_state.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
