"""
Tests for src/config.py — centralized settings module.
"""

import pytest


def fresh_settings(monkeypatch, **overrides):
    """
    Create a Settings instance isolated from any real .env file.
    Clears all optional env vars to their defaults, then applies overrides.
    The .env file is suppressed by pointing env_file at a non-existent path.
    """
    # Suppress .env file loading by patching the model_config inside Settings
    from src.config import Settings
    from pydantic_settings import SettingsConfigDict

    # Build a Settings subclass that ignores the .env file
    class IsolatedSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=None,          # <-- no file loading
            extra="ignore",
        )

    # Clear env vars that could bleed in from the shell
    for var in [
        "GOOGLE_MODEL", "EMBEDDING_MODEL", "CHROMA_COLLECTION_NAME",
        "CHROMA_DB_PATH", "DATA_DIR", "TAVILY_API_KEY",
        "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT",
        "CONVERSATION_DB_PATH", "CONVERSATION_MAX_TURNS", "UPDATER_DB_PATH",
    ]:
        monkeypatch.delenv(var, raising=False)

    for k, v in overrides.items():
        monkeypatch.setenv(k.upper(), str(v))

    return IsolatedSettings()


def test_defaults_applied(monkeypatch):
    s = fresh_settings(monkeypatch)
    assert s.google_model == "gemini-2.5-flash"
    assert s.embedding_model == "sentence-transformers/all-mpnet-base-v2"
    assert s.chroma_collection_name == "uk_immigration_docs"
    assert s.chroma_db_path == "./chroma_db"
    assert s.data_dir == "data"
    assert s.conversation_max_turns == 20


def test_google_model_override(monkeypatch):
    s = fresh_settings(monkeypatch, GOOGLE_MODEL="gemini-2.5-flash")
    assert s.google_model == "gemini-2.5-flash"


def test_missing_google_api_key_defaults_to_empty(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    s = fresh_settings(monkeypatch)
    assert s.google_api_key == ""


def test_langsmith_endpoint_default_is_global(monkeypatch):
    s = fresh_settings(monkeypatch)
    assert "eu." not in s.langsmith_endpoint
    assert "api.smith.langchain.com" in s.langsmith_endpoint


def test_data_dir_override(monkeypatch):
    s = fresh_settings(monkeypatch, DATA_DIR="custom_data")
    assert s.data_dir == "custom_data"


def test_chroma_collection_name_default(monkeypatch):
    s = fresh_settings(monkeypatch)
    assert s.chroma_collection_name == "uk_immigration_docs"
    assert s.chroma_collection_name != "rag_documents"


def test_langsmith_endpoint_eu_override(monkeypatch):
    eu = "https://eu.api.smith.langchain.com"
    s = fresh_settings(monkeypatch, LANGSMITH_ENDPOINT=eu)
    assert s.langsmith_endpoint == eu


def test_conversation_max_turns_override(monkeypatch):
    s = fresh_settings(monkeypatch, CONVERSATION_MAX_TURNS="10")
    assert s.conversation_max_turns == 10
