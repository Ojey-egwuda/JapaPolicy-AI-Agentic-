"""
Tests for src/persistence.py — SQLite-backed ConversationStore.
"""

import pytest
from src.persistence import ConversationStore


class TestConversationStore:
    def test_save_and_load_conversation(self, persistence_store):
        persistence_store.add_turn("thread-1", "What is the salary?", "It is £38,700.")
        history = persistence_store.get_history("thread-1")
        assert len(history) == 1
        assert history[0]["question"] == "What is the salary?"
        assert history[0]["answer"] == "It is £38,700."

    def test_get_history_empty_thread_returns_empty_list(self, persistence_store):
        history = persistence_store.get_history("nonexistent-thread")
        assert history == []

    def test_max_turns_enforced(self, persistence_store):
        from src.config import settings
        max_turns = settings.conversation_max_turns
        # Insert more turns than the cap
        for i in range(max_turns + 5):
            persistence_store.add_turn("thread-2", f"Q{i}", f"A{i}")
        history = persistence_store.get_history("thread-2")
        assert len(history) <= max_turns

    def test_keeps_most_recent_turns(self, persistence_store):
        for i in range(25):
            persistence_store.add_turn("thread-3", f"Q{i}", f"A{i}")
        history = persistence_store.get_history("thread-3")
        # Oldest entries should be trimmed; most recent should be present
        questions = [h["question"] for h in history]
        assert "Q24" in questions  # last one must be kept

    def test_clear_thread_removes_only_that_thread(self, persistence_store):
        persistence_store.add_turn("thread-A", "QA", "AA")
        persistence_store.add_turn("thread-B", "QB", "AB")
        persistence_store.clear_thread("thread-A")
        assert persistence_store.get_history("thread-A") == []
        assert len(persistence_store.get_history("thread-B")) == 1

    def test_clear_all_removes_everything(self, persistence_store):
        persistence_store.add_turn("thread-X", "Q1", "A1")
        persistence_store.add_turn("thread-Y", "Q2", "A2")
        persistence_store.clear_all()
        assert persistence_store.get_history("thread-X") == []
        assert persistence_store.get_history("thread-Y") == []

    def test_metadata_stored_and_retrieved(self, persistence_store):
        metadata = {
            "confidence": "high",
            "query_type": "visa_eligibility",
            "visa_category": "skilled_worker",
        }
        persistence_store.add_turn("thread-meta", "Q?", "A.", metadata=metadata)
        history = persistence_store.get_history("thread-meta")
        assert history[0]["confidence"] == "high"
        assert history[0]["query_type"] == "visa_eligibility"
        assert history[0]["visa_category"] == "skilled_worker"

    def test_get_all_thread_ids(self, persistence_store):
        persistence_store.add_turn("alpha", "Q", "A")
        persistence_store.add_turn("beta", "Q", "A")
        ids = persistence_store.get_all_thread_ids()
        assert "alpha" in ids
        assert "beta" in ids

    def test_multiple_turns_ordered_oldest_first(self, persistence_store):
        persistence_store.add_turn("thread-order", "first", "A1")
        persistence_store.add_turn("thread-order", "second", "A2")
        persistence_store.add_turn("thread-order", "third", "A3")
        history = persistence_store.get_history("thread-order")
        questions = [h["question"] for h in history]
        assert questions == ["first", "second", "third"]

    def test_in_memory_store_no_file_created(self, tmp_path):
        """ConversationStore(db_path=':memory:') must not create any file."""
        store = ConversationStore(db_path=":memory:")
        store.add_turn("t", "Q", "A")
        assert len(list(tmp_path.iterdir())) == 0
