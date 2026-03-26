"""
JapaPolicy AI — Conversation Persistence

SQLite-backed store for conversation history.
Replaces the in-memory dict in AgenticRAGAssistant so history
survives process restarts.

Usage:
    from .persistence import ConversationStore

    store = ConversationStore()                        # uses settings.conversation_db_path
    store = ConversationStore(db_path=":memory:")      # in-memory (tests)

    store.add_turn(thread_id, question, answer, metadata)
    turns = store.get_history(thread_id)
    store.clear_thread(thread_id)
    store.clear_all()
"""

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from .config import settings

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id     TEXT    NOT NULL,
    turn_index    INTEGER NOT NULL,
    question      TEXT    NOT NULL,
    answer        TEXT    NOT NULL,
    confidence    TEXT,
    query_type    TEXT,
    visa_category TEXT,
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_thread ON conversations(thread_id);
"""


class ConversationStore:
    """
    Persistent conversation store backed by SQLite.

    Thread-safe for single-process use (Streamlit / CLI).
    Pass db_path=":memory:" for isolated test instances.

    Implementation note: for ":memory:" databases a single connection is kept
    open for the lifetime of the store because each new connection to ":memory:"
    creates a fresh, empty database.  For file-based paths a new connection is
    opened per operation (standard practice).
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path if db_path is not None else settings.conversation_db_path
        self._lock = threading.Lock()
        # Keep a persistent connection open for in-memory databases
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        self._init_db()

    @contextmanager
    def _get_conn(self):
        if self._persistent_conn is not None:
            # In-memory mode: reuse the single open connection
            try:
                yield self._persistent_conn
                self._persistent_conn.commit()
            except Exception:
                self._persistent_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_db(self) -> None:
        with self._lock:
            with self._get_conn() as conn:
                conn.executescript(_CREATE_TABLE)

    def add_turn(
        self,
        thread_id: str,
        question: str,
        answer: str,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """
        Append a conversation turn for the given thread.
        Automatically trims old turns to stay within settings.conversation_max_turns.
        """
        meta = metadata or {}
        created_at = datetime.now(timezone.utc).isoformat()

        with self._lock:
            with self._get_conn() as conn:
                # Determine next turn index
                row = conn.execute(
                    "SELECT COALESCE(MAX(turn_index), -1) FROM conversations WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()
                next_index = (row[0] or 0) + 1

                conn.execute(
                    """
                    INSERT INTO conversations
                        (thread_id, turn_index, question, answer,
                         confidence, query_type, visa_category, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        next_index,
                        question,
                        answer,
                        meta.get("confidence"),
                        meta.get("query_type"),
                        meta.get("visa_category"),
                        created_at,
                    ),
                )

                # Trim to max_turns — keep the most recent N rows
                max_turns = settings.conversation_max_turns
                conn.execute(
                    """
                    DELETE FROM conversations
                    WHERE thread_id = ?
                      AND id NOT IN (
                          SELECT id FROM conversations
                          WHERE thread_id = ?
                          ORDER BY id DESC
                          LIMIT ?
                      )
                    """,
                    (thread_id, thread_id, max_turns),
                )

    def get_history(
        self, thread_id: str, max_turns: int = None
    ) -> List[Dict[str, Any]]:
        """
        Return conversation turns for the given thread, oldest first.
        """
        limit = max_turns or settings.conversation_max_turns
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT question, answer, confidence, query_type, visa_category, created_at
                FROM (
                    SELECT * FROM conversations
                    WHERE thread_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (thread_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_thread(self, thread_id: str) -> None:
        """Delete all turns for a single thread."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "DELETE FROM conversations WHERE thread_id = ?", (thread_id,)
                )

    def clear_all(self) -> None:
        """Delete all conversation history."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM conversations")

    def get_all_thread_ids(self) -> List[str]:
        """Return a list of all known thread IDs."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT thread_id FROM conversations ORDER BY thread_id"
            ).fetchall()
        return [r[0] for r in rows]
