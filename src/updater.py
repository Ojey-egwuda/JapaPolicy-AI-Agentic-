"""
JapaPolicy AI — Incremental Document Updater

Polls GOV.UK Atom feeds for UKVI/Home Office changes and incrementally
updates ChromaDB — only the changed documents, no full rebuild needed.

HOW IT WORKS
────────────
1. Poll Atom feeds every N minutes
2. For each new/updated entry, fetch structured content via Content API
3. Delete existing ChromaDB chunks for that source (identified by metadata)
4. Re-embed and re-index the fresh content
5. Track every processed document in SQLite so we never double-process

CONTENT HANDLING
────────────────
• HTML guidance pages  → extract body text, chunk, embed
• PDF attachments      → download to temp file, extract pages via PyPDF, embed
• Withdrawn documents  → remove their chunks from ChromaDB

USAGE
─────
  # One-off check (run manually or from cron):
  python -m src.updater

  # Continuous scheduled polling (blocks — run in background):
  python -m src.updater --schedule 1440   # once per day (default)
  python -m src.updater --schedule 720    # twice per day

  # From code:
  from src.updater import IncrementalUpdater
  updater = IncrementalUpdater()
  summary = updater.run_once()
"""

import hashlib
import html
import io
import os
import re
import sqlite3
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pypdf
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

from .config import settings

# ── Constants ─────────────────────────────────────────────────────────────────

FEEDS = [
    "https://www.gov.uk/government/organisations/uk-visas-and-immigration.atom",
    "https://www.gov.uk/government/organisations/home-office.atom",
]

CONTENT_API_BASE = "https://www.gov.uk/api/content"
ASSETS_BASE = "https://assets.publishing.service.gov.uk"

# GOV.UK paths to check proactively on every run (high-value immigration pages)
PROACTIVE_PATHS = [
    "/guidance/skilled-worker-visa",
    "/guidance/student-visa",
    "/guidance/family-visas-apply-for-one-extend-or-switch",
    "/guidance/graduate-visa",
    "/guidance/health-and-care-worker-visa",
    "/guidance/global-talent-visa",
    "/guidance/indefinite-leave-to-remain-in-the-uk",
    "/guidance/naturalisation-as-a-british-citizen-if-you-have-lived-in-the-uk",
    "/guidance/immigration-rules",
]

# Chunk size for HTML content (characters, ~250 tokens)
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

REQUEST_TIMEOUT = 30      # seconds
REQUEST_HEADERS = {
    "User-Agent": "JapaPolicy-AI-Updater/1.0 (immigration guidance monitor)",
    "Accept": "application/json",
}

# ── SQLite tracking schema ─────────────────────────────────────────────────────

_CREATE_TRACKER = """
CREATE TABLE IF NOT EXISTS tracked_documents (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    govuk_path        TEXT    NOT NULL UNIQUE,
    content_id        TEXT,
    title             TEXT,
    public_updated_at TEXT,
    last_processed_at TEXT    NOT NULL,
    chunk_count       INTEGER DEFAULT 0,
    status            TEXT    DEFAULT 'active'
);
CREATE TABLE IF NOT EXISTS update_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    added         INTEGER DEFAULT 0,
    updated       INTEGER DEFAULT 0,
    removed       INTEGER DEFAULT 0,
    errors        INTEGER DEFAULT 0,
    note          TEXT
);
"""


# ── Update tracker ─────────────────────────────────────────────────────────────

class UpdateTracker:
    """
    SQLite-backed record of every document we have indexed.
    Tells us whether a document is new, updated, or already current.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.updater_db_path
        self._lock = threading.Lock()
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        self._init_db()

    @contextmanager
    def _get_conn(self):
        if self._persistent_conn is not None:
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
                conn.executescript(_CREATE_TRACKER)

    def needs_update(self, govuk_path: str, public_updated_at: str) -> bool:
        """Return True if this path is new or has a newer timestamp than recorded."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT public_updated_at FROM tracked_documents WHERE govuk_path = ?",
                (govuk_path,),
            ).fetchone()
        if row is None:
            return True  # never seen before
        return row["public_updated_at"] != public_updated_at

    def record(
        self,
        govuk_path: str,
        content_id: str,
        title: str,
        public_updated_at: str,
        chunk_count: int,
        status: str = "active",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO tracked_documents
                        (govuk_path, content_id, title, public_updated_at,
                         last_processed_at, chunk_count, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(govuk_path) DO UPDATE SET
                        content_id        = excluded.content_id,
                        title             = excluded.title,
                        public_updated_at = excluded.public_updated_at,
                        last_processed_at = excluded.last_processed_at,
                        chunk_count       = excluded.chunk_count,
                        status            = excluded.status
                    """,
                    (govuk_path, content_id, title, public_updated_at, now, chunk_count, status),
                )

    def get_all(self) -> List[Dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tracked_documents ORDER BY last_processed_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def log_run(
        self,
        started_at: str,
        added: int,
        updated: int,
        removed: int,
        errors: int,
        note: str = "",
    ) -> None:
        finished_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO update_runs
                        (started_at, finished_at, added, updated, removed, errors, note)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (started_at, finished_at, added, updated, removed, errors, note),
                )

    def get_recent_runs(self, limit: int = 10) -> List[Dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM update_runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


# ── HTML text extraction ───────────────────────────────────────────────────────

def _strip_html(raw: str) -> str:
    """Strip HTML tags and decode entities, returning plain text."""
    # Remove script/style blocks entirely
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    # Replace block-level tags with newlines for readability
    raw = re.sub(r"<(br|p|div|li|h[1-6]|tr|td|th)[^>]*>", "\n", raw, flags=re.IGNORECASE)
    # Remove all remaining tags
    raw = re.sub(r"<[^>]+>", "", raw)
    # Decode HTML entities
    raw = html.unescape(raw)
    # Collapse whitespace
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r" {2,}", " ", raw)
    return raw.strip()


def _chunk_text(text: str, source: str, title: str, govuk_path: str) -> List[Document]:
    """
    Split plain text into overlapping chunks and wrap as LangChain Documents.
    Uses the same metadata fields that the retriever and VectorDB expect.
    """
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(Document(
                page_content=chunk.strip(),
                metadata={
                    "source":     source,
                    "title":      title,
                    "govuk_path": govuk_path,
                    "page":       chunk_index,
                    "chunk_type": "html",
                },
            ))
            chunk_index += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── GOV.UK API helpers ─────────────────────────────────────────────────────────

def fetch_feed(feed_url: str) -> List[Dict[str, str]]:
    """
    Fetch and parse an Atom feed. Returns a list of
    {id, title, link, updated} dicts, newest first.
    """
    resp = requests.get(feed_url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": REQUEST_HEADERS["User-Agent"]})
    resp.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    entries = []
    for entry in root.findall("atom:entry", ns):
        link_el  = entry.find("atom:link", ns)
        entries.append({
            "id":      (entry.findtext("atom:id", default="", namespaces=ns) or "").strip(),
            "title":   (entry.findtext("atom:title", default="", namespaces=ns) or "").strip(),
            "link":    link_el.get("href", "") if link_el is not None else "",
            "updated": (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip(),
        })
    return entries


def fetch_content_api(govuk_path: str) -> Optional[Dict]:
    """
    Call the GOV.UK Content API for a given path.
    Returns the parsed JSON dict or None on error.
    """
    url = f"{CONTENT_API_BASE}{govuk_path}"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"   ⚠️  Content API error for {govuk_path}: {e}")
        return None


def download_pdf(url: str) -> Optional[bytes]:
    """Download a PDF from assets.publishing.service.gov.uk."""
    try:
        resp = requests.get(url, timeout=60, headers={"User-Agent": REQUEST_HEADERS["User-Agent"]})
        resp.raise_for_status()
        if "pdf" not in resp.headers.get("Content-Type", "").lower():
            return None
        return resp.content
    except Exception as e:
        print(f"   ⚠️  PDF download error ({url}): {e}")
        return None


def pdf_bytes_to_documents(
    pdf_bytes: bytes,
    source: str,
    title: str,
    govuk_path: str,
) -> List[Document]:
    """Extract pages from PDF bytes using PyPDF and return Document objects."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        docs = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    page_content=text.strip(),
                    metadata={
                        "source":     source,
                        "title":      title,
                        "govuk_path": govuk_path,
                        "page":       page_num,
                        "chunk_type": "pdf",
                    },
                ))
        return docs
    except Exception as e:
        print(f"   ⚠️  PDF parsing error: {e}")
        return []


# ── Main updater ───────────────────────────────────────────────────────────────

class IncrementalUpdater:
    """
    Polls GOV.UK feeds and incrementally updates ChromaDB.

    Incremental strategy:
      • Delete all chunks for a given govuk_path (matched via metadata filter)
      • Re-embed and re-add the fresh content
    This is safe because ChromaDB supports metadata-based deletion.
    """

    def __init__(self, tracker_db_path: Optional[str] = None):
        self.tracker = UpdateTracker(db_path=tracker_db_path)
        self._vector_db = None   # lazy — don't load embedding model until needed

    @property
    def vector_db(self):
        if self._vector_db is None:
            from .vectordb import VectorDB
            self._vector_db = VectorDB()
        return self._vector_db

    # ── ChromaDB helpers ───────────────────────────────────────────────────────

    def _delete_existing_chunks(self, govuk_path: str) -> int:
        """
        Remove all ChromaDB chunks whose metadata.govuk_path matches.
        Returns the number of chunks deleted.
        """
        try:
            results = self.vector_db.collection.get(
                where={"govuk_path": govuk_path},
                include=[],
            )
            ids = results.get("ids", [])
            if ids:
                self.vector_db.collection.delete(ids=ids)
            return len(ids)
        except Exception as e:
            print(f"   ⚠️  Could not delete chunks for {govuk_path}: {e}")
            return 0

    def _add_chunks(self, docs: List[Document]) -> int:
        """
        Embed and add Document objects to ChromaDB.
        Uses stable IDs derived from govuk_path + chunk index.
        Returns the number of chunks added.
        """
        if not docs:
            return 0

        all_texts      = []
        all_embeddings = []
        all_metadatas  = []
        all_ids        = []

        for doc in docs:
            govuk_path = doc.metadata.get("govuk_path", "unknown")
            page       = doc.metadata.get("page", 0)
            # Stable, collision-resistant chunk ID
            path_hash  = hashlib.md5(govuk_path.encode()).hexdigest()[:10]
            chunk_id   = f"govuk_{path_hash}_{page}"

            embedding = self.vector_db.embedding_model.encode(
                [doc.page_content],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0].tolist()

            all_texts.append(doc.page_content)
            all_embeddings.append(embedding)
            all_metadatas.append(doc.metadata)
            all_ids.append(chunk_id)

        try:
            self.vector_db.collection.add(
                documents=all_texts,
                embeddings=all_embeddings,
                metadatas=all_metadatas,
                ids=all_ids,
            )
            return len(all_ids)
        except Exception as e:
            print(f"   ⚠️  ChromaDB add error: {e}")
            return 0

    # ── Document processing ────────────────────────────────────────────────────

    def process_path(self, govuk_path: str) -> Tuple[str, int]:
        """
        Fetch, parse, delete old chunks, and add new chunks for one GOV.UK path.

        Returns (status, chunk_count) where status is one of:
          'added', 'updated', 'withdrawn', 'skipped', 'error'
        """
        data = fetch_content_api(govuk_path)
        if data is None:
            return "error", 0

        # Withdrawn content — remove from ChromaDB
        if data.get("withdrawn_notice"):
            deleted = self._delete_existing_chunks(govuk_path)
            if deleted:
                print(f"   🗑️  Withdrawn: {govuk_path} ({deleted} chunks removed)")
            self.tracker.record(
                govuk_path=govuk_path,
                content_id=data.get("content_id", ""),
                title=data.get("title", govuk_path),
                public_updated_at=data.get("public_updated_at", ""),
                chunk_count=0,
                status="withdrawn",
            )
            return "withdrawn", 0

        public_updated_at = data.get("public_updated_at", "")
        content_id        = data.get("content_id", "")
        title             = data.get("title", govuk_path)

        # Skip if already up-to-date
        if not self.tracker.needs_update(govuk_path, public_updated_at):
            return "skipped", 0

        is_new = not any(
            d["govuk_path"] == govuk_path for d in self.tracker.get_all()
        )

        docs: List[Document] = []

        # 1. Extract HTML body text
        body_html = (data.get("details") or {}).get("body", "")
        if body_html:
            plain = _strip_html(body_html)
            if len(plain) > 100:
                docs.extend(_chunk_text(plain, govuk_path, title, govuk_path))

        # 2. Extract PDF attachments
        attachments = (data.get("details") or {}).get("attachments") or []
        for att in attachments:
            att_url = att.get("url", "")
            if not att_url.lower().endswith(".pdf"):
                continue
            att_title = att.get("title", title)
            print(f"   📄 Downloading PDF: {att_title[:60]}")
            pdf_bytes = download_pdf(att_url)
            if pdf_bytes:
                pdf_docs = pdf_bytes_to_documents(pdf_bytes, att_url, att_title, govuk_path)
                docs.extend(pdf_docs)
                print(f"      ↳ {len(pdf_docs)} pages extracted")

        if not docs:
            # Page exists but has no extractable content (e.g. index pages)
            return "skipped", 0

        # Delete stale chunks then add fresh ones
        deleted = self._delete_existing_chunks(govuk_path)
        added   = self._add_chunks(docs)

        self.tracker.record(
            govuk_path=govuk_path,
            content_id=content_id,
            title=title,
            public_updated_at=public_updated_at,
            chunk_count=added,
            status="active",
        )

        label = "➕ Added" if is_new else "🔄 Updated"
        print(
            f"   {label}: {title[:60]}"
            f" | {deleted} old → {added} new chunks"
        )
        return ("added" if is_new else "updated"), added

    # ── Feed polling ───────────────────────────────────────────────────────────

    def _paths_from_feeds(self) -> List[Tuple[str, str]]:
        """
        Poll all configured Atom feeds and return a deduplicated list of
        (govuk_path, updated_at) tuples from feed entries.
        """
        seen  = set()
        paths = []
        for feed_url in FEEDS:
            try:
                entries = fetch_feed(feed_url)
                for entry in entries:
                    link = entry.get("link", "")
                    # Extract path from full URL
                    path = link.replace("https://www.gov.uk", "").split("?")[0].strip()
                    if path and path not in seen:
                        seen.add(path)
                        paths.append((path, entry.get("updated", "")))
            except Exception as e:
                print(f"   ⚠️  Feed error ({feed_url}): {e}")
        return paths

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_once(self) -> Dict[str, Any]:
        """
        Run one full update cycle:
          1. Poll Atom feeds for new/updated documents
          2. Check proactive paths
          3. Process any that need updating

        Returns a summary dict.
        """
        started_at = datetime.now(timezone.utc).isoformat()
        print(f"\n{'='*60}")
        print(f"🔄 JapaPolicy Updater — {started_at[:19]}Z")
        print(f"{'='*60}")

        counts = {"added": 0, "updated": 0, "withdrawn": 0, "skipped": 0, "errors": 0}

        # Gather candidate paths
        feed_paths = self._paths_from_feeds()
        print(f"\n📡 Feed entries: {len(feed_paths)}")

        # Merge with proactive paths (no timestamp — tracker will decide if update needed)
        all_paths: List[Tuple[str, str]] = list(feed_paths)
        feed_path_set = {p for p, _ in feed_paths}
        for p in PROACTIVE_PATHS:
            if p not in feed_path_set:
                all_paths.append((p, ""))

        # Filter to only paths that need updating
        candidates = []
        for path, updated_at in all_paths:
            if updated_at:
                # We have a timestamp from the feed — quick pre-filter
                if self.tracker.needs_update(path, updated_at):
                    candidates.append(path)
            else:
                # Proactive path — check via Content API
                candidates.append(path)

        print(f"📋 Candidates to process: {len(candidates)}")

        if not candidates:
            print("✅ Everything is up to date.")
            self.tracker.log_run(started_at, 0, 0, 0, 0, "no changes detected")
            return {**counts, "total_candidates": 0}

        print()
        for i, path in enumerate(candidates, 1):
            print(f"[{i}/{len(candidates)}] {path}")
            try:
                status, _ = self.process_path(path)
                counts[status] = counts.get(status, 0) + 1
            except Exception as e:
                print(f"   ❌ Error: {e}")
                counts["errors"] += 1

            # Respect GOV.UK rate limit: max 10 req/sec across feed + content API calls
            time.sleep(0.15)

        print(f"\n{'='*60}")
        print(f"✅ Run complete")
        print(f"   ➕ Added:     {counts['added']}")
        print(f"   🔄 Updated:   {counts['updated']}")
        print(f"   🗑️  Withdrawn: {counts['withdrawn']}")
        print(f"   ⏭️  Skipped:   {counts['skipped']}")
        print(f"   ❌ Errors:    {counts['errors']}")
        print(f"{'='*60}\n")

        self.tracker.log_run(
            started_at=started_at,
            added=counts["added"],
            updated=counts["updated"],
            removed=counts["withdrawn"],
            errors=counts["errors"],
            note=f"{len(candidates)} candidates checked",
        )

        return {**counts, "total_candidates": len(candidates)}

    def run_scheduled(self, interval_minutes: int = 1440) -> None:
        """
        Run update cycles indefinitely on a fixed interval.
        Default is 1440 minutes (once per day). Blocks the calling thread.
        """
        print(f"⏰ Scheduled updater started — interval: {interval_minutes} min")
        while True:
            try:
                self.run_once()
            except Exception as e:
                print(f"❌ Updater cycle failed: {e}")
            print(f"💤 Sleeping {interval_minutes} min until next check…\n")
            time.sleep(interval_minutes * 60)

    def status(self) -> None:
        """Print a human-readable status report."""
        docs  = self.tracker.get_all()
        runs  = self.tracker.get_recent_runs(5)

        print(f"\n{'='*60}")
        print(f"📊 JapaPolicy Updater Status")
        print(f"{'='*60}")
        print(f"Tracked documents: {len(docs)}")

        active    = [d for d in docs if d["status"] == "active"]
        withdrawn = [d for d in docs if d["status"] == "withdrawn"]
        total_chunks = sum(d.get("chunk_count", 0) for d in active)

        print(f"  Active:    {len(active)}  ({total_chunks} total chunks)")
        print(f"  Withdrawn: {len(withdrawn)}")

        if runs:
            print(f"\nRecent runs:")
            for r in runs:
                ts = (r.get("started_at") or "")[:19]
                print(
                    f"  {ts}Z  "
                    f"+{r['added']} updated:{r['updated']} "
                    f"removed:{r['removed']} errors:{r['errors']}"
                )
        print(f"{'='*60}\n")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JapaPolicy incremental document updater")
    parser.add_argument(
        "--schedule", type=int, metavar="MINUTES", default=None,
        help="Run on a recurring schedule. Default interval is 1440 (once per day). Omit for a single run.",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print current tracker status and exit.",
    )
    args = parser.parse_args()

    updater = IncrementalUpdater()

    if args.status:
        updater.status()
    elif args.schedule:
        updater.run_scheduled(interval_minutes=args.schedule)
    else:
        updater.run_once()
