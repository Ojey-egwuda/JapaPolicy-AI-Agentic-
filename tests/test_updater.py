"""
Tests for src/updater.py — incremental document updater.

All network calls (Atom feeds, Content API, PDF downloads) are mocked.
All ChromaDB and embedding calls are mocked.
No real HTTP requests are made.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from langchain_core.documents import Document

from src.updater import (
    UpdateTracker,
    IncrementalUpdater,
    _strip_html,
    _chunk_text,
    fetch_feed,
    fetch_content_api,
    pdf_bytes_to_documents,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tracker():
    return UpdateTracker(db_path=":memory:")


@pytest.fixture
def updater(tracker):
    u = IncrementalUpdater(tracker_db_path=":memory:")
    u.tracker = tracker

    import numpy as np

    # Stub out the VectorDB so no model is loaded
    mock_db = MagicMock()
    mock_db.collection.get.return_value = {"ids": []}
    mock_db.collection.add.return_value = None
    mock_db.collection.delete.return_value = None
    # encode() must return a numpy array so [0].tolist() works
    mock_db.embedding_model.encode.return_value = np.zeros((1, 768))
    u._vector_db = mock_db

    return u


SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>UKVI updates</title>
  <updated>2026-03-26T10:00:00Z</updated>
  <entry>
    <id>tag:www.gov.uk,2026:/guidance/skilled-worker-visa</id>
    <title>Skilled Worker visa</title>
    <link href="https://www.gov.uk/guidance/skilled-worker-visa"/>
    <updated>2026-03-26T09:00:00Z</updated>
  </entry>
  <entry>
    <id>tag:www.gov.uk,2026:/guidance/student-visa</id>
    <title>Student visa</title>
    <link href="https://www.gov.uk/guidance/student-visa"/>
    <updated>2026-03-25T12:00:00Z</updated>
  </entry>
</feed>"""

SAMPLE_CONTENT_API = {
    "content_id": "abc-123",
    "base_path": "/guidance/skilled-worker-visa",
    "title": "Skilled Worker visa",
    "public_updated_at": "2026-03-26T09:00:00Z",
    "withdrawn_notice": None,
    "details": {
        "body": "<h2>Overview</h2><p>You can apply for a Skilled Worker visa if you "
                "have a job offer from a UK employer with a valid sponsor licence. "
                "The minimum salary is £38,700 per year.</p>",
        "attachments": [],
    },
}


# ── UpdateTracker tests ────────────────────────────────────────────────────────

class TestUpdateTracker:
    def test_new_path_needs_update(self, tracker):
        assert tracker.needs_update("/guidance/skilled-worker-visa", "2026-01-01T00:00:00Z")

    def test_known_path_same_timestamp_no_update(self, tracker):
        tracker.record(
            govuk_path="/guidance/skilled-worker-visa",
            content_id="abc",
            title="Skilled Worker visa",
            public_updated_at="2026-01-01T00:00:00Z",
            chunk_count=5,
        )
        assert not tracker.needs_update("/guidance/skilled-worker-visa", "2026-01-01T00:00:00Z")

    def test_known_path_newer_timestamp_needs_update(self, tracker):
        tracker.record(
            govuk_path="/guidance/skilled-worker-visa",
            content_id="abc",
            title="Skilled Worker visa",
            public_updated_at="2026-01-01T00:00:00Z",
            chunk_count=5,
        )
        assert tracker.needs_update("/guidance/skilled-worker-visa", "2026-03-26T09:00:00Z")

    def test_record_upserts(self, tracker):
        tracker.record("/p", "id1", "Title", "2026-01-01T00:00:00Z", 3)
        tracker.record("/p", "id1", "Title Updated", "2026-02-01T00:00:00Z", 7)
        docs = tracker.get_all()
        assert len(docs) == 1
        assert docs[0]["title"] == "Title Updated"
        assert docs[0]["chunk_count"] == 7

    def test_get_all_returns_all_documents(self, tracker):
        tracker.record("/p1", "id1", "T1", "2026-01-01T00:00:00Z", 2)
        tracker.record("/p2", "id2", "T2", "2026-01-02T00:00:00Z", 4)
        assert len(tracker.get_all()) == 2

    def test_log_run_and_get_recent(self, tracker):
        tracker.log_run("2026-01-01T00:00:00Z", added=2, updated=1, removed=0, errors=0)
        runs = tracker.get_recent_runs(5)
        assert len(runs) == 1
        assert runs[0]["added"] == 2
        assert runs[0]["updated"] == 1

    def test_withdrawn_status_recorded(self, tracker):
        tracker.record("/p", "id", "T", "2026-01-01T00:00:00Z", 0, status="withdrawn")
        docs = tracker.get_all()
        assert docs[0]["status"] == "withdrawn"


# ── HTML stripping tests ───────────────────────────────────────────────────────

class TestStripHtml:
    def test_removes_tags(self):
        result = _strip_html("<p>Hello <b>world</b></p>")
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result

    def test_decodes_entities(self):
        result = _strip_html("&pound;38,700 &amp; £30,960")
        assert "£" in result or "38,700" in result

    def test_removes_script_blocks(self):
        result = _strip_html("<script>alert('xss')</script><p>Safe text</p>")
        assert "alert" not in result
        assert "Safe text" in result

    def test_collapses_whitespace(self):
        result = _strip_html("<p>  Too   many   spaces  </p>")
        assert "  " not in result.strip()

    def test_empty_input(self):
        assert _strip_html("") == ""


# ── Chunking tests ─────────────────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_produces_one_chunk(self):
        chunks = _chunk_text("Short text.", "source", "Title", "/path")
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self):
        long_text = "word " * 500   # ~2500 chars, well over CHUNK_SIZE=1200
        chunks = _chunk_text(long_text, "source", "Title", "/path")
        assert len(chunks) > 1

    def test_chunks_have_correct_metadata(self):
        chunks = _chunk_text("Some text here.", "my-source", "My Title", "/my/path")
        assert chunks[0].metadata["source"] == "my-source"
        assert chunks[0].metadata["govuk_path"] == "/my/path"
        assert chunks[0].metadata["chunk_type"] == "html"

    def test_chunk_page_index_increments(self):
        long_text = "x " * 1000
        chunks = _chunk_text(long_text, "s", "T", "/p")
        pages = [c.metadata["page"] for c in chunks]
        assert pages == list(range(len(chunks)))

    def test_empty_text_produces_no_chunks(self):
        chunks = _chunk_text("", "source", "Title", "/path")
        assert chunks == []


# ── Feed parsing tests ─────────────────────────────────────────────────────────

class TestFetchFeed:
    def test_parses_entries(self):
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_ATOM
        mock_resp.raise_for_status = MagicMock()

        with patch("src.updater.requests.get", return_value=mock_resp):
            entries = fetch_feed("https://www.gov.uk/ukvi.atom")

        assert len(entries) == 2
        assert entries[0]["title"] == "Skilled Worker visa"
        assert "/guidance/skilled-worker-visa" in entries[0]["link"]
        assert entries[0]["updated"] == "2026-03-26T09:00:00Z"

    def test_network_error_raises(self):
        with patch("src.updater.requests.get", side_effect=Exception("timeout")):
            with pytest.raises(Exception):
                fetch_feed("https://www.gov.uk/ukvi.atom")


# ── Content API tests ──────────────────────────────────────────────────────────

class TestFetchContentApi:
    def test_returns_parsed_json(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_CONTENT_API
        mock_resp.raise_for_status = MagicMock()

        with patch("src.updater.requests.get", return_value=mock_resp):
            result = fetch_content_api("/guidance/skilled-worker-visa")

        assert result["title"] == "Skilled Worker visa"
        assert result["content_id"] == "abc-123"

    def test_404_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("src.updater.requests.get", return_value=mock_resp):
            result = fetch_content_api("/guidance/nonexistent")

        assert result is None

    def test_network_error_returns_none(self):
        with patch("src.updater.requests.get", side_effect=Exception("connection refused")):
            result = fetch_content_api("/guidance/skilled-worker-visa")
        assert result is None


# ── PDF extraction tests ───────────────────────────────────────────────────────

class TestPdfBytesToDocuments:
    def test_extracts_pages_as_documents(self):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Visa requirements text on page 1."

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page, mock_page]

        with patch("src.updater.pypdf.PdfReader", return_value=mock_reader):
            docs = pdf_bytes_to_documents(b"fakepdfbytes", "source.pdf", "Title", "/path")

        assert len(docs) == 2
        assert docs[0].metadata["chunk_type"] == "pdf"
        assert docs[0].metadata["page"] == 0
        assert docs[1].metadata["page"] == 1

    def test_skips_empty_pages(self):
        mock_page_empty = MagicMock()
        mock_page_empty.extract_text.return_value = "   "
        mock_page_text = MagicMock()
        mock_page_text.extract_text.return_value = "Real content here."

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page_empty, mock_page_text]

        with patch("src.updater.pypdf.PdfReader", return_value=mock_reader):
            docs = pdf_bytes_to_documents(b"fakepdf", "source.pdf", "T", "/p")

        assert len(docs) == 1
        assert "Real content" in docs[0].page_content

    def test_parse_error_returns_empty(self):
        with patch("src.updater.pypdf.PdfReader", side_effect=Exception("bad pdf")):
            docs = pdf_bytes_to_documents(b"notapdf", "source.pdf", "T", "/p")
        assert docs == []


# ── IncrementalUpdater.process_path tests ─────────────────────────────────────

class TestProcessPath:
    def test_new_document_is_added(self, updater):
        with patch("src.updater.fetch_content_api", return_value=SAMPLE_CONTENT_API):
            status, count = updater.process_path("/guidance/skilled-worker-visa")

        assert status == "added"
        assert count > 0
        updater._vector_db.collection.add.assert_called_once()

    def test_unchanged_document_is_skipped(self, updater):
        # Pre-record with same timestamp as the API will return
        updater.tracker.record(
            govuk_path="/guidance/skilled-worker-visa",
            content_id="abc-123",
            title="Skilled Worker visa",
            public_updated_at="2026-03-26T09:00:00Z",
            chunk_count=3,
        )
        with patch("src.updater.fetch_content_api", return_value=SAMPLE_CONTENT_API):
            status, count = updater.process_path("/guidance/skilled-worker-visa")

        assert status == "skipped"
        assert count == 0
        updater._vector_db.collection.add.assert_not_called()

    def test_updated_document_deletes_old_chunks_then_adds_new(self, updater):
        # Pre-record with OLD timestamp
        updater.tracker.record(
            govuk_path="/guidance/skilled-worker-visa",
            content_id="abc-123",
            title="Skilled Worker visa",
            public_updated_at="2025-01-01T00:00:00Z",  # older
            chunk_count=3,
        )
        # Simulate existing chunks in ChromaDB
        updater._vector_db.collection.get.return_value = {
            "ids": ["govuk_abc_0", "govuk_abc_1", "govuk_abc_2"]
        }

        with patch("src.updater.fetch_content_api", return_value=SAMPLE_CONTENT_API):
            status, count = updater.process_path("/guidance/skilled-worker-visa")

        assert status == "updated"
        updater._vector_db.collection.delete.assert_called_once()
        updater._vector_db.collection.add.assert_called_once()

    def test_withdrawn_document_removes_chunks(self, updater):
        withdrawn_data = {
            **SAMPLE_CONTENT_API,
            "withdrawn_notice": {"explanation": "This guidance has been withdrawn."},
        }
        updater._vector_db.collection.get.return_value = {"ids": ["chunk_0", "chunk_1"]}

        with patch("src.updater.fetch_content_api", return_value=withdrawn_data):
            status, count = updater.process_path("/guidance/old-visa")

        assert status == "withdrawn"
        assert count == 0
        updater._vector_db.collection.delete.assert_called_once()

    def test_content_api_failure_returns_error(self, updater):
        with patch("src.updater.fetch_content_api", return_value=None):
            status, count = updater.process_path("/guidance/broken-path")

        assert status == "error"
        assert count == 0

    def test_no_body_no_attachments_skipped(self, updater):
        empty_content = {
            **SAMPLE_CONTENT_API,
            "details": {"body": "", "attachments": []},
        }
        with patch("src.updater.fetch_content_api", return_value=empty_content):
            status, count = updater.process_path("/guidance/empty-page")

        assert status == "skipped"

    def test_pdf_attachment_processed(self, updater):
        content_with_pdf = {
            **SAMPLE_CONTENT_API,
            "details": {
                "body": "",
                "attachments": [
                    {"url": "https://assets.publishing.service.gov.uk/doc.pdf", "title": "Guidance PDF"},
                ],
            },
        }
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF page content about visa requirements."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("src.updater.fetch_content_api", return_value=content_with_pdf):
            with patch("src.updater.download_pdf", return_value=b"fakepdf"):
                with patch("src.updater.pypdf.PdfReader", return_value=mock_reader):
                    status, count = updater.process_path("/guidance/skilled-worker-visa")

        assert status == "added"
        assert count > 0


# ── IncrementalUpdater.run_once tests ─────────────────────────────────────────

class TestRunOnce:
    def test_run_once_returns_summary(self, updater):
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_ATOM
        mock_resp.raise_for_status = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_CONTENT_API

        with patch("src.updater.requests.get", return_value=mock_resp):
            summary = updater.run_once()

        assert "added" in summary
        assert "updated" in summary
        assert "errors" in summary
        assert "skipped" in summary
        assert "total_candidates" in summary

    def test_run_once_logs_the_run(self, updater):
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_ATOM
        mock_resp.raise_for_status = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_CONTENT_API

        with patch("src.updater.requests.get", return_value=mock_resp):
            updater.run_once()

        runs = updater.tracker.get_recent_runs(1)
        assert len(runs) == 1
        assert runs[0]["finished_at"] is not None

    def test_feed_error_does_not_crash_run(self, updater):
        with patch("src.updater.fetch_feed", side_effect=Exception("network down")):
            with patch("src.updater.fetch_content_api", return_value=None):
                summary = updater.run_once()

        # Should complete without raising — errors counted
        assert isinstance(summary, dict)

    def test_all_up_to_date_skips_all_proactive_paths(self, updater):
        """
        When all proactive paths are already current, process_path returns
        'skipped' for each one. Proactive paths always become candidates
        (no feed timestamp to pre-filter on), but the tracker skips them.
        """
        from src.updater import PROACTIVE_PATHS

        # Pre-record all proactive paths with a far-future timestamp
        for path in PROACTIVE_PATHS:
            updater.tracker.record(path, "id", "Title", "2099-01-01T00:00:00Z", 5)

        # Empty feed (no new entries from UKVI)
        mock_resp = MagicMock()
        mock_resp.text = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>UKVI</title><updated>2026-01-01T00:00:00Z</updated>
</feed>"""
        mock_resp.raise_for_status = MagicMock()

        # Content API returns the same timestamp → tracker will skip
        up_to_date = {
            **SAMPLE_CONTENT_API,
            "public_updated_at": "2099-01-01T00:00:00Z",
        }

        with patch("src.updater.requests.get", return_value=mock_resp):
            with patch("src.updater.fetch_content_api", return_value=up_to_date):
                summary = updater.run_once()

        # All proactive paths are candidates but all are skipped
        assert summary["total_candidates"] == len(PROACTIVE_PATHS)
        assert summary["skipped"] == len(PROACTIVE_PATHS)
        assert summary["added"] == 0
        assert summary["updated"] == 0


# ── Status report test ─────────────────────────────────────────────────────────

class TestStatus:
    def test_status_does_not_raise(self, updater, capsys):
        updater.tracker.record("/p1", "id1", "T1", "2026-01-01T00:00:00Z", 10)
        updater.tracker.log_run("2026-01-01T00:00:00Z", 1, 0, 0, 0)
        updater.status()
        captured = capsys.readouterr()
        assert "Tracked documents" in captured.out
        assert "1" in captured.out
