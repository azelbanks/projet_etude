"""
Tests etendus pour collection/ — fonctions utilitaires et mocking.
"""

import os
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from collection.collect_bluesky import (
    validate_text,
    compute_word_count,
    detect_language_hint,
    extract_metadata,
)


# ============================================================
#  validate_text — returns (bool, str)
# ============================================================


class TestValidateText:
    def test_valid_text(self):
        ok, result = validate_text("Hello world this is a normal post")
        assert ok is True

    def test_empty_text(self):
        ok, reason = validate_text("")
        assert ok is False

    def test_none_text(self):
        ok, reason = validate_text(None)
        assert ok is False

    def test_too_short_text(self):
        ok, reason = validate_text("ab")
        assert ok is False
        assert 'short' in reason

    def test_url_only_text(self):
        ok, reason = validate_text("https://example.com")
        assert ok is False
        assert 'url' in reason


class TestComputeWordCount:
    def test_normal_text(self):
        assert compute_word_count("hello world foo bar") == 4

    def test_empty_text(self):
        assert compute_word_count("") == 0

    def test_text_with_url(self):
        count = compute_word_count("hello https://example.com world")
        assert count == 2  # URL stripped


class TestDetectLanguageHint:
    def test_french_text(self):
        lang = detect_language_hint("le president de la republique est dans les rues")
        assert lang == 'fr'

    def test_english_text(self):
        lang = detect_language_hint("the president announced new policy measures today")
        assert lang == 'en'

    def test_empty_text(self):
        lang = detect_language_hint("")
        assert lang == 'en'  # default


class TestExtractMetadata:
    def test_extract_basic_post_no_embed(self):
        mock_post = MagicMock()
        mock_post.embed = None
        mock_post.record.langs = ['fr']

        has_image, image_url, langs = extract_metadata(mock_post)
        assert has_image is False
        assert image_url is None
        assert langs == ['fr']

    def test_extract_post_with_image(self):
        mock_post = MagicMock()
        mock_post.embed.images = [MagicMock(fullsize="https://img.example.com/1.jpg")]
        mock_post.record.langs = ['en']

        has_image, image_url, langs = extract_metadata(mock_post)
        assert has_image is True
        assert image_url == "https://img.example.com/1.jpg"


# ============================================================
#  PipelineMonitor
# ============================================================


class TestPipelineMonitorExtended:
    def test_monitor_init(self):
        from collection.pipeline_monitor import PipelineMonitor
        monitor = PipelineMonitor()
        assert monitor is not None
        assert monitor.posts_new == 0

    def test_record_keyword(self):
        from collection.pipeline_monitor import PipelineMonitor
        monitor = PipelineMonitor()
        monitor.start_cycle()
        monitor.record_keyword("test", "en", added=10, duplicates=2)
        assert monitor.posts_new == 10
        assert monitor.duplicates_skipped == 2
        assert monitor.keywords_processed == 1

    def test_record_keyword_with_error(self):
        from collection.pipeline_monitor import PipelineMonitor
        monitor = PipelineMonitor()
        monitor.start_cycle()
        monitor.record_keyword("fail", "fr", errors=1, error_msg="rate limit")
        assert monitor.errors == 1
        assert len(monitor.error_details) == 1

    def test_end_cycle(self):
        from collection.pipeline_monitor import PipelineMonitor
        monitor = PipelineMonitor()
        monitor.start_cycle()
        monitor.record_keyword("kw1", "en", added=5, duplicates=1)
        monitor.record_keyword("kw2", "fr", added=3, duplicates=0)
        summary = monitor.end_cycle()
        assert summary is not None
        assert monitor.posts_new == 8


# ============================================================
#  setup_indexes
# ============================================================


class TestSetupIndexes:
    @patch('collection.setup_indexes.MongoClient')
    def test_connect_db_no_auth(self, mock_client_cls):
        from collection.setup_indexes import connect_db
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        with patch.dict(os.environ, {}, clear=True):
            db = connect_db()
            assert db is not None

    @patch('collection.setup_indexes.MongoClient')
    def test_connect_db_with_auth(self, mock_client_cls):
        from collection.setup_indexes import connect_db
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        with patch.dict(os.environ, {
            'MONGO_USER': 'admin', 'MONGO_PASSWORD': 'secret', 'MONGO_HOST': 'db.host'
        }):
            db = connect_db()
            assert db is not None

    def test_setup_indexes_creates_all(self):
        from collection.setup_indexes import setup_indexes
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.create_index.return_value = "idx_test"

        setup_indexes(mock_db)
        # Should create 7 indexes
        assert mock_collection.create_index.call_count == 7

    def test_setup_schema_validation(self):
        from collection.setup_indexes import setup_schema_validation
        mock_db = MagicMock()
        setup_schema_validation(mock_db)
        mock_db.command.assert_called_once()
