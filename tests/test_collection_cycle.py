"""Tests pour run_collection_cycle et extract_metadata avec mocks."""

import sys
import os
import pytest
import datetime
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from collection.collect_bluesky import extract_metadata, run_collection_cycle


class TestExtractMetadata:
    def test_post_with_image(self):
        post = MagicMock()
        post.embed = MagicMock()
        post.embed.images = [MagicMock(fullsize="https://img.example.com/1.jpg")]
        post.record.langs = ['en']
        has_image, image_url, langs = extract_metadata(post)
        assert has_image is True
        assert image_url == "https://img.example.com/1.jpg"
        assert langs == ['en']

    def test_post_without_image(self):
        post = MagicMock()
        post.embed = None
        post.record.langs = ['fr']
        has_image, image_url, langs = extract_metadata(post)
        assert has_image is False
        assert image_url is None

    def test_post_with_empty_images(self):
        post = MagicMock()
        post.embed = MagicMock()
        post.embed.images = []
        post.record.langs = []
        has_image, image_url, langs = extract_metadata(post)
        assert has_image is True  # has_image is True because embed.images attr exists
        assert image_url is None  # but no image to extract

    def test_post_without_embed_images(self):
        post = MagicMock()
        post.embed = MagicMock(spec=[])  # no 'images' attribute
        post.record.langs = ['en', 'fr']
        has_image, image_url, langs = extract_metadata(post)
        assert has_image is False
        assert langs == ['en', 'fr']


class TestRunCollectionCycle:
    @patch('collection.collect_bluesky.time')
    @patch('collection.collect_bluesky.random')
    @patch('collection.collect_bluesky.SEARCH_CONFIG', {'en': ['test_kw']})
    @patch('collection.collect_bluesky.reload_excluded_handles')
    def test_basic_cycle(self, mock_reload, mock_random, mock_time):
        mock_random.uniform.return_value = 1.0
        mock_time.sleep = MagicMock()

        # Mock collection
        mock_collection = MagicMock()
        mock_bulk_result = MagicMock()
        mock_bulk_result.upserted_count = 3
        mock_bulk_result.modified_count = 1
        mock_collection.bulk_write.return_value = mock_bulk_result

        # Mock bluesky client
        mock_client = MagicMock()
        mock_post = MagicMock()
        mock_post.uri = "at://did:plc:test/app.bsky.feed.post/123"
        mock_post.cid = "bafytest"
        mock_post.record.text = "Test post about climate change"
        mock_post.record.created_at = "2026-05-01T12:00:00Z"
        mock_post.author.did = "did:plc:test"
        mock_post.author.handle = "test.bsky.social"
        mock_post.author.display_name = "Test User"
        mock_post.embed = None
        mock_post.record.langs = ['en']
        mock_post.reply_count = 0
        mock_post.repost_count = 0
        mock_post.like_count = 0
        mock_client.app.bsky.feed.search_posts.return_value = MagicMock(posts=[mock_post])

        run_collection_cycle(mock_collection, mock_client)
        mock_collection.bulk_write.assert_called_once()
        mock_reload.assert_called_once()

    @patch('collection.collect_bluesky.time')
    @patch('collection.collect_bluesky.random')
    @patch('collection.collect_bluesky.SEARCH_CONFIG', {'en': ['test_kw']})
    @patch('collection.collect_bluesky.reload_excluded_handles')
    def test_cycle_with_monitor(self, mock_reload, mock_random, mock_time):
        mock_random.uniform.return_value = 1.0
        mock_time.sleep = MagicMock()

        mock_collection = MagicMock()
        mock_bulk_result = MagicMock()
        mock_bulk_result.upserted_count = 2
        mock_bulk_result.modified_count = 0
        mock_collection.bulk_write.return_value = mock_bulk_result

        mock_client = MagicMock()
        mock_post = MagicMock()
        mock_post.uri = "at://test"
        mock_post.cid = "test"
        mock_post.record.text = "Valid post"
        mock_post.record.created_at = "2026-05-01T12:00:00Z"
        mock_post.author.did = "did:plc:test"
        mock_post.author.handle = "user.bsky.social"
        mock_post.author.display_name = "User"
        mock_post.embed = None
        mock_post.record.langs = ['en']
        mock_post.reply_count = 0
        mock_post.repost_count = 0
        mock_post.like_count = 0
        mock_client.app.bsky.feed.search_posts.return_value = MagicMock(posts=[mock_post])

        mock_monitor = MagicMock()
        run_collection_cycle(mock_collection, mock_client, monitor=mock_monitor)
        mock_monitor.start_cycle.assert_called_once()
        mock_monitor.record_keyword.assert_called()
        mock_monitor.end_cycle.assert_called_once()

    @patch('collection.collect_bluesky.time')
    @patch('collection.collect_bluesky.random')
    @patch('collection.collect_bluesky.SEARCH_CONFIG', {'en': ['error_kw']})
    @patch('collection.collect_bluesky.reload_excluded_handles')
    def test_cycle_handles_api_error(self, mock_reload, mock_random, mock_time):
        mock_random.uniform.return_value = 1.0
        mock_time.sleep = MagicMock()

        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.app.bsky.feed.search_posts.side_effect = Exception("Network error")

        # Should not raise
        run_collection_cycle(mock_collection, mock_client)

    @patch('collection.collect_bluesky.time')
    @patch('collection.collect_bluesky.random')
    @patch('collection.collect_bluesky.SEARCH_CONFIG', {'en': ['rate_kw']})
    @patch('collection.collect_bluesky.reload_excluded_handles')
    def test_cycle_handles_rate_limit(self, mock_reload, mock_random, mock_time):
        mock_random.uniform.return_value = 30.0
        mock_time.sleep = MagicMock()

        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.app.bsky.feed.search_posts.side_effect = Exception("429 Too Many Requests")

        run_collection_cycle(mock_collection, mock_client)
        # Should have slept for rate limit
        assert mock_time.sleep.called

    @patch('collection.collect_bluesky.time')
    @patch('collection.collect_bluesky.random')
    @patch('collection.collect_bluesky.SEARCH_CONFIG', {'en': ['test']})
    @patch('collection.collect_bluesky.EXCLUDED_HANDLES', {'excluded.bsky.social'})
    @patch('collection.collect_bluesky.reload_excluded_handles')
    def test_excluded_handles_skipped(self, mock_reload, mock_random, mock_time):
        mock_random.uniform.return_value = 1.0
        mock_time.sleep = MagicMock()

        mock_collection = MagicMock()
        mock_client = MagicMock()

        mock_post = MagicMock()
        mock_post.author.handle = "excluded.bsky.social"
        mock_post.record.text = "Should be skipped"
        mock_client.app.bsky.feed.search_posts.return_value = MagicMock(posts=[mock_post])

        run_collection_cycle(mock_collection, mock_client)
        # No bulk_write because only post was excluded
        mock_collection.bulk_write.assert_not_called()
