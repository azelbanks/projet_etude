"""
Tests for pipeline/mongo_aggregations.py — MongoDB aggregation helpers.

Uses unittest.mock to simulate MongoDB responses without requiring
a running database. Tests the data transformation logic, not MongoDB itself.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pipeline.mongo_aggregations import get_overview_stats, get_recent_posts, get_score_distribution


# -----------------------------------------------------------------------
#  Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def mock_collection():
    """Return a MagicMock that behaves like a pymongo Collection."""
    return MagicMock()


# -----------------------------------------------------------------------
#  get_overview_stats
# -----------------------------------------------------------------------

class TestGetOverviewStats:
    """Tests for the get_overview_stats() aggregation."""

    def test_returns_all_expected_keys(self, mock_collection):
        mock_collection.aggregate.return_value = [{
            "total": [{"count": 1000}],
            "by_label": [
                {"_id": "FIABLE", "count": 800},
                {"_id": "SUSPECT", "count": 200},
            ],
            "by_emotion": [
                {"_id": "neutre", "count": 500},
                {"_id": "colere", "count": 300},
            ],
            "by_language": [
                {"_id": "fr", "count": 600},
                {"_id": "en", "count": 400},
            ],
            "avg_cred": [{"_id": None, "avg": 0.72}],
        }]

        stats = get_overview_stats(mock_collection)

        assert stats["total_posts"] == 1000
        assert stats["by_label"]["FIABLE"] == 800
        assert stats["by_label"]["SUSPECT"] == 200
        assert stats["by_emotion"]["neutre"] == 500
        assert stats["by_language"]["fr"] == 600
        assert abs(stats["avg_credibility"] - 0.72) < 0.001

    def test_empty_collection(self, mock_collection):
        mock_collection.aggregate.return_value = [{
            "total": [],
            "by_label": [],
            "by_emotion": [],
            "by_language": [],
            "avg_cred": [],
        }]

        stats = get_overview_stats(mock_collection)

        assert stats["total_posts"] == 0
        assert stats["by_label"] == {}
        assert stats["avg_credibility"] is None

    def test_null_ids_excluded(self, mock_collection):
        mock_collection.aggregate.return_value = [{
            "total": [{"count": 100}],
            "by_label": [
                {"_id": "FIABLE", "count": 90},
                {"_id": None, "count": 10},
            ],
            "by_emotion": [],
            "by_language": [],
            "avg_cred": [],
        }]

        stats = get_overview_stats(mock_collection)
        assert None not in stats["by_label"]
        assert "FIABLE" in stats["by_label"]

    def test_aggregation_error_returns_empty(self, mock_collection):
        mock_collection.aggregate.side_effect = Exception("Connection lost")

        stats = get_overview_stats(mock_collection)

        assert stats["total_posts"] == 0
        assert stats["by_label"] == {}
        assert stats["avg_credibility"] is None


# -----------------------------------------------------------------------
#  get_recent_posts
# -----------------------------------------------------------------------

class TestGetRecentPosts:
    """Tests for the get_recent_posts() query."""

    def test_returns_list_of_dicts(self, mock_collection):
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([
            {"text": "Post 1", "ai_emotion": "neutre"},
            {"text": "Post 2", "ai_emotion": "colere"},
        ]))
        mock_collection.find.return_value.sort.return_value.limit.return_value = mock_cursor

        result = get_recent_posts(mock_collection, limit=2)

        assert len(result) == 2
        assert result[0]["text"] == "Post 1"

    def test_query_error_returns_empty_list(self, mock_collection):
        mock_collection.find.side_effect = Exception("Timeout")

        result = get_recent_posts(mock_collection)

        assert result == []

    def test_respects_limit_parameter(self, mock_collection):
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_collection.find.return_value.sort.return_value.limit.return_value = mock_cursor

        get_recent_posts(mock_collection, limit=10)

        mock_collection.find.return_value.sort.return_value.limit.assert_called_with(10)


# -----------------------------------------------------------------------
#  get_score_distribution
# -----------------------------------------------------------------------

class TestGetScoreDistribution:
    """Tests for the get_score_distribution() histogram."""

    def test_returns_list_of_bins(self, mock_collection):
        mock_collection.aggregate.return_value = [
            {"_id": 0.0, "count": 50},
            {"_id": 0.05, "count": 30},
            {"_id": 0.5, "count": 100},
        ]

        result = get_score_distribution(mock_collection, bins=20)

        assert isinstance(result, list)
        assert len(result) == 3
        for item in result:
            assert "bin_start" in item
            assert "bin_end" in item
            assert "count" in item

    def test_other_bucket_excluded(self, mock_collection):
        mock_collection.aggregate.return_value = [
            {"_id": 0.0, "count": 50},
            {"_id": "_other", "count": 5},
        ]

        result = get_score_distribution(mock_collection, bins=20)

        assert len(result) == 1  # "_other" excluded

    def test_aggregation_error_returns_empty(self, mock_collection):
        mock_collection.aggregate.side_effect = Exception("Error")

        result = get_score_distribution(mock_collection)

        assert result == []

    def test_bin_boundaries_are_correct(self, mock_collection):
        mock_collection.aggregate.return_value = [
            {"_id": 0.0, "count": 10},
        ]

        result = get_score_distribution(mock_collection, bins=10)

        assert result[0]["bin_start"] == 0.0
        assert result[0]["bin_end"] == 0.1
