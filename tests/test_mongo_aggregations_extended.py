"""
Tests etendus pour mongo_aggregations.py — branches non couvertes.

Couvre les fonctions avec mocking MongoDB.
"""

from unittest.mock import patch, MagicMock
import pytest

from pipeline.mongo_aggregations import (
    get_mongo_collection,
    get_overview_stats,
    get_recent_posts,
    get_score_distribution,
)


class TestGetMongoCollection:
    @patch('pymongo.MongoClient')
    def test_returns_collection_on_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.server_info.return_value = {'version': '6.0'}
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection

        result = get_mongo_collection()
        assert result is mock_collection

    @patch('pymongo.MongoClient')
    def test_connection_failure_returns_none(self, mock_client_cls):
        mock_client_cls.return_value.server_info.side_effect = Exception("timeout")
        result = get_mongo_collection()
        assert result is None

    @patch.dict('os.environ', {'MONGODB_URI': 'mongodb://custom:27017/'})
    @patch('pymongo.MongoClient')
    def test_uses_env_uri(self, mock_client_cls):
        mock_client_cls.return_value.server_info.side_effect = Exception("fail")
        get_mongo_collection()
        # First call should use the env URI
        call_args = mock_client_cls.call_args_list[0]
        assert 'custom' in call_args[0][0]


class TestGetOverviewStats:
    def test_with_facet_data(self):
        mock_collection = MagicMock()
        mock_collection.aggregate.return_value = iter([{
            'total': [{'count': 1000}],
            'by_label': [
                {'_id': 'FIABLE', 'count': 700},
                {'_id': 'SUSPECT', 'count': 300},
            ],
            'by_emotion': [
                {'_id': 'neutre', 'count': 500},
            ],
            'by_language': [
                {'_id': 'fr', 'count': 600},
                {'_id': 'en', 'count': 400},
            ],
            'avg_cred': [{'avg': 0.72}],
        }])

        result = get_overview_stats(mock_collection)
        assert result['total_posts'] == 1000
        assert result['by_label']['FIABLE'] == 700
        assert result['avg_credibility'] == 0.72

    def test_empty_collection(self):
        mock_collection = MagicMock()
        mock_collection.aggregate.return_value = iter([{
            'total': [],
            'by_label': [],
            'by_emotion': [],
            'by_language': [],
            'avg_cred': [],
        }])

        result = get_overview_stats(mock_collection)
        assert result['total_posts'] == 0
        assert result['avg_credibility'] is None

    def test_aggregation_error(self):
        mock_collection = MagicMock()
        mock_collection.aggregate.side_effect = Exception("aggregation error")

        result = get_overview_stats(mock_collection)
        assert result['total_posts'] == 0


class TestGetRecentPosts:
    def test_returns_list(self):
        mock_collection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.limit.return_value = iter([
            {'text': 'hello', 'ai_score_credibility': 0.8},
            {'text': 'world', 'ai_score_credibility': 0.3},
        ])
        mock_collection.find.return_value = mock_cursor

        result = get_recent_posts(mock_collection, limit=10)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_query_error_returns_empty(self):
        mock_collection = MagicMock()
        mock_collection.find.side_effect = Exception("query error")

        result = get_recent_posts(mock_collection, limit=10)
        assert result == []


class TestGetScoreDistribution:
    def test_with_bucket_data(self):
        mock_collection = MagicMock()
        mock_collection.aggregate.return_value = iter([
            {'_id': 0.0, 'count': 10},
            {'_id': 0.5, 'count': 50},
        ])

        result = get_score_distribution(mock_collection, bins=20)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]['bin_start'] == 0.0
        assert result[1]['bin_start'] == 0.5

    def test_aggregation_error(self):
        mock_collection = MagicMock()
        mock_collection.aggregate.side_effect = Exception("bucket error")

        result = get_score_distribution(mock_collection, bins=20)
        assert result == []

    def test_other_bucket_skipped(self):
        mock_collection = MagicMock()
        mock_collection.aggregate.return_value = iter([
            {'_id': 0.0, 'count': 10},
            {'_id': '_other', 'count': 5},
        ])

        result = get_score_distribution(mock_collection, bins=20)
        assert len(result) == 1  # _other is excluded
