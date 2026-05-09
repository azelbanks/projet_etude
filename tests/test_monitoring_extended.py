"""
Tests etendus pour le monitoring (weekly_score_check).

Couvre les fonctions : run_scoring, write_report, _build_mongo_uri,
fetch_recent_posts, main.
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from monitoring.weekly_score_check import (
    run_scoring,
    write_report,
    _build_mongo_uri,
    fetch_recent_posts,
    main,
    SAMPLE_SIZE,
    SUSPECT_RATE_ALERT_THRESHOLD,
)


class TestBuildMongoUri:
    def test_default_no_auth(self):
        with patch.dict(os.environ, {}, clear=True):
            uri = _build_mongo_uri()
            assert 'localhost' in uri or 'mongodb://' in uri

    def test_with_auth(self):
        with patch.dict(os.environ, {
            'MONGO_USER': 'testuser',
            'MONGO_PASSWORD': 'testpass',
            'MONGO_HOST': 'db.example.com:27017',
        }):
            uri = _build_mongo_uri()
            assert 'testuser' in uri
            assert 'db.example.com' in uri

    def test_special_chars_in_password(self):
        with patch.dict(os.environ, {
            'MONGO_USER': 'user',
            'MONGO_PASSWORD': 'p@ss:w0rd',
            'MONGO_HOST': 'localhost:27017',
        }):
            uri = _build_mongo_uri()
            assert '%40' in uri or 'p@ss' in uri

    def test_custom_host(self):
        with patch.dict(os.environ, {'MONGO_HOST': 'myhost:12345'}, clear=True):
            uri = _build_mongo_uri()
            assert 'myhost:12345' in uri


class TestRunScoring:
    def test_run_scoring_basic(self):
        df = pd.DataFrame({
            'text': [
                'Scientific study published results',
                'SHOCKING SCANDAL exposed!!!',
                'Normal weather report today',
            ],
        })
        mock_detector = MagicMock()
        mock_detector.predict.return_value = pd.DataFrame({
            'prediction_label': [0, 1, 0],
            'ai_score_credibility': [0.85, 0.2, 0.75],
            'language': ['en', 'en', 'fr'],
        })
        result = run_scoring(df, mock_detector)
        assert 'suspect_rate' in result
        assert 'mean_credibility' in result
        assert 'score_percentiles' in result
        assert 'language_breakdown' in result
        assert result['n_sampled'] == 3
        assert result['n_suspect'] == 1
        assert 'timestamp' in result
        assert 'avg_text_length' in result

    def test_run_scoring_all_fiable(self):
        df = pd.DataFrame({'text': ['a', 'b']})
        mock_detector = MagicMock()
        mock_detector.predict.return_value = pd.DataFrame({
            'prediction_label': [0, 0],
            'ai_score_credibility': [0.9, 0.95],
            'language': ['en', 'en'],
        })
        result = run_scoring(df, mock_detector)
        assert result['n_suspect'] == 0
        assert result['suspect_rate'] == 0.0

    def test_percentiles_ordered(self):
        df = pd.DataFrame({'text': ['a', 'b', 'c', 'd', 'e']})
        mock_detector = MagicMock()
        mock_detector.predict.return_value = pd.DataFrame({
            'prediction_label': [0, 0, 1, 0, 1],
            'ai_score_credibility': [0.9, 0.7, 0.2, 0.85, 0.15],
            'language': ['en', 'en', 'fr', 'fr', 'en'],
        })
        result = run_scoring(df, mock_detector)
        p = result['score_percentiles']
        assert p['p10'] <= p['p25'] <= p['p50'] <= p['p75'] <= p['p90']

    def test_language_breakdown_sums(self):
        df = pd.DataFrame({'text': ['a', 'b']})
        mock_detector = MagicMock()
        mock_detector.predict.return_value = pd.DataFrame({
            'prediction_label': [0, 1],
            'ai_score_credibility': [0.8, 0.3],
            'language': ['fr', 'en'],
        })
        result = run_scoring(df, mock_detector)
        lb = result['language_breakdown']
        total = lb['pct_fr'] + lb['pct_en'] + lb['pct_other']
        assert abs(total - 100.0) < 0.1


class TestWriteReport:
    def test_write_creates_file(self, tmp_path):
        path = str(tmp_path / "reports" / "test.jsonl")
        report = {'n_sampled': 100, 'suspect_rate': 0.3}
        write_report(report, path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.loads(f.readline())
            assert data['n_sampled'] == 100

    def test_write_appends(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        write_report({'a': 1}, path)
        write_report({'b': 2}, path)
        with open(path) as f:
            lines = f.readlines()
            assert len(lines) == 2


class TestFetchRecentPosts:
    @patch('monitoring.weekly_score_check.MongoClient')
    def test_fetch_returns_dataframe(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.limit.return_value = [
            {'text': 'hello', 'collected_at': '2026-01-01'},
            {'text': 'world', 'collected_at': '2026-01-02'},
        ]
        mock_client.__getitem__.return_value.__getitem__.return_value.find.return_value = mock_cursor

        df = fetch_recent_posts(n=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    @patch('monitoring.weekly_score_check.MongoClient')
    def test_fetch_empty_raises(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.limit.return_value = []
        mock_client.__getitem__.return_value.__getitem__.return_value.find.return_value = mock_cursor

        with pytest.raises(RuntimeError, match="No posts"):
            fetch_recent_posts(n=10)


class TestMain:
    @patch('monitoring.weekly_score_check.write_report')
    @patch('monitoring.weekly_score_check.run_scoring')
    @patch('monitoring.weekly_score_check.ExpertFakeNewsDetector')
    @patch('monitoring.weekly_score_check.fetch_recent_posts')
    def test_main_normal_run(self, mock_fetch, mock_detector_cls, mock_scoring, mock_write):
        mock_fetch.return_value = pd.DataFrame({'text': ['hello world']})
        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        mock_scoring.return_value = {
            'suspect_rate': 0.1,
            'mean_credibility': 0.8,
            'std_credibility': 0.1,
            'score_percentiles': {'p10': 0.5, 'p25': 0.6, 'p50': 0.7, 'p75': 0.8, 'p90': 0.9},
            'language_breakdown': {'pct_fr': 50.0, 'pct_en': 50.0, 'pct_other': 0.0},
            'avg_text_length': 42.0,
        }

        main()
        mock_fetch.assert_called_once()
        mock_scoring.assert_called_once()
        mock_write.assert_called_once()

    @patch('monitoring.weekly_score_check.write_report')
    @patch('monitoring.weekly_score_check.run_scoring')
    @patch('monitoring.weekly_score_check.ExpertFakeNewsDetector')
    @patch('monitoring.weekly_score_check.fetch_recent_posts')
    def test_main_alert_triggered(self, mock_fetch, mock_detector_cls, mock_scoring, mock_write):
        mock_fetch.return_value = pd.DataFrame({'text': ['test']})
        mock_detector_cls.return_value = MagicMock()
        mock_scoring.return_value = {
            'suspect_rate': 0.50,  # above threshold
            'mean_credibility': 0.4,
            'std_credibility': 0.2,
            'score_percentiles': {'p10': 0.1, 'p25': 0.2, 'p50': 0.4, 'p75': 0.6, 'p90': 0.8},
            'language_breakdown': {'pct_fr': 50.0, 'pct_en': 50.0, 'pct_other': 0.0},
            'avg_text_length': 30.0,
        }

        main()  # should not crash, alert is just logged


class TestConstants:
    def test_sample_size(self):
        assert SAMPLE_SIZE == 1000

    def test_alert_threshold(self):
        assert 0 < SUSPECT_RATE_ALERT_THRESHOLD < 1
