"""
Tests for monitoring/weekly_score_check.py — scoring logic and report writing.

Uses mock objects to avoid requiring MongoDB or trained models.
Focuses on the data transformation and output format correctness.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from monitoring.weekly_score_check import run_scoring, write_report


# -----------------------------------------------------------------------
#  Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """A small DataFrame simulating MongoDB posts."""
    return pd.DataFrame({
        'text': [
            'Breaking news: scientists discover new planet',
            'SCANDALE ! Le gouvernement cache la verite !',
            'A nice day for a walk in the park.',
            'EXPOSED: they lied about everything!!!',
            'La meteo est belle aujourd hui.',
        ]
    })


@pytest.fixture
def mock_detector(sample_df):
    """A mock detector that returns plausible predictions."""
    from unittest.mock import MagicMock

    detector = MagicMock()
    n = len(sample_df)

    detector.predict.return_value = pd.DataFrame({
        'prediction_label': [0, 1, 0, 1, 0],
        'ai_score_credibility': [0.85, 0.30, 0.90, 0.25, 0.88],
        'language': ['en', 'fr', 'en', 'en', 'fr'],
        'ai_analysis_log': ['ok'] * n,
    })
    return detector


# -----------------------------------------------------------------------
#  run_scoring
# -----------------------------------------------------------------------

class TestRunScoring:
    """Tests for the run_scoring() function."""

    def test_returns_all_required_keys(self, sample_df, mock_detector):
        report = run_scoring(sample_df, mock_detector)

        required_keys = [
            'timestamp', 'n_sampled', 'n_suspect', 'suspect_rate',
            'mean_credibility', 'std_credibility', 'score_percentiles',
            'language_breakdown', 'avg_text_length',
        ]
        for key in required_keys:
            assert key in report, f"Missing key: {key}"

    def test_n_sampled_matches_input(self, sample_df, mock_detector):
        report = run_scoring(sample_df, mock_detector)
        assert report['n_sampled'] == len(sample_df)

    def test_suspect_count_correct(self, sample_df, mock_detector):
        report = run_scoring(sample_df, mock_detector)
        assert report['n_suspect'] == 2  # labels [0, 1, 0, 1, 0]

    def test_suspect_rate_in_range(self, sample_df, mock_detector):
        report = run_scoring(sample_df, mock_detector)
        assert 0.0 <= report['suspect_rate'] <= 1.0

    def test_mean_credibility_reasonable(self, sample_df, mock_detector):
        report = run_scoring(sample_df, mock_detector)
        assert 0.0 <= report['mean_credibility'] <= 1.0

    def test_percentiles_ordered(self, sample_df, mock_detector):
        report = run_scoring(sample_df, mock_detector)
        p = report['score_percentiles']
        assert p['p10'] <= p['p25'] <= p['p50'] <= p['p75'] <= p['p90']

    def test_language_breakdown_sums_to_100(self, sample_df, mock_detector):
        report = run_scoring(sample_df, mock_detector)
        lb = report['language_breakdown']
        total = lb['pct_fr'] + lb['pct_en'] + lb.get('pct_other', 0)
        assert abs(total - 100.0) < 0.1

    def test_avg_text_length_positive(self, sample_df, mock_detector):
        report = run_scoring(sample_df, mock_detector)
        assert report['avg_text_length'] > 0


# -----------------------------------------------------------------------
#  write_report
# -----------------------------------------------------------------------

class TestWriteReport:
    """Tests for the write_report() function."""

    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'logs', 'test.jsonl')
            write_report({"key": "value"}, path)
            assert os.path.exists(path)

    def test_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.jsonl')
            report = {"n_suspect": 5, "suspect_rate": 0.05}
            write_report(report, path)

            with open(path) as f:
                line = f.readline()
            parsed = json.loads(line)
            assert parsed["n_suspect"] == 5

    def test_appends_multiple_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.jsonl')
            write_report({"run": 1}, path)
            write_report({"run": 2}, path)

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2
