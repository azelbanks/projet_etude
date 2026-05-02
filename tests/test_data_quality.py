"""
Tests for collection/data_quality_check.py — quality report logic.

Uses mock objects to simulate MongoDB without requiring a running instance.
"""

import pytest
from unittest.mock import MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from collection.data_quality_check import run_quality_check, print_report


# -----------------------------------------------------------------------
#  Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def mock_collection():
    return MagicMock()


# -----------------------------------------------------------------------
#  run_quality_check
# -----------------------------------------------------------------------

class TestRunQualityCheck:
    """Tests for the run_quality_check() function."""

    def test_empty_collection(self, mock_collection):
        mock_collection.count_documents.return_value = 0

        report = run_quality_check(mock_collection)

        assert report["total_documents"] == 0

    def test_returns_total_documents(self, mock_collection):
        # count_documents is called multiple times: first for total, then for each required field
        mock_collection.count_documents.side_effect = [5000, 0, 0, 0, 0]
        mock_collection.aggregate.return_value = []

        report = run_quality_check(mock_collection)

        assert report["total_documents"] == 5000

    def test_detects_missing_fields(self, mock_collection):
        # total=1000, text_missing=5, uri_missing=0, collected_at_missing=2, empty_text=0
        mock_collection.count_documents.side_effect = [1000, 5, 0, 2, 0]
        mock_collection.aggregate.return_value = []

        report = run_quality_check(mock_collection)

        assert report["missing_required_fields"]["text"] == 5
        assert report["missing_required_fields"]["collected_at"] == 2

    def test_no_missing_fields(self, mock_collection):
        mock_collection.count_documents.side_effect = [1000, 0, 0, 0, 0]
        mock_collection.aggregate.return_value = []

        report = run_quality_check(mock_collection)

        assert report["missing_required_fields"] == "none"

    def test_duplicate_detection(self, mock_collection):
        mock_collection.count_documents.side_effect = [100, 0, 0, 0, 0]
        # First two aggregates are lang and label, third is duplicate check
        mock_collection.aggregate.side_effect = [
            [],  # lang
            [],  # label
            [{"duplicate_uri_groups": 3}],  # duplicates
        ]

        report = run_quality_check(mock_collection)

        assert report["duplicate_uri_groups"] == 3


# -----------------------------------------------------------------------
#  print_report (smoke test)
# -----------------------------------------------------------------------

class TestPrintReport:
    """Smoke tests for print_report() — verifies it doesn't crash."""

    def test_prints_empty_report(self, capsys):
        print_report({"total_documents": 0})
        captured = capsys.readouterr()
        assert "DATA QUALITY REPORT" in captured.out

    def test_prints_full_report(self, capsys):
        report = {
            "total_documents": 1000,
            "missing_required_fields": "none",
            "empty_text_count": 0,
            "documents_by_ai_language": {"fr": 600, "en": 400},
            "documents_by_prediction_label": {"FIABLE": 800, "SUSPECT": 200},
            "duplicate_uri_groups": 0,
        }
        print_report(report)
        captured = capsys.readouterr()
        assert "1000" in captured.out
        assert "fr" in captured.out
