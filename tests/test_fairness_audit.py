"""Tests pour src/monitoring/fairness_audit.py"""

import json
import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detector(scores, labels):
    """Mock detector qui retourne des prédictions fixes."""
    detector = MagicMock()
    results = pd.DataFrame(
        {
            "prediction_label": labels,
            "ai_score_credibility": scores,
        }
    )
    detector.predict.return_value = results
    return detector


# ---------------------------------------------------------------------------
# compute_fairness_metrics
# ---------------------------------------------------------------------------


class TestComputeFairnessMetrics:
    def setup_method(self):
        from monitoring.fairness_audit import compute_fairness_metrics

        self.compute = compute_fairness_metrics

    def test_returns_dict_with_required_keys(self):
        texts = ["text"] * 30
        labels = [0] * 15 + [1] * 15
        languages = ["fr"] * 15 + ["en"] * 15
        lengths = [10] * 30
        detector = _make_detector([0.3] * 15 + [0.8] * 15, [0] * 15 + [1] * 15)

        result = self.compute(texts, labels, languages, lengths, detector)

        assert "timestamp" in result
        assert "n_total" in result
        assert "groups" in result
        assert result["n_total"] == 30

    def test_groups_by_language(self):
        texts = ["text"] * 30
        labels = [0] * 15 + [1] * 15
        languages = ["fr"] * 15 + ["en"] * 15
        lengths = [20] * 30
        detector = _make_detector([0.3] * 15 + [0.8] * 15, [0] * 15 + [1] * 15)

        result = self.compute(texts, labels, languages, lengths, detector)

        assert "lang_fr" in result["groups"]
        assert "lang_en" in result["groups"]
        assert result["groups"]["lang_fr"]["n"] == 15
        assert result["groups"]["lang_en"]["n"] == 15

    def test_groups_by_text_length(self):
        texts = ["text"] * 40
        labels = [0, 1] * 20
        languages = ["fr"] * 40
        # Répartition sur plusieurs buckets de longueur
        lengths = [5] * 10 + [20] * 10 + [50] * 10 + [200] * 10
        preds = [0, 1] * 20
        scores = [0.3, 0.8] * 20
        detector = _make_detector(scores, preds)

        result = self.compute(texts, labels, languages, lengths, detector)

        length_keys = [k for k in result["groups"] if k.startswith("length_")]
        assert len(length_keys) >= 1

    def test_demographic_parity_computed(self):
        texts = ["text"] * 30
        labels = [0] * 15 + [1] * 15
        languages = ["fr"] * 15 + ["en"] * 15
        lengths = [20] * 30
        # FR prédit tout à 0, EN prédit tout à 1
        preds = [0] * 15 + [1] * 15
        scores = [0.2] * 15 + [0.9] * 15
        detector = _make_detector(scores, preds)

        result = self.compute(texts, labels, languages, lengths, detector)

        assert "demographic_parity_diff" in result
        assert result["demographic_parity_diff"] == pytest.approx(1.0, abs=0.01)

    def test_equalized_odds_computed(self):
        texts = ["text"] * 30
        labels = [0] * 15 + [1] * 15
        languages = ["fr"] * 15 + ["en"] * 15
        lengths = [20] * 30
        preds = [0] * 15 + [1] * 15
        scores = [0.2] * 15 + [0.9] * 15
        detector = _make_detector(scores, preds)

        result = self.compute(texts, labels, languages, lengths, detector)

        assert "equalized_odds_tpr_diff" in result
        assert "equalized_odds_fpr_diff" in result

    def test_group_skipped_if_less_than_10(self):
        # Seulement 5 posts EN — doit être ignoré
        texts = ["text"] * 25
        labels = [0] * 20 + [1] * 5
        languages = ["fr"] * 20 + ["en"] * 5
        lengths = [20] * 25
        detector = _make_detector([0.3] * 25, [0] * 25)

        result = self.compute(texts, labels, languages, lengths, detector)

        assert "lang_fr" in result["groups"]
        assert "lang_en" not in result["groups"]

    def test_per_group_metrics_structure(self):
        texts = ["text"] * 30
        labels = [0] * 15 + [1] * 15
        languages = ["fr"] * 15 + ["en"] * 15
        lengths = [20] * 30
        detector = _make_detector([0.5] * 30, [0] * 15 + [1] * 15)

        result = self.compute(texts, labels, languages, lengths, detector)

        for group in result["groups"].values():
            assert "n" in group
            assert "positive_rate" in group
            assert "tpr" in group
            assert "fpr" in group
            assert "mean_score" in group

    def test_tpr_zero_when_no_positive(self):
        texts = ["text"] * 20
        labels = [0] * 20  # aucun positif
        languages = ["fr"] * 20
        lengths = [20] * 20
        detector = _make_detector([0.3] * 20, [0] * 20)

        result = self.compute(texts, labels, languages, lengths, detector)

        if "lang_fr" in result["groups"]:
            assert result["groups"]["lang_fr"]["tpr"] == 0.0


# ---------------------------------------------------------------------------
# write_fairness_report
# ---------------------------------------------------------------------------


class TestWriteFairnessReport:
    def setup_method(self):
        from monitoring.fairness_audit import write_fairness_report

        self.write = write_fairness_report

    def test_creates_file(self, tmp_path):
        report = {"timestamp": "2026-01-01", "n_total": 10, "groups": {}}
        path = str(tmp_path / "fairness.jsonl")
        self.write(report, path=path)
        assert os.path.exists(path)

    def test_writes_valid_json(self, tmp_path):
        report = {"timestamp": "2026-01-01", "n_total": 10, "groups": {}}
        path = str(tmp_path / "fairness.jsonl")
        self.write(report, path=path)
        with open(path) as f:
            data = json.loads(f.readline())
        assert data["n_total"] == 10

    def test_appends_multiple_reports(self, tmp_path):
        path = str(tmp_path / "fairness.jsonl")
        for i in range(3):
            self.write({"run": i}, path=path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 3
