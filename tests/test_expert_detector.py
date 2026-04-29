"""
Tests for ExpertFakeNewsDetector from expert_detector.py.

Validates model loading, prediction output format, label correctness,
and credibility score ranges.  Tests that require a trained model are
skipped when the model files are not present on disk.
"""

import os
import numpy as np
import pandas as pd
import pytest

from pipeline.expert_detector import ExpertFakeNewsDetector

_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Use the latest model suffix that matches current feature count (15 linguistic features).
# model_expert_v5 expects 30015 features (30000 TF-IDF + 15 linguistic).
_MODEL_SUFFIX = 'expert_v5'
_MODEL_EXISTS = os.path.exists(os.path.join(_MODEL_DIR, f'model_{_MODEL_SUFFIX}.pkl'))


@pytest.fixture(scope='module')
def detector():
    """Load the ExpertFakeNewsDetector once for all tests in this module."""
    det = ExpertFakeNewsDetector(model_dir=_MODEL_DIR)
    det.load(suffix=_MODEL_SUFFIX)
    return det


@pytest.mark.skipif(not _MODEL_EXISTS, reason="Model files not found in models/")
class TestExpertFakeNewsDetector:
    """Tests for ExpertFakeNewsDetector (require trained model on disk)."""

    def test_model_loads_successfully(self, detector):
        """After load(), is_trained should be True."""
        assert detector.is_trained is True

    def test_predict_returns_dataframe(self, detector, sample_texts):
        """predict() should return a DataFrame with the required columns."""
        result = detector.predict(sample_texts)
        assert isinstance(result, pd.DataFrame)
        for col in ['language', 'prediction_label', 'ai_score_credibility', 'ai_analysis_log']:
            assert col in result.columns, f"Missing column: {col}"
        assert len(result) == len(sample_texts)

    def test_predict_suspect_text_flagged(self, detector):
        """An obviously suspect text should be predicted as label=1 (suspect)."""
        suspect = pd.Series([
            'EXPOSED: Secret government labs are using 5G towers to spread mind-control chemicals!!!'
        ])
        result = detector.predict(suspect)
        assert result['prediction_label'].iloc[0] == 1, (
            "Obviously suspect text should be flagged as 1 (suspect)"
        )

    def test_predict_fiable_text_passes(self, detector):
        """An obviously reliable text should be predicted as label=0 (fiable)."""
        fiable = pd.Series([
            'New study published in Nature confirms the effectiveness of the updated vaccine formula.'
        ])
        result = detector.predict(fiable)
        assert result['prediction_label'].iloc[0] == 0, (
            "Obviously reliable text should be classified as 0 (fiable)"
        )

    def test_credibility_score_range(self, detector, sample_texts):
        """ai_score_credibility should be between 0 and 1 for all predictions."""
        result = detector.predict(sample_texts)
        scores = result['ai_score_credibility'].values
        assert np.all(scores >= 0.0), f"Found score < 0: {scores.min()}"
        assert np.all(scores <= 1.0), f"Found score > 1: {scores.max()}"
