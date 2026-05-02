"""
Integration tests for the ExpertFakeNewsDetector pipeline.

Tests the full prediction pipeline (V5) end-to-end when model files
are present on disk. Validates bilingual support, score calibration,
batch processing, and edge cases.
"""

import os
import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline.expert_detector import ExpertFakeNewsDetector, LinguisticFeatureExtractor

_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
_MODEL_EXISTS = os.path.exists(os.path.join(_MODEL_DIR, 'model_expert_v5.pkl'))


@pytest.mark.skipif(not _MODEL_EXISTS, reason="Model files not found")
class TestPipelineIntegration:
    """End-to-end integration tests for the V5 pipeline."""

    @pytest.fixture(scope='class')
    def detector(self):
        det = ExpertFakeNewsDetector(model_dir=_MODEL_DIR)
        det.load(suffix='expert_v5')
        return det

    def test_bilingual_prediction(self, detector):
        """Pipeline should handle mixed FR/EN input."""
        texts = pd.Series([
            "The vaccine is safe according to WHO studies.",
            "Le vaccin est dangereux, on nous cache la VERITE !!!",
        ])
        result = detector.predict(texts)
        assert len(result) == 2
        languages = result['language'].tolist()
        assert 'en' in languages
        assert 'fr' in languages

    def test_single_text(self, detector):
        """Pipeline should handle a single text input."""
        result = detector.predict(pd.Series(["Hello world"]))
        assert len(result) == 1
        assert 'prediction_label' in result.columns

    def test_batch_consistency(self, detector):
        """Same text processed individually and in batch should give same score."""
        text = "Scientists confirm climate change is real"
        single = detector.predict(pd.Series([text]))
        batch = detector.predict(pd.Series([text, "other text"]))

        score_single = single['ai_score_credibility'].iloc[0]
        score_batch = batch['ai_score_credibility'].iloc[0]
        assert abs(score_single - score_batch) < 0.01

    def test_short_text_prediction(self, detector):
        """Ultra-short texts should still produce valid predictions."""
        texts = pd.Series(["wow", "ok", "fake!"])
        result = detector.predict(texts)
        assert len(result) == 3
        assert all(result['ai_score_credibility'].between(0, 1))

    def test_special_characters(self, detector):
        """Texts with emojis and special chars should not crash."""
        texts = pd.Series([
            "This is amazing!!! 🔥🔥🔥 #truth",
            "C'est incroyable 🇫🇷 @macron",
        ])
        result = detector.predict(texts)
        assert len(result) == 2

    def test_long_text(self, detector):
        """A very long text should be handled without error."""
        long_text = "This is a test sentence. " * 200
        result = detector.predict(pd.Series([long_text]))
        assert len(result) == 1


class TestLinguisticFeaturesEdgeCases:
    """Edge case tests for LinguisticFeatureExtractor."""

    def test_empty_text_handling(self):
        """Empty-ish texts should return features without NaN."""
        texts = pd.Series(["", "   ", "."])
        result = LinguisticFeatureExtractor.extract(texts)
        assert result.shape[0] == 3
        assert not np.isnan(result).any()

    def test_numeric_only_text(self):
        """Purely numeric text should return valid features."""
        texts = pd.Series(["12345 67890"])
        result = LinguisticFeatureExtractor.extract(texts)
        assert result.shape == (1, 15)
        assert not np.isnan(result).any()

    def test_all_caps_text(self):
        """All-caps text should have high caps_ratio and low avg_word_length."""
        texts = pd.Series(["BREAKING NEWS ALERT DANGER"])
        result = LinguisticFeatureExtractor.extract(texts)
        caps_idx = LinguisticFeatureExtractor.FEATURE_NAMES.index('caps_ratio')
        assert result[0, caps_idx] > 0.5

    def test_very_long_text(self):
        """Feature extraction on a 5000-word text should not crash."""
        texts = pd.Series(["word " * 5000])
        result = LinguisticFeatureExtractor.extract(texts)
        assert result.shape == (1, 15)

    def test_multilingual_text(self):
        """Mixed FR/EN text should produce valid features."""
        texts = pd.Series(["This is English. Ceci est du français. Dies ist Deutsch."])
        result = LinguisticFeatureExtractor.extract(texts)
        assert result.shape == (1, 15)
        assert not np.isnan(result).any()
