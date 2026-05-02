"""
Tests for StyleFeatureExtractorV6 from dashboard/app.py.

Validates feature count, sensationalism detection, source citation
detection, and numeric integrity of the extracted features.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dashboard'))
from app import StyleFeatureExtractorV6


class TestStyleFeatureExtractorV6:
    """Tests for StyleFeatureExtractorV6.extract()."""

    def test_extract_returns_correct_length(self):
        """extract() should return 28 features per text."""
        texts = pd.Series(['This is a simple test sentence for feature extraction.'])
        result = StyleFeatureExtractorV6.extract(texts)
        assert result.shape[1] == 28, (
            f"Expected 28 features, got {result.shape[1]}"
        )

    def test_sensationalism_score_high_for_suspect(self):
        """Text with BREAKING/EXPOSED keywords should have sensationalism_score > 0."""
        texts = pd.Series([
            'BREAKING: EXPOSED scandal reveals shocking truth about the conspiracy!!!'
        ])
        result = StyleFeatureExtractorV6.extract(texts)
        sens_idx = StyleFeatureExtractorV6.FEATURE_NAMES.index('sensationalism_score')
        assert result[0, sens_idx] > 0, (
            f"sensationalism_score should be > 0 for suspect text, got {result[0, sens_idx]}"
        )

    def test_has_source_citation(self):
        """Text with 'according to' should have has_source_citation > 0."""
        texts = pd.Series([
            'According to Reuters, the new policy will take effect next month.'
        ])
        result = StyleFeatureExtractorV6.extract(texts)
        src_idx = StyleFeatureExtractorV6.FEATURE_NAMES.index('has_source_citation')
        assert result[0, src_idx] > 0, (
            f"has_source_citation should be > 0, got {result[0, src_idx]}"
        )

    def test_all_features_numeric(self, sample_texts):
        """All extracted features should be finite numeric values (no None/NaN)."""
        result = StyleFeatureExtractorV6.extract(sample_texts)
        assert not np.isnan(result).any(), "Feature matrix contains NaN values"
        assert np.isfinite(result).all(), "Feature matrix contains non-finite values"

    def test_suspect_vs_fiable_features_differ(self):
        """Suspect and fiable texts should produce measurably different features."""
        suspect = pd.Series(["BREAKING: EXPOSED!!! Government LIES about VACCINES!!! Share NOW!!!"])
        fiable = pd.Series(["The central bank published its quarterly economic report today."])
        feat_s = StyleFeatureExtractorV6.extract(suspect)
        feat_f = StyleFeatureExtractorV6.extract(fiable)
        # Suspect should have higher sensationalism, more caps, more punctuation
        sens_idx = StyleFeatureExtractorV6.FEATURE_NAMES.index('sensationalism_score')
        caps_idx = StyleFeatureExtractorV6.FEATURE_NAMES.index('all_caps_words_ratio')
        assert feat_s[0, sens_idx] > feat_f[0, sens_idx], "Suspect should have higher sensationalism"
        assert feat_s[0, caps_idx] > feat_f[0, caps_idx], "Suspect should have more caps"

    def test_empty_text_handled(self):
        """Empty text should produce features without crashing."""
        texts = pd.Series(["", "   "])
        result = StyleFeatureExtractorV6.extract(texts)
        assert result.shape == (2, 28)
        assert np.isfinite(result).all()

    def test_feature_names_match_output(self):
        """FEATURE_NAMES list should match the number of extracted features."""
        texts = pd.Series(["Test text"])
        result = StyleFeatureExtractorV6.extract(texts)
        assert len(StyleFeatureExtractorV6.FEATURE_NAMES) == result.shape[1]
