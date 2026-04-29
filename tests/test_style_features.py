"""
Tests for StyleFeatureExtractorV6 from dashboard/app.py.

Validates feature count, sensationalism detection, source citation
detection, and numeric integrity of the extracted features.
"""

import numpy as np
import pandas as pd
import pytest

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
