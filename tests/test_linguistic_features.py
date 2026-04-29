"""
Tests for LinguisticFeatureExtractor from expert_detector.py.

Validates that linguistic feature extraction produces correct shapes,
sensible values, and no missing data.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.expert_detector import LinguisticFeatureExtractor


class TestLinguisticFeatureExtractor:
    """Tests for LinguisticFeatureExtractor.extract()."""

    def test_extract_returns_correct_shape(self, sample_texts):
        """extract() on 5 texts should return a (5, 15) array."""
        result = LinguisticFeatureExtractor.extract(sample_texts)
        assert result.shape == (5, len(LinguisticFeatureExtractor.FEATURE_NAMES))
        assert result.shape == (5, 15)

    def test_caps_ratio_high_for_all_caps(self):
        """A fully capitalised text should have caps_ratio > 0.5."""
        texts = pd.Series(['THIS IS ALL CAPS'])
        result = LinguisticFeatureExtractor.extract(texts)
        # caps_ratio is at index 1 in FEATURE_NAMES
        caps_idx = LinguisticFeatureExtractor.FEATURE_NAMES.index('caps_ratio')
        assert result[0, caps_idx] > 0.5, (
            f"caps_ratio should be > 0.5 for all-caps text, got {result[0, caps_idx]}"
        )

    def test_exclamation_count(self):
        """A text with '!!!' should have exclamation_count > 0."""
        texts = pd.Series(['This is outrageous!!!'])
        result = LinguisticFeatureExtractor.extract(texts)
        excl_idx = LinguisticFeatureExtractor.FEATURE_NAMES.index('exclamation_count')
        assert result[0, excl_idx] > 0, (
            f"exclamation_count should be > 0, got {result[0, excl_idx]}"
        )

    def test_no_nan_values(self, sample_texts):
        """Output should contain no NaN values."""
        result = LinguisticFeatureExtractor.extract(sample_texts)
        assert not np.isnan(result).any(), "Feature matrix contains NaN values"

    def test_word_count_positive(self, sample_texts):
        """word_count should be > 0 for all non-empty texts."""
        result = LinguisticFeatureExtractor.extract(sample_texts)
        wc_idx = LinguisticFeatureExtractor.FEATURE_NAMES.index('word_count')
        for i in range(len(sample_texts)):
            assert result[i, wc_idx] > 0, (
                f"word_count for text {i} should be > 0, got {result[i, wc_idx]}"
            )
