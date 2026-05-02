"""Tests pour ExpertFakeNewsDetector — init, save/load, health_check, predict."""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pipeline.expert_detector import (
    ExpertFakeNewsDetector, LinguisticFeatureExtractor, LanguageRouter,
    DatasetCleaner,
)


class TestExpertFakeNewsDetectorInit:
    def test_default_threshold(self):
        d = ExpertFakeNewsDetector(model_dir='/nonexistent', threshold=0.44)
        assert d.threshold == 0.44
        assert d.is_trained is False
        assert d.model is None

    def test_custom_thresholds(self):
        d = ExpertFakeNewsDetector(
            model_dir='/nonexistent',
            threshold=0.50,
            threshold_fr=0.42,
            threshold_en=0.48,
        )
        assert d.threshold_fr == 0.42
        assert d.threshold_en == 0.48

    def test_emotions_disabled_when_model_missing(self):
        d = ExpertFakeNewsDetector(model_dir='/nonexistent', use_emotions=True)
        assert d.use_emotions is False
        assert d.emotion_extractor is None

    def test_health_check_cases_exist(self):
        assert len(ExpertFakeNewsDetector.HEALTH_CHECK_CASES) == 5
        for text, label, lo, hi in ExpertFakeNewsDetector.HEALTH_CHECK_CASES:
            assert isinstance(text, str)
            assert label in (0, 1)
            assert 0 <= lo <= hi <= 1

    def test_load_nonexistent_model(self):
        d = ExpertFakeNewsDetector(model_dir='/nonexistent')
        with pytest.raises(FileNotFoundError):
            d.load(suffix='expert_v999')


class TestLanguageRouter:
    def test_detect_french(self):
        result = LanguageRouter.detect_language(
            "Le président de la République a annoncé de nouvelles mesures pour la santé publique"
        )
        assert result == 'fr'

    def test_detect_english(self):
        result = LanguageRouter.detect_language(
            "The president announced new healthcare measures today"
        )
        assert result == 'en'

    def test_detect_batch(self):
        texts = pd.Series([
            "Bonjour le monde, comment allez-vous aujourd'hui en France",
            "Hello world, how are you doing today in America",
        ])
        results = LanguageRouter.detect_batch(texts)
        assert len(results) == 2
        assert results.iloc[0] == 'fr'
        assert results.iloc[1] == 'en'

    def test_handles_empty_text(self):
        result = LanguageRouter.detect_language("")
        assert result in ('en', 'fr', 'other')

    def test_handles_short_text(self):
        result = LanguageRouter.detect_language("ok")
        assert result in ('en', 'fr', 'other')


class TestLinguisticFeatureExtractorExtended:
    def test_feature_count(self):
        assert len(LinguisticFeatureExtractor.FEATURE_NAMES) == 15

    def test_sensationalism_detection(self):
        texts = pd.Series(["BREAKING: Shocking conspiracy exposed by deep state!"])
        features = LinguisticFeatureExtractor.extract(texts)
        # sensationalism_score is index 6
        assert features[0, 6] > 0

    def test_url_detection(self):
        texts = pd.Series([
            "Check https://example.com for more",
            "No url here",
        ])
        features = LinguisticFeatureExtractor.extract(texts)
        assert features[0, 7] == 1.0  # has_url
        assert features[1, 7] == 0.0

    def test_interpellation_detection(self):
        texts = pd.Series(["Réveillez-vous ! Partagez avant censure !"])
        features = LinguisticFeatureExtractor.extract(texts)
        # interpellation_score is index 13
        assert features[0, 13] > 0

    def test_short_text_flag(self):
        short = pd.Series(["Hello world"])
        long_ = pd.Series(["This is a much longer text with many words that exceeds twenty words in total for testing purposes only and also more filler content here"])
        f_short = LinguisticFeatureExtractor.extract(short)
        f_long = LinguisticFeatureExtractor.extract(long_)
        assert f_short[0, 14] == 1.0  # is_short_text (< 20 words)
        assert f_long[0, 14] == 0.0  # >= 20 words

    def test_numeric_density(self):
        texts = pd.Series(["In 2024, 50% of users had 100 posts"])
        features = LinguisticFeatureExtractor.extract(texts)
        assert features[0, 8] > 0  # numeric_density

    def test_lexical_diversity(self):
        low_div = pd.Series(["the the the the the the the the the the"])
        high_div = pd.Series(["the quick brown fox jumps over lazy dog sleeping cat"])
        f_low = LinguisticFeatureExtractor.extract(low_div)
        f_high = LinguisticFeatureExtractor.extract(high_div)
        assert f_high[0, 9] > f_low[0, 9]  # lexical_diversity

    def test_all_caps_ratio(self):
        texts = pd.Series(["THIS IS ALL CAPS WORDS HERE"])
        features = LinguisticFeatureExtractor.extract(texts)
        assert features[0, 12] > 0.5  # all_caps_words_ratio

    def test_batch_processing(self):
        texts = pd.Series(["Text one", "Text two", "Text three"])
        features = LinguisticFeatureExtractor.extract(texts)
        assert features.shape == (3, 15)

    def test_french_sensationalism(self):
        texts = pd.Series(["SCANDALE: le gouvernement cache la vérité sur ce complot!"])
        features = LinguisticFeatureExtractor.extract(texts)
        assert features[0, 6] > 0  # sensationalism_score


class TestDatasetCleanerFrAugmentation:
    def test_handles_short_articles(self):
        df = pd.DataFrame({
            'text_original': ["Court."],
            'text_clean': ["court"],
            'label': [0],
        })
        result = DatasetCleaner.generate_fr_short_augmentation(df)
        # Should produce something even from short text
        assert isinstance(result, pd.DataFrame)
