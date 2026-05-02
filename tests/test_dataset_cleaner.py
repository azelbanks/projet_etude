"""Tests pour DatasetCleaner — nettoyage biais Reuters et préparation datasets."""

import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pipeline.expert_detector import DatasetCleaner


class TestRemoveAgencyBias:
    def test_removes_reuters_prefix(self):
        text = "WASHINGTON (Reuters) - The president announced new measures."
        result = DatasetCleaner.remove_agency_bias(text)
        assert "(Reuters)" not in result
        assert "The president" in result

    def test_removes_ap_prefix(self):
        text = "NEW YORK (AP) - Markets surged today."
        result = DatasetCleaner.remove_agency_bias(text)
        assert "(AP)" not in result

    def test_removes_byline(self):
        text = "Some news. Reporting by John Smith; Editing by Jane Doe"
        result = DatasetCleaner.remove_agency_bias(text)
        assert "Reporting by" not in result
        assert "Editing by" not in result

    def test_removes_trust_principles(self):
        text = "Article text. Our Standards: The Thomson Reuters Trust Principles."
        result = DatasetCleaner.remove_agency_bias(text)
        assert "Thomson Reuters" not in result

    def test_preserves_normal_text(self):
        text = "The weather is nice today in Paris."
        result = DatasetCleaner.remove_agency_bias(text)
        assert result == text

    def test_handles_non_string(self):
        assert DatasetCleaner.remove_agency_bias(None) == ""
        assert DatasetCleaner.remove_agency_bias(123) == ""

    def test_handles_empty_string(self):
        assert DatasetCleaner.remove_agency_bias("") == ""

    def test_removes_afp_pattern(self):
        text = "Un article important. Avec AFP"
        result = DatasetCleaner.remove_agency_bias(text)
        assert "Avec AFP" not in result

    def test_removes_source_attribution(self):
        text = "Breaking news. Source : Reuters"
        result = DatasetCleaner.remove_agency_bias(text)
        assert "Source" not in result or "Reuters" not in result


class TestCleanForML:
    def test_lowercases_text(self):
        assert DatasetCleaner.clean_for_ml("HELLO WORLD") == "hello world"

    def test_removes_urls(self):
        text = "Check this https://example.com and www.test.org"
        result = DatasetCleaner.clean_for_ml(text)
        assert "https://" not in result
        assert "www." not in result

    def test_removes_mentions(self):
        text = "Hello @user123 how are you?"
        result = DatasetCleaner.clean_for_ml(text)
        assert "@user123" not in result

    def test_normalizes_hashtags(self):
        text = "This is #trending"
        result = DatasetCleaner.clean_for_ml(text)
        assert "#" not in result
        assert "trending" in result

    def test_preserves_french_accents(self):
        text = "L'été à Montréal est très chaud"
        result = DatasetCleaner.clean_for_ml(text)
        assert "été" in result
        assert "très" in result

    def test_handles_non_string(self):
        assert DatasetCleaner.clean_for_ml(None) == ""

    def test_collapses_whitespace(self):
        text = "Hello    world   !"
        result = DatasetCleaner.clean_for_ml(text)
        assert "  " not in result


class TestGenerateFrShortAugmentation:
    def test_generates_short_texts(self):
        df = pd.DataFrame({
            'text_original': [
                "Ceci est une première phrase. Et voici la seconde phrase avec plus de mots.",
                "Un article long avec beaucoup de contenu. La deuxième phrase aussi.",
            ],
            'text_clean': ["ceci est une premiere phrase", "un article long"],
            'label': [0, 1],
        })
        result = DatasetCleaner.generate_fr_short_augmentation(df)
        assert len(result) > 0
        assert 'label' in result.columns
        assert 'text_original' in result.columns

    def test_preserves_labels(self):
        df = pd.DataFrame({
            'text_original': [
                "Phrase un. Phrase deux. Phrase trois.",
            ],
            'text_clean': ["phrase un phrase deux phrase trois"],
            'label': [1],
        })
        result = DatasetCleaner.generate_fr_short_augmentation(df)
        if len(result) > 0:
            assert all(result['label'] == 1)


class TestQuantifyLeakage:
    def test_quantify_on_sample(self):
        df_true = pd.DataFrame({
            'text': [
                "WASHINGTON (Reuters) - The president said something important today.",
                "Normal article text without Reuters markers at all here.",
                "NEW YORK (Reuters) - Markets surged. Reporting by John Smith",
            ]
        })
        result = DatasetCleaner.audit_reuters_leakage(df_true)
        assert 'total_articles' in result
        assert result['total_articles'] == 3
        assert result['has_reuters_marker'] >= 2
        assert 'has_reuters_pct' in result
        assert result['has_reuters_pct'] > 50
