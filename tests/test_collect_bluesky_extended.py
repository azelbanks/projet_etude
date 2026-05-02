"""Tests etendus pour collect_bluesky — fonctions utilitaires + mocks."""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from collection.collect_bluesky import (
    validate_text, compute_word_count, detect_language_hint,
    load_excluded_handles, SEARCH_CONFIG,
)


class TestValidateTextExtended:
    def test_valid_text(self):
        ok, result = validate_text("This is a valid post about politics")
        assert ok is True
        assert result == "This is a valid post about politics"

    def test_strips_whitespace(self):
        ok, result = validate_text("  Hello world  ")
        assert ok is True
        assert result == "Hello world"

    def test_none_input(self):
        ok, reason = validate_text(None)
        assert ok is False
        assert reason == "empty_or_missing"

    def test_empty_string(self):
        ok, reason = validate_text("")
        assert ok is False
        assert reason == "empty_or_missing"

    def test_too_short(self):
        ok, reason = validate_text("ab")
        assert ok is False
        assert reason == "too_short"

    def test_url_only(self):
        ok, reason = validate_text("https://example.com")
        assert ok is False
        assert reason == "url_only"

    def test_url_with_text(self):
        ok, result = validate_text("Check this link https://example.com for more info")
        assert ok is True

    def test_integer_input(self):
        ok, reason = validate_text(123)
        assert ok is False

    def test_boolean_input(self):
        ok, reason = validate_text(True)
        assert ok is False


class TestComputeWordCountExtended:
    def test_normal_text(self):
        assert compute_word_count("Hello world foo bar") == 4

    def test_text_with_urls(self):
        count = compute_word_count("Hello https://example.com world")
        assert count == 2  # URL removed

    def test_empty_text(self):
        assert compute_word_count("") == 0

    def test_only_url(self):
        assert compute_word_count("https://example.com") == 0

    def test_multiple_urls(self):
        count = compute_word_count("Visit https://a.com and https://b.com today")
        assert count == 3  # "Visit", "and", "today"


class TestDetectLanguageHintExtended:
    def test_french_text(self):
        result = detect_language_hint("Je suis dans le jardin de la maison")
        assert result == "fr"

    def test_english_text(self):
        result = detect_language_hint("The cat is on the table and it is nice")
        assert result == "en"

    def test_empty_text(self):
        result = detect_language_hint("")
        assert result == "en"

    def test_mixed_text(self):
        # Mostly French
        result = detect_language_hint("Le chat est sur la table et il est content de la vie")
        assert result == "fr"

    def test_url_ignored(self):
        result = detect_language_hint("https://example.com")
        assert result == "en"


class TestSearchConfig:
    def test_has_en_and_fr(self):
        assert 'en' in SEARCH_CONFIG
        assert 'fr' in SEARCH_CONFIG

    def test_en_terms_non_empty(self):
        assert len(SEARCH_CONFIG['en']) > 0

    def test_fr_terms_non_empty(self):
        assert len(SEARCH_CONFIG['fr']) > 0

    def test_no_emotional_bias_terms(self):
        """Verify 'happy', 'amazing', 'joie' were removed (collecteur V3)."""
        all_terms = SEARCH_CONFIG['en'] + SEARCH_CONFIG['fr']
        for term in ['happy', 'amazing', 'thank you', 'joie']:
            assert term not in all_terms, f"Biased term '{term}' should be removed"

    def test_has_desinformation_terms(self):
        en_terms = SEARCH_CONFIG['en']
        assert 'conspiracy' in en_terms
        assert 'vaccine' in en_terms


class TestLoadExcludedHandles:
    def test_returns_set(self):
        result = load_excluded_handles()
        assert isinstance(result, set)

    def test_nonexistent_file_returns_empty(self, tmp_path, monkeypatch):
        import collection.collect_bluesky as cb
        monkeypatch.setattr(cb, '_EXCLUDED_HANDLES_FILE', str(tmp_path / 'nope.txt'))
        result = cb.load_excluded_handles()
        assert result == set()


class TestConnectDbMaxRetries:
    @patch('collection.collect_bluesky.MongoClient')
    @patch('collection.collect_bluesky.time')
    def test_raises_after_max_retries(self, mock_time, mock_client):
        mock_time.sleep = MagicMock()
        mock_client.return_value.admin.command.side_effect = Exception("connection refused")

        from collection.collect_bluesky import connect_db
        with pytest.raises(ConnectionError, match="inaccessible"):
            connect_db()

        assert mock_time.sleep.call_count == 20  # MAX_RETRIES
