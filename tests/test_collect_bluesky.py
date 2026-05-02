"""
Tests for collection/collect_bluesky.py — pure utility functions.

Validates text validation, word counting, and language detection heuristics
without requiring any external service (no Bluesky API, no MongoDB).
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from collection.collect_bluesky import validate_text, compute_word_count, detect_language_hint


# -----------------------------------------------------------------------
#  validate_text
# -----------------------------------------------------------------------

class TestValidateText:
    """Tests for the validate_text() function."""

    def test_valid_text(self):
        ok, result = validate_text("This is a valid post about climate change.")
        assert ok is True
        assert result == "This is a valid post about climate change."

    def test_valid_text_stripped(self):
        ok, result = validate_text("  some whitespace  ")
        assert ok is True
        assert result == "some whitespace"

    def test_empty_string(self):
        ok, reason = validate_text("")
        assert ok is False
        assert reason == "empty_or_missing"

    def test_none_text(self):
        ok, reason = validate_text(None)
        assert ok is False
        assert reason == "empty_or_missing"

    def test_non_string(self):
        ok, reason = validate_text(12345)
        assert ok is False
        assert reason == "empty_or_missing"

    def test_too_short(self):
        ok, reason = validate_text("ab")
        assert ok is False
        assert reason == "too_short"

    def test_exactly_3_chars(self):
        ok, result = validate_text("abc")
        assert ok is True
        assert result == "abc"

    def test_url_only(self):
        ok, reason = validate_text("https://example.com/some/path")
        assert ok is False
        assert reason == "url_only"

    def test_url_with_real_content(self):
        ok, result = validate_text("Check this out https://example.com interesting article")
        assert ok is True
        assert "Check this out" in result

    def test_multiple_urls_only(self):
        ok, reason = validate_text("https://a.com https://b.com")
        assert ok is False
        assert reason == "url_only"


# -----------------------------------------------------------------------
#  compute_word_count
# -----------------------------------------------------------------------

class TestComputeWordCount:
    """Tests for the compute_word_count() function."""

    def test_simple_sentence(self):
        assert compute_word_count("Hello world today") == 3

    def test_empty_text(self):
        assert compute_word_count("") == 0

    def test_url_excluded(self):
        count = compute_word_count("Visit https://example.com for info")
        assert count == 3  # "Visit", "for", "info"

    def test_only_url(self):
        assert compute_word_count("https://example.com") == 0

    def test_mixed_content(self):
        text = "Breaking news https://t.co/abc this is important https://bit.ly/xyz"
        count = compute_word_count(text)
        assert count == 5  # "Breaking", "news", "this", "is", "important"


# -----------------------------------------------------------------------
#  detect_language_hint
# -----------------------------------------------------------------------

class TestDetectLanguageHint:
    """Tests for the detect_language_hint() heuristic."""

    def test_french_text(self):
        text = "Le gouvernement est dans une situation qui est difficile pour les citoyens"
        assert detect_language_hint(text) == "fr"

    def test_english_text(self):
        text = "The government announced new policies about climate change today"
        assert detect_language_hint(text) == "en"

    def test_url_ignored_in_detection(self):
        text = "Le gouvernement est https://example.com dans une situation"
        assert detect_language_hint(text) == "fr"

    def test_empty_returns_en(self):
        assert detect_language_hint("") == "en"

    def test_mixed_but_mostly_french(self):
        text = "Les experts sont dans le domaine de la santé et du digital"
        assert detect_language_hint(text) == "fr"

    def test_pure_english(self):
        text = "Scientists discover breakthrough technology for renewable energy"
        assert detect_language_hint(text) == "en"

    def test_short_french(self):
        text = "Il est un bon élève"
        assert detect_language_hint(text) == "fr"
