"""
Tests de validation des entrees aux frontieres du systeme.

Verifie que les textes utilisateur sont correctement valides et sanitizes
avant d'etre traites par le pipeline ou affiches dans le dashboard.
"""

import os
import sys
import html

import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from collection.collect_bluesky import validate_text


# -----------------------------------------------------------------------
#  Tests de securite — injection et XSS
# -----------------------------------------------------------------------

class TestSecurityValidation:
    """Verify that potentially dangerous inputs are handled safely."""

    def test_html_injection_in_text(self):
        """HTML tags in text should be escapable."""
        malicious = '<script>alert("XSS")</script> Hello world'
        ok, result = validate_text(malicious)
        assert ok is True
        # The text passes validation (it's valid text), but should be
        # escaped before rendering in HTML context
        escaped = html.escape(result)
        assert '<script>' not in escaped
        assert '&lt;script&gt;' in escaped

    def test_very_long_text_validation(self):
        """Extremely long text should still be handled."""
        long_text = "A" * 100_000
        ok, result = validate_text(long_text)
        assert ok is True
        assert len(result) == 100_000

    def test_null_bytes(self):
        """Text with null bytes should be handled."""
        text = "Hello\x00world test text"
        ok, result = validate_text(text)
        # Should either accept or reject, but not crash
        assert isinstance(ok, bool)

    def test_unicode_edge_cases(self):
        """Various unicode should not crash validation."""
        cases = [
            "Normal text",
            "Texte avec accents : e\u0301te\u0301",  # combining accents
            "\u200b\u200bHidden zero-width spaces\u200b",  # zero-width
            "\U0001f4a9 Pile of poo emoji text",  # 4-byte emoji
            "RTL text \u0627\u0644\u0639\u0631\u0628\u064a\u0629",  # Arabic
        ]
        for text in cases:
            ok, result = validate_text(text)
            assert isinstance(ok, bool), f"Failed on: {repr(text)}"


# -----------------------------------------------------------------------
#  Tests de validation pipeline — entrees limites
# -----------------------------------------------------------------------

_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
_MODEL_EXISTS = os.path.exists(os.path.join(_MODEL_DIR, 'model_expert_v5.pkl'))


@pytest.mark.skipif(not _MODEL_EXISTS, reason="Model files not found")
class TestPipelineInputValidation:
    """Verify pipeline handles edge-case inputs gracefully."""

    @pytest.fixture(scope='class')
    def detector(self):
        from pipeline.expert_detector import ExpertFakeNewsDetector
        det = ExpertFakeNewsDetector(model_dir=_MODEL_DIR)
        det.load(suffix='expert_v5')
        return det

    def test_html_in_prediction(self, detector):
        """HTML tags affect TF-IDF features — verify prediction still valid."""
        with_html = "<b>Scientists</b> discover <a href='x'>new</a> treatment for disease"
        result = detector.predict(pd.Series([with_html]))
        # Should produce valid output (no crash) with score in range
        assert 0 <= result['ai_score_credibility'].iloc[0] <= 1
        assert result['prediction_label'].iloc[0] in (0, 1)

    def test_repeated_text_stability(self, detector):
        """Same text predicted twice should give identical results."""
        text = pd.Series(["The president announced new economic measures today."])
        r1 = detector.predict(text)['ai_score_credibility'].iloc[0]
        r2 = detector.predict(text)['ai_score_credibility'].iloc[0]
        assert r1 == r2, f"Non-deterministic prediction: {r1} vs {r2}"

    def test_single_word_input(self, detector):
        """Single word should produce valid output."""
        result = detector.predict(pd.Series(["hello"]))
        assert len(result) == 1
        assert 0 <= result['ai_score_credibility'].iloc[0] <= 1
