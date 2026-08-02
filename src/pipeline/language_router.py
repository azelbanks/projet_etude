"""
ThumaCheck — Language Router
=============================

Detects text language (FR/EN) for routing to the appropriate
model pipeline (CamemBERT for FR, RoBERTa for EN).

Auteur : Niamato Consulting (pour Thumalien)
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

class LanguageRouter:
    """Détecte la langue de chaque post et route vers le traitement adapté."""

    @staticmethod
    def detect_language(text: str) -> str:
        """Retourne 'fr', 'en', ou 'other'."""
        if not LANGDETECT_AVAILABLE:
            return 'en'
        try:
            lang = detect(str(text)[:500])
            if lang == 'fr':
                return 'fr'
            if lang == 'en':
                return 'en'
            return 'other'
        except Exception:
            return 'en'

    @classmethod
    def detect_batch(cls, texts: pd.Series) -> pd.Series:
        """Détecte la langue pour une série de textes."""
        return texts.apply(cls.detect_language)


