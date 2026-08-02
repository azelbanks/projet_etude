"""
ThumaCheck — Expert Bilingual Fake News Detector (Facade)
=========================================================

This module re-exports all pipeline components for backward compatibility.
The implementation has been split into focused modules:

- dataset_cleaner.py    — DatasetCleaner (Reuters bias removal, multi-source loading)
- linguistic_features.py — LinguisticFeatureExtractor (45+ NLP features)
- emotion_classifier.py — EmotionFeatureExtractor, _EmotionMLP (7-class emotion)
- language_router.py    — LanguageRouter (FR/EN detection)
- detector.py           — ExpertFakeNewsDetector (V9 cascade pipeline)

Auteur : Niamato Consulting (pour Thumalien)
"""

from .dataset_cleaner import DatasetCleaner
from .detector import ExpertFakeNewsDetector
from .emotion_classifier import EmotionFeatureExtractor, _EmotionMLP
from .language_router import LanguageRouter
from .linguistic_features import LinguisticFeatureExtractor

__all__ = [
    "DatasetCleaner",
    "EmotionFeatureExtractor",
    "ExpertFakeNewsDetector",
    "LanguageRouter",
    "LinguisticFeatureExtractor",
    "_EmotionMLP",
]
