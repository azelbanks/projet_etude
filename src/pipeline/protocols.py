"""
ThumaCheck — Model Protocols
=============================

Defines the interface contract for all classification models.
Enables swapping CamemBERT for another transformer without
modifying the detector pipeline (Strategy pattern).

Usage:
    Any new classifier must implement ClassifierProtocol.
    Use isinstance checks or Protocol-based type hints.
"""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ClassifierProtocol(Protocol):
    """Interface for all ThumaCheck classifiers.

    Implementations: CamemBERTClassifier, RoBERTaClassifier,
    ExpertFakeNewsDetector (V5 TF-IDF pipeline).
    """

    def load(self) -> bool:
        """Load pre-trained model from disk."""
        ...

    def predict(self, texts: list[str]) -> pd.DataFrame:
        """Predict labels and scores for a list of texts.

        Returns a DataFrame with at least columns:
            - label (int): 0=reliable, 1=suspect
            - score (float): confidence score [0, 1]
        """
        ...

    def save(self, suffix: str = "") -> None:
        """Save model to disk."""
        ...


@runtime_checkable
class ExplainableModel(Protocol):
    """Interface for models that support explainability."""

    def explain_prediction(self, text: str, top_n: int = 10) -> dict:
        """Return word-level attribution for a single text.

        Returns a dict with at least:
            - top_features: list of (word, weight) tuples
            - prediction: label
            - score: confidence
        """
        ...
