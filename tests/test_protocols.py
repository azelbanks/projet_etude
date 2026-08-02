"""
Tests du contrat d'interface (Strategy pattern).

`protocols.py` n'etait couvert par aucun test alors qu'il definit le contrat
que tout nouveau classifieur doit respecter : une regression y passait
inapercue jusqu'a l'integration.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline.protocols import ClassifierProtocol, ExplainableModel


class _ConformingClassifier:
    """Implementation minimale conforme a ClassifierProtocol."""

    def load(self) -> bool:
        return True

    def predict(self, texts: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"label": [0] * len(texts), "score": [0.5] * len(texts)})

    def save(self, suffix: str = "") -> None:
        return None


class _MissingSave:
    """Il manque save() : ne doit pas satisfaire le protocole."""

    def load(self) -> bool:
        return True

    def predict(self, texts: list[str]) -> pd.DataFrame:
        return pd.DataFrame()


class _ConformingExplainable:
    def explain_prediction(self, text: str, top_n: int = 10) -> dict:
        return {"top_features": [("mot", 0.5)], "prediction": 0, "score": 0.5}


class TestClassifierProtocol:
    def test_conforming_class_satisfies_protocol(self):
        assert isinstance(_ConformingClassifier(), ClassifierProtocol)

    def test_missing_method_fails_protocol(self):
        assert not isinstance(_MissingSave(), ClassifierProtocol)

    def test_unrelated_object_fails_protocol(self):
        assert not isinstance(object(), ClassifierProtocol)

    def test_predict_returns_documented_columns(self):
        """Le protocole exige au minimum les colonnes label et score."""
        df = _ConformingClassifier().predict(["a", "b"])
        assert {"label", "score"}.issubset(df.columns)
        assert len(df) == 2

    def test_real_detector_satisfies_protocol(self):
        """ExpertFakeNewsDetector est annonce comme implementation du contrat."""
        from pipeline.detector import ExpertFakeNewsDetector

        assert hasattr(ExpertFakeNewsDetector, "load")
        assert hasattr(ExpertFakeNewsDetector, "predict")
        assert hasattr(ExpertFakeNewsDetector, "save")


class TestExplainableModel:
    def test_conforming_class_satisfies_protocol(self):
        assert isinstance(_ConformingExplainable(), ExplainableModel)

    def test_missing_method_fails_protocol(self):
        assert not isinstance(_ConformingClassifier(), ExplainableModel)

    def test_explain_returns_documented_keys(self):
        out = _ConformingExplainable().explain_prediction("texte")
        assert {"top_features", "prediction", "score"}.issubset(out)
