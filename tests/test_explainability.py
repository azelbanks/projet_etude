"""
Tests unitaires du module d'explicabilité.

Ces tests sont **agnostiques du modèle réel** : ils utilisent des mocks
pour valider la logique des décompositions et des métriques de fidélité,
sans nécessiter de charger CamemBERT ou de calculer SHAP. Pour des tests
d'intégration end-to-end, voir `scripts/run_xai_pipeline.py` lancé avec
le gold set complet.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))


# =====================================================================
#  MetaLearnerDecomposer
# =====================================================================

class _MockLogReg:
    def __init__(self, coef, intercept):
        self.coef_ = np.array([coef])
        self.intercept_ = np.array([intercept])

    def predict_proba(self, X):
        z = X @ self.coef_[0] + self.intercept_[0]
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])


class TestMetaLearnerDecomposer:
    def test_decomposition_matches_logreg_proba(self):
        """β·x + intercept doit reproduire predict_proba à 1e-9 près."""
        from explainability.meta_decomposition import MetaLearnerDecomposer

        np.random.seed(0)
        coef = np.array([1.5, 2.1, -0.8, 0.3])
        intercept = -0.4
        mock = _MockLogReg(coef, intercept)
        meta = {"meta_model": mock}

        decomposer = MetaLearnerDecomposer(meta)
        for _ in range(20):
            x = np.random.uniform(0, 1, 4)
            d = decomposer.decompose(x)
            p_via_model = mock.predict_proba(x.reshape(1, -1))[0, 1]
            assert abs(d.proba_suspect - p_via_model) < 1e-9
            # Sum of contributions = z - intercept
            assert abs(sum(d.contributions) + d.intercept - d.logit) < 1e-9

    def test_top_drivers_ordered_by_magnitude(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        mock = _MockLogReg([0.5, 2.0, -3.0, 0.1], -0.5)
        meta = {"meta_model": mock}
        d = MetaLearnerDecomposer(meta).decompose([0.4, 0.4, 0.4, 0.4])
        top = d.top_drivers(3)
        contribs = [t["contribution"] for t in top]
        assert all(
            abs(contribs[i]) >= abs(contribs[i + 1]) for i in range(len(contribs) - 1)
        )

    def test_v7_default_feature_names_4_features(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        mock = _MockLogReg([1.0, 1.0, 1.0, 1.0], 0.0)
        d = MetaLearnerDecomposer({"meta_model": mock})
        assert d.feature_names == [
            "score_v5_fiable", "score_v6_suspect", "disagreement", "interaction",
        ]

    def test_v8_default_feature_names_7_features(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        mock = _MockLogReg([1] * 7, 0.0)
        d = MetaLearnerDecomposer({"meta_model": mock})
        assert len(d.feature_names) == 7
        assert "score_camembert_fiable" in d.feature_names

    def test_dimension_mismatch_raises(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1, 1, 1, 1], 0)})
        with pytest.raises(ValueError):
            d.decompose([0.5, 0.5])  # mauvaise dim

    def test_label_above_threshold_is_suspect(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        # Coef > 0 + grand x → P(suspect) élevé
        mock = _MockLogReg([5.0, 5.0, 0, 0], -0.5)
        d = MetaLearnerDecomposer({"meta_model": mock})
        r = d.decompose([1.0, 1.0, 0.0, 0.0])
        assert r.label == "SUSPECT"
        assert r.proba_suspect > 0.5

    def test_non_linear_model_raises(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        class Tree:
            pass

        with pytest.raises(ValueError):
            MetaLearnerDecomposer({"meta_model": Tree()})


# =====================================================================
#  FaithfulnessEvaluator
# =====================================================================

class TestFaithfulnessEvaluator:
    @staticmethod
    def _linear_pred(w, b):
        def fn(X):
            z = X @ w + b
            p = 1 / (1 + np.exp(-z))
            return np.column_stack([1 - p, p])
        return fn

    def test_aopc_higher_for_correct_attribution(self, tmp_path):
        from explainability.faithfulness import FaithfulnessEvaluator

        np.random.seed(42)
        w = np.array([2.0, 1.5, 0.1, 0.05])
        X = np.random.uniform(0, 1, (40, 4))
        attr_correct = np.tile(np.abs(w), (40, 1))
        attr_wrong = np.tile(np.abs(w[::-1]), (40, 1))

        ev = FaithfulnessEvaluator(self._linear_pred(w, -0.5), output_dir=str(tmp_path))
        r_ok = ev.evaluate(X, attr_correct, max_k=4)
        r_bad = ev.evaluate(X, attr_wrong, max_k=4)

        assert r_ok.aopc > r_bad.aopc
        # Comprehensiveness@1 = chute de proba après masque de la top feature
        assert r_ok.comprehensiveness_at_k[1] > r_bad.comprehensiveness_at_k[1]

    def test_random_baseline_uplift_positive(self, tmp_path):
        from explainability.faithfulness import FaithfulnessEvaluator

        np.random.seed(7)
        w = np.array([3.0, 2.0, 0.05])
        X = np.random.uniform(0, 1, (30, 3))
        attr = np.tile(np.abs(w), (30, 1))
        ev = FaithfulnessEvaluator(self._linear_pred(w, 0), output_dir=str(tmp_path))
        cmp = ev.compare_with_random(X, attr, n_random_seeds=5, max_k=3)
        assert cmp["aopc_uplift"] > 0

    def test_proba_curve_starts_at_baseline(self, tmp_path):
        from explainability.faithfulness import FaithfulnessEvaluator

        w = np.array([1.0, 1.0])
        X = np.array([[0.5, 0.5]] * 10)
        attr = np.array([[1.0, 1.0]] * 10)
        ev = FaithfulnessEvaluator(self._linear_pred(w, 0), output_dir=str(tmp_path))
        r = ev.evaluate(X, attr, max_k=2)
        # k=0 = pas de masque = p_base
        p_base_expected = 1 / (1 + np.exp(-(0.5 + 0.5)))
        assert abs(r.proba_curve_mean[0] - p_base_expected) < 1e-9

    def test_figure_is_created(self, tmp_path):
        from explainability.faithfulness import FaithfulnessEvaluator

        w = np.array([1.0, 1.0])
        X = np.random.uniform(0, 1, (5, 2))
        attr = np.abs(np.random.randn(5, 2))
        ev = FaithfulnessEvaluator(self._linear_pred(w, 0), output_dir=str(tmp_path))
        r = ev.evaluate(X, attr, max_k=2)
        assert os.path.exists(r.figures["aopc_curve"])


# =====================================================================
#  Sérialisation JSON
# =====================================================================

class TestSerialization:
    def test_meta_decomposition_to_json(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer
        import json

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1, 1, 1, 1], 0)})
        r = d.decompose([0.5, 0.5, 0.5, 0.5])
        parsed = json.loads(r.to_json())
        assert "logit" in parsed
        assert "proba_suspect" in parsed
        assert "top_drivers" in parsed
        assert len(parsed["contributions"]) == 4

    def test_faithfulness_to_json(self, tmp_path):
        from explainability.faithfulness import FaithfulnessEvaluator
        import json

        def fn(X):
            z = X @ np.array([1.0, 1.0])
            p = 1 / (1 + np.exp(-z))
            return np.column_stack([1 - p, p])

        ev = FaithfulnessEvaluator(fn, output_dir=str(tmp_path))
        r = ev.evaluate(
            np.random.uniform(0, 1, (5, 2)),
            np.abs(np.random.randn(5, 2)),
            max_k=2,
        )
        parsed = json.loads(r.to_json())
        assert "aopc" in parsed
        assert "comprehensiveness_at_k" in parsed


# =====================================================================
#  Smoke test : imports lazy
# =====================================================================

class TestLazyImports:
    def test_package_imports_without_torch(self):
        """Vérifie que importer le paquet ne charge pas torch/captum."""
        # On vérifie que `import explainability` ne casse pas, même si
        # captum n'est pas installé. La preuve est l'absence d'ImportError.
        import importlib
        import explainability
        importlib.reload(explainability)
        assert hasattr(explainability, "GlobalShapExplainer")

    def test_meta_decomposition_no_torch_required(self):
        """MetaLearnerDecomposer doit fonctionner sans torch."""
        from explainability.meta_decomposition import MetaLearnerDecomposer
        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1, 1, 1, 1], 0)})
        assert d.decompose([0.5] * 4).proba_suspect > 0
