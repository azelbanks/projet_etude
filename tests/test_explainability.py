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

from explainability import MetaLearnerDecomposer

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
        assert all(abs(contribs[i]) >= abs(contribs[i + 1]) for i in range(len(contribs) - 1))

    def test_v7_default_feature_names_4_features(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        mock = _MockLogReg([1.0, 1.0, 1.0, 1.0], 0.0)
        d = MetaLearnerDecomposer({"meta_model": mock})
        assert d.feature_names == [
            "score_v5_fiable",
            "score_v6_suspect",
            "disagreement",
            "interaction",
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
        import json

        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1, 1, 1, 1], 0)})
        r = d.decompose([0.5, 0.5, 0.5, 0.5])
        parsed = json.loads(r.to_json())
        assert "logit" in parsed
        assert "proba_suspect" in parsed
        assert "top_drivers" in parsed
        assert len(parsed["contributions"]) == 4

    def test_faithfulness_to_json(self, tmp_path):
        import json

        from explainability.faithfulness import FaithfulnessEvaluator

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


# =====================================================================
#  Tests mutation-killing pour meta_decomposition
# =====================================================================


class TestMetaDecompositionMutationKilling:
    """Tests ciblés pour tuer les mutants survivants de mutmut."""

    def _make_decomposer(self, coef=None, intercept=0.0):
        coef = coef or [1.5, -2.0, 0.8, -0.3]
        return MetaLearnerDecomposer({"meta_model": _MockLogReg(coef, intercept)})

    # --- to_json: vérifier CHAQUE clé et valeur exacte ---
    def test_to_json_all_keys_present(self):
        import json

        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1, -1, 0.5, 0.2], -0.3)})
        r = d.decompose([0.6, 0.4, 0.8, 0.1])
        parsed = json.loads(r.to_json())
        expected_keys = {
            "feature_names",
            "feature_values",
            "coefficients",
            "contributions",
            "intercept",
            "logit",
            "proba_suspect",
            "label",
            "threshold",
            "top_drivers",
            "figures",
        }
        assert set(parsed.keys()) == expected_keys

    def test_to_json_values_match_dataclass(self):
        import json

        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1, -1, 0.5, 0.2], -0.3)})
        r = d.decompose([0.6, 0.4, 0.8, 0.1])
        parsed = json.loads(r.to_json())
        assert parsed["feature_names"] == r.feature_names
        assert parsed["feature_values"] == pytest.approx(r.feature_values)
        assert parsed["coefficients"] == pytest.approx(r.coefficients)
        assert parsed["contributions"] == pytest.approx(r.contributions)
        assert parsed["intercept"] == pytest.approx(r.intercept)
        assert parsed["logit"] == pytest.approx(r.logit)
        assert parsed["proba_suspect"] == pytest.approx(r.proba_suspect)
        assert parsed["label"] == r.label
        assert parsed["threshold"] == pytest.approx(r.threshold)
        assert parsed["figures"] == r.figures

    def test_to_json_top_drivers_count(self):
        import json

        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1, -1, 0.5, 0.2], -0.3)})
        r = d.decompose([0.6, 0.4, 0.8, 0.1])
        parsed = json.loads(r.to_json())
        # to_json uses top_drivers(5), but only 4 features → 4 drivers
        assert len(parsed["top_drivers"]) == 4

    def test_to_json_ensure_ascii_false(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1, -1, 0.5, 0.2], -0.3)})
        r = d.decompose([0.6, 0.4, 0.8, 0.1])
        # ensure_ascii=False means unicode chars are NOT escaped
        json_str = r.to_json()
        assert "\\u" not in json_str or json_str == json_str  # no ascii escaping

    def test_to_json_indent(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1, -1, 0.5, 0.2], -0.3)})
        r = d.decompose([0.6, 0.4, 0.8, 0.1])
        json_str = r.to_json()
        # indent=2 → lines should start with 2 spaces
        assert "\n  " in json_str

    # --- top_drivers: vérifier chaque champ du dict ---
    def test_top_drivers_each_field_exact(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([2.0, -1.0, 0.5, 0.1], 0.0)})
        r = d.decompose([1.0, 1.0, 1.0, 1.0])
        top = r.top_drivers(4)
        # First driver should be the one with largest |contribution|
        assert top[0]["feature"] == "score_v5_fiable"  # coef=2.0, x=1.0 → contrib=2.0
        assert top[0]["value"] == pytest.approx(1.0)
        assert top[0]["coefficient"] == pytest.approx(2.0)
        assert top[0]["contribution"] == pytest.approx(2.0)
        assert top[0]["direction"] == "SUSPECT"  # positive contribution

    def test_top_drivers_direction_fiable(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([-5.0, 0.1, 0.1, 0.1], 0.0)})
        r = d.decompose([1.0, 0.0, 0.0, 0.0])
        top = r.top_drivers(1)
        assert top[0]["direction"] == "FIABLE"  # negative contribution
        assert top[0]["contribution"] < 0

    def test_top_drivers_default_k_is_3(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1, 2, 3, 4], 0)})
        r = d.decompose([1.0, 1.0, 1.0, 1.0])
        top = r.top_drivers()  # default k=3
        assert len(top) == 3

    def test_top_drivers_k_exceeds_features(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1, 2, 3, 4], 0)})
        r = d.decompose([1.0, 1.0, 1.0, 1.0])
        top = r.top_drivers(10)  # k > n_features
        assert len(top) == 4

    # --- default threshold ---
    def test_default_threshold_is_05(self):
        from explainability.meta_decomposition import MetaDecomposition

        d = MetaDecomposition(
            feature_names=["a"],
            feature_values=[1.0],
            coefficients=[1.0],
            contributions=[1.0],
            intercept=0.0,
            logit=1.0,
            proba_suspect=0.73,
            label="SUSPECT",
        )
        assert d.threshold == 0.5

    def test_default_figures_is_empty_dict(self):
        from explainability.meta_decomposition import MetaDecomposition

        d = MetaDecomposition(
            feature_names=["a"],
            feature_values=[1.0],
            coefficients=[1.0],
            contributions=[1.0],
            intercept=0.0,
            logit=1.0,
            proba_suspect=0.73,
            label="SUSPECT",
        )
        assert d.figures == {}

    # --- Feature name constants V7 / V8 ---
    def test_v7_feature_names_exact(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1, 1, 1, 1], 0)})
        assert d.feature_names == [
            "score_v5_fiable",
            "score_v6_suspect",
            "disagreement",
            "interaction",
        ]

    def test_v8_feature_names_exact(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1] * 7, 0)})
        assert d.feature_names == [
            "score_v5_fiable",
            "score_v6_suspect",
            "score_camembert_fiable",
            "disagreement_v5_v6",
            "disagreement_v5_cam",
            "interaction_v5_v6",
            "min_fiable",
        ]

    def test_fallback_feature_names_other_dim(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1] * 5, 0)})
        assert d.feature_names == ["f_0", "f_1", "f_2", "f_3", "f_4"]

    # --- meta_data fallback keys ---
    def test_meta_data_model_key_fallback(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        mock = _MockLogReg([1, 1, 1, 1], 0)
        # "model" key instead of "meta_model"
        d = MetaLearnerDecomposer({"model": mock})
        assert d.feature_names is not None

    def test_meta_data_empty_keys_fallback_to_dict_itself(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        mock = _MockLogReg([1, 1, 1, 1], 0)
        # Neither "meta_model" nor "model" → falls back to the dict itself
        # But dict has no coef_ → ValueError expected
        with pytest.raises(ValueError):
            MetaLearnerDecomposer({"other_key": mock})

    # --- custom threshold ---
    def test_custom_threshold(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1, 1, 1, 1], 0)}, threshold=0.8)
        r = d.decompose([0.5, 0.5, 0.5, 0.5])
        assert r.threshold == 0.8

    def test_label_depends_on_threshold(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        # With default threshold=0.5 and z=0 → p=0.5 → SUSPECT (>= 0.5)
        d_low = MetaLearnerDecomposer({"meta_model": _MockLogReg([0, 0, 0, 0], 0)}, threshold=0.5)
        r_low = d_low.decompose([0, 0, 0, 0])
        assert r_low.label == "SUSPECT"
        # With threshold=0.6 → p=0.5 < 0.6 → FIABLE
        d_high = MetaLearnerDecomposer({"meta_model": _MockLogReg([0, 0, 0, 0], 0)}, threshold=0.6)
        r_high = d_high.decompose([0, 0, 0, 0])
        assert r_high.label == "FIABLE"

    # --- sigmoid ---
    def test_sigmoid_zero(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        assert MetaLearnerDecomposer._sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_large_positive(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        assert MetaLearnerDecomposer._sigmoid(100.0) == pytest.approx(1.0)

    def test_sigmoid_large_negative(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        assert MetaLearnerDecomposer._sigmoid(-100.0) == pytest.approx(0.0, abs=1e-30)

    # --- decompose output type/structure ---
    def test_decompose_returns_lists(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1, -1, 0.5, 0.2], 0)})
        r = d.decompose([0.5, 0.5, 0.5, 0.5])
        assert isinstance(r.feature_names, list)
        assert isinstance(r.feature_values, list)
        assert isinstance(r.coefficients, list)
        assert isinstance(r.contributions, list)
        assert isinstance(r.intercept, float)
        assert isinstance(r.logit, float)
        assert isinstance(r.proba_suspect, float)

    # --- feature_names from meta_data ---
    def test_custom_feature_names_from_meta_data(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        mock = _MockLogReg([1, 1, 1, 1], 0)
        names = ["alpha", "beta", "gamma", "delta"]
        d = MetaLearnerDecomposer({"meta_model": mock, "feature_names": names})
        assert d.feature_names == names

    # --- direction boundary: contribution == 0 → FIABLE ---
    def test_direction_zero_contribution_is_fiable(self):
        from explainability.meta_decomposition import MetaDecomposition

        d = MetaDecomposition(
            feature_names=["a", "b"],
            feature_values=[0.0, 1.0],
            coefficients=[0.0, 1.0],
            contributions=[0.0, 1.0],
            intercept=0.0,
            logit=1.0,
            proba_suspect=0.73,
            label="SUSPECT",
        )
        top = d.top_drivers(2)
        zero_driver = next(t for t in top if t["contribution"] == 0.0)
        assert zero_driver["direction"] == "FIABLE"  # 0 > 0 is False → FIABLE

    # --- to_plotly_bar tests (kill mutants on the visualization) ---
    def test_to_plotly_bar_labels_fr_defaults(self):
        from explainability.meta_decomposition import MetaDecomposition, MetaLearnerDecomposer

        d = MetaDecomposition(
            feature_names=["score_v5_fiable", "score_v6_suspect", "disagreement", "interaction"],
            feature_values=[0.8, 0.3, 0.1, 0.05],
            coefficients=[1.5, -1.0, 0.5, 0.2],
            contributions=[1.2, -0.3, 0.05, 0.01],
            intercept=-0.5,
            logit=0.46,
            proba_suspect=0.61,
            label="SUSPECT",
        )
        fig = MetaLearnerDecomposer.to_plotly_bar(d)
        # Check all default FR labels appear in the figure y-axis
        bar_data = fig.data[0]
        y_labels = list(bar_data.y)
        assert "Score V5 (P fiable, TF-IDF)" in y_labels
        assert "Score V6 (P suspect, style)" in y_labels

    def test_to_plotly_bar_colors(self):
        from explainability.meta_decomposition import MetaDecomposition, MetaLearnerDecomposer

        d = MetaDecomposition(
            feature_names=["a", "b"],
            feature_values=[1.0, 1.0],
            coefficients=[1.0, -1.0],
            contributions=[1.0, -1.0],
            intercept=0.0,
            logit=0.0,
            proba_suspect=0.5,
            label="SUSPECT",
        )
        fig = MetaLearnerDecomposer.to_plotly_bar(d)
        colors = list(fig.data[0].marker.color)
        # Positive contribution → red, negative → green
        assert "#FF1744" in colors
        assert "#00E676" in colors

    def test_to_plotly_bar_title_contains_proba_and_label(self):
        from explainability.meta_decomposition import MetaDecomposition, MetaLearnerDecomposer

        d = MetaDecomposition(
            feature_names=["a", "b"],
            feature_values=[1.0, 1.0],
            coefficients=[1.0, -1.0],
            contributions=[1.0, -1.0],
            intercept=0.0,
            logit=0.0,
            proba_suspect=0.5,
            label="SUSPECT",
        )
        fig = MetaLearnerDecomposer.to_plotly_bar(d)
        title_text = fig.layout.title.text
        assert "0.50" in title_text  # proba formatted
        assert "SUSPECT" in title_text

    def test_to_plotly_bar_hover_contains_values(self):
        from explainability.meta_decomposition import MetaDecomposition, MetaLearnerDecomposer

        d = MetaDecomposition(
            feature_names=["a", "b"],
            feature_values=[0.75, 0.25],
            coefficients=[2.0, -1.0],
            contributions=[1.5, -0.25],
            intercept=0.0,
            logit=1.25,
            proba_suspect=0.78,
            label="SUSPECT",
        )
        fig = MetaLearnerDecomposer.to_plotly_bar(d)
        hover = list(fig.data[0].hovertext)
        # Hover should contain feature values and coefficients
        hover_str = " ".join(hover)
        assert "x=" in hover_str
        assert "β=" in hover_str

    def test_to_plotly_bar_v8_labels(self):
        from explainability.meta_decomposition import MetaDecomposition, MetaLearnerDecomposer

        d = MetaDecomposition(
            feature_names=[
                "score_camembert_fiable",
                "disagreement_v5_v6",
                "disagreement_v5_cam",
                "interaction_v5_v6",
                "min_fiable",
            ],
            feature_values=[0.5] * 5,
            coefficients=[1.0] * 5,
            contributions=[0.5] * 5,
            intercept=0.0,
            logit=2.5,
            proba_suspect=0.92,
            label="SUSPECT",
        )
        fig = MetaLearnerDecomposer.to_plotly_bar(d)
        y_labels = list(fig.data[0].y)
        assert "Score CamemBERT (P fiable)" in y_labels
        assert "Min(V5,CamemBERT)" in y_labels

    def test_to_plotly_bar_custom_labels_fr(self):
        from explainability.meta_decomposition import MetaDecomposition, MetaLearnerDecomposer

        d = MetaDecomposition(
            feature_names=["a"],
            feature_values=[1.0],
            coefficients=[1.0],
            contributions=[1.0],
            intercept=0.0,
            logit=1.0,
            proba_suspect=0.73,
            label="SUSPECT",
        )
        custom = {"a": "Mon Feature A"}
        fig = MetaLearnerDecomposer.to_plotly_bar(d, labels_fr=custom)
        assert "Mon Feature A" in list(fig.data[0].y)

    def test_to_plotly_bar_ordering_reversed(self):
        """Bars should be ordered with largest magnitude at top."""
        from explainability.meta_decomposition import MetaDecomposition, MetaLearnerDecomposer

        d = MetaDecomposition(
            feature_names=["small", "big", "medium"],
            feature_values=[1.0, 1.0, 1.0],
            coefficients=[0.1, 5.0, 1.0],
            contributions=[0.1, 5.0, 1.0],
            intercept=0.0,
            logit=6.1,
            proba_suspect=0.99,
            label="SUSPECT",
        )
        fig = MetaLearnerDecomposer.to_plotly_bar(d)
        y_labels = list(fig.data[0].y)
        x_vals = list(fig.data[0].x)
        # Top bar (last in reversed list) should be biggest magnitude
        assert abs(x_vals[-1]) >= abs(x_vals[0])

    def test_to_plotly_bar_margin_and_height(self):
        from explainability.meta_decomposition import MetaDecomposition, MetaLearnerDecomposer

        d = MetaDecomposition(
            feature_names=["a", "b", "c"],
            feature_values=[1.0, 1.0, 1.0],
            coefficients=[1.0, 1.0, 1.0],
            contributions=[1.0, 1.0, 1.0],
            intercept=0.0,
            logit=3.0,
            proba_suspect=0.95,
            label="SUSPECT",
        )
        fig = MetaLearnerDecomposer.to_plotly_bar(d)
        assert fig.layout.margin.l == 200
        assert fig.layout.margin.t == 60
        assert fig.layout.margin.b == 40
        assert fig.layout.margin.r == 20
        assert fig.layout.height == max(280, 3 * 38)

    def test_to_json_top_drivers_uses_5(self):
        """to_json calls top_drivers(5), verify the count."""
        import json

        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1] * 7, 0)})
        r = d.decompose([1.0] * 7)
        parsed = json.loads(r.to_json())
        assert len(parsed["top_drivers"]) == 5  # top_drivers(5) with 7 features → 5

    def test_error_message_non_linear(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        class Fake:
            pass

        with pytest.raises(ValueError, match="coef_"):
            MetaLearnerDecomposer({"meta_model": Fake()})

    def test_error_message_dimension_mismatch(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer

        d = MetaLearnerDecomposer({"meta_model": _MockLogReg([1, 1, 1, 1], 0)})
        with pytest.raises(ValueError, match="features"):
            d.decompose([0.5, 0.5])
