"""
Tests etendus pour le module src/explainability/.

Couvre : AttentionResult, IGResult, GlobalShapResult (dataclasses),
CamembertAttentionExplainer, IGExplainer, SHAPGlobalExplainer,
MetaLearnerDecomposer, FaithfulnessValidator.
"""

import json
import os
import tempfile

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from explainability.attention_viz import AttentionResult
from explainability.integrated_gradients import IGResult
from explainability.shap_global import GlobalShapResult
from explainability.meta_decomposition import MetaLearnerDecomposer


# ============================================================
#  Dataclass serialization
# ============================================================


class TestAttentionResult:
    def test_to_json(self):
        r = AttentionResult(
            text="test",
            tokens=["<s>", "test", "</s>"],
            cls_attention_per_head=[[[0.5, 0.3, 0.2]]],
            cls_attention_avg_heads=[[0.5, 0.3, 0.2]],
            cls_attention_last_layer=[0.5, 0.3, 0.2],
            prediction_label="FIABLE",
            prediction_proba_suspect=0.2,
            prediction_proba_fiable=0.8,
        )
        j = json.loads(r.to_json())
        assert j["text"] == "test"
        assert j["prediction_label"] == "FIABLE"
        assert len(j["tokens"]) == 3

    def test_with_ground_truth(self):
        r = AttentionResult(
            text="t", tokens=["t"],
            cls_attention_per_head=[], cls_attention_avg_heads=[],
            cls_attention_last_layer=[],
            prediction_label="SUSPECT",
            prediction_proba_suspect=0.7,
            prediction_proba_fiable=0.3,
            ground_truth="SUSPECT",
            error_type="TP",
        )
        j = json.loads(r.to_json())
        assert j["ground_truth"] == "SUSPECT"
        assert j["error_type"] == "TP"


class TestIGResult:
    def test_to_json(self):
        r = IGResult(
            text="hello",
            tokens=["<s>", "hello", "</s>"],
            attributions=[0.1, 0.8, 0.1],
            convergence_delta=0.001,
            target_class=1,
            target_class_name="SUSPECT",
            n_steps=50,
            prediction_proba=[0.3, 0.7],
        )
        j = json.loads(r.to_json())
        assert j["convergence_delta"] == 0.001
        assert j["target_class_name"] == "SUSPECT"
        assert len(j["attributions"]) == 3


class TestGlobalShapResult:
    def test_to_json(self):
        r = GlobalShapResult(
            feature_names=["caps_ratio", "excl_count"],
            mean_abs_shap=[0.05, 0.12],
            sorted_indices=[1, 0],
            n_samples=100,
            n_features=2,
            model_name="GradientBoosting",
        )
        j = json.loads(r.to_json())
        assert j["model_name"] == "GradientBoosting"
        assert j["n_samples"] == 100


# ============================================================
#  MetaLearnerDecomposer
# ============================================================


class TestMetaLearnerDecomposer:
    def test_decompose_linear(self):
        """Decompose should work with a mock linear model."""
        mock_model = MagicMock()
        mock_model.coef_ = np.array([[0.5, -0.3, 0.2, 0.1, -0.05, 0.08, 0.04]])
        mock_model.intercept_ = np.array([-0.1])

        meta_data = {
            'meta_model': mock_model,
            'feature_names': [
                'score_v5', 'score_v6', 'score_cam',
                'disagreement_v5_v6', 'disagreement_v5_cam',
                'interaction_v5_v6', 'min_fiable',
            ],
        }

        decomposer = MetaLearnerDecomposer(meta_data)

        x = np.array([0.8, 0.3, 0.7, 0.1, 0.05, 0.24, 0.7])
        result = decomposer.decompose(x)

        assert hasattr(result, 'contributions')
        assert hasattr(result, 'logit')
        assert hasattr(result, 'proba_suspect')
        assert len(result.contributions) == 7

    def test_decompose_no_coef_raises(self):
        """Should raise ValueError for model without coef_."""
        mock_model = MagicMock(spec=[])  # no coef_ attribute
        if hasattr(mock_model, 'coef_'):
            del mock_model.coef_

        meta_data = {'meta_model': mock_model}

        with pytest.raises((ValueError, AttributeError)):
            MetaLearnerDecomposer(meta_data)


# ============================================================
#  SHAPGlobalExplainer — unit tests (mocked)
# ============================================================


class TestSHAPGlobalExplainerMocked:
    def test_init_stores_params(self):
        """Init should store model and feature_names."""
        from explainability.shap_global import GlobalShapExplainer
        import tempfile
        explainer = GlobalShapExplainer(
            model=MagicMock(),
            feature_names=['a', 'b', 'c'],
            output_dir=tempfile.mkdtemp(),
        )
        assert explainer.feature_names == ['a', 'b', 'c']
        assert explainer.class_index == 1

    def test_explain_feature_mismatch_raises(self):
        """Should raise ValueError when X columns != feature_names."""
        from explainability.shap_global import GlobalShapExplainer
        import tempfile
        explainer = GlobalShapExplainer(
            model=MagicMock(),
            feature_names=['a', 'b'],
            output_dir=tempfile.mkdtemp(),
        )
        with pytest.raises(ValueError, match="Mismatch"):
            explainer.explain(np.zeros((10, 5)))


# ============================================================
#  CamembertAttentionExplainer — init guard
# ============================================================


class TestAttentionExplainerInit:
    def test_init_not_loaded_raises(self):
        """Should raise RuntimeError when classifier not loaded."""
        from explainability.attention_viz import CamembertAttentionExplainer
        mock_clf = MagicMock()
        mock_clf._loaded = False
        with pytest.raises(RuntimeError, match="non charg"):
            CamembertAttentionExplainer(classifier=mock_clf)


# ============================================================
#  IGExplainer — init guard
# ============================================================


class TestIGExplainerInit:
    def test_init_not_loaded_raises(self):
        """Should raise RuntimeError when classifier not loaded."""
        from explainability.integrated_gradients import IGExplainer
        mock_clf = MagicMock()
        mock_clf._loaded = False
        with pytest.raises(RuntimeError, match="non charg"):
            IGExplainer(classifier=mock_clf)


# ============================================================
#  MetaDecomposition dataclass
# ============================================================


class TestMetaDecomposition:
    def test_top_drivers(self):
        from explainability.meta_decomposition import MetaDecomposition
        d = MetaDecomposition(
            feature_names=['a', 'b', 'c'],
            feature_values=[0.5, 0.3, 0.8],
            coefficients=[1.0, -0.5, 0.2],
            contributions=[0.5, -0.15, 0.16],
            intercept=-0.1,
            logit=0.41,
            proba_suspect=0.6,
            label='SUSPECT',
        )
        top = d.top_drivers(2)
        assert len(top) == 2
        assert top[0]['feature'] == 'a'  # highest |contribution|
        assert top[0]['direction'] == 'SUSPECT'

    def test_to_json(self):
        from explainability.meta_decomposition import MetaDecomposition
        d = MetaDecomposition(
            feature_names=['x'], feature_values=[1.0],
            coefficients=[0.5], contributions=[0.5],
            intercept=0.0, logit=0.5, proba_suspect=0.62,
            label='SUSPECT',
        )
        j = json.loads(d.to_json())
        assert j['label'] == 'SUSPECT'
        assert 'top_drivers' in j

    def test_to_plotly_bar(self):
        from explainability.meta_decomposition import MetaDecomposition, MetaLearnerDecomposer
        d = MetaDecomposition(
            feature_names=['score_v5', 'score_v6', 'score_cam'],
            feature_values=[0.8, 0.3, 0.7],
            coefficients=[0.5, -0.3, 0.2],
            contributions=[0.4, -0.09, 0.14],
            intercept=-0.1,
            logit=0.35,
            proba_suspect=0.59,
            label='SUSPECT',
        )
        fig = MetaLearnerDecomposer.to_plotly_bar(d)
        assert fig is not None


class TestMetaLearnerDecomposerExtended:
    def test_default_feature_names_v7(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer
        mock_model = MagicMock()
        mock_model.coef_ = np.array([[0.5, -0.3, 0.2, 0.1]])
        mock_model.intercept_ = np.array([-0.1])
        decomposer = MetaLearnerDecomposer({'meta_model': mock_model})
        assert len(decomposer.feature_names) == 4
        assert 'score_v5_fiable' in decomposer.feature_names

    def test_default_feature_names_v8(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer
        mock_model = MagicMock()
        mock_model.coef_ = np.array([[0.5, -0.3, 0.2, 0.1, -0.05, 0.08, 0.04]])
        mock_model.intercept_ = np.array([-0.1])
        decomposer = MetaLearnerDecomposer({'meta_model': mock_model})
        assert len(decomposer.feature_names) == 7
        assert 'score_camembert_fiable' in decomposer.feature_names

    def test_decompose_wrong_size_raises(self):
        from explainability.meta_decomposition import MetaLearnerDecomposer
        mock_model = MagicMock()
        mock_model.coef_ = np.array([[0.5, -0.3]])
        mock_model.intercept_ = np.array([0.0])
        decomposer = MetaLearnerDecomposer({'meta_model': mock_model})
        with pytest.raises(ValueError, match="features"):
            decomposer.decompose(np.array([1, 2, 3]))


# ============================================================
#  Lazy imports in explainability/__init__.py
# ============================================================


class TestExplainabilityInit:
    def test_lazy_import_global_shap(self):
        from explainability import GlobalShapExplainer
        assert GlobalShapExplainer is not None

    def test_lazy_import_meta_decomposer(self):
        from explainability import MetaLearnerDecomposer
        assert MetaLearnerDecomposer is not None

    def test_lazy_import_faithfulness(self):
        from explainability import FaithfulnessEvaluator
        assert FaithfulnessEvaluator is not None

    def test_lazy_import_unknown_raises(self):
        import explainability
        with pytest.raises(AttributeError):
            _ = explainability.NonExistentClass
