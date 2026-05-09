"""
Tests etendus pour src/explainability/shap_global.py.
"""

import json
import tempfile

import numpy as np
import pytest

from explainability.shap_global import GlobalShapExplainer, GlobalShapResult


class TestGlobalShapExplainerWithModel:
    @pytest.fixture
    def mock_model(self):
        from sklearn.ensemble import GradientBoostingClassifier
        np.random.seed(42)
        X = np.random.rand(50, 5)
        y = (X[:, 0] > 0.5).astype(int)
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture
    def feature_names(self):
        return ['feat_a', 'feat_b', 'feat_c', 'feat_d', 'feat_e']

    def test_init(self, mock_model, feature_names):
        explainer = GlobalShapExplainer(
            model=mock_model,
            feature_names=feature_names,
            output_dir=tempfile.mkdtemp(),
        )
        assert explainer is not None
        assert explainer.feature_names == feature_names

    def test_explain_returns_result(self, mock_model, feature_names):
        X = np.random.rand(20, 5)
        explainer = GlobalShapExplainer(
            model=mock_model,
            feature_names=feature_names,
            output_dir=tempfile.mkdtemp(),
        )
        result = explainer.explain(X)
        assert isinstance(result, GlobalShapResult)
        assert len(result.mean_abs_shap) == 5
        assert result.n_features == 5
        assert result.n_samples == 20

    def test_result_json_valid(self, mock_model, feature_names):
        X = np.random.rand(20, 5)
        explainer = GlobalShapExplainer(
            model=mock_model,
            feature_names=feature_names,
            output_dir=tempfile.mkdtemp(),
        )
        result = explainer.explain(X)
        j = json.loads(result.to_json())
        assert 'feature_names' in j
        assert len(j['feature_names']) == 5

    def test_sorted_indices_descending(self, mock_model, feature_names):
        X = np.random.rand(30, 5)
        explainer = GlobalShapExplainer(
            model=mock_model,
            feature_names=feature_names,
            output_dir=tempfile.mkdtemp(),
        )
        result = explainer.explain(X)
        for i in range(len(result.sorted_indices) - 1):
            idx_a = result.sorted_indices[i]
            idx_b = result.sorted_indices[i + 1]
            assert result.mean_abs_shap[idx_a] >= result.mean_abs_shap[idx_b]


class TestGlobalShapResultSerialization:
    def test_to_json(self):
        r = GlobalShapResult(
            feature_names=["a", "b"],
            mean_abs_shap=[0.1, 0.2],
            sorted_indices=[1, 0],
            n_samples=10,
            n_features=2,
            model_name="TestModel",
        )
        j = json.loads(r.to_json())
        assert j['model_name'] == 'TestModel'
        assert j['n_samples'] == 10

    def test_figures_default_empty(self):
        r = GlobalShapResult(
            feature_names=[], mean_abs_shap=[], sorted_indices=[],
            n_samples=0, n_features=0, model_name="test",
        )
        assert r.figures == {}
