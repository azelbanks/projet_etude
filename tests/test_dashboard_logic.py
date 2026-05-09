"""
Tests pour la logique metier du dashboard (sans Streamlit).

Couvre : _normalize_mongo_df, predict_v7_hybrid, DEMO_POSTS,
EMOTION_LABELS, CSS constants, helper functions.
"""

import os
import sys
import importlib
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

# ---------------------------------------------------------------------------
#  Import the actual dashboard module with Streamlit mocked
# ---------------------------------------------------------------------------


def _get_app_module():
    """Import dashboard/app.py with Streamlit mocked out."""
    # Mock Streamlit decorators
    st_mock = MagicMock()
    st_mock.cache_resource = lambda f=None, **kw: (lambda fn: fn) if f is None else f
    st_mock.cache_data = lambda f=None, **kw: (lambda fn: fn) if f is None else f

    saved = {}
    mocks = {
        'streamlit': st_mock,
        'streamlit_authenticator': MagicMock(),
        'yaml': MagicMock(),
    }
    for mod_name, mock_obj in mocks.items():
        saved[mod_name] = sys.modules.get(mod_name)
        sys.modules[mod_name] = mock_obj

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dashboard'))

    # Force reimport
    if 'app' in sys.modules:
        del sys.modules['app']

    import app as dashboard_app
    return dashboard_app


# Import once
app = _get_app_module()


# ============================================================
#  _normalize_mongo_df
# ============================================================


class TestNormalizeMongoDf:
    def test_label_fiable(self):
        df = pd.DataFrame({'text': ['hello'], 'prediction_label': [0], 'ai_score_credibility': [0.8], 'ai_emotion': ['joie']})
        result = app._normalize_mongo_df(df)
        assert result['prediction_label'].iloc[0] == 'FIABLE'

    def test_label_suspect_int(self):
        df = pd.DataFrame({'text': ['test'], 'prediction_label': [1], 'ai_score_credibility': [0.2]})
        result = app._normalize_mongo_df(df)
        assert result['prediction_label'].iloc[0] == 'SUSPECT'

    def test_label_suspect_string(self):
        df = pd.DataFrame({'text': ['t'], 'prediction_label': ['SUSPECT']})
        result = app._normalize_mongo_df(df)
        assert result['prediction_label'].iloc[0] == 'SUSPECT'

    def test_missing_label_becomes_non_analyse(self):
        df = pd.DataFrame({'text': ['t']})
        result = app._normalize_mongo_df(df)
        assert result['prediction_label'].iloc[0] == 'NON ANALYSE'

    def test_v9_label_takes_priority(self):
        df = pd.DataFrame({'text': ['t'], 'ai_v9_label': ['SUSPECT'], 'prediction_label': ['FIABLE']})
        result = app._normalize_mongo_df(df)
        assert result['prediction_label'].iloc[0] == 'SUSPECT'

    def test_fillna_credibility(self):
        df = pd.DataFrame({'text': ['t'], 'prediction_label': [0]})
        result = app._normalize_mongo_df(df)
        assert result['ai_score_credibility'].iloc[0] == 0.5

    def test_fillna_emotion(self):
        df = pd.DataFrame({'text': ['t'], 'prediction_label': [0], 'ai_emotion': [None]})
        result = app._normalize_mongo_df(df)
        assert result['ai_emotion'].iloc[0] == 'neutre'

    def test_datetime_parsing(self):
        df = pd.DataFrame({'text': ['t'], 'prediction_label': [0], 'collected_at': ['2026-01-01T10:00:00']})
        result = app._normalize_mongo_df(df)
        assert pd.api.types.is_datetime64_any_dtype(result['collected_at'])

    def test_adds_default_language(self):
        df = pd.DataFrame({'text': ['hello world test'], 'prediction_label': [0]})
        result = app._normalize_mongo_df(df)
        assert 'ai_language' in result.columns


# ============================================================
#  DEMO_POSTS
# ============================================================


class TestDemoPosts:
    def test_count(self):
        assert len(app.DEMO_POSTS) >= 10

    def test_keys(self):
        required = {'uri', 'text', 'ai_score_credibility', 'ai_emotion', 'ai_language', 'prediction_label'}
        for post in app.DEMO_POSTS:
            assert required.issubset(post.keys()), f"Missing keys in: {post['uri']}"

    def test_labels_valid(self):
        for post in app.DEMO_POSTS:
            assert post['prediction_label'] in ('FIABLE', 'SUSPECT')

    def test_scores_in_range(self):
        for post in app.DEMO_POSTS:
            assert 0 <= post['ai_score_credibility'] <= 1

    def test_languages(self):
        langs = {p['ai_language'] for p in app.DEMO_POSTS}
        assert 'fr' in langs
        assert 'en' in langs


# ============================================================
#  Constants
# ============================================================


class TestConstants:
    def test_emotion_labels(self):
        assert len(app.EMOTION_LABELS) == 7
        assert 'neutre' in app.EMOTION_LABELS

    def test_emotion_emojis_all_present(self):
        for e in app.EMOTION_LABELS:
            assert e in app.EMOTION_EMOJIS

    def test_emotion_colors_all_present(self):
        for e in app.EMOTION_LABELS:
            assert e in app.EMOTION_COLORS

    def test_emotion_display_all_present(self):
        for e in app.EMOTION_LABELS:
            assert e in app.EMOTION_DISPLAY

    def test_css_contains_glass_card(self):
        assert 'glass-card' in app.CUSTOM_CSS

    def test_css_contains_verdict(self):
        assert 'verdict-fiable' in app.CUSTOM_CSS

    def test_default_threshold(self):
        assert app.DEFAULT_THRESHOLD_V5 == 0.44

    def test_fallback_threshold(self):
        assert app.FALLBACK_THRESHOLD_V7 == 0.42


# ============================================================
#  Helper functions
# ============================================================


class TestMetricCard:
    def test_returns_html(self):
        html = app.metric_card('icon', 'Label', '42', '#FFF')
        assert 'glass-card' in html
        assert '42' in html
        assert 'Label' in html

    def test_with_delta(self):
        html = app.metric_card('i', 'L', '10', '#FFF', delta='+5%')
        assert '+5%' in html
        assert 'metric-delta' in html

    def test_without_delta(self):
        html = app.metric_card('i', 'L', '10', '#FFF')
        assert 'metric-delta' not in html


class TestPlotlyLayout:
    def test_returns_dict(self):
        layout = app._plotly_layout()
        assert isinstance(layout, dict)
        assert 'paper_bgcolor' in layout

    def test_merges_xaxis(self):
        layout = app._plotly_layout(xaxis=dict(title='Score'))
        assert layout['xaxis']['title'] == 'Score'
        assert 'gridcolor' in layout['xaxis']  # default preserved

    def test_overrides_extra_kwargs(self):
        layout = app._plotly_layout(height=500)
        assert layout['height'] == 500


class TestMakeGauge:
    def test_high_score_green(self):
        fig = app.make_gauge(0.85)
        assert fig is not None

    def test_medium_score(self):
        fig = app.make_gauge(0.55)
        assert fig is not None

    def test_low_score(self):
        fig = app.make_gauge(0.15)
        assert fig is not None


class TestMakeRadar:
    def test_returns_figure(self):
        fig = app.make_radar([0.1, 0.2, 0.5, 0.3, 0.1, 0.05, 0.15], 'Test Radar')
        assert fig is not None


# ============================================================
#  predict_v7_hybrid
# ============================================================


class TestPredictV7Hybrid:
    def _make_mocks(self):
        detector = MagicMock()
        detector.predict.return_value = pd.DataFrame({
            'ai_score_credibility': [0.85],
            'language': ['en'],
        })

        emo = MagicMock()
        emo.get_emotion_features.return_value = np.zeros((1, 7))

        v6_model = MagicMock()
        v6_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        v6_data = {'model': v6_model, 'scaler': MagicMock(transform=lambda x: x), 'model_name': 'LogReg'}

        return detector, emo, v6_data

    def test_fallback_no_v7(self):
        detector, emo, v6_data = self._make_mocks()
        result = app.predict_v7_hybrid("test text", detector, emo, v6_data, v7_data=None)
        assert 'score_v7' in result
        assert result['version'] == 'v7_fallback'
        assert result['label_v7'] in ('FIABLE', 'SUSPECT')

    def test_with_meta_model(self):
        detector, emo, v6_data = self._make_mocks()
        meta_model = MagicMock()
        meta_model.predict_proba.return_value = np.array([[0.4, 0.6]])
        v7_data = {'meta_model': meta_model, 'uses_camembert': False, 'version': 'v7_hybrid'}

        result = app.predict_v7_hybrid("test text", detector, emo, v6_data, v7_data)
        assert result['score_v7'] == 0.6
        assert result['label_v7'] == 'SUSPECT'
        assert result['version'] == 'v7_hybrid'

    def test_with_camembert_meta(self):
        detector, emo, v6_data = self._make_mocks()
        # Override language to fr for camembert path
        detector.predict.return_value = pd.DataFrame({
            'ai_score_credibility': [0.85],
            'language': ['fr'],
        })
        meta_model = MagicMock()
        meta_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        v7_data = {'meta_model': meta_model, 'uses_camembert': True, 'version': 'v8_meta', 'camembert_suffix': 'camembert_fr'}

        cam = MagicMock()
        cam.predict_credibility_scores.return_value = [0.9]

        result = app.predict_v7_hybrid("test text", detector, emo, v6_data, v7_data, cam_classifier=cam)
        assert result['score_v7'] == 0.2
        assert result['label_v7'] == 'FIABLE'

    def test_scores_in_range(self):
        detector, emo, v6_data = self._make_mocks()
        result = app.predict_v7_hybrid("a simple test text", detector, emo, v6_data, v7_data=None)
        assert 0 <= result['score_v5'] <= 1
        assert 0 <= result['score_v6'] <= 1
        assert 0 <= result['score_v7'] <= 1

    def test_returns_lang(self):
        detector, emo, v6_data = self._make_mocks()
        result = app.predict_v7_hybrid("test", detector, emo, v6_data, v7_data=None)
        assert result['lang'] == 'en'


# ============================================================
#  get_data fallback
# ============================================================


class TestGetData:
    @patch.object(app, '_fetch_mongo_data', return_value=None)
    def test_demo_fallback(self, mock_fetch):
        df, is_demo = app.get_data()
        assert is_demo is True
        assert len(df) == len(app.DEMO_POSTS)
        assert 'prediction_label' in df.columns

    @patch.object(app, '_fetch_mongo_data')
    def test_mongo_data(self, mock_fetch):
        docs = [
            {'text': 'hello', 'ai_score_credibility': 0.8, 'prediction_label': 'FIABLE',
             'ai_emotion': 'neutre', 'ai_language': 'en', 'uri': 'x', 'collected_at': '2026-01-01'}
        ]
        stats = {'total_posts': 100, 'by_label': {}, 'by_emotion': {}, 'by_language': {}, 'avg_credibility': 0.7}
        mock_fetch.return_value = (docs, 100, stats, [])

        df, is_demo = app.get_data()
        assert is_demo is False
        assert len(df) == 1
        assert df.attrs.get('n_total_mongo') == 100


class TestApplyLayout:
    def test_apply_layout(self):
        fig = MagicMock()
        result = app._apply_layout(fig, height=300)
        fig.update_layout.assert_called_once()
        assert result is fig


class TestHeroFooter:
    def test_hero(self):
        app.hero('Title', 'Subtitle')

    def test_footer(self):
        app.footer()


class TestFetchMongoData:
    @patch.object(app, 'get_mongo_collection', return_value=None)
    def test_no_connection(self, mock_col):
        result = app._fetch_mongo_data()
        assert result is None

    @patch.object(app, '_HAS_MONGO_AGG', False)
    def test_no_mongo_module(self):
        result = app._fetch_mongo_data()
        assert result is None


class TestPredictV7Emo:
    """Test predict_v7_hybrid when emotion features fail."""

    def test_emo_failure_fallback(self):
        detector = MagicMock()
        detector.predict.return_value = pd.DataFrame({
            'ai_score_credibility': [0.85],
            'language': ['en'],
        })

        emo = MagicMock()
        emo.get_emotion_features.side_effect = Exception("emo failed")

        v6_model = MagicMock()
        v6_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        v6_data = {'model': v6_model, 'scaler': MagicMock(transform=lambda x: x), 'model_name': 'LogReg'}

        result = app.predict_v7_hybrid("test text", detector, emo, v6_data, v7_data=None)
        assert 'score_v7' in result  # should still work via fallback


class TestPredictV7Shap:
    """Test predict_v7_hybrid when SHAP is available."""

    def test_with_gradient_boosting_model(self):
        detector = MagicMock()
        detector.predict.return_value = pd.DataFrame({
            'ai_score_credibility': [0.7],
            'language': ['en'],
        })

        emo = MagicMock()
        emo.get_emotion_features.return_value = np.zeros((1, 7))

        v6_model = MagicMock()
        v6_model.predict_proba.return_value = np.array([[0.6, 0.4]])
        v6_data = {'model': v6_model, 'scaler': MagicMock(transform=lambda x: x), 'model_name': 'GradientBoosting'}

        result = app.predict_v7_hybrid("test", detector, emo, v6_data, v7_data=None)
        assert 'score_v5' in result
