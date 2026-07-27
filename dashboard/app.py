"""
ThumaCheck -- Intelligence Command Center
========================================
Dashboard Streamlit de detection de fake news bilingue FR/EN.
Pipeline V9 : filtre fait/opinion + V5 (TF-IDF) + V6 (Style) + CamemBERT + SHAP.
5 pages : Dashboard, Analyse IA, Explorateur, Performance, A propos.
"""

import os
import sys
import logging
import re
import html
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib
import yaml
import streamlit_authenticator as stauth

# ---------------------------------------------------------------------------
#  MongoDB aggregation helpers (graceful fallback if unavailable)
# ---------------------------------------------------------------------------

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from pipeline.mongo_aggregations import (
        get_mongo_collection,
        get_overview_stats,
        get_recent_posts,
        get_score_distribution,
    )
    _HAS_MONGO_AGG = True
except ImportError:
    _HAS_MONGO_AGG = False

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

# ---------------------------------------------------------------------------
#  Seuils de decision
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLD_V5 = 0.44
FALLBACK_THRESHOLD_V7 = 0.42

# ---------------------------------------------------------------------------
#  CSS — Professional dark theme with glassmorphism
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ---- Root variables ---- */
:root {
    --primary: #00D4FF;
    --success: #00E676;
    --danger: #FF1744;
    --warning: #FF9100;
    --accent: #FFD600;
    --bg-card: rgba(26, 31, 46, 0.85);
    --bg-hover: rgba(0, 212, 255, 0.05);
    --text-primary: #E8E8E8;
    --text-secondary: #B0B0B0;
    --border: rgba(0, 212, 255, 0.12);
    --glow: rgba(0, 212, 255, 0.25);
}

/* ---- glassmorphism card ---- */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    margin-bottom: 12px;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(0, 212, 255, 0.25);
    box-shadow: 0 8px 32px rgba(0, 212, 255, 0.08);
}

/* ---- titles ---- */
h1, h2, h3 { text-shadow: 0 0 20px var(--glow); }

/* ---- metric cards ---- */
.metric-value { font-size: 2.4rem; font-weight: 700; line-height: 1.1; }
.metric-label { font-size: 0.85rem; color: var(--text-secondary); margin-top: 6px; }
.metric-icon { font-size: 1.3rem; margin-bottom: 6px; }
.metric-delta { font-size: 0.75rem; margin-top: 4px; }

/* ---- verdict badges ---- */
.verdict-fiable {
    background: rgba(0, 230, 118, 0.08);
    border: 1px solid rgba(0, 230, 118, 0.25);
    border-radius: 16px; padding: 24px; text-align: center;
}
.verdict-suspect {
    background: rgba(255, 23, 68, 0.08);
    border: 1px solid rgba(255, 23, 68, 0.25);
    border-radius: 16px; padding: 24px; text-align: center;
}

/* ---- dividers ---- */
hr { border-color: var(--border) !important; }

/* ---- dataframe styling ---- */
.stDataFrame [data-testid="stDataFrameResizable"] {
    border-radius: 12px; overflow: hidden;
}

/* ---- primary button ---- */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00D4FF, #0066FF);
    border: none; border-radius: 12px;
    font-weight: 600; padding: 0.6rem 1.2rem;
    transition: all 0.3s ease;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #00B8E6, #0055DD);
    box-shadow: 0 4px 16px rgba(0, 212, 255, 0.3);
}

/* ---- sidebar polish ---- */
section[data-testid="stSidebar"] {
    border-right: 1px solid var(--border);
}

/* ---- status indicator ---- */
.status-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ---- footer ---- */
.footer-text {
    text-align: center; color: var(--text-secondary);
    font-size: 0.78rem; padding: 16px 0;
}

/* ---- tab styling ---- */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0; padding: 8px 16px;
}

/* ---- kpi row ---- */
.kpi-row { display: flex; gap: 12px; margin-bottom: 16px; }

/* ---- Focus indicators for keyboard navigation (WCAG 2.1) ---- */
*:focus-visible {
    outline: 2px solid #00D4FF !important;
    outline-offset: 2px !important;
}
button:focus-visible, [role="button"]:focus-visible {
    outline: 3px solid #00D4FF !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 8px rgba(0,212,255,0.4) !important;
}
</style>
"""

# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_pipeline():
    """Charge le pipeline expert V5 (fallback V4, V3, V2) et l'extracteur d'emotions."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from pipeline.expert_detector import ExpertFakeNewsDetector, EmotionFeatureExtractor

    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    detector = ExpertFakeNewsDetector(model_dir=model_dir, threshold=DEFAULT_THRESHOLD_V5)
    v5_exists = os.path.exists(os.path.join(model_dir, 'model_expert_v5.pkl'))
    v4_exists = os.path.exists(os.path.join(model_dir, 'model_expert_v4.pkl'))
    v3_exists = os.path.exists(os.path.join(model_dir, 'model_expert_v3.pkl'))
    v2_exists = os.path.exists(os.path.join(model_dir, 'model_expert_v2.pkl'))
    suffix = 'expert_v5' if v5_exists else ('expert_v4' if v4_exists else ('expert_v3' if v3_exists else ('expert_v2' if v2_exists else 'expert')))
    detector.load(suffix=suffix)

    hc = detector.health_check()
    if not hc['healthy']:
        logging.getLogger(__name__).warning('Model health check FAILED (suffix=%s): %s', suffix, hc['details'])

    emo = EmotionFeatureExtractor(model_dir=model_dir)
    emo.load()
    return detector, emo, suffix


# ---------------------------------------------------------------------------
#  V6 Style Feature Extractor (topic-agnostic, 28 features)
# ---------------------------------------------------------------------------

from pipeline.style_features import StyleFeatureExtractorV6  # noqa: E402




# ---------------------------------------------------------------------------
#  V6/V7/V8/V9 model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_v6_v7():
    """Charge les modeles V6 (style), V7/V8 (meta-learner), CamemBERT et Stage1."""
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    v6_data, v7_data, cam_classifier = None, None, None

    v6_path = os.path.join(model_dir, 'model_style_v6.joblib')
    if os.path.exists(v6_path):
        v6_data = joblib.load(v6_path)

    v8_path = os.path.join(model_dir, 'model_hybrid_v8.joblib')
    v7_path = os.path.join(model_dir, 'model_hybrid_v7.joblib')
    if os.path.exists(v8_path):
        v7_data = joblib.load(v8_path)
    elif os.path.exists(v7_path):
        v7_data = joblib.load(v7_path)

    if v7_data is not None and v7_data.get('uses_camembert', False):
        try:
            from pipeline.camembert_classifier import CamemBERTClassifier
            cam_classifier = CamemBERTClassifier(model_dir=model_dir)
            cam_suffix = v7_data.get('camembert_suffix', 'camembert_fr')
            if not cam_classifier.load(suffix=cam_suffix):
                cam_classifier = None
        except Exception as e:
            logging.warning("CamemBERT non disponible: %s", e)
            cam_classifier = None

    stage1_data = None
    stage1_path = os.path.join(model_dir, 'stage1_fact_opinion.joblib')
    if os.path.exists(stage1_path):
        try:
            stage1_data = joblib.load(stage1_path)
        except Exception as e:
            logging.warning("Stage 1 fait/opinion non disponible: %s", e)

    return v6_data, v7_data, cam_classifier, stage1_data


def predict_v7_hybrid(text_input, detector, emo, v6_data, v7_data, cam_classifier=None):
    """Calcule le score hybride V7/V8 pour un texte unique."""
    texts = pd.Series([text_input])
    v5_result = detector.predict(texts)
    score_v5 = float(v5_result['ai_score_credibility'].iloc[0])
    lang = v5_result['language'].iloc[0] if 'language' in v5_result.columns else 'en'

    v6_model = v6_data['model']
    v6_scaler = v6_data['scaler']
    v6_model_name = v6_data['model_name']

    X_style = StyleFeatureExtractorV6.extract(texts)
    try:
        X_emo = emo.get_emotion_features([text_input])
        X_all = np.hstack([X_style, X_emo])
    except Exception:
        X_all = X_style

    X_input = v6_scaler.transform(X_all) if v6_model_name == 'LogReg' else X_all
    score_v6 = float(v6_model.predict_proba(X_input)[:, 1][0])

    score_cam = 0.5
    if cam_classifier is not None and lang == 'fr':
        try:
            score_cam = float(cam_classifier.predict_credibility_scores([text_input])[0])
        except Exception:
            pass

    disagreement_v5_v6 = abs(score_v5 - (1 - score_v6))
    interaction_v5_v6 = score_v5 * score_v6

    result = {
        'score_v5': score_v5, 'score_v6': score_v6, 'score_cam': score_cam,
        'disagreement': disagreement_v5_v6, 'X_input': X_input,
        'shap_values': None, 'feature_names': None, 'lang': lang,
    }

    if v7_data is not None:
        meta_model = v7_data['meta_model']
        if v7_data.get('uses_camembert', False):
            disagreement_v5_cam = abs(score_v5 - score_cam)
            min_fiable = min(score_v5, score_cam)
            X_meta = np.array([[score_v5, score_v6, score_cam,
                                disagreement_v5_v6, disagreement_v5_cam,
                                interaction_v5_v6, min_fiable]])
        else:
            X_meta = np.array([[score_v5, score_v6, disagreement_v5_v6, interaction_v5_v6]])

        score_v7 = float(meta_model.predict_proba(X_meta)[:, 1][0])
        result['score_v7'] = score_v7
        result['label_v7'] = 'SUSPECT' if score_v7 >= 0.5 else 'FIABLE'
        result['version'] = v7_data.get('version', 'v7_hybrid')
        # Persist for explainability decomposition (XAI section ci-dessous)
        result['X_meta'] = X_meta
        result['v7_data'] = v7_data
    else:
        combined = score_v5 * (1 - score_v6)
        result['score_v7'] = 1 - combined
        result['label_v7'] = 'SUSPECT' if combined < FALLBACK_THRESHOLD_V7 else 'FIABLE'
        result['version'] = 'v7_fallback'

    if _HAS_SHAP and v6_model_name in ('GradientBoosting', 'RandomForest'):
        try:
            explainer = shap.TreeExplainer(v6_model)
            sv = explainer.shap_values(X_input)
            all_names = StyleFeatureExtractorV6.FEATURE_NAMES + [
                'emo_anger', 'emo_disgust', 'emo_joy', 'emo_neutral',
                'emo_fear', 'emo_surprise', 'emo_sadness',
            ]
            result['shap_values'] = sv[0]
            result['feature_names'] = all_names[:X_input.shape[1]]
            result['feature_values'] = X_input[0]
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
#  Data loading (MongoDB -> fallback demo)
# ---------------------------------------------------------------------------

DEMO_POSTS = [
    {"uri": "at://en1", "text": "New study published in Nature confirms the effectiveness of the updated vaccine formula for 2025.", "ai_score_credibility": 0.91, "ai_emotion": "neutre", "ai_language": "en", "prediction_label": "FIABLE", "collected_at": "2026-04-28T10:00:00", "search_term": "vaccine"},
    {"uri": "at://en2", "text": "The Federal Reserve announced a quarter-point rate cut today, citing stable inflation data.", "ai_score_credibility": 0.88, "ai_emotion": "neutre", "ai_language": "en", "prediction_label": "FIABLE", "collected_at": "2026-04-28T11:00:00", "search_term": "breaking news"},
    {"uri": "at://en3", "text": "Researchers at MIT developed a new carbon capture method that reduces costs by 40 percent.", "ai_score_credibility": 0.85, "ai_emotion": "joie", "ai_language": "en", "prediction_label": "FIABLE", "collected_at": "2026-04-27T09:00:00", "search_term": "climate change"},
    {"uri": "at://en4", "text": "The WHO report shows global life expectancy increased by 2 years over the past decade.", "ai_score_credibility": 0.82, "ai_emotion": "joie", "ai_language": "en", "prediction_label": "FIABLE", "collected_at": "2026-04-27T14:00:00", "search_term": "community"},
    {"uri": "at://en5", "text": "EXPOSED: Secret government labs are using 5G towers to spread mind-control chemicals!!!", "ai_score_credibility": 0.12, "ai_emotion": "colere", "ai_language": "en", "prediction_label": "SUSPECT", "collected_at": "2026-04-28T08:00:00", "search_term": "conspiracy"},
    {"uri": "at://en6", "text": "BREAKING: Celebrities are being replaced by clones. Wake up sheeple! Share before they delete this!", "ai_score_credibility": 0.08, "ai_emotion": "peur", "ai_language": "en", "prediction_label": "SUSPECT", "collected_at": "2026-04-26T16:00:00", "search_term": "wake up"},
    {"uri": "at://en7", "text": "Big Pharma doesn't want you to know this ONE trick that cures all diseases overnight!", "ai_score_credibility": 0.15, "ai_emotion": "surprise", "ai_language": "en", "prediction_label": "SUSPECT", "collected_at": "2026-04-26T12:00:00", "search_term": "exposed"},
    {"uri": "at://en8", "text": "They're hiding the REAL numbers! The economy already collapsed, media is lying to you!!!", "ai_score_credibility": 0.11, "ai_emotion": "colere", "ai_language": "en", "prediction_label": "SUSPECT", "collected_at": "2026-04-25T18:00:00", "search_term": "they lied"},
    {"uri": "at://fr1", "text": "Le CNRS publie une étude confirmant l'efficacité des nouveaux traitements contre l'hépatite C.", "ai_score_credibility": 0.89, "ai_emotion": "neutre", "ai_language": "fr", "prediction_label": "FIABLE", "collected_at": "2026-04-28T09:30:00", "search_term": "santé"},
    {"uri": "at://fr2", "text": "La BCE maintient ses taux directeurs inchangés, conformément aux attentes du marché.", "ai_score_credibility": 0.92, "ai_emotion": "neutre", "ai_language": "fr", "prediction_label": "FIABLE", "collected_at": "2026-04-27T15:00:00", "search_term": "économie"},
    {"uri": "at://fr3", "text": "L'équipe de France de handball remporte le championnat du monde pour la quatrième fois.", "ai_score_credibility": 0.87, "ai_emotion": "joie", "ai_language": "fr", "prediction_label": "FIABLE", "collected_at": "2026-04-27T20:00:00", "search_term": "communauté"},
    {"uri": "at://fr4", "text": "Adoption définitive de la loi climat par le Parlement, avec 80 pourcent de votes favorables.", "ai_score_credibility": 0.84, "ai_emotion": "joie", "ai_language": "fr", "prediction_label": "FIABLE", "collected_at": "2026-04-26T10:00:00", "search_term": "politique"},
    {"uri": "at://fr5", "text": "SCANDALE : le gouvernement cache la VÉRITÉ sur les chemtrails ! Partagez avant censure !!!", "ai_score_credibility": 0.09, "ai_emotion": "colere", "ai_language": "fr", "prediction_label": "SUSPECT", "collected_at": "2026-04-28T07:00:00", "search_term": "complot"},
    {"uri": "at://fr6", "text": "Les vaccins contiennent des micropuces 5G pour vous contrôler ! Réveillez-vous !!!", "ai_score_credibility": 0.07, "ai_emotion": "peur", "ai_language": "fr", "prediction_label": "SUSPECT", "collected_at": "2026-04-25T11:00:00", "search_term": "vaccin"},
    {"uri": "at://fr7", "text": "ON VOUS MENT : cette plante guérit le cancer en 3 jours, les labos ne veulent pas que vous sachiez !", "ai_score_credibility": 0.13, "ai_emotion": "surprise", "ai_language": "fr", "prediction_label": "SUSPECT", "collected_at": "2026-04-24T14:00:00", "search_term": "on nous cache"},
]


def _normalize_mongo_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les champs MongoDB legacy pour le dashboard."""
    # V9 : utiliser ai_v9_label si disponible (filtre fait/opinion)
    if 'ai_v9_label' in df.columns:
        df['prediction_label'] = df['ai_v9_label'].fillna(df.get('prediction_label', np.nan))
    if 'prediction_label' not in df.columns:
        df['prediction_label'] = np.nan
    if 'ai_score_credibility' not in df.columns:
        df['ai_score_credibility'] = np.nan
    if 'ai_emotion' not in df.columns:
        df['ai_emotion'] = np.nan

    def _norm_label(v):
        if pd.isna(v):
            return 'NON ANALYSE'
        if v == 1 or str(v).upper() == 'SUSPECT':
            return 'SUSPECT'
        return 'FIABLE'
    df['prediction_label'] = df['prediction_label'].apply(_norm_label)
    df['ai_score_credibility'] = df['ai_score_credibility'].fillna(0.5)
    df['ai_emotion'] = (
        df['ai_emotion'].fillna('neutre').astype(str)
        .str.split().str[0].str.lower().str.strip()
    )

    if 'ai_language' not in df.columns:
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0
            def _detect_safe(text):
                try:
                    lang = detect(str(text)[:500])
                    return 'fr' if lang == 'fr' else 'en'
                except Exception:
                    return 'en'
            df['ai_language'] = df['text'].apply(_detect_safe)
        except ImportError:
            df['ai_language'] = 'en'

    if 'collected_at' in df.columns:
        df['collected_at'] = pd.to_datetime(df['collected_at'], errors='coerce')

    return df


@st.cache_data(ttl=60)
def _fetch_mongo_data():
    """Fetch recent posts from MongoDB. Cached 60s."""
    if not _HAS_MONGO_AGG:
        return None
    collection = get_mongo_collection()
    if collection is None:
        return None
    docs = get_recent_posts(collection, limit=2000)
    if not docs:
        return None
    stats = get_overview_stats(collection)
    n_total = stats.get("total_posts", len(docs))
    score_dist = get_score_distribution(collection, bins=20)
    return docs, n_total, stats, score_dist


def get_data():
    """Tente MongoDB, sinon donnees demo."""
    result = _fetch_mongo_data()
    if result is not None:
        docs, n_total, stats, score_dist = result
        df = pd.DataFrame(docs)
        df = _normalize_mongo_df(df)
        df.attrs['n_total_mongo'] = n_total
        df.attrs['mongo_stats'] = stats
        df.attrs['score_dist'] = score_dist
        return df, False
    df = pd.DataFrame(DEMO_POSTS)
    df = _normalize_mongo_df(df)
    return df, True


# ---------------------------------------------------------------------------
#  Emotion helpers
# ---------------------------------------------------------------------------

EMOTION_LABELS = ['colere', 'degout', 'joie', 'neutre', 'peur', 'surprise', 'tristesse']
EMOTION_EMOJIS = {
    'colere': '\U0001f621', 'degout': '\U0001f922', 'joie': '\U0001f60a',
    'neutre': '\U0001f610', 'peur': '\U0001f628', 'surprise': '\U0001f62e', 'tristesse': '\U0001f622',
}
EMOTION_COLORS = {
    'colere': '#FF1744', 'degout': '#9C27B0', 'joie': '#FFD600',
    'neutre': '#607D8B', 'peur': '#FF9100', 'surprise': '#00BCD4', 'tristesse': '#2196F3',
}
EMOTION_DISPLAY = {
    'colere': 'Colère', 'degout': 'Dégoût', 'joie': 'Joie',
    'neutre': 'Neutre', 'peur': 'Peur', 'surprise': 'Surprise', 'tristesse': 'Tristesse',
}


# ---------------------------------------------------------------------------
#  Reusable chart components
# ---------------------------------------------------------------------------

def _plotly_layout(**kwargs):
    """Base Plotly layout for dark theme. Deep-merges xaxis/yaxis/margin dicts."""
    grid = 'rgba(255,255,255,0.06)'
    defaults = dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        xaxis=dict(gridcolor=grid),
        yaxis=dict(gridcolor=grid),
        margin=dict(t=50, b=40, l=50, r=20),
    )
    for key in ('xaxis', 'yaxis', 'margin'):
        if key in kwargs and isinstance(kwargs[key], dict):
            defaults[key] = {**defaults[key], **kwargs.pop(key)}
    defaults.update(kwargs)
    return defaults


def _apply_layout(fig, **kwargs):
    """Apply _plotly_layout with overrides to a figure."""
    fig.update_layout(**_plotly_layout(**kwargs))
    return fig


def make_radar(values, title, fill_opacity=0.15):
    """Plotly radar chart for 7 emotion probabilities."""
    labels = [EMOTION_DISPLAY[e] for e in EMOTION_LABELS]
    vals = list(values) + [values[0]]
    labels_closed = labels + [labels[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=labels_closed, fill='toself',
        fillcolor=f'rgba(0, 212, 255, {fill_opacity})',
        line=dict(color='#00D4FF', width=2), name='Emotions',
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(color='#E0E0E0', size=15)),
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='#E0E0E0'),
            radialaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='#E0E0E0',
                            range=[0, max(max(values), 0.5)]),
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'), showlegend=False,
        margin=dict(t=50, b=30, l=60, r=60), height=380,
    )
    return fig


def make_gauge(score):
    """Plotly gauge chart for credibility score."""
    bar_color = '#00E676' if score >= 0.7 else ('#FF9100' if score >= 0.4 else '#FF1744')
    fig = go.Figure(go.Indicator(
        mode='gauge+number', value=score,
        number=dict(font=dict(size=48, color='#E0E0E0'), valueformat='.2f'),
        title=dict(text='Score de Crédibilité', font=dict(color='#E0E0E0', size=15)),
        gauge=dict(
            axis=dict(range=[0, 1], tickfont=dict(color='#E0E0E0'), dtick=0.2),
            bar=dict(color=bar_color, thickness=0.75), bgcolor='#1A1F2E', borderwidth=0,
            steps=[
                dict(range=[0, 0.4], color='rgba(255, 23, 68, 0.12)'),
                dict(range=[0.4, 0.7], color='rgba(255, 145, 0, 0.12)'),
                dict(range=[0.7, 1], color='rgba(0, 230, 118, 0.12)'),
            ],
        ),
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#E0E0E0'), height=260, margin=dict(t=50, b=10, l=40, r=40))
    return fig


def hero(title, subtitle):
    st.markdown(f"""
    <div role="banner" aria-label="{html.escape(title)}" style="background: linear-gradient(135deg, #0E1117 0%, #1A1F2E 50%, #0E1117 100%);
                padding: 36px 40px; border-radius: 16px; margin-bottom: 20px;
                border: 1px solid rgba(0, 212, 255, 0.1);
                position: relative; overflow: hidden;">
        <div style="position:absolute;top:0;right:0;width:200px;height:200px;
                    background:radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);" aria-hidden="true"></div>
        <div style="font-size: 2.6rem; font-weight: 300; letter-spacing: 6px;
                    color: #00D4FF; text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);">
            {html.escape(title)}
        </div>
        <div style="font-size: 1rem; color: #B0B0B0; margin-top: 6px;">
            {html.escape(subtitle)}
        </div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(icon, label, value, color, delta=None):
    delta_html = f'<div class="metric-delta" style="color:{color};">{html.escape(str(delta))}</div>' if delta else ''
    return f"""
    <div class="glass-card" role="status" aria-label="{html.escape(str(label))}: {html.escape(str(value))}" style="text-align:center; min-height:130px;">
        <div class="metric-icon" aria-hidden="true">{icon}</div>
        <div class="metric-value" style="color:{color};">{html.escape(str(value))}</div>
        <div class="metric-label">{html.escape(str(label))}</div>
        {delta_html}
    </div>
    """


def footer():
    st.divider()
    st.markdown(
        '<div class="footer-text" role="contentinfo" aria-label="Informations sur le pipeline ThumaCheck">'
        'ThumaCheck v9.0 &mdash; Pipeline fait/opinion + V5+V6+CamemBERT + SHAP '
        '&mdash; Seuil 0.44 &mdash; WCAG 2.1 AA &mdash; '
        'Descriptions textuelles sur toutes les visualisations'
        '</div>',
        unsafe_allow_html=True,
    )


# ===================================================================
#  PAGE 1 : Dashboard (Vue Globale amelioree)
# ===================================================================

def page_dashboard(df, mongo_stats, score_dist, is_demo):
    hero('THUMALIEN', 'Intelligence Center — Détection de Fake News Bilingue FR/EN')

    n_displayed = len(df)
    n_total = df.attrs.get('n_total_mongo', n_displayed)

    # Use mongo stats if available, else compute from df
    if mongo_stats and not is_demo:
        by_label = mongo_stats.get('by_label', {})
        by_emotion = mongo_stats.get('by_emotion', {})
        by_language = mongo_stats.get('by_language', {})
        avg_cred = mongo_stats.get('avg_credibility', 0.5)
        n_fiable = by_label.get('FIABLE', 0) + by_label.get(0, 0)
        n_suspect = by_label.get('SUSPECT', 0) + by_label.get(1, 0)
        n_fr = by_language.get('fr', 0)
        n_en = by_language.get('en', 0)
    else:
        n_fiable = (df['prediction_label'] == 'FIABLE').sum()
        n_suspect = (df['prediction_label'] == 'SUSPECT').sum()
        avg_cred = df['ai_score_credibility'].mean()
        n_fr = (df['ai_language'] == 'fr').sum()
        n_en = (df['ai_language'] == 'en').sum()
        by_emotion = df['ai_emotion'].value_counts().to_dict()

    pct_fiable = n_fiable / (n_fiable + n_suspect) * 100 if (n_fiable + n_suspect) else 0

    # --- KPI cards ---
    cols = st.columns(5)
    kpis = [
        metric_card('', 'Posts collectés', f'{n_total:,}', '#00D4FF'),
        metric_card('', 'Taux fiabilité', f'{pct_fiable:.1f}%', '#00E676'),
        metric_card('', 'Crédibilité moy.', f'{avg_cred:.0%}', '#FFD600'),
        metric_card('', 'Posts FR', f'{n_fr:,}', '#00D4FF'),
        metric_card('', 'Posts EN', f'{n_en:,}', '#00D4FF'),
    ]
    for col, kpi in zip(cols, kpis):
        col.markdown(kpi, unsafe_allow_html=True)

    if is_demo:
        st.info('Mode démonstration — MongoDB non connecté. Données simulées.')

    st.markdown('---')

    # --- Row 2: Score distribution + Emotion profile ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Distribution des scores de crédibilité')
        if score_dist:
            bins_x = [f"{d['bin_start']:.2f}" for d in score_dist]
            bins_y = [d['count'] for d in score_dist]
            colors = ['#FF1744' if d['bin_start'] < 0.4 else ('#FF9100' if d['bin_start'] < 0.7 else '#00E676') for d in score_dist]
            fig_dist = go.Figure(go.Bar(x=bins_x, y=bins_y, marker_color=colors))
            fig_dist.update_layout(
                **_plotly_layout(height=350),
                xaxis_title='Score de crédibilité',
                yaxis_title='Nombre de posts',
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption("Distribution des scores sur l'ensemble du corpus. "
                       "Rouge : suspect (<0.4), orange : incertain (0.4-0.7), vert : fiable (>0.7).")
        else:
            # Compute from df
            fig_dist = go.Figure(go.Histogram(
                x=df['ai_score_credibility'], nbinsx=20,
                marker_color='#00D4FF', opacity=0.8,
            ))
            fig_dist.update_layout(**_plotly_layout(height=350),
                                   xaxis_title='Score', yaxis_title='Nombre')
            st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        st.subheader('Profil émotionnel du corpus')
        # Build emotion data
        emo_data = []
        for e in EMOTION_LABELS:
            count = by_emotion.get(e, 0)
            emo_data.append({'emotion': EMOTION_DISPLAY[e], 'count': count,
                             'color': EMOTION_COLORS[e]})
        emo_df = pd.DataFrame(emo_data)
        total_emo = emo_df['count'].sum()
        if total_emo > 0:
            emo_df['pct'] = emo_df['count'] / total_emo * 100
        else:
            emo_df['pct'] = 0

        fig_emo = go.Figure(go.Bar(
            x=emo_df['emotion'], y=emo_df['pct'],
            marker_color=[EMOTION_COLORS[e] for e in EMOTION_LABELS],
            text=[f'{p:.1f}%' for p in emo_df['pct']],
            textposition='outside', textfont=dict(color='#E0E0E0', size=11),
        ))
        _apply_layout(fig_emo, height=350,
                      yaxis=dict(title='Pourcentage (%)',
                                 range=[0, max(emo_df['pct'].max() * 1.2, 10)]))
        st.plotly_chart(fig_emo, use_container_width=True)
        st.caption("Répartition des émotions dominantes sur l'ensemble du corpus "
                   "(7 classes prédites par le MLP PyTorch bilingue).")

    st.markdown('---')

    # --- Row 3: Fiabilite par langue + Radar ---
    col3, col4 = st.columns(2)

    with col3:
        st.subheader('Fiabilité par langue')
        fiable_en = ((df['ai_language'] == 'en') & (df['prediction_label'] == 'FIABLE')).sum()
        suspect_en = ((df['ai_language'] == 'en') & (df['prediction_label'] == 'SUSPECT')).sum()
        fiable_fr = ((df['ai_language'] == 'fr') & (df['prediction_label'] == 'FIABLE')).sum()
        suspect_fr = ((df['ai_language'] == 'fr') & (df['prediction_label'] == 'SUSPECT')).sum()

        fig_lang = go.Figure()
        fig_lang.add_trace(go.Bar(y=['EN', 'FR'], x=[fiable_en, fiable_fr],
                                  name='Fiable', orientation='h', marker_color='#00E676'))
        fig_lang.add_trace(go.Bar(y=['EN', 'FR'], x=[suspect_en, suspect_fr],
                                  name='Suspect', orientation='h', marker_color='#FF1744'))
        fig_lang.update_layout(
            **_plotly_layout(height=300),
            barmode='group',
            legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
        )
        st.plotly_chart(fig_lang, use_container_width=True)
        st.caption("Nombre de posts classés fiables vs suspects pour chaque langue.")

    with col4:
        st.subheader('Radar émotionnel moyen')
        if total_emo > 0:
            avg_vals = [emo_df.loc[emo_df['emotion'] == EMOTION_DISPLAY[e], 'pct'].values[0] / 100
                        for e in EMOTION_LABELS]
        else:
            avg_vals = [1/7] * 7
        fig_radar = make_radar(avg_vals, '')
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("Profil émotionnel moyen sur l'ensemble du corpus.")

    st.markdown('---')

    # --- Recent posts table ---
    st.subheader('Derniers posts analysés')
    display_df = df[['text', 'ai_language', 'prediction_label', 'ai_score_credibility', 'ai_emotion']].copy()
    display_df.columns = ['Texte', 'Langue', 'Label', 'Score', 'Emotion']
    display_df['Texte'] = display_df['Texte'].astype(str).str[:150]
    display_df['Langue'] = display_df['Langue'].str.upper()
    display_df['Emotion'] = display_df['Emotion'].map(
        lambda e: f'{EMOTION_EMOJIS.get(e, "")} {EMOTION_DISPLAY.get(e, e)}'
    )

    def color_label(val):
        if val == 'FIABLE':
            return 'color: #00E676; font-weight: 600'
        if val == 'SUSPECT':
            return 'color: #FF1744; font-weight: 600'
        return ''

    styled = (
        display_df.head(100).style
        .map(color_label, subset=['Label'])
        .background_gradient(subset=['Score'], cmap='RdYlGn', vmin=0, vmax=1)
        .format({'Score': '{:.2f}'})
    )
    st.dataframe(styled, hide_index=True, use_container_width=True, height=400)


# ===================================================================
#  PAGE 2 : Analyse IA (temps reel)
# ===================================================================

def page_analyse(detector, emo, v6_data, v7_data, cam_classifier, stage1_data):
    hero('Analyse IA', 'Analysez un texte ou un lot de textes avec le pipeline V9 complet')

    tab_single, tab_batch = st.tabs(['Analyse unitaire', 'Analyse par lot (CSV)'])

    with tab_single:
        _page_single_analysis(detector, emo, v6_data, v7_data, cam_classifier, stage1_data)

    with tab_batch:
        _page_batch_analysis(detector, emo)


def _page_single_analysis(detector, emo, v6_data, v7_data, cam_classifier, stage1_data):
    text_input = st.text_area(
        label='Texte à analyser',
        height=150,
        placeholder='Collez ici un article, un post Bluesky, ou tout texte FR/EN...',
    )

    col_btn, col_examples = st.columns([1, 3])
    with col_btn:
        clicked = st.button('Analyser', type='primary', use_container_width=True)
    with col_examples:
        example = st.selectbox('Ou choisir un exemple :', [
            '', 'Post fiable FR', 'Post suspect FR', 'Post fiable EN', 'Post suspect EN',
        ], label_visibility='collapsed')

    examples = {
        'Post fiable FR': "Le CNRS publie une étude confirmant l'efficacité des nouveaux traitements contre l'hépatite C.",
        'Post suspect FR': "SCANDALE : le gouvernement cache la VÉRITÉ sur les chemtrails ! Partagez avant censure !!!",
        'Post fiable EN': "Researchers at MIT developed a new carbon capture method that reduces costs by 40 percent.",
        'Post suspect EN': "EXPOSED: Secret government labs are using 5G towers to spread mind-control chemicals!!!",
    }
    if example and example in examples:
        text_input = examples[example]

    if (clicked or example) and text_input.strip():
        # Sanitize: limit input length and strip HTML tags for safety
        text_input = text_input[:10_000]
        text_input = re.sub(r'<[^>]+>', '', text_input)
        with st.spinner('Analyse en cours...'):
            result = detector.predict(pd.Series([text_input]))
            score = float(result['ai_score_credibility'].iloc[0])
            label = result['prediction_label'].iloc[0]
            lang = result['language'].iloc[0]

            probas = emo.get_emotion_features([text_input])[0]
            dominant_idx = int(np.argmax(probas))
            dominant_emotion = EMOTION_LABELS[dominant_idx]
            dominant_proba = float(probas[dominant_idx])
            explanation = detector.explain_prediction(text_input)

            # Stage 1
            post_type, post_type_proba = None, None
            if stage1_data is not None:
                try:
                    s1_pipe = stage1_data['pipeline']
                    s1_th = stage1_data.get('threshold', 0.40)
                    s1_proba = s1_pipe.predict_proba([text_input])[0]
                    p_factuel = float(s1_proba[1])
                    post_type_proba = p_factuel
                    post_type = 'factuel' if p_factuel >= s1_th else 'opinion'
                except Exception:
                    pass

            # V7/V8
            v7_result = None
            if v6_data is not None:
                v7_result = predict_v7_hybrid(text_input, detector, emo, v6_data, v7_data, cam_classifier)
                if v7_result is not None:
                    v7_result['post_type'] = post_type
                    v7_result['post_type_proba'] = post_type_proba

        # === Results display ===
        st.markdown('---')

        # Stage 1 banner
        if post_type is not None:
            pt_color = '#FFD600' if post_type == 'factuel' else '#00D4FF'
            pt_label = 'Factuel' if post_type == 'factuel' else 'Opinion'
            pt_icon = '' if post_type == 'factuel' else ''
            note = '' if post_type == 'factuel' else ' — les opinions ne sont pas évaluées comme désinformation'
            st.markdown(
                f'<div role="status" aria-live="polite" aria-label="Stage 1: {html.escape(pt_label)}" style="background:rgba(255,255,255,0.03);border-radius:12px;padding:12px 16px;'
                f'border-left:4px solid {pt_color};margin-bottom:16px;">'
                f'{pt_icon} <b>Stage 1 :</b> <span style="color:{pt_color}">{html.escape(pt_label)}</span>'
                f' (P(factuel) = {post_type_proba:.2f}){note}</div>',
                unsafe_allow_html=True,
            )

        # 3-column verdict
        c1, c2, c3 = st.columns([2, 1, 1])

        with c1:
            st.plotly_chart(make_gauge(score), use_container_width=True)
            st.caption("Score entre 0 (probablement faux) et 1 (probablement fiable). Seuil : 0.44")
            # Uncertainty band when score is near threshold
            if 0.34 <= score <= 0.54:
                st.markdown(
                    '<div role="alert" style="background:rgba(255,214,0,0.10);border:1px solid rgba(255,214,0,0.4);'
                    'border-radius:8px;padding:8px 12px;margin-top:4px;text-align:center;">'
                    '<span style="color:#FFD600;font-weight:600;">⚠ ZONE D\'INCERTITUDE</span><br>'
                    '<span style="color:#B0B0B0;font-size:0.85rem;">Le score est proche du seuil (0.44). '
                    'Le verdict pourrait basculer avec des informations complémentaires.</span></div>',
                    unsafe_allow_html=True,
                )

        with c2:
            is_fiable = (label == 0) or (str(label).upper() == 'FIABLE') or (isinstance(label, (int, np.integer)) and label == 0)
            css_class = 'verdict-fiable' if is_fiable else 'verdict-suspect'
            verdict_text = 'FIABLE' if is_fiable else 'SUSPECT'
            verdict_color = '#00E676' if is_fiable else '#FF1744'
            verdict_icon = '' if is_fiable else ''
            safe_lang = html.escape(str(lang).upper())
            st.markdown(
                f'<div class="{css_class}" role="status" aria-live="polite" aria-label="Verdict: {verdict_text}">'
                f'<div style="font-size:2.5rem;" aria-hidden="true">{verdict_icon}</div>'
                f'<div style="font-size:1.8rem;font-weight:700;color:{verdict_color};margin-top:8px;">{verdict_text}</div>'
                f'<div style="margin-top:12px;color:#B0B0B0;">Langue : {safe_lang}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with c3:
            emoji = EMOTION_EMOJIS.get(dominant_emotion, '')
            safe_emotion = html.escape(str(EMOTION_DISPLAY.get(dominant_emotion, dominant_emotion)))
            st.markdown(
                f'<div class="glass-card" role="status" aria-live="polite" aria-label="Emotion dominante: {safe_emotion} ({dominant_proba:.1%})" style="text-align:center;">'
                f'<div style="font-size:2.5rem;" aria-hidden="true">{emoji}</div>'
                f'<div style="font-size:1.4rem;font-weight:600;color:#00D4FF;margin-top:8px;">'
                f'{safe_emotion}</div>'
                f'<div style="margin-top:8px;color:#B0B0B0;">{dominant_proba:.1%}</div>'
                '<div class="metric-label">Émotion dominante</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Full 7-emotion distribution + Radar side by side
        emo_col1, emo_col2 = st.columns(2)
        with emo_col1:
            emo_display_names = [EMOTION_DISPLAY[e] for e in EMOTION_LABELS]
            emo_pct = [float(probas[i]) * 100 for i in range(len(EMOTION_LABELS))]
            emo_colors = [EMOTION_COLORS[e] for e in EMOTION_LABELS]
            sorted_emo = sorted(zip(emo_display_names, emo_pct, emo_colors), key=lambda x: x[1])
            names_s, pcts_s, colors_s = zip(*sorted_emo)
            fig_emo_dist = go.Figure(go.Bar(
                y=list(names_s), x=list(pcts_s), orientation='h',
                marker_color=list(colors_s),
                text=[f'{p:.1f}%' for p in pcts_s],
                textposition='outside',
            ))
            _apply_layout(fig_emo_dist,
                height=280,
                title=dict(text='Distribution des 7 émotions', x=0.5,
                           font=dict(color='#E0E0E0', size=14)),
                xaxis=dict(title='Probabilité (%)', range=[0, max(emo_pct) * 1.25]),
                margin=dict(t=45, b=35, l=90, r=40),
            )
            st.plotly_chart(fig_emo_dist, use_container_width=True)
        with emo_col2:
            st.plotly_chart(make_radar(probas.tolist(), 'Profil émotionnel', fill_opacity=0.25),
                            use_container_width=True)

        # Explainability
        if explanation.get('explainable'):
            st.subheader('Pourquoi ce verdict ?')
            all_words, all_contribs, all_colors = [], [], []
            for item in explanation['top_suspect_words'][:7]:
                all_words.append(item['word'])
                all_contribs.append(item['contribution'])
                all_colors.append('#FF1744')
            for item in explanation['top_fiable_words'][:7]:
                all_words.append(item['word'])
                all_contribs.append(item['contribution'])
                all_colors.append('#00E676')

            if all_words:
                sorted_data = sorted(zip(all_words, all_contribs, all_colors), key=lambda x: x[1])
                words_s, contribs_s, colors_s = zip(*sorted_data)
                fig_explain = go.Figure(go.Bar(
                    y=list(words_s), x=list(contribs_s), orientation='h',
                    marker_color=list(colors_s),
                ))
                _apply_layout(fig_explain,
                    height=max(280, len(all_words) * 28),
                    title=dict(text='Contribution des mots au verdict', x=0.5,
                               font=dict(color='#E0E0E0', size=15)),
                    xaxis=dict(title='Contribution (+ = suspect, - = fiable)',
                               zeroline=True, zerolinecolor='rgba(255,255,255,0.2)'),
                    margin=dict(t=50, b=40, l=120, r=20),
                )
                st.plotly_chart(fig_explain, use_container_width=True)
                st.caption("Coefficient de régression logistique × valeur TF-IDF de chaque mot.")

            if explanation.get('sensationalist_words'):
                pills = ' '.join(
                    f'<span style="background:rgba(255,23,68,0.15);border:1px solid rgba(255,23,68,0.3);'
                    f'border-radius:8px;padding:4px 12px;margin:3px;display:inline-block;font-size:0.88rem;">'
                    f'{html.escape(s["word"])}</span>'
                    for s in explanation['sensationalist_words']
                )
                st.markdown(
                    '<div class="glass-card" role="region" aria-label="Mots sensationnalistes detectes">'
                    '<div style="font-size:1rem;font-weight:600;color:#FF1744;margin-bottom:10px;">'
                    'Mots sensationnalistes détectés</div>'
                    f'<div>{pills}</div></div>',
                    unsafe_allow_html=True,
                )

        # V8 Hybrid scores
        if v7_result is not None:
            st.markdown('---')
            st.subheader('Analyse multi-modèle V9')

            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.markdown(metric_card('', 'V5 TF-IDF', f'{v7_result["score_v5"]:.2f}', '#00D4FF'), unsafe_allow_html=True)
            hc2.markdown(metric_card('', 'V6 Style', f'{v7_result["score_v6"]:.2f}', '#FFD600'), unsafe_allow_html=True)
            v7_color = '#FF1744' if v7_result['label_v7'] == 'SUSPECT' else '#00E676'
            hc3.markdown(metric_card('', 'V8 Hybride', f'{v7_result["score_v7"]:.2f}', v7_color), unsafe_allow_html=True)
            hc4.markdown(metric_card('', 'Désaccord', f'{v7_result["disagreement"]:.2f}',
                                     '#FF9100' if v7_result['disagreement'] > 0.3 else '#00E676'), unsafe_allow_html=True)

            st.caption("V5 = P(fiable) TF-IDF. V6 = P(suspect) style. V8 = méta-learner V5+V6+CamemBERT. "
                       "Le désaccord V5/V6 mesure la divergence entre les deux approches.")

            # SHAP
            if v7_result.get('shap_values') is not None:
                st.subheader('Explication SHAP -- Features de style')
                shap_vals = v7_result['shap_values']
                feat_names = v7_result['feature_names']
                feat_values = v7_result['feature_values']
                top_idx = np.argsort(np.abs(shap_vals))[::-1][:12]

                bar_names, bar_shap, bar_colors, bar_hover = [], [], [], []
                for idx in reversed(top_idx):
                    fname = feat_names[idx]
                    label_fr = StyleFeatureExtractorV6.FEATURE_LABELS_FR.get(fname, fname)
                    bar_names.append(label_fr)
                    bar_shap.append(shap_vals[idx])
                    bar_colors.append('#FF1744' if shap_vals[idx] > 0 else '#00E676')
                    bar_hover.append(f'{label_fr}<br>Valeur={feat_values[idx]:.3f}<br>SHAP={shap_vals[idx]:+.4f}')

                fig_shap = go.Figure(go.Bar(
                    y=bar_names, x=bar_shap, orientation='h',
                    marker_color=bar_colors, hovertext=bar_hover, hoverinfo='text',
                ))
                _apply_layout(fig_shap,
                    height=max(320, len(top_idx) * 30),
                    title=dict(text='Impact SHAP sur le score V6', x=0.5,
                               font=dict(color='#E0E0E0', size=15)),
                    xaxis=dict(title='SHAP value (+ = suspect, - = fiable)',
                               zeroline=True, zerolinecolor='rgba(255,255,255,0.2)'),
                    margin=dict(t=50, b=40, l=180, r=20),
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                st.caption("SHAP décompose la prédiction V6 en contributions par feature. "
                           "Seul le STYLE est analysé, indépendamment du sujet.")

            # ----------------------------------------------------------
            # Décomposition exacte du méta-learner V7/V8 (LogReg)
            # ----------------------------------------------------------
            if v7_result.get('X_meta') is not None and v7_result.get('v7_data') is not None:
                try:
                    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
                    from explainability.meta_decomposition import MetaLearnerDecomposer

                    decomposer = MetaLearnerDecomposer(v7_result['v7_data'])
                    decomp = decomposer.decompose(v7_result['X_meta'][0])

                    st.subheader("Décomposition de l'ensemble (méta-learner V8)")
                    fig_meta = MetaLearnerDecomposer.to_plotly_bar(decomp)
                    st.plotly_chart(fig_meta, use_container_width=True)

                    drivers = decomp.top_drivers(3)
                    driver_strs = [
                        f"<b>{d['feature']}</b> "
                        f"(<span style='color:{'#FF1744' if d['direction']=='SUSPECT' else '#00E676'}'>"
                        f"{d['direction']} {d['contribution']:+.3f}</span>)"
                        for d in drivers
                    ]
                    st.markdown(
                        f"**logit z = {decomp.logit:+.3f}** → "
                        f"σ(z) = P(suspect) = **{decomp.proba_suspect:.3f}** → "
                        f"décision **{decomp.label}** (seuil {decomp.threshold:.2f})",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "Top 3 contributeurs : " + " · ".join(driver_strs),
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Décomposition exacte d'un modèle linéaire : z = β₀ + Σ βᵢ·xᵢ. "
                        "Chaque barre = contribution d'une feature au logit. "
                        "Rouge = pousse vers SUSPECT, vert = pousse vers FIABLE. "
                        "Cette décomposition complète SHAP : SHAP explique V6 (style), "
                        "celle-ci explique l'agrégation V5+V6+CamemBERT."
                    )
                except Exception as e:
                    logging.warning("Décomposition méta-learner échouée: %s", e)

            # ----------------------------------------------------------
            # CamemBERT Attention Heatmap (XAI-03/XAI-04)
            # ----------------------------------------------------------
            if cam_classifier is not None:
                try:
                    attn_data = cam_classifier.extract_attention(text_input)
                    tokens = attn_data['tokens']
                    weights = attn_data['attention_weights']

                    # Filter special tokens and subword prefixes
                    display_tokens, display_weights = [], []
                    for t, w in zip(tokens, weights):
                        if t in ('<s>', '</s>', '<pad>'):
                            continue
                        # Clean subword markers
                        clean = t.replace('\u2581', ' ').strip()
                        if clean:
                            display_tokens.append(clean)
                            display_weights.append(w)

                    if display_tokens:
                        st.markdown('---')
                        st.subheader('Heatmap d\'attention CamemBERT')

                        # Build heatmap as colored HTML spans
                        # Normalize weights to amplify contrast
                        w_min = min(display_weights)
                        w_max = max(display_weights)
                        w_range = w_max - w_min if w_max > w_min else 1e-6
                        norm_weights = [(w - w_min) / w_range for w in display_weights]

                        spans = []
                        for tok, w_norm in zip(display_tokens, norm_weights):
                            # Amplify: apply power curve to boost high-attention tokens
                            w_boost = w_norm ** 0.5
                            r = int(220 + 35 * w_boost)
                            g = int(240 * (1 - w_boost))
                            b = int(240 * (1 - w_boost))
                            a = 0.15 + 0.85 * w_boost
                            safe_tok = html.escape(tok)
                            spans.append(
                                f'<span style="background:rgba({r},{g},{b},{a:.2f});'
                                f'padding:3px 5px;margin:2px;border-radius:4px;'
                                f'display:inline-block;font-size:0.95rem;font-weight:{"600" if w_boost > 0.6 else "400"};"'
                                f' title="attention: {w_boost:.3f}">{safe_tok}</span>'
                            )
                        st.markdown(
                            '<div class="glass-card" role="region" aria-label="Heatmap attention CamemBERT">'
                            '<div style="line-height:2.2;">' + ' '.join(spans) + '</div>'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            "Poids d'attention du token [CLS] (derniere couche, moyenne sur 12 tetes). "
                            "Rouge intense = le modele concentre son attention sur ce token."
                        )
                except Exception as e:
                    logging.debug("Attention CamemBERT non disponible: %s", e)


def _page_batch_analysis(detector, emo):
    st.markdown("Importez un fichier CSV avec une colonne `text` pour analyser plusieurs textes.")
    uploaded = st.file_uploader('Fichier CSV', type=['csv'], label_visibility='collapsed')

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f'Erreur de lecture CSV : {e}')
            return

        if 'text' not in batch_df.columns:
            st.error("Le CSV doit contenir une colonne `text`.")
            return

        n = len(batch_df)
        st.info(f'{n} textes détectés dans le fichier.')

        if st.button(f'Analyser {n} textes', type='primary'):
            progress_bar = st.progress(0, text='Preparation...')
            BATCH_SIZE = 200

            texts = pd.Series(batch_df['text'].astype(str).values)
            all_results = []
            total_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

            for i in range(0, n, BATCH_SIZE):
                batch_end = min(i + BATCH_SIZE, n)
                batch_texts = texts.iloc[i:batch_end].reset_index(drop=True)
                batch_result = detector.predict(batch_texts)
                all_results.append(batch_result)
                progress = min((i + BATCH_SIZE) / n, 1.0)
                progress_bar.progress(progress, text=f'Analyse : {batch_end}/{n} textes')

            results = pd.concat(all_results, ignore_index=True)
            progress_bar.progress(1.0, text='Extraction des emotions...')

            batch_df['Score'] = results['ai_score_credibility'].values
            batch_df['Label'] = results['prediction_label'].map({0: 'FIABLE', 1: 'SUSPECT'}).values
            batch_df['Langue'] = results['language'].values

            try:
                emo_features = emo.get_emotion_features(texts.tolist())
                batch_df['Emotion'] = [
                    EMOTION_LABELS[int(np.argmax(p))] for p in emo_features
                ]
            except Exception:
                batch_df['Emotion'] = 'neutre'

            progress_bar.progress(1.0, text=f'Termine : {n} textes analyses')

            # Summary
            n_fiable = (batch_df['Label'] == 'FIABLE').sum()
            n_suspect = (batch_df['Label'] == 'SUSPECT').sum()

            c1, c2, c3 = st.columns(3)
            c1.metric('Fiables', n_fiable, f'{n_fiable/n*100:.0f}%')
            c2.metric('Suspects', n_suspect, f'{n_suspect/n*100:.0f}%')
            c3.metric('Score moyen', f'{batch_df["Score"].mean():.2f}')

            # Results table
            display_cols = ['text', 'Score', 'Label', 'Langue', 'Emotion']
            available = [c for c in display_cols if c in batch_df.columns]
            st.dataframe(batch_df[available].head(200), hide_index=True, use_container_width=True, height=400)

            # Download
            csv_output = batch_df[available].to_csv(index=False).encode('utf-8')
            st.download_button(
                'Télécharger les résultats (CSV)',
                csv_output, 'thumalien_resultats.csv', 'text/csv',
                use_container_width=True,
            )

    # --- Glossaire pédagogique ---
    st.markdown('---')
    with st.expander('Glossaire pédagogique — Comprendre les concepts clés', expanded=False):
        st.markdown("""
| Concept | Explication |
|---|---|
| **TF-IDF** | *Term Frequency – Inverse Document Frequency.* Mesure l'importance d'un mot dans un texte par rapport à un corpus. Plus un mot est rare dans le corpus mais fréquent dans le texte, plus son score TF-IDF est élevé. C'est la base de notre modèle de détection. |
| **SHAP** | *SHapley Additive exPlanations.* Méthode d'explicabilité IA issue de la théorie des jeux. Chaque mot reçoit une valeur SHAP qui mesure sa contribution au verdict final (positif = pousse vers « suspect », négatif = pousse vers « fiable »). |
| **Seuil 0.44** | Notre modèle produit un score de crédibilité entre 0 et 1. En dessous de 0.44, le texte est classé **SUSPECT** ; au-dessus, **FIABLE**. Ce seuil a été optimisé par validation croisée pour maximiser le F1-score. |
| **Stage 1 (Fait/Opinion)** | Premier filtre du pipeline : un classifieur sépare les textes factuels des opinions. Seuls les textes factuels passent par la détection de désinformation complète, car les opinions ne sont pas vérifiables de la même manière. |
| **Score de crédibilité** | Probabilité estimée qu'un texte soit fiable (0 = probablement faux, 1 = probablement vrai). Combine les features TF-IDF, linguistiques et émotionnelles. |
| **Émotions (7 classes)** | Notre modèle MLP bilingue détecte 7 émotions : colère, dégoût, joie, neutre, peur, surprise, tristesse. Les fake news exploitent souvent la peur et la colère pour maximiser l'engagement. |
| **CamemBERT / RoBERTa** | Modèles de langage pré-entraînés (Transformer). CamemBERT pour le français, RoBERTa pour l'anglais. Utilisés pour capter le sens contextuel des textes courts (< 30 mots). |
| **Régression logistique** | Modèle de classification simple et interprétable. Chaque mot a un coefficient : positif = contribue au verdict « suspect », négatif = contribue au verdict « fiable ». |
""")


# ===================================================================
#  PAGE 3 : Explorateur de donnees
# ===================================================================

def page_explorer(df, is_demo):
    hero('Explorateur', 'Explorez et filtrez les posts collectés depuis Bluesky')

    if is_demo:
        st.info('Mode démonstration — données simulées.')

    # --- Filters in sidebar ---
    st.sidebar.markdown('### Filtres')

    lang_filter = st.sidebar.multiselect(
        'Langue', ['fr', 'en'],
        default=['fr', 'en'],
    )
    label_filter = st.sidebar.multiselect(
        'Label', ['FIABLE', 'SUSPECT', 'NON ANALYSE'],
        default=['FIABLE', 'SUSPECT'],
    )
    emotion_filter = st.sidebar.multiselect(
        'Émotion', EMOTION_LABELS,
        default=EMOTION_LABELS,
    )
    score_range = st.sidebar.slider(
        'Score de crédibilité', 0.0, 1.0, (0.0, 1.0), 0.05,
    )

    search_text = st.text_input('Recherche dans le texte', placeholder='Mot-clé...')

    # Apply filters
    mask = (
        df['ai_language'].isin(lang_filter) &
        df['prediction_label'].isin(label_filter) &
        df['ai_emotion'].isin(emotion_filter) &
        df['ai_score_credibility'].between(score_range[0], score_range[1])
    )
    if search_text.strip():
        mask = mask & df['text'].str.contains(search_text.strip(), case=False, na=False)

    filtered = df[mask]

    # Stats
    st.markdown(f'**{len(filtered):,}** posts correspondants sur **{len(df):,}** affichés')

    col1, col2, col3 = st.columns(3)
    n_f = (filtered['prediction_label'] == 'FIABLE').sum()
    n_s = (filtered['prediction_label'] == 'SUSPECT').sum()
    col1.metric('Fiables', f'{n_f:,}')
    col2.metric('Suspects', f'{n_s:,}')
    col3.metric('Score moyen', f'{filtered["ai_score_credibility"].mean():.2f}' if len(filtered) else '--')

    st.markdown('---')

    # Emotion breakdown of filtered data
    if len(filtered) > 0:
        emo_counts = filtered['ai_emotion'].value_counts()
        fig_emo = go.Figure(go.Pie(
            labels=[EMOTION_DISPLAY.get(e, e) for e in emo_counts.index],
            values=emo_counts.values,
            marker_colors=[EMOTION_COLORS.get(e, '#666') for e in emo_counts.index],
            hole=0.4, textinfo='label+percent',
        ))
        fig_emo.update_layout(
            **_plotly_layout(height=350),
            showlegend=False,
            title=dict(text='Émotions (sélection filtrée)', x=0.5,
                       font=dict(color='#E0E0E0', size=15)),
        )
        st.plotly_chart(fig_emo, use_container_width=True)

    # Table with full text access
    display_cols = ['text', 'ai_language', 'prediction_label', 'ai_score_credibility', 'ai_emotion']
    if 'search_term' in filtered.columns:
        display_cols.append('search_term')
    available = [c for c in display_cols if c in filtered.columns]

    disp = filtered[available].copy()
    disp.columns = ['Texte', 'Langue', 'Label', 'Score', 'Emotion'] + (['Terme'] if 'search_term' in available else [])
    disp['Langue'] = disp['Langue'].str.upper()

    def color_label(val):
        if val == 'FIABLE':
            return 'color: #00E676; font-weight: 600'
        if val == 'SUSPECT':
            return 'color: #FF1744; font-weight: 600'
        return ''

    styled = (
        disp.head(500).style
        .map(color_label, subset=['Label'])
        .background_gradient(subset=['Score'], cmap='RdYlGn', vmin=0, vmax=1)
        .format({'Score': '{:.2f}'})
    )
    st.dataframe(styled, hide_index=True, use_container_width=True, height=500)

    # Export
    if len(filtered) > 0:
        csv = filtered[available].to_csv(index=False).encode('utf-8')
        st.download_button(
            f'Exporter {len(filtered):,} posts (CSV)',
            csv, 'thumalien_export.csv', 'text/csv',
        )


# ===================================================================
#  PAGE 4 : Performance & Transparence
# ===================================================================

def page_performance():
    hero('Performance & Transparence', 'Validation expérimentale, bilan carbone et conformité')

    # --- Tabs ---
    tab_perf, tab_carbon, tab_compliance = st.tabs([
        'Validation expérimentale', 'Bilan carbone', 'Conformité'
    ])

    with tab_perf:
        _section_ablation()
        st.markdown('---')
        _section_roadmap()

    with tab_carbon:
        _section_carbon()

    with tab_compliance:
        _section_compliance()


def _section_ablation():
    st.subheader('Ablation Study (7 conditions)')

    ablation_data = {
        '#': [1, 2, 3, 4, 5, 6, 7],
        'Condition': [
            'EN seul (TF-IDF)', 'FR seul', 'Bilingue naïf',
            'Bilingue + oversampling x3', 'Bilingue + class_weight',
            'Bilingue SVM', 'Bilingue + émotions (V1.5)',
        ],
        'F1 EN': [0.9956, None, 0.9884, 0.9842, 0.9824, 0.9820, 0.9852],
        'F1 FR': [None, 0.5423, 0.5394, 0.9807, 0.9780, 0.9795, 0.9818],
        'Features': ['30K', '30K', '30K + 12', '30K + 12', '30K + 12', '30K + 12', '30K + 19'],
    }
    abl_df = pd.DataFrame(ablation_data)

    def highlight_best(s):
        styles = [''] * len(s)
        valid = s.dropna()
        if len(valid) > 0:
            best = valid.max()
            for i in s.index:
                if pd.notna(s[i]) and s[i] == best:
                    styles[i] = 'color: #00D4FF; font-weight: 700'
        return styles

    styled_abl = (
        abl_df.style
        .apply(highlight_best, subset=['F1 EN'])
        .apply(highlight_best, subset=['F1 FR'])
        .format({'F1 EN': lambda v: f'{v:.4f}' if pd.notna(v) else '\u2014',
                 'F1 FR': lambda v: f'{v:.4f}' if pd.notna(v) else '\u2014'})
        .background_gradient(subset=['F1 EN'], cmap='Blues', vmin=0.95, vmax=1.0)
        .background_gradient(subset=['F1 FR'], cmap='Blues', vmin=0.50, vmax=1.0)
    )
    st.dataframe(styled_abl, hide_index=True, use_container_width=True)

    conditions = abl_df['Condition']
    fig_abl = go.Figure()
    fig_abl.add_trace(go.Bar(x=conditions, y=abl_df['F1 EN'], name='F1 EN', marker_color='#00D4FF'))
    fig_abl.add_trace(go.Bar(x=conditions, y=abl_df['F1 FR'], name='F1 FR', marker_color='#FFD600'))
    _apply_layout(fig_abl, height=400, barmode='group',
        yaxis=dict(range=[0, 1.05], title='F1-score'),
        xaxis=dict(tickangle=-25),
        legend=dict(orientation='h', y=-0.25, x=0.5, xanchor='center'),
        margin=dict(t=30, b=100, l=50, r=20),
    )
    st.plotly_chart(fig_abl, use_container_width=True)
    st.caption("F1-scores sur holdout pour les 7 conditions de l'ablation study.")

    st.markdown('---')

    # Performance par longueur
    st.subheader('Performance par longueur de texte')
    length_cats = ['Ultra-court\n(<15 mots)', 'Court\n(15-30)', 'Moyen\n(30-100)', 'Long\n(>300)']
    length_f1 = [0.747, 0.844, 0.942, 0.990]
    colors = ['#FF9100', '#FFD600', '#00E676', '#00D4FF']

    fig_len = go.Figure(go.Bar(
        x=length_cats, y=length_f1, marker_color=colors,
        text=[f'{v:.3f}' for v in length_f1],
        textposition='outside', textfont=dict(color='#E0E0E0', size=12),
    ))
    _apply_layout(fig_len, height=350,
        yaxis=dict(range=[0, 1.08], title='F1-score'),
    )
    st.plotly_chart(fig_len, use_container_width=True)
    st.caption("F1-score par catégorie de longueur. Les textes ultra-courts (<15 mots) restent le cas le plus difficile.")

    # SHAP beeswarm global
    st.subheader('SHAP Beeswarm — Importance globale des features')
    _shap_beeswarm_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'shap_global_values.pkl')
    if os.path.exists(_shap_beeswarm_path):
        try:
            import joblib as _jl
            _shap_data = _jl.load(_shap_beeswarm_path)
            _sv = _shap_data['shap_values']  # (n_samples, n_features)
            _fn = _shap_data['feature_names']

            # Mean |SHAP| per feature
            _mean_abs = np.mean(np.abs(_sv), axis=0)
            _top_idx = np.argsort(_mean_abs)[::-1][:20]

            _bee_names = [_fn[i] for i in _top_idx][::-1]
            _bee_vals = [_mean_abs[i] for i in _top_idx][::-1]
            _bee_colors = ['#00E676' if v < np.median(_mean_abs[_top_idx]) else '#FF1744' for v in _bee_vals]

            fig_bee = go.Figure(go.Bar(
                y=_bee_names, x=_bee_vals, orientation='h',
                marker_color=_bee_colors,
                text=[f'{v:.4f}' for v in _bee_vals],
                textposition='outside', textfont=dict(color='#E0E0E0', size=10),
            ))
            _apply_layout(fig_bee, height=500,
                title=dict(text='Mean |SHAP| — Top 20 features', x=0.5,
                           font=dict(color='#E0E0E0', size=14)),
                xaxis=dict(title='Mean |SHAP value|', gridcolor='#333'),
            )
            st.plotly_chart(fig_bee, use_container_width=True)
            st.caption("Beeswarm global : importance moyenne des features sur l'ensemble du jeu de validation. "
                       "Les features en rouge ont l'impact le plus fort sur la prédiction.")
        except Exception as _e:
            st.info(f"SHAP beeswarm non disponible : {_e}")
    else:
        # Static fallback with known feature importances from V6 GradientBoosting
        _static_features = [
            'tfidf_top1', 'caps_ratio', 'exclamation_count', 'word_count',
            'avg_word_length', 'emo_joie', 'question_marks', 'emo_colere',
            'emo_neutre', 'punctuation_density', 'unique_ratio', 'sentence_count',
        ]
        _static_importance = [0.142, 0.098, 0.087, 0.076, 0.068, 0.061, 0.055, 0.052, 0.048, 0.044, 0.039, 0.035]
        fig_bee_static = go.Figure(go.Bar(
            y=_static_features[::-1], x=_static_importance[::-1], orientation='h',
            marker_color=['#FF1744' if v > 0.06 else '#00E676' for v in _static_importance[::-1]],
            text=[f'{v:.3f}' for v in _static_importance[::-1]],
            textposition='outside', textfont=dict(color='#E0E0E0', size=10),
        ))
        _apply_layout(fig_bee_static, height=400,
            title=dict(text='Feature Importance globale (statique)', x=0.5,
                       font=dict(color='#E0E0E0', size=14)),
            xaxis=dict(title='Importance relative', gridcolor='#333'),
        )
        st.plotly_chart(fig_bee_static, use_container_width=True)
        st.caption("Importance statique des features. Generez shap_global_values.pkl pour le beeswarm dynamique.")

    # Limitations
    st.subheader('Limitations connues')
    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card('', 'FR ultra-court', 'F1 = 0,747', '#FF9100'), unsafe_allow_html=True)
    c2.markdown(metric_card('', 'Calibration', 'ECE = 0,049', '#FF9100'), unsafe_allow_html=True)
    c3.markdown(metric_card('', 'Erreurs neutres', '84 %', '#FF9100'), unsafe_allow_html=True)
    st.caption("84% des erreurs portent sur des contenus neutres/factuels sans marqueur fort. "
               "Le modèle ne fait pas de fact-checking : il détecte des patterns stylistiques.")


def _section_roadmap():
    st.subheader("Roadmap d'évolution")

    versions = [
        ('V1', 'TF-IDF + LogReg baseline', 'F1 EN = 0.9956 (biaisé Reuters). Aucune capacité FR.'),
        ('V1.5', 'Bilingue + émotions MLP', 'F1 EN = 0.9852, F1 FR = 0.9818. Domain shift : 77% suspects sur Bluesky.'),
        ('V2', 'Datasets sociaux + seuil 0.44', '145K textes, CV F1 = 0.897, 73.4% fiable sur Bluesky.'),
        ('V3', 'Correction preprocessing', 'Bug fix 5/12 features nulles. F1 = 0.900, Précision +19.3%.'),
        ('V4', 'Amélioration FR court', 'FR court F1 : 0.65 → 0.86 (+32%). 15 features linguistiques.'),
        ('V5', '+10K posts FR synthétiques', 'F1 global = 0.913, FR ultra-court = 0.904. 197K textes.'),
        ('V6', 'Style-only topic-agnostic', '28 features stylistiques, GradientBoosting CV F1 = 0.830.'),
        ('V7', 'Ensemble V5+V6 + SHAP', 'Méta-learner, FP 57 → 25 sur gold set. Explicabilité SHAP.'),
        ('V8', '+CamemBERT (3e signal)', 'Méta-learner 7 features, F1 suspect +28%.'),
        ('V9', 'Pipeline fait/opinion', 'Stage 1 filtre opinions. FP -67% (186 → 62). Kappa ×3.'),
    ]

    for name, title, desc in versions:
        with st.expander(f'{name} -- {title}'):
            st.markdown(desc)


def _load_all_emissions():
    """Load emissions from all CSV files (main + backups) and deduplicate by run_id."""
    base = os.path.join(os.path.dirname(__file__), '..')
    csv_files = [
        os.path.join(base, 'emissions.csv'),
        os.path.join(base, 'emissions.csv.bak'),
        os.path.join(base, 'emissions.csv_0.bak'),
        os.path.join(base, 'emissions.csv_1.bak'),
        os.path.join(base, 'emissions.csv_2.bak'),
    ]
    frames = []
    for f in csv_files:
        try:
            if os.path.exists(f):
                frames.append(pd.read_csv(f))
        except Exception:
            pass
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if 'run_id' in df.columns:
        df = df.drop_duplicates(subset='run_id')
    return df


# Runs selected for the rapport (matching the 6 documented trainings)
_RAPPORT_RUN_IDS = {
    'b747dcc0-8836-4fdc-95b8-f0e4127be5a5',  # V1.5 LogReg (fév.)
    '81b9cc2c-03fa-452f-a963-8f5dde71d16a',  # V5 LogReg
    'a75d296a-47e1-4a73-9b1d-863b6203cbfa',  # V5 LogReg retrain
    'ad76e332-0e49-41ba-b0c9-be3306c70e74',  # CamemBERT
    '9a310be3-e72e-4a61-baca-b7a63bbe19d0',  # RoBERTa EN V1
    '5981e174-9b93-41a3-8b16-a8bcc57cc215',  # RoBERTa EN V2
}

# Estimated emissions for components not tracked by CodeCarbon
# Rate: 7.55e-7 kg CO2/s (Apple M4 Pro, France grid, stable across all runs)
_ESTIMATED_ITEMS = pd.DataFrame([
    {'project_name': 'V6 GradientBoost (estimé)', 'duration': 480.0,
     'emissions': 0.000362, 'energy_consumed': 0.00648},
    {'project_name': 'Pipeline XAI (estimé)', 'duration': 300.0,
     'emissions': 0.000227, 'energy_consumed': 0.00405},
    {'project_name': 'Inférence cumulée (estimé)', 'duration': 200.0,
     'emissions': 0.000151, 'energy_consumed': 0.00270},
])


def _section_carbon():
    st.subheader('Bilan Carbone (Green IT)')

    all_df = _load_all_emissions()

    # Filter to the 6 documented runs
    if all_df is not None and 'run_id' in all_df.columns:
        em_df = all_df[all_df['run_id'].isin(_RAPPORT_RUN_IDS)].copy()
    elif all_df is not None and 'emissions' in all_df.columns:
        em_df = all_df
    else:
        em_df = None

    if em_df is not None and 'emissions' in em_df.columns and len(em_df) > 0:
        measured_co2_g = em_df['emissions'].sum() * 1000
        estimated_co2_g = _ESTIMATED_ITEMS['emissions'].sum() * 1000
        total_co2_g = measured_co2_g + estimated_co2_g
        total_energy = (em_df['energy_consumed'].sum() if 'energy_consumed' in em_df.columns else 0) + _ESTIMATED_ITEMS['energy_consumed'].sum()
        total_duration = (em_df['duration'].sum() if 'duration' in em_df.columns else 0) + _ESTIMATED_ITEMS['duration'].sum()
        n_runs = len(em_df) + len(_ESTIMATED_ITEMS)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card('', 'CO2 total', f'≈ {total_co2_g:.2f} g', '#00E676'), unsafe_allow_html=True)
        c2.markdown(metric_card('', 'Énergie', f'{total_energy*1000:.1f} Wh', '#FFD600'), unsafe_allow_html=True)
        c3.markdown(metric_card('', 'Durée totale', f'{total_duration/60:.1f} min', '#00D4FF'), unsafe_allow_html=True)
        c4.markdown(metric_card('', 'Runs', f'{len(em_df)} mesurés + 3 estimés', '#E0E0E0'), unsafe_allow_html=True)

        km_voiture = total_co2_g / 170  # 170g CO2/km (ADEME 2024)
        smartphones = total_energy * 1000 / 12
        st.markdown(
            '<div class="glass-card" role="region" aria-label="Equivalences ecologiques">'
            '<div style="font-size:1rem;font-weight:600;color:#00D4FF;margin-bottom:10px;">'
            'Équivalences écologiques</div>'
            '<div style="display:flex;gap:30px;flex-wrap:wrap;font-size:0.9rem;line-height:1.8;">'
            f'<div><strong>{km_voiture:.4f} km</strong> en voiture (≈ {km_voiture*1000:.0f} m)</div>'
            f'<div><strong>{smartphones:.2f}</strong> charges smartphone</div>'
            f'<div>Mix électrique France : <strong>56 g CO2/kWh</strong></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

        # Pie chart of emissions breakdown
        labels = ['RoBERTa EN (V1+V2)', 'V5 LogReg (3 runs)', 'CamemBERT',
                  'V6 GradientBoost*', 'Pipeline XAI*', 'Inférence*']
        values = [3.47, 2.18, 0.48, 0.36, 0.23, 0.15]
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values,
            marker=dict(colors=['#00D4FF', '#00B8D4', '#FFD600', '#00E676', '#9E9E9E', '#757575']),
            textinfo='label+percent', textfont_size=13,
            hole=0.4,
        )])
        fig.update_layout(
            title='Répartition des émissions CO₂ (≈ 6,88 g total)',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), height=400,
            legend=dict(font=dict(size=11)),
            annotations=[dict(text='6,88 g', x=0.5, y=0.5, font_size=18, showarrow=False, font_color='white')],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption('\\* Estimé via le taux CodeCarbon moyen (7,55×10⁻⁷ kg CO₂/s, Apple M4 Pro, mix FR)')

        with st.expander('Détail par run CodeCarbon (6 mesurés)'):
            detail_cols = ['project_name', 'duration', 'emissions', 'energy_consumed']
            available = [c for c in detail_cols if c in em_df.columns]
            detail = em_df[available].copy()
            if 'duration' in detail.columns:
                detail['duration'] = detail['duration'].apply(lambda x: f'{x:.1f}s')
            if 'emissions' in detail.columns:
                detail['emissions'] = detail['emissions'].apply(lambda x: f'{x*1000:.4f} g')
            if 'energy_consumed' in detail.columns:
                detail['energy_consumed'] = detail['energy_consumed'].apply(lambda x: f'{x*1000:.2f} Wh')
            detail.rename(columns={'project_name': 'Projet', 'duration': 'Durée',
                                   'emissions': 'CO2', 'energy_consumed': 'Énergie'}, inplace=True)
            st.dataframe(detail, hide_index=True, use_container_width=True)

        with st.expander('Estimations (3 composants non instrumentés)'):
            est = _ESTIMATED_ITEMS.copy()
            est['duration'] = est['duration'].apply(lambda x: f'{x:.0f}s (~{x/60:.0f} min)')
            est['emissions'] = est['emissions'].apply(lambda x: f'~{x*1000:.2f} g')
            est['energy_consumed'] = est['energy_consumed'].apply(lambda x: f'~{x*1000:.2f} Wh')
            est.rename(columns={'project_name': 'Composant', 'duration': 'Durée estimée',
                                'emissions': 'CO2 estimé', 'energy_consumed': 'Énergie estimée'}, inplace=True)
            st.dataframe(est, hide_index=True, use_container_width=True)
            st.caption('Méthode : durée estimée × taux d\'émission moyen CodeCarbon (constant sur Apple M4 Pro)')
    else:
        st.info('Aucune donnée CodeCarbon disponible. Lancez un entraînement pour mesurer les émissions.')


def _section_compliance():
    st.subheader('Conformité réglementaire')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="glass-card" role="region" aria-label="Conformite RGPD">'
            '<div style="font-size:1.3rem;margin-bottom:8px;color:#00D4FF;">RGPD</div>'
            '<ul style="font-size:0.88rem;line-height:1.8;color:#B0B0B0;">'
            '<li>Données publiques uniquement (posts Bluesky)</li>'
            '<li>Pas de profilage individuel</li>'
            '<li>Droit à l\'effacement via suppression MongoDB</li>'
            '<li>Aucune donnée personnelle stockée</li>'
            '<li>Collecte via API Bluesky (serveurs US), traitement local UE</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="glass-card" role="region" aria-label="Conformite AI Act europeen">'
            '<div style="font-size:1.3rem;margin-bottom:8px;color:#00D4FF;">AI Act européen</div>'
            '<ul style="font-size:0.88rem;line-height:1.8;color:#B0B0B0;">'
            '<li>Risque limité (art. 50/52) — applicable 2 août 2026</li>'
            '<li>Scores présentés comme aide à la décision</li>'
            '<li>Biais documentés (ablation study, gold set)</li>'
            '<li>Explicabilité SHAP intégrée</li>'
            '<li>Empreinte carbone mesurée (CodeCarbon)</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )

    st.markdown('---')
    st.subheader('Accessibilité (WCAG 2.1 AA)')
    st.markdown(
        '- **Contraste** : tous les textes respectent un ratio >= 4.5:1 sur fond sombre\n'
        '- **Descriptions** : chaque visualisation est accompagnee d\'une legende textuelle\n'
        '- **Navigation** : structure par pages avec navigation par sidebar\n'
        '- **Couleur** : les labels utilisent couleur + texte (FIABLE/SUSPECT), pas la couleur seule\n'
        '- **Focus visible** : indicateurs CSS `:focus-visible` sur tous les elements interactifs\n'
        '- **ARIA landmarks** : `role="banner"`, `role="status"`, `role="region"` sur tous les composants\n'
        '- **Langue** : `lang="fr"` declare via script pour les lecteurs d\'ecran (NVDA, VoiceOver)\n'
        '- **aria-live** : zones de resultats marquees `aria-live="polite"` pour annonce dynamique\n'
        '- **Decoratif** : icones decoratives marquees `aria-hidden="true"`\n'
        '- **Mobile** : layout responsive via `use_container_width=True` sur tous les graphiques\n'
        '- **Clavier** : navigation complete possible sans souris (sidebar + tabs + boutons)'
    )
    with st.expander('Checklist detaillee lecteurs d\'ecran'):
        st.markdown(
            '| Critere | Status | Details |\n'
            '|---------|--------|---------|\n'
            '| `lang="fr"` sur `<html>` | OK | Script injection au demarrage |\n'
            '| ARIA landmarks sur sections | OK | banner, status, region, contentinfo |\n'
            '| Focus visible CSS | OK | Outline 2px #00D4FF sur `:focus-visible` |\n'
            '| alt-text sur images | OK | Aucune image bitmap, SVG inline avec aria-label |\n'
            '| Annonce resultats dynamiques | OK | `aria-live="polite"` sur verdicts |\n'
            '| Contraste >= 4.5:1 | OK | Fond #0E1117, texte #E0E0E0 (ratio ~12:1) |\n'
            '| Pas de couleur seule | OK | Toujours texte + couleur pour FIABLE/SUSPECT |\n'
            '| Graphiques avec legende | OK | `st.caption()` sous chaque graphique |\n'
            '| Responsive mobile | Partiel | Streamlit responsive natif, graphiques adaptatifs |'
        )


# ===================================================================
#  PAGE 5 : A propos
# ===================================================================

def page_about():
    hero('À propos', 'Architecture, équipe et vision du projet ThumaCheck')

    st.subheader('Architecture du pipeline')
    st.markdown("""
```
                    Bluesky API (AT Protocol)
                           |
                    Collecteur V3 Python
                    (28 FR + 16 EN termes)
                           |
                    MongoDB (245K+ posts)
                    (healthcheck, auth)
                           |
              +------------+------------+
              |                         |
        Inference auto              Dashboard Streamlit
        (emotions + V5)             (5 pages, dark theme)
              |                         |
        MLP Emotions (7 cls)    +-------+-------+-------+
        ExpertDetector V5       |       |       |       |
                             Vue    Analyse  Explore  Perf
                             Globale  IA     Donnees  Carbon
```
    """)

    st.subheader('Pipeline de détection V9')
    st.markdown("""
```
Texte input
    |
    v
[Stage 1] Classifieur fait/opinion (TF-IDF + LogReg)
    |                    |
    v                    v
  FACTUEL             OPINION
    |                 → "Non évaluable"
    v
[V5] TF-IDF 30K + 15 features ling. + 7 émotions → LogReg
    |
    v
[V6] 28 features stylistiques + 7 émotions → GradientBoosting
    |
    v
[V8] Méta-learner(V5, V6, CamemBERT, désaccord) → LogReg
    |
    v
Score final + Explication SHAP
```
    """)

    st.subheader('Stack technologique')
    tech_data = {
        'Composant': ['Langage', 'Collecte', 'Stockage', 'ML classique', 'Deep Learning',
                      'Transformers', 'Dashboard', 'Conteneurisation', 'Green IT'],
        'Technologie': ['Python 3.13', 'atproto (AT Protocol)', 'MongoDB 8',
                        'scikit-learn', 'PyTorch', 'CamemBERT / RoBERTa',
                        'Streamlit + Plotly', 'Docker Compose', 'CodeCarbon'],
        'Rôle': ['Pipeline complet', 'API Bluesky', 'Base NoSQL documentaire',
                 'TF-IDF + LogReg + GradientBoosting', 'MLP émotions 7 classes',
                 'Embeddings sémantiques FR/EN', 'Interface utilisateur',
                 '4 services orchestrés', 'Mesure empreinte carbone'],
    }
    st.dataframe(pd.DataFrame(tech_data), hide_index=True, use_container_width=True)

    st.subheader('Équipe')
    st.markdown(
        '- **Azélie Bernard** — Lead technique (Data Engineer, ML Engineer, DevOps, Dashboard)\n'
        '- Formation : Master Big Data & IA, Sup de Vinci\n'
        '- 40+ commits, 28 notebooks, 10+ documents techniques'
    )

    st.subheader('Chiffres clés')
    c1, c2, c3, c4 = st.columns(4)
    # Chiffre dynamique depuis MongoDB si disponible
    _data = _fetch_mongo_data()
    _n_posts = f'{_data[1]:,}' if _data else '245K+'
    c1.markdown(metric_card('', 'Posts collectés', _n_posts, '#00D4FF'), unsafe_allow_html=True)
    c2.markdown(metric_card('', 'Versions modèle', '9', '#FFD600'), unsafe_allow_html=True)
    c3.markdown(metric_card('', 'Features', '30K+50', '#00E676'), unsafe_allow_html=True)
    c4.markdown(metric_card('', 'Notebooks', '28', '#00D4FF'), unsafe_allow_html=True)


# ===================================================================
#  Main
# ===================================================================

def _load_auth_config():
    """Charge la configuration d'authentification depuis auth_config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), 'auth_config.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    st.set_page_config(
        page_title='ThumaCheck — Détection de Fake News',
        page_icon='',
        layout='wide',
        initial_sidebar_state='expanded',
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Declare page language for screen readers (WCAG)
    st.markdown('<script>document.documentElement.lang="fr";</script>', unsafe_allow_html=True)

    # --- Authentication ---
    auth_config = _load_auth_config()
    authenticator = stauth.Authenticate(
        auth_config['credentials'],
        auth_config['cookie']['name'],
        auth_config['cookie']['key'],
        auth_config['cookie']['expiry_days'],
    )

    authenticator.login()

    if st.session_state.get('authentication_status') is None:
        st.info('Veuillez vous connecter pour accéder au dashboard.')
        return
    if st.session_state.get('authentication_status') is False:
        st.error('Identifiants incorrects.')
        return

    # --- AI Act Art. 50 — Banniere de transparence IA ---
    st.markdown(
        '<div role="status" aria-label="Avertissement intelligence artificielle" '
        'style="background:rgba(0,212,255,0.08);border-left:3px solid #00D4FF;'
        'padding:8px 14px;margin-bottom:12px;font-size:0.82rem;color:#B0BEC5;">'
        'Ce systeme utilise l\'intelligence artificielle pour analyser des contenus. '
        'Les scores sont indicatifs et ne constituent pas une decision automatisee. '
        'Conforme AI Act art. 50 (UE 2024/1689).</div>',
        unsafe_allow_html=True,
    )

    # --- Sidebar ---
    authenticator.logout('Déconnexion', 'sidebar')
    st.sidebar.markdown(
        '<div role="banner" aria-label="ThumaCheck Intelligence Center" style="text-align:center;padding:12px 0;">'
        '<span style="font-size:1.4rem;letter-spacing:4px;color:#00D4FF;font-weight:300;">'
        'THUMACHECK</span><br>'
        '<span style="font-size:0.7rem;color:#607D8B;">Intelligence Center v9.0</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    page = st.sidebar.radio(
        'Navigation',
        ['Dashboard', 'Analyse IA', 'Explorateur', 'Performance', 'À propos'],
        label_visibility='collapsed',
    )

    st.sidebar.divider()

    # Load resources
    detector, emo, model_suffix = load_pipeline()
    v6_data, v7_data, cam_classifier, stage1_data = load_v6_v7()
    df, is_demo = get_data()

    mongo_stats = df.attrs.get('mongo_stats', {})
    score_dist = df.attrs.get('score_dist', [])

    # Sidebar status
    st.sidebar.markdown(
        '<div role="status" aria-label="Statut des composants du pipeline" style="font-size:0.8rem;color:#B0B0B0;">'
        f'<span class="status-dot" style="background:#00E676;" aria-hidden="true"></span> Pipeline {model_suffix.replace("expert_", "").upper()}<br>'
        f'<span class="status-dot" style="background:{"#00E676" if v6_data else "#FF1744"};" aria-hidden="true"></span> '
        f'V6 Style {"OK" if v6_data else "N/A"}<br>'
        f'<span class="status-dot" style="background:{"#00E676" if v7_data else "#FF1744"};" aria-hidden="true"></span> '
        f'V8 Hybride {"OK" if v7_data else "N/A"}<br>'
        f'<span class="status-dot" style="background:{"#00E676" if cam_classifier else "#FFD600"};" aria-hidden="true"></span> '
        f'CamemBERT {"OK" if cam_classifier else "N/A"}<br>'
        f'<span class="status-dot" style="background:{"#00E676" if stage1_data else "#FF1744"};" aria-hidden="true"></span> '
        f'Stage 1 {"OK" if stage1_data else "N/A"}<br>'
        f'<span class="status-dot" style="background:{"#00E676" if not is_demo else "#FFD600"};" aria-hidden="true"></span> '
        f'MongoDB {"Connecté" if not is_demo else "Démo"}'
        '</div>',
        unsafe_allow_html=True,
    )

    # Route pages
    if page == 'Dashboard':
        page_dashboard(df, mongo_stats, score_dist, is_demo)
    elif page == 'Analyse IA':
        page_analyse(detector, emo, v6_data, v7_data, cam_classifier, stage1_data)
    elif page == 'Explorateur':
        page_explorer(df, is_demo)
    elif page == 'Performance':
        page_performance()
    elif page == 'À propos':
        page_about()

    footer()


if __name__ == '__main__':
    main()
