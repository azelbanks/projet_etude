"""
Thumalien — Intelligence Command Center
========================================
Dashboard Streamlit de detection de fake news bilingue FR/EN.
Pipeline V5 : TF-IDF(30K) + 15 linguistiques + 7 emotions MLP PyTorch -> LogReg calibre.
Ameliorations V5 : +10K posts FR sociaux synthetiques, 198K textes, FR ultra-court F1=0.90.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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

# ---------------------------------------------------------------------------
#  CSS glassmorphism + dark theme overrides
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ---- glassmorphism card ---- */
.glass-card {
    background: rgba(26, 31, 46, 0.8);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    margin-bottom: 12px;
}

/* ---- glow titles ---- */
h1, h2, h3 {
    text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
}

/* ---- metric cards ---- */
.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.85rem;
    opacity: 0.7;
    margin-top: 4px;
}
.metric-icon {
    font-size: 1.4rem;
    margin-bottom: 4px;
}

/* ---- verdict badges ---- */
.verdict-fiable {
    background: rgba(0, 230, 118, 0.1);
    border: 1px solid rgba(0, 230, 118, 0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.verdict-suspect {
    background: rgba(255, 23, 68, 0.1);
    border: 1px solid rgba(255, 23, 68, 0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

/* ---- dividers ---- */
hr {
    border-color: rgba(0, 212, 255, 0.2) !important;
}

/* ---- dataframe styling ---- */
.stDataFrame [data-testid="stDataFrameResizable"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ---- primary button ---- */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00D4FF, #0066FF);
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #00B8E6, #0055DD);
}

/* ---- footer ---- */
.footer-text {
    text-align: center;
    opacity: 0.5;
    font-size: 0.8rem;
    padding: 16px 0;
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
    detector = ExpertFakeNewsDetector(model_dir=model_dir, threshold=0.44)
    # V5 = +10K FR social synthetique, FR ultra-court F1=0.90
    # V4 = FR court ameliore + 15 features linguistiques + augmentation FR
    # V3 = features linguistiques corrigees
    # V2 = fallback (avant correction features)
    v5_exists = os.path.exists(os.path.join(model_dir, 'model_expert_v5.pkl'))
    v4_exists = os.path.exists(os.path.join(model_dir, 'model_expert_v4.pkl'))
    v3_exists = os.path.exists(os.path.join(model_dir, 'model_expert_v3.pkl'))
    v2_exists = os.path.exists(os.path.join(model_dir, 'model_expert_v2.pkl'))
    suffix = 'expert_v5' if v5_exists else ('expert_v4' if v4_exists else ('expert_v3' if v3_exists else ('expert_v2' if v2_exists else 'expert')))
    detector.load(suffix=suffix)

    # --- Health check after loading ---
    hc = detector.health_check()
    if not hc['healthy']:
        _logger = logging.getLogger(__name__)
        _logger.warning(
            'Model health check FAILED after loading (suffix=%s). Details: %s',
            suffix,
            hc['details'],
        )

    emo = EmotionFeatureExtractor(model_dir=model_dir)
    emo.load()
    return detector, emo, suffix


# ---------------------------------------------------------------------------
#  Data loading (MongoDB -> fallback demo)
# ---------------------------------------------------------------------------

DEMO_POSTS = [
    # EN fiables
    {"uri": "at://en1", "text": "New study published in Nature confirms the effectiveness of the updated vaccine formula for 2025.", "ai_score_credibility": 0.91, "ai_emotion": "neutre", "ai_language": "en", "prediction_label": "FIABLE"},
    {"uri": "at://en2", "text": "The Federal Reserve announced a quarter-point rate cut today, citing stable inflation data.", "ai_score_credibility": 0.88, "ai_emotion": "neutre", "ai_language": "en", "prediction_label": "FIABLE"},
    {"uri": "at://en3", "text": "Researchers at MIT developed a new carbon capture method that reduces costs by 40 percent.", "ai_score_credibility": 0.85, "ai_emotion": "joie", "ai_language": "en", "prediction_label": "FIABLE"},
    {"uri": "at://en4", "text": "The WHO report shows global life expectancy increased by 2 years over the past decade.", "ai_score_credibility": 0.82, "ai_emotion": "joie", "ai_language": "en", "prediction_label": "FIABLE"},
    # EN suspects
    {"uri": "at://en5", "text": "EXPOSED: Secret government labs are using 5G towers to spread mind-control chemicals!!!", "ai_score_credibility": 0.12, "ai_emotion": "colere", "ai_language": "en", "prediction_label": "SUSPECT"},
    {"uri": "at://en6", "text": "BREAKING: Celebrities are being replaced by clones. Wake up sheeple! Share before they delete this!", "ai_score_credibility": 0.08, "ai_emotion": "peur", "ai_language": "en", "prediction_label": "SUSPECT"},
    {"uri": "at://en7", "text": "Big Pharma doesn't want you to know this ONE trick that cures all diseases overnight!", "ai_score_credibility": 0.15, "ai_emotion": "surprise", "ai_language": "en", "prediction_label": "SUSPECT"},
    {"uri": "at://en8", "text": "They're hiding the REAL numbers! The economy already collapsed, media is lying to you!!!", "ai_score_credibility": 0.11, "ai_emotion": "colere", "ai_language": "en", "prediction_label": "SUSPECT"},
    # FR fiables
    {"uri": "at://fr1", "text": "Le CNRS publie une etude confirmant l'efficacite des nouveaux traitements contre l'hepatite C.", "ai_score_credibility": 0.89, "ai_emotion": "neutre", "ai_language": "fr", "prediction_label": "FIABLE"},
    {"uri": "at://fr2", "text": "La BCE maintient ses taux directeurs inchanges, conformement aux attentes du marche.", "ai_score_credibility": 0.92, "ai_emotion": "neutre", "ai_language": "fr", "prediction_label": "FIABLE"},
    {"uri": "at://fr3", "text": "L'equipe de France de handball remporte le championnat du monde pour la quatrieme fois.", "ai_score_credibility": 0.87, "ai_emotion": "joie", "ai_language": "fr", "prediction_label": "FIABLE"},
    {"uri": "at://fr4", "text": "Adoption definitive de la loi climat par le Parlement, avec 80 pourcent de votes favorables.", "ai_score_credibility": 0.84, "ai_emotion": "joie", "ai_language": "fr", "prediction_label": "FIABLE"},
    # FR suspects
    {"uri": "at://fr5", "text": "SCANDALE : le gouvernement cache la VERITE sur les chemtrails ! Partagez avant censure !!!", "ai_score_credibility": 0.09, "ai_emotion": "colere", "ai_language": "fr", "prediction_label": "SUSPECT"},
    {"uri": "at://fr6", "text": "Les vaccins contiennent des micropuces 5G pour vous controler ! Reveillez-vous !!!", "ai_score_credibility": 0.07, "ai_emotion": "peur", "ai_language": "fr", "prediction_label": "SUSPECT"},
    {"uri": "at://fr7", "text": "ON VOUS MENT : cette plante guerit le cancer en 3 jours, les labos ne veulent pas que vous sachiez !", "ai_score_credibility": 0.13, "ai_emotion": "surprise", "ai_language": "fr", "prediction_label": "SUSPECT"},
]


def _normalize_mongo_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les champs MongoDB legacy pour le dashboard."""
    # Ensure required columns exist with defaults
    if 'prediction_label' not in df.columns:
        df['prediction_label'] = np.nan
    if 'ai_score_credibility' not in df.columns:
        df['ai_score_credibility'] = np.nan
    if 'ai_emotion' not in df.columns:
        df['ai_emotion'] = np.nan

    # prediction_label : int 0/1 -> str FIABLE/SUSPECT, NaN -> 'NON ANALYSE'
    def _norm_label(v):
        if pd.isna(v):
            return 'NON ANALYSE'
        if v == 1 or str(v).upper() == 'SUSPECT':
            return 'SUSPECT'
        return 'FIABLE'
    df['prediction_label'] = df['prediction_label'].apply(_norm_label)

    # ai_score_credibility : fill NaN with 0.5 (unknown)
    df['ai_score_credibility'] = df['ai_score_credibility'].fillna(0.5)

    # ai_emotion : 'joie 😂' -> 'joie', NaN -> 'neutre'
    df['ai_emotion'] = (
        df['ai_emotion'].fillna('neutre').astype(str)
        .str.split().str[0]
        .str.lower()
        .str.strip()
    )

    # ai_language : creer si absent via langdetect
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

    return df


@st.cache_data(ttl=60)
def _fetch_mongo_data() -> tuple[list[dict], int] | None:
    """Fetch recent posts and total count from MongoDB via mongo_aggregations.

    Returns (docs, total_count) or None if MongoDB is unavailable.
    Cached for 60 seconds to avoid hammering the database on every rerun.
    """
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
    return docs, n_total


def get_data() -> tuple[pd.DataFrame, bool]:
    """Tente MongoDB via mongo_aggregations, sinon retourne les donnees demo."""
    result = _fetch_mongo_data()
    if result is not None:
        docs, n_total = result
        df = pd.DataFrame(docs)
        df = _normalize_mongo_df(df)
        df.attrs['n_total_mongo'] = n_total
        return df, False
    return pd.DataFrame(DEMO_POSTS), True


# ---------------------------------------------------------------------------
#  V1.5 analysis on loaded posts
# ---------------------------------------------------------------------------

def _apply_v15_analysis(df: pd.DataFrame, detector, emo, model_suffix: str = 'expert_v5') -> pd.DataFrame:
    """Applique le pipeline V5 sur les posts MongoDB (cache en session_state)."""
    # Hash based on text content + model version to detect changes
    df_hash = hash(tuple(df['text'].values[:50]))
    cache_key = f'analyzed_df_{df_hash}_{model_suffix}_t{detector.threshold}'

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    with st.spinner(f'Analyse V5 de {len(df)} posts Bluesky en cours...'):
        texts = pd.Series(df['text'].values)

        # --- Credibility predictions ---
        results = detector.predict(texts)
        df = df.copy()
        df['prediction_label'] = results['prediction_label'].map(
            {0: 'FIABLE', 1: 'SUSPECT'}
        ).fillna('SUSPECT')
        df['ai_score_credibility'] = results['ai_score_credibility'].values
        df['ai_language'] = results['language'].values

        # --- Emotion analysis ---
        try:
            emo_features = emo.get_emotion_features(texts.tolist())
            df['ai_emotion'] = [
                EMOTION_LABELS[int(np.argmax(p))] for p in emo_features
            ]
        except Exception:
            df['ai_emotion'] = 'neutre'

        # Preserve n_total_mongo attribute
        n_total = df.attrs.get('n_total_mongo', len(df))
        df.attrs['n_total_mongo'] = n_total

    st.session_state[cache_key] = df
    return df


# ---------------------------------------------------------------------------
#  Emotion helpers
# ---------------------------------------------------------------------------

EMOTION_LABELS = ['colere', 'degout', 'joie', 'neutre', 'peur', 'surprise', 'tristesse']
EMOTION_EMOJIS = {
    'colere': '😡', 'degout': '🤢', 'joie': '😊',
    'neutre': '😐', 'peur': '😨', 'surprise': '😮', 'tristesse': '😢',
}
EMOTION_DISPLAY = {
    'colere': 'Colere', 'degout': 'Degout', 'joie': 'Joie',
    'neutre': 'Neutre', 'peur': 'Peur', 'surprise': 'Surprise', 'tristesse': 'Tristesse',
}


def make_radar(values, title, fill_opacity=0.15):
    """Plotly radar chart for 7 emotion probabilities."""
    labels = [EMOTION_DISPLAY[e] for e in EMOTION_LABELS]
    vals = list(values) + [values[0]]
    labels_closed = labels + [labels[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=labels_closed,
        fill='toself',
        fillcolor=f'rgba(0, 212, 255, {fill_opacity})',
        line=dict(color='#00D4FF', width=2),
        name='Emotions',
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(color='#E0E0E0', size=16)),
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='#E0E0E0'),
            radialaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='#E0E0E0', range=[0, max(max(values), 0.5)]),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        showlegend=False,
        margin=dict(t=60, b=30, l=60, r=60),
        height=400,
    )
    return fig


def make_gauge(score):
    """Plotly gauge chart for credibility score."""
    if score >= 0.7:
        bar_color = '#00E676'
    elif score >= 0.4:
        bar_color = '#FF9100'
    else:
        bar_color = '#FF1744'

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=score,
        number=dict(font=dict(size=48, color='#E0E0E0'), valueformat='.2f'),
        title=dict(text='Score de Credibilite', font=dict(color='#E0E0E0', size=16)),
        gauge=dict(
            axis=dict(range=[0, 1], tickfont=dict(color='#E0E0E0'), dtick=0.2),
            bar=dict(color=bar_color, thickness=0.75),
            bgcolor='#1A1F2E',
            borderwidth=0,
            steps=[
                dict(range=[0, 0.4], color='rgba(255, 23, 68, 0.15)'),
                dict(range=[0.4, 0.7], color='rgba(255, 145, 0, 0.15)'),
                dict(range=[0.7, 1], color='rgba(0, 230, 118, 0.15)'),
            ],
        ),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        height=280,
        margin=dict(t=60, b=20, l=40, r=40),
    )
    return fig


# ---------------------------------------------------------------------------
#  Hero header
# ---------------------------------------------------------------------------

def hero(title, subtitle):
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0E1117 0%, #1A1F2E 100%);
                padding: 40px; border-radius: 16px; margin-bottom: 24px;
                border: 1px solid rgba(0, 212, 255, 0.1);">
        <div style="font-size: 3rem; font-weight: 300; letter-spacing: 8px;
                    color: #00D4FF; text-shadow: 0 0 30px rgba(0, 212, 255, 0.4);">
            {title}
        </div>
        <div style="font-size: 1.1rem; color: #E0E0E0; margin-top: 8px; opacity: 0.85;">
            {subtitle}
        </div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(icon, label, value, color):
    return f"""
    <div class="glass-card" style="text-align:center; min-height:140px;">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def footer():
    st.divider()
    st.markdown(
        '<div class="footer-text">'
        'Thumalien v5.0 &bull; Pipeline bilingue FR/EN &bull; Seuil 0.44 &bull; '
        'WCAG 2.1 AA &bull; Descriptions textuelles sur toutes les visualisations'
        '</div>',
        unsafe_allow_html=True,
    )


# ===================================================================
#  PAGE 1 : Vue Globale
# ===================================================================

def page_overview(df, detector, emo):
    hero('THUMALIEN', 'Detection de Fake News Bilingue FR/EN')

    # --- metrics ---
    n_total = len(df)
    n_mongo_total = df.attrs.get('n_total_mongo', n_total)
    n_fiable = (df['prediction_label'] == 'FIABLE').sum()
    n_analysed = (df['prediction_label'] != 'NON ANALYSE').sum()
    pct_fiable = n_fiable / n_analysed * 100 if n_analysed else 0
    mean_cred = df.loc[df['prediction_label'] != 'NON ANALYSE', 'ai_score_credibility'].mean() if n_analysed else 0
    n_fr = (df['ai_language'] == 'fr').sum()
    n_en = (df['ai_language'] == 'en').sum()

    cols = st.columns(4)
    total_label = f'{n_total}' if n_mongo_total == n_total else f'{n_total} / {n_mongo_total:,}'
    cards = [
        metric_card('📊', 'Posts affiches / total', total_label, '#00D4FF'),
        metric_card('✅', 'Fiabilite', f'{pct_fiable:.0f}%', '#00E676'),
        metric_card('🎯', 'Credibilite moy.', f'{mean_cred:.0%}', '#FFD600'),
        metric_card('🌍', 'Bilingue', f'{n_fr} FR / {n_en} EN', '#00D4FF'),
    ]
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)

    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

    # --- charts ---
    col_left, col_right = st.columns(2)

    with col_left:
        # Radar: average emotion profile
        if 'ai_emotion' in df.columns:
            emo_counts = df['ai_emotion'].value_counts(normalize=True)
            avg_vals = [float(emo_counts.get(e, 0)) for e in EMOTION_LABELS]
        else:
            avg_vals = [1 / 7] * 7
        fig_radar = make_radar(avg_vals, 'Profil Emotionnel')
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption(
            "Moyenne des 7 probabilites emotionnelles sur l'ensemble des posts analyses. "
            "Les axes representent : colere, degout, joie, neutre, peur, surprise, tristesse."
        )

    with col_right:
        # Grouped bar: reliability by language
        fiable_en = ((df['ai_language'] == 'en') & (df['prediction_label'] == 'FIABLE')).sum()
        suspect_en = ((df['ai_language'] == 'en') & (df['prediction_label'] == 'SUSPECT')).sum()
        na_en = ((df['ai_language'] == 'en') & (df['prediction_label'] == 'NON ANALYSE')).sum()
        fiable_fr = ((df['ai_language'] == 'fr') & (df['prediction_label'] == 'FIABLE')).sum()
        suspect_fr = ((df['ai_language'] == 'fr') & (df['prediction_label'] == 'SUSPECT')).sum()
        na_fr = ((df['ai_language'] == 'fr') & (df['prediction_label'] == 'NON ANALYSE')).sum()

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=['EN', 'FR'], x=[fiable_en, fiable_fr],
            name='Fiable', orientation='h',
            marker_color='#00E676',
        ))
        fig_bar.add_trace(go.Bar(
            y=['EN', 'FR'], x=[suspect_en, suspect_fr],
            name='Suspect', orientation='h',
            marker_color='#FF1744',
        ))
        if na_en + na_fr > 0:
            fig_bar.add_trace(go.Bar(
                y=['EN', 'FR'], x=[na_en, na_fr],
                name='Non analyse', orientation='h',
                marker_color='#666666',
            ))
        fig_bar.update_layout(
            barmode='group',
            title=dict(text='Fiabilite par Langue', x=0.5, font=dict(color='#E0E0E0', size=16)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0E0E0'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
            margin=dict(t=60, b=50, l=40, r=20),
            height=400,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption(
            "Repartition des posts classes fiables et suspects pour chaque langue detectee."
        )

    # --- Performance par longueur de texte ---
    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
    st.subheader('Performance du modele par longueur de texte')

    length_categories = ['Ultra-court\n(<15 mots)', 'Court\n(15-30 mots)',
                         'Moyen\n(30-100 mots)', 'Long\n(>300 mots)']
    length_f1 = [0.747, 0.844, 0.942, 0.990]
    length_colors = ['#FF9100', '#FFD600', '#00E676', '#00D4FF']

    fig_length = go.Figure(go.Bar(
        x=length_categories,
        y=length_f1,
        marker_color=length_colors,
        text=[f'{v:.3f}' for v in length_f1],
        textposition='outside',
        textfont=dict(color='#E0E0E0', size=13),
    ))
    fig_length.update_layout(
        title=dict(text='F1-score par categorie de longueur', x=0.5,
                   font=dict(color='#E0E0E0', size=16)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)', range=[0, 1.08],
                   title='F1-score'),
        margin=dict(t=60, b=40, l=50, r=20),
        height=380,
    )
    st.plotly_chart(fig_length, use_container_width=True)
    st.caption(
        "F1-score mesure sur le jeu de test selon la longueur du texte. "
        "Les textes ultra-courts (<15 mots) restent le cas le plus difficile (F1=0.747)."
    )

    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

    # --- recent posts table ---
    st.subheader('Derniers posts analyses')
    display_df = df[['text', 'ai_language', 'prediction_label', 'ai_score_credibility']].copy()

    def _brief_reason(row):
        label, score = row['prediction_label'], row['ai_score_credibility']
        if label == 'FIABLE':
            if score >= 0.8:
                return f'Credibilite elevee ({score:.0%})'
            return f'Credibilite moderee ({score:.0%})'
        else:
            if score < 0.2:
                return f'Tres suspect ({score:.0%})'
            return f'Suspect ({score:.0%})'

    display_df['Raison'] = display_df.apply(_brief_reason, axis=1)
    display_df = display_df[['text', 'ai_language', 'prediction_label', 'ai_score_credibility', 'Raison']]
    display_df.columns = ['Texte', 'Langue', 'Label', 'Score', 'Raison']
    display_df['Texte'] = display_df['Texte'].astype(str).str[:120]
    display_df['Langue'] = display_df['Langue'].str.upper()

    def color_label(val):
        if val == 'FIABLE':
            return 'color: #00E676; font-weight: 600'
        return 'color: #FF1744; font-weight: 600'

    styled = (
        display_df.style
        .map(color_label, subset=['Label'])
        .background_gradient(subset=['Score'], cmap='RdYlGn', vmin=0, vmax=1)
        .format({'Score': '{:.2f}'})
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)


# ===================================================================
#  PAGE 2 : Analyse en temps reel
# ===================================================================

def page_realtime(detector, emo):
    hero('Analyse en temps reel',
         'Soumettez un texte pour obtenir son profil de credibilite et son empreinte emotionnelle')

    text_input = st.text_area(
        label='Texte a analyser',
        height=180,
        placeholder='Collez ici un article, un post Bluesky, ou tout texte FR/EN a analyser...',
    )

    clicked = st.button('🔍 Analyser', type='primary', use_container_width=True)

    if clicked and text_input.strip():
        with st.spinner('Analyse en cours...'):
            result = detector.predict(pd.Series([text_input]))
            score = float(result['ai_score_credibility'].iloc[0])
            label = result['prediction_label'].iloc[0]
            lang = result['language'].iloc[0]

            probas = emo.get_emotion_features([text_input])[0]
            dominant_idx = int(np.argmax(probas))
            dominant_emotion = EMOTION_LABELS[dominant_idx]
            dominant_proba = float(probas[dominant_idx])

            # Explainability per-instance
            explanation = detector.explain_prediction(text_input)

        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

        # --- 3 columns ---
        c1, c2, c3 = st.columns([2, 1, 1])

        with c1:
            st.plotly_chart(make_gauge(score), use_container_width=True)
            st.caption(
                "Score entre 0 (probablement faux) et 1 (probablement fiable). "
                "Seuil de decision : 0.44"
            )

        with c2:
            is_fiable = (label == 0) or (str(label).upper() == 'FIABLE') or (isinstance(label, (int, np.integer)) and label == 0)
            if is_fiable:
                st.markdown(
                    '<div class="verdict-fiable">'
                    '<div style="font-size:2.5rem;">✅</div>'
                    '<div style="font-size:1.8rem; font-weight:700; color:#00E676; margin-top:8px;">FIABLE</div>'
                    f'<div style="margin-top:12px; opacity:0.7;">Langue : {lang.upper()}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="verdict-suspect">'
                    '<div style="font-size:2.5rem;">⚠️</div>'
                    '<div style="font-size:1.8rem; font-weight:700; color:#FF1744; margin-top:8px;">SUSPECT</div>'
                    f'<div style="margin-top:12px; opacity:0.7;">Langue : {lang.upper()}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        with c3:
            emoji = EMOTION_EMOJIS.get(dominant_emotion, '❓')
            st.markdown(
                '<div class="glass-card" style="text-align:center;">'
                f'<div style="font-size:2.5rem;">{emoji}</div>'
                f'<div style="font-size:1.5rem; font-weight:600; color:#00D4FF; margin-top:8px;">'
                f'{EMOTION_DISPLAY[dominant_emotion]}</div>'
                f'<div style="margin-top:8px; opacity:0.7;">{dominant_proba:.1%}</div>'
                '<div class="metric-label">Emotion dominante</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # --- radar detail ---
        st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
        fig_detail = make_radar(probas.tolist(), 'Profil Emotionnel du Texte', fill_opacity=0.25)
        st.plotly_chart(fig_detail, use_container_width=True)
        st.caption(
            "Probabilites predites par le modele MLP PyTorch bilingue "
            "pour chacune des 7 classes emotionnelles."
        )

        with st.expander('Voir les probabilites detaillees'):
            prob_df = pd.DataFrame({
                'Emotion': [f'{EMOTION_EMOJIS[e]} {EMOTION_DISPLAY[e]}' for e in EMOTION_LABELS],
                'Probabilite': [f'{p:.4f}' for p in probas],
            })
            st.dataframe(prob_df, hide_index=True, use_container_width=True)

        # --- Explainability : Pourquoi ce verdict ? ---
        if explanation.get('explainable'):
            st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
            st.subheader('Pourquoi ce verdict ?')

            # 1. Bar chart divergent des contributions mots
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
                sorted_data = sorted(
                    zip(all_words, all_contribs, all_colors),
                    key=lambda x: x[1],
                )
                words_s, contribs_s, colors_s = zip(*sorted_data)
                fig_explain = go.Figure(go.Bar(
                    y=list(words_s),
                    x=list(contribs_s),
                    orientation='h',
                    marker_color=list(colors_s),
                ))
                fig_explain.update_layout(
                    title=dict(
                        text='Contribution des mots au verdict',
                        x=0.5,
                        font=dict(color='#E0E0E0', size=16),
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E0E0E0'),
                    xaxis=dict(
                        title='Contribution (+ = suspect, - = fiable)',
                        gridcolor='rgba(255,255,255,0.08)',
                        zeroline=True,
                        zerolinecolor='rgba(255,255,255,0.3)',
                    ),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                    margin=dict(t=60, b=40, l=120, r=20),
                    height=max(300, len(all_words) * 28),
                )
                st.plotly_chart(fig_explain, use_container_width=True)
                st.caption(
                    "Contribution de chaque mot au verdict. Les barres rouges (positives) "
                    "poussent vers SUSPECT, les vertes (negatives) vers FIABLE. "
                    "Calcul exact : coefficient de regression logistique x valeur TF-IDF du mot."
                )

            # 2. Mots sensationnalistes (si détectés)
            if explanation.get('sensationalist_words'):
                pills = ' '.join(
                    f'<span style="background:rgba(255,23,68,0.2); '
                    f'border:1px solid rgba(255,23,68,0.4); '
                    f'border-radius:8px; padding:4px 12px; margin:4px; '
                    f'display:inline-block; font-size:0.9rem;">'
                    f'{s["word"]}</span>'
                    for s in explanation['sensationalist_words']
                )
                st.markdown(
                    '<div class="glass-card">'
                    '<div style="font-size:1.1rem; font-weight:600; '
                    'color:#FF1744; margin-bottom:12px;">'
                    'Mots sensationnalistes detectes</div>'
                    f'<div>{pills}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            # 3. Signaux linguistiques détaillés
            LING_LABELS_FR = {
                'word_count': 'Nombre de mots',
                'caps_ratio': 'Ratio majuscules',
                'exclamation_count': 'Points d\'exclamation',
                'question_count': 'Points d\'interrogation',
                'punct_density': 'Densite de ponctuation',
                'avg_word_length': 'Longueur moyenne des mots',
                'sensationalism_score': 'Score de sensationnalisme',
                'has_url': 'Presence d\'URL',
                'numeric_density': 'Densite numerique',
                'lexical_diversity': 'Diversite lexicale (TTR)',
                'sentence_count': 'Nombre de phrases',
                'avg_sentence_length': 'Longueur moyenne des phrases',
            }

            with st.expander('Signaux linguistiques detailles'):
                ling_data = explanation['linguistic_signals']
                ling_df = pd.DataFrame(ling_data)
                ling_df['Signal'] = ling_df['feature'].map(LING_LABELS_FR)
                ling_df['Valeur'] = ling_df['value'].apply(lambda v: f'{v:.3f}')
                ling_df['Contribution'] = ling_df['contribution'].apply(
                    lambda v: f'{v:+.4f}'
                )
                ling_df['Direction'] = ling_df['direction']
                display_ling = ling_df[['Signal', 'Valeur', 'Contribution', 'Direction']]

                def color_direction(val):
                    if val == 'SUSPECT':
                        return 'color: #FF1744; font-weight: 600'
                    return 'color: #00E676; font-weight: 600'

                styled_ling = display_ling.style.map(
                    color_direction, subset=['Direction']
                )
                st.dataframe(styled_ling, hide_index=True, use_container_width=True)
                st.caption(
                    "Contribution = coefficient du modele x valeur de la feature. "
                    "Positif pousse vers SUSPECT, negatif vers FIABLE."
                )

    elif clicked:
        st.warning('Veuillez saisir un texte a analyser.')


# ===================================================================
#  PAGE 3 : Metriques & Transparence
# ===================================================================

def page_metrics():
    hero('Metriques & Transparence',
         'Validation experimentale, bilan carbone et conformite reglementaire')

    # --- Ablation study ---
    st.subheader('Validation experimentale — Ablation Study (7 conditions)')

    ablation_data = {
        '#': [1, 2, 3, 4, 5, 6, 7],
        'Condition': [
            'EN seul (TF-IDF)',
            'FR seul',
            'Bilingue naif',
            'Bilingue + oversampling x3',
            'Bilingue + class_weight',
            'Bilingue SVM',
            'Bilingue + emotions (V1.5)',
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
        .format({'F1 EN': lambda v: f'{v:.4f}' if pd.notna(v) else '—',
                 'F1 FR': lambda v: f'{v:.4f}' if pd.notna(v) else '—'})
        .background_gradient(subset=['F1 EN'], cmap='Blues', vmin=0.95, vmax=1.0)
        .background_gradient(subset=['F1 FR'], cmap='Blues', vmin=0.50, vmax=1.0)
    )
    st.dataframe(styled_abl, hide_index=True, use_container_width=True)

    # Grouped bar
    conditions = abl_df['Condition']
    f1_en = abl_df['F1 EN'].tolist()
    f1_fr = abl_df['F1 FR'].tolist()

    fig_abl = go.Figure()
    fig_abl.add_trace(go.Bar(
        x=conditions, y=f1_en, name='F1 EN', marker_color='#00D4FF',
    ))
    fig_abl.add_trace(go.Bar(
        x=conditions, y=f1_fr, name='F1 FR', marker_color='#FFD600',
    ))
    fig_abl.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.08)', tickangle=-30),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)', range=[0, 1.05], title='F1-score'),
        legend=dict(orientation='h', y=-0.3, x=0.5, xanchor='center'),
        margin=dict(t=30, b=120, l=50, r=20),
        height=420,
    )
    st.plotly_chart(fig_abl, use_container_width=True)
    st.caption(
        "F1-scores sur holdout pour les 7 conditions de l'ablation study. "
        "La condition 7 (V1.5) integre les 7 features emotionnelles du MLP PyTorch."
    )

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # --- Green IT ---
    st.subheader('Bilan Carbone')

    emissions_path = os.path.join(os.path.dirname(__file__), '..', 'emissions.csv')
    em_df = None
    try:
        if os.path.exists(emissions_path):
            em_df = pd.read_csv(emissions_path)
    except Exception:
        pass

    if em_df is not None and 'emissions' in em_df.columns and len(em_df) > 0:
        total_co2_kg = em_df['emissions'].sum()
        total_co2_g = total_co2_kg * 1000
        total_energy = em_df['energy_consumed'].sum() if 'energy_consumed' in em_df.columns else 0
        total_duration = em_df['duration'].sum() if 'duration' in em_df.columns else 0
        n_runs = len(em_df)

        # --- Metric cards ---
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.markdown(metric_card('🌿', 'Emissions totales', f'{total_co2_g:.2f} g', '#00E676'), unsafe_allow_html=True)
        cc2.markdown(metric_card('⚡', 'Energie consommee', f'{total_energy*1000:.1f} Wh', '#FFD600'), unsafe_allow_html=True)
        cc3.markdown(metric_card('⏱️', 'Temps total', f'{total_duration/60:.1f} min', '#00D4FF'), unsafe_allow_html=True)
        cc4.markdown(metric_card('🔄', 'Runs CodeCarbon', f'{n_runs}', '#E0E0E0'), unsafe_allow_html=True)

        # --- Comparison card ---
        km_voiture = total_co2_kg / 0.12  # ~120g CO2/km en voiture
        smartphones = total_energy * 1000 / 12  # ~12 Wh pour charger un smartphone
        st.markdown(
            '<div class="glass-card">'
            '<div style="font-size:1.1rem; font-weight:600; color:#00D4FF; margin-bottom:12px;">'
            'Equivalences ecologiques</div>'
            '<div style="display:flex; gap:40px; flex-wrap:wrap; font-size:0.95rem; line-height:1.8;">'
            f'<div>🚗 <strong>{km_voiture:.4f} km</strong> en voiture thermique</div>'
            f'<div>📱 <strong>{smartphones:.2f}</strong> charges de smartphone</div>'
            f'<div>🇫🇷 Mix electrique France : <strong>56 g CO2/kWh</strong> (nucleaire dominant)</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

        # --- Detail per run ---
        with st.expander('Detail par run CodeCarbon'):
            detail_cols = ['project_name', 'duration', 'emissions', 'energy_consumed', 'cpu_model', 'ram_total_size']
            available_cols = [c for c in detail_cols if c in em_df.columns]
            detail_df = em_df[available_cols].copy()
            if 'duration' in detail_df.columns:
                detail_df['duration'] = detail_df['duration'].apply(lambda x: f'{x:.1f}s')
            if 'emissions' in detail_df.columns:
                detail_df['emissions'] = detail_df['emissions'].apply(lambda x: f'{x*1000:.4f} g')
            if 'energy_consumed' in detail_df.columns:
                detail_df['energy_consumed'] = detail_df['energy_consumed'].apply(lambda x: f'{x*1000:.2f} Wh')
            if 'ram_total_size' in detail_df.columns:
                detail_df['ram_total_size'] = detail_df['ram_total_size'].apply(lambda x: f'{x:.0f} GB')
            renames = {
                'project_name': 'Projet', 'duration': 'Duree',
                'emissions': 'CO2 (g)', 'energy_consumed': 'Energie (Wh)',
                'cpu_model': 'CPU', 'ram_total_size': 'RAM',
            }
            detail_df.rename(columns=renames, inplace=True)
            st.dataframe(detail_df, hide_index=True, use_container_width=True)

    else:
        st.markdown(
            '<div class="glass-card" style="text-align:center;">'
            '<div style="font-size:2rem;">🌿</div>'
            '<div class="metric-value" style="color:#888;">Aucune donnee</div>'
            '<div class="metric-label">Lancez un entrainement avec CodeCarbon pour mesurer les emissions</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # --- Limitations connues ---
    st.subheader('Limitations connues')

    lim_col1, lim_col2, lim_col3 = st.columns(3)
    with lim_col1:
        st.markdown(
            '<div class="glass-card" style="text-align:center;">'
            '<div style="font-size:2rem;">🇫🇷</div>'
            '<div class="metric-value" style="color:#FF9100; font-size:2rem;">F1 = 0.65</div>'
            '<div class="metric-label">Textes ultra-courts en francais (<15 mots).<br>'
            'Le manque de contexte degrade fortement la detection.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with lim_col2:
        st.markdown(
            '<div class="glass-card" style="text-align:center;">'
            '<div style="font-size:2rem;">🎯</div>'
            '<div class="metric-value" style="color:#FF9100; font-size:2rem;">ECE = 0.049</div>'
            '<div class="metric-label">Erreur de calibration attendue.<br>'
            'Les probabilites predites devient legerement des frequences reelles.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with lim_col3:
        st.markdown(
            '<div class="glass-card" style="text-align:center;">'
            '<div style="font-size:2rem;">⚠️</div>'
            '<div class="metric-value" style="color:#FF9100; font-size:2rem;">84%</div>'
            '<div class="metric-label">des erreurs portent sur des contenus neutres.<br>'
            'Les textes factuels sans marqueur fort sont les plus difficiles a classer.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # --- Roadmap ---
    st.subheader("Roadmap d'evolution")

    with st.expander('✅ V1 — TF-IDF + LogReg baseline (completee)'):
        st.markdown(
            '- TF-IDF 30K features sur corpus ISOT anglais\n'
            '- LogReg avec calibration de probabilites\n'
            '- F1 EN : 0.9956 sur holdout\n'
            '- Limitation : aucune capacite francaise'
        )
    with st.expander('✅ V1.5 — Features emotionnelles MLP PyTorch (completee)'):
        st.markdown(
            '- Dataset Kaggle FrenchFakeNewsDetector (~9 500 articles FR reels)\n'
            '- MLP PyTorch bilingue 7 classes emotionnelles\n'
            '- 12 features linguistiques + 7 probas emotionnelles\n'
            '- F1 EN : 0.9852 | F1 FR : 0.9818\n'
            '- Limitation : domain shift sur posts courts (77% suspects)'
        )
    with st.expander('✅ V2 — Integration datasets sociaux + seuil 0.44'):
        st.markdown(
            '- 3 datasets sociaux : FakeNewsNet (23K titres), CONSTRAINT (10K tweets), Credibility Corpus (11K tweets)\n'
            '- 145 703 textes d\'entrainement (63% courts < 50 mots)\n'
            '- Seuil de decision ajuste a 0.44 (optimise sur Bluesky)\n'
            '- CV F1 : 0.897 | Limitation : FR court F1=0.65'
        )
    with st.expander('✅ V3 — Correction features linguistiques'):
        st.markdown(
            '- Bug fix : features linguistiques (caps_ratio, exclamation, etc.) calculees sur texte original\n'
            '- Retraining avec features corrigees\n'
            '- CV F1 : 0.900 | Precision +19.3%\n'
            '- Limitation : FR court toujours faible (F1=0.65)'
        )
    with st.expander('✅ V4 — Amelioration FR court + augmentation donnees'):
        st.markdown(
            '- 187 782 textes d\'entrainement | FR=76K (40%) vs EN=112K (60%)\n'
            '- Augmentation FR courte : 27K textes courts generes depuis articles\n'
            '- 3 nouvelles features : all_caps_words_ratio, interpellation_score, is_short_text\n'
            '- Vocabulaire sensationnaliste FR enrichi (+16 termes social media)\n'
            '- **FR court F1 : 0.65 -> 0.86 (+32%)** | FR global F1 : 0.935\n'
            '- Health check : PASS (5/5)'
        )
    with st.expander('🔵 V5 — Integration FR social + 10K posts synthetiques (actuelle)'):
        st.markdown(
            '- 197 782 textes d\'entrainement | FR=86K (43.5%) vs EN=112K (56.5%)\n'
            '- +10K posts FR sociaux synthetiques (5K suspect + 5K fiable)\n'
            '- **FR ultra-court F1 : 0.86 -> 0.90 (+10.4%)** | FR global F1 : 0.944\n'
            '- Test bilingue : 12/12 (vs 9/10 en V4)\n'
            '- Health check : PASS (5/5) | Temps entrainement : 30 min'
        )
    with st.expander('🟡 V6 — Pipeline hybride + RoBERTa EN (prevu)'):
        st.markdown(
            '- Pipeline hybride stacking TF-IDF V5 + CamemBERT (P1)\n'
            '- Re-fine-tuning CamemBERT avec donnees FR sociales (P2)\n'
            '- Seuil adaptatif par langue FR/EN (P3)\n'
            '- RoBERTa fine-tune pour l\'anglais court (P4)\n'
            '- Objectif : F1 FR court > 0.92 | F1 EN court > 0.85'
        )

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # --- Conformite ---
    st.subheader('Conformite reglementaire')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="glass-card">'
            '<div style="font-size:1.5rem; margin-bottom:8px;">🔒 RGPD</div>'
            '<div style="font-size:0.9rem; line-height:1.6;">'
            'Donnees publiques uniquement. Pas de profilage individuel. '
            'Droit a l\'effacement via suppression MongoDB. '
            'Aucune donnee personnelle stockee.'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="glass-card">'
            '<div style="font-size:1.5rem; margin-bottom:8px;">⚖️ IA Act</div>'
            '<div style="font-size:0.9rem; line-height:1.6;">'
            'Systeme classe risque limite (art. 52). '
            'Scores presentes comme aide a la decision. '
            'Biais documentes dans l\'ablation study.'
            '</div></div>',
            unsafe_allow_html=True,
        )


# ===================================================================
#  Main
# ===================================================================

def main():
    st.set_page_config(
        page_title='Thumalien — Fake News Detection',
        page_icon='🔍',
        layout='wide',
        initial_sidebar_state='expanded',
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.markdown(
        '<div style="text-align:center; padding:16px 0;">'
        '<span style="font-size:1.5rem; letter-spacing:4px; color:#00D4FF; font-weight:300;">'
        'THUMALIEN</span></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    page = st.sidebar.radio(
        'Navigation',
        ['🔍 Vue Globale', '⚡ Analyse en temps reel', '📊 Metriques & Transparence'],
        label_visibility='collapsed',
    )

    st.sidebar.divider()
    st.sidebar.caption('v5.0 — Pipeline bilingue FR/EN + 10K FR social (seuil 0.44)')

    # Load resources
    detector, emo, model_suffix = load_pipeline()
    df, is_demo = get_data()

    if is_demo:
        st.sidebar.info('📋 Mode demo — donnees simulees (MongoDB non connecte)')
    else:
        # Analyse V5 sur les posts MongoDB (cache en session_state)
        df = _apply_v15_analysis(df, detector, emo, model_suffix)

    # Route pages
    if page == '🔍 Vue Globale':
        page_overview(df, detector, emo)
    elif page == '⚡ Analyse en temps reel':
        page_realtime(detector, emo)
    else:
        page_metrics()

    footer()


if __name__ == '__main__':
    main()
