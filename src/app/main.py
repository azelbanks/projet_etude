import streamlit as st
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Thumalien Dashboard",
    page_icon="👽",
    layout="wide",
)

st.title("🕵️ Thumalien : Détecteur de Fake News & Émotions")
st.markdown("### *Monitoring en temps réel du flux Bluesky — Pipeline Expert Bilingue*")

# --- CONNEXION MONGO ---
@st.cache_resource
def init_connection():
    mongo_user = os.getenv('MONGO_USER')
    mongo_password = os.getenv('MONGO_PASSWORD')
    mongo_host = os.getenv('MONGO_HOST', 'mongodb')
    if mongo_user and mongo_password:
        from urllib.parse import quote_plus
        uri = f"mongodb://{quote_plus(mongo_user)}:{quote_plus(mongo_password)}@{mongo_host}:27017/?authSource=admin"
    else:
        uri = f"mongodb://{mongo_host}:27017/"
    return MongoClient(uri)

try:
    client = init_connection()
    db = client['thumalien_db']
    collection = db['raw_posts']
    collection.find_one()
except Exception as e:
    st.error(f"Erreur de connexion à MongoDB : {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Section",
    ["Tableau de bord", "Performance du modèle", "Green IT"],
)
st.sidebar.divider()
if st.sidebar.button("Rafraîchir les données"):
    st.cache_data.clear()
    st.rerun()

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data(ttl=30)
def load_data():
    cursor = collection.find({}, {"_id": 0})
    df = pd.DataFrame(list(cursor))
    return df

df = load_data()

if df.empty:
    st.warning("Aucune donnée dans la base. Vérifiez que le collecteur tourne.")
    st.stop()

# Colonnes par défaut
if 'ai_emotion' not in df.columns:
    df['ai_emotion'] = "Inconnu"
if 'ai_score_credibility' not in df.columns:
    df['ai_score_credibility'] = 0.5
if 'prediction_label' not in df.columns:
    df['prediction_label'] = 0
if 'ai_language' not in df.columns:
    df['ai_language'] = "en"

# ====================================================================
#  PAGE 1 : TABLEAU DE BORD PRINCIPAL
# ====================================================================
if page == "Tableau de bord":

    # --- KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Posts", f"{len(df):,}")
    with col2:
        nb_fakes = df[df['prediction_label'] == 1].shape[0]
        pct_fake = nb_fakes / len(df) * 100 if len(df) > 0 else 0
        st.metric("Alertes Fake News", f"{nb_fakes:,}", f"{pct_fake:.1f}%")
    with col3:
        avg_cred = df['ai_score_credibility'].mean()
        st.metric("Crédibilité Moyenne", f"{avg_cred:.1%}")
    with col4:
        top_emotion = (
            df['ai_emotion'].mode()[0]
            if not df['ai_emotion'].empty
            else "N/A"
        )
        st.metric("Ambiance Dominante", top_emotion)

    st.divider()

    # --- GRAPHIQUES LIGNE 1 ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Répartition des Émotions")
        if not df['ai_emotion'].empty:
            emotion_counts = (
                df['ai_emotion']
                .value_counts()
                .reset_index()
            )
            emotion_counts.columns = ['Emotion', 'Count']
            fig_pie = px.pie(
                emotion_counts,
                values='Count',
                names='Emotion',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Score de Crédibilité")
        fig_hist = px.histogram(
            df,
            x="ai_score_credibility",
            nbins=30,
            color_discrete_sequence=['#2ecc71'],
            labels={"ai_score_credibility": "Score (0=Fake, 1=Fiable)"},
        )
        fig_hist.add_vline(
            x=0.5, line_dash="dash", line_color="red",
            annotation_text="Seuil 50%",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # --- GRAPHIQUES LIGNE 2 ---
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Distribution des Langues")
        if 'ai_language' in df.columns:
            lang_map = {'en': 'Anglais', 'fr': 'Français', 'other': 'Autre'}
            lang_counts = (
                df['ai_language']
                .map(lang_map)
                .fillna('Autre')
                .value_counts()
                .reset_index()
            )
            lang_counts.columns = ['Langue', 'Count']
            fig_lang = px.pie(
                lang_counts,
                values='Count',
                names='Langue',
                color_discrete_sequence=['#3498db', '#e74c3c', '#95a5a6'],
                hole=0.4,
            )
            st.plotly_chart(fig_lang, use_container_width=True)

    with c4:
        st.subheader("Fake News par Langue")
        if 'ai_language' in df.columns:
            cross = (
                df.groupby('ai_language')['prediction_label']
                .value_counts()
                .unstack(fill_value=0)
                .rename(columns={0: 'Fiable', 1: 'Suspect'})
            )
            cross.index = cross.index.map(
                {'en': 'Anglais', 'fr': 'Français', 'other': 'Autre'}
            )
            fig_cross = px.bar(
                cross.reset_index(),
                x=cross.index.name or 'ai_language',
                y=['Fiable', 'Suspect'],
                barmode='group',
                color_discrete_map={
                    'Fiable': '#2ecc71',
                    'Suspect': '#e74c3c',
                },
            )
            st.plotly_chart(fig_cross, use_container_width=True)

    # --- DERNIERS POSTS ---
    st.divider()
    st.subheader("Derniers Posts Analysés")

    cols_to_show = [
        'text', 'ai_language', 'ai_emotion',
        'ai_score_credibility', 'prediction_label', 'ai_analysis_log',
    ]
    valid_cols = [c for c in cols_to_show if c in df.columns]

    if not df.empty:
        display_df = df[valid_cols].tail(15).iloc[::-1].copy()
        if 'prediction_label' in display_df.columns:
            display_df['prediction_label'] = display_df['prediction_label'].map(
                {0: 'Fiable', 1: 'Suspect'}
            )
        st.dataframe(display_df, use_container_width=True, height=400)


# ====================================================================
#  PAGE 2 : PERFORMANCE DU MODÈLE
# ====================================================================
elif page == "Performance du modèle":
    st.header("Performance du Modèle Expert")

    # Charger les métriques sauvegardées
    metrics_path = '/app/models/metrics_expert.pkl'
    eval_img_path = '/app/models/evaluation_expert.png'

    try:
        import joblib
        metrics = joblib.load(metrics_path)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Accuracy (CV)",
                f"{metrics['cv_accuracy_mean']:.2%}",
                f"± {metrics['cv_accuracy_std']:.2%}",
            )
        with col2:
            st.metric(
                "F1-Score (CV)",
                f"{metrics['cv_f1_mean']:.2%}",
                f"± {metrics['cv_f1_std']:.2%}",
            )
        with col3:
            st.metric("Precision (CV)", f"{metrics['cv_precision_mean']:.2%}")
        with col4:
            st.metric("ROC-AUC (CV)", f"{metrics['cv_roc_auc_mean']:.2%}")

        st.divider()

        # Détails du modèle
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Configuration")
            st.markdown(f"""
            - **Type de modèle** : `{metrics['model_type'].upper()}`
            - **Échantillons** : {metrics['n_samples']:,}
            - **Features TF-IDF** : {metrics['n_features_tfidf']:,}
            - **Features linguistiques** : {metrics['n_features_linguistic']}
            - **Validation croisée** : {metrics['n_folds']}-fold stratifiée
            - **Biais Reuters** : Supprimé
            - **Support bilingue** : FR + EN
            """)

        with c2:
            st.subheader("Scores par Fold")
            fold_data = pd.DataFrame({
                'Fold': [f"Fold {i+1}" for i in range(metrics['n_folds'])],
                'Accuracy': metrics['cv_accuracy_per_fold'],
                'F1-Score': metrics['cv_f1_per_fold'],
            })
            fig_folds = px.bar(
                fold_data.melt(id_vars='Fold', var_name='Métrique', value_name='Score'),
                x='Fold', y='Score', color='Métrique',
                barmode='group',
                color_discrete_map={'Accuracy': '#3498db', 'F1-Score': '#e74c3c'},
            )
            fig_folds.update_yaxes(range=[0.5, 1.0])
            st.plotly_chart(fig_folds, use_container_width=True)

        # Image d'évaluation (matrice de confusion + ROC)
        if os.path.exists(eval_img_path):
            st.divider()
            st.subheader("Matrice de Confusion & Courbe ROC")
            st.image(eval_img_path, use_container_width=True)

    except FileNotFoundError:
        st.warning(
            "Métriques du modèle expert non trouvées. "
            "Exécutez le Notebook 05 pour entraîner le modèle."
        )
    except Exception as e:
        st.error(f"Erreur de chargement des métriques : {e}")


# ====================================================================
#  PAGE 3 : GREEN IT
# ====================================================================
elif page == "Green IT":
    st.header("Monitoring Green IT — Empreinte Carbone IA")

    emissions_path = '/app/emissions.csv'

    if os.path.exists(emissions_path):
        df_em = pd.read_csv(emissions_path)

        if not df_em.empty:
            # KPIs Green IT
            total_co2 = df_em['emissions'].sum()
            total_energy = df_em['energy_consumed'].sum()
            total_duration = df_em['duration'].sum()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("CO2 Total", f"{total_co2 * 1000:.4f} g")
            with col2:
                st.metric("Énergie Totale", f"{total_energy * 1000:.4f} Wh")
            with col3:
                st.metric("Durée Totale", f"{total_duration:.1f} s")
            with col4:
                n_runs = len(df_em)
                st.metric("Exécutions trackées", n_runs)

            st.divider()

            # Graphiques
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Émissions CO2 par Opération")
                summary = (
                    df_em.groupby('project_name')
                    .agg({'emissions': 'sum', 'energy_consumed': 'sum', 'duration': 'sum'})
                    .reset_index()
                )
                summary['emissions_g'] = summary['emissions'] * 1000

                fig_co2 = px.bar(
                    summary,
                    x='project_name',
                    y='emissions_g',
                    color_discrete_sequence=['#27ae60'],
                    labels={
                        'project_name': 'Opération',
                        'emissions_g': 'CO2 (grammes)',
                    },
                )
                fig_co2.update_xaxes(tickangle=45)
                st.plotly_chart(fig_co2, use_container_width=True)

            with c2:
                st.subheader("Énergie Consommée par Opération")
                summary['energy_wh'] = summary['energy_consumed'] * 1000

                fig_energy = px.bar(
                    summary,
                    x='project_name',
                    y='energy_wh',
                    color_discrete_sequence=['#2980b9'],
                    labels={
                        'project_name': 'Opération',
                        'energy_wh': 'Énergie (Wh)',
                    },
                )
                fig_energy.update_xaxes(tickangle=45)
                st.plotly_chart(fig_energy, use_container_width=True)

            # Équivalences pédagogiques
            st.divider()
            st.subheader("Mise en Perspective")

            eq_col1, eq_col2, eq_col3 = st.columns(3)
            with eq_col1:
                km_car = total_co2 / 0.120
                st.metric(
                    "Équivalent voiture",
                    f"{km_car * 1000:.2f} m",
                    help="Basé sur 120g CO2/km (voiture moyenne)",
                )
            with eq_col2:
                hours_streaming = total_co2 / 0.036
                st.metric(
                    "Équivalent streaming",
                    f"{hours_streaming * 3600:.1f} s",
                    help="Basé sur 36g CO2/h de streaming vidéo",
                )
            with eq_col3:
                google_searches = total_co2 / 0.0002
                st.metric(
                    "Équivalent recherches Google",
                    f"{google_searches:.1f}",
                    help="Basé sur 0.2g CO2 par recherche Google",
                )

            st.info(
                "Le projet Thumalien a une empreinte carbone très faible. "
                "L'utilisation de modèles TF-IDF + LogReg plutôt que de grands "
                "transformers (GPT, LLaMA) réduit la consommation par un facteur 100x."
            )

            # Tableau détaillé
            st.divider()
            st.subheader("Détail des Exécutions")
            display_cols = [
                'timestamp', 'project_name', 'duration',
                'emissions', 'energy_consumed',
            ]
            valid_display = [c for c in display_cols if c in df_em.columns]
            st.dataframe(
                df_em[valid_display].sort_values('timestamp', ascending=False),
                use_container_width=True,
            )

        else:
            st.warning("Le fichier emissions.csv est vide.")
    else:
        st.warning(
            "Fichier emissions.csv non trouvé. "
            "Exécutez les notebooks avec CodeCarbon activé."
        )

    # Image du rapport Green IT
    green_img = '/app/models/green_it_report.png'
    if os.path.exists(green_img):
        st.divider()
        st.subheader("Rapport Visuel Green IT")
        st.image(green_img, use_container_width=True)
