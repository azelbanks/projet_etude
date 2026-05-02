"""
Thumalien — Expert Bilingual Fake News Detector
================================================

Pipeline de détection de désinformation bilingue (FR/EN) de niveau production.

Problème résolu :
    Le modèle baseline (LogReg + TF-IDF sur Kaggle Fake/True News) apprenait
    à reconnaître le style Reuters (99% accuracy artificielle) au lieu de
    détecter les fake news. Appliqué à des posts Bluesky, il classait tout
    comme FAKE.

Solution :
    1. Nettoyage du biais Reuters dans le dataset d'entraînement
    2. Feature engineering linguistique (signaux de désinformation)
    3. TF-IDF amélioré (n-grams, sublinear TF)
    4. Validation croisée stratifiée rigoureuse
    5. Détection de langue pour routage FR/EN
    6. Monitoring CodeCarbon intégré

Auteur  : Thumalien Team
Version : 2.0 (Expert)
"""

import re
import os
import logging
import pickle
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

logger = logging.getLogger(__name__)


# ================================================================
#  1. NETTOYAGE DU DATASET — Suppression du biais Reuters
# ================================================================

class DatasetCleaner:
    """
    Nettoie le dataset Kaggle Fake/True News pour supprimer
    le biais d'attribution Reuters.

    Pourquoi ?
        True.csv = 100% articles Reuters → le modèle apprend à
        reconnaître "WASHINGTON (Reuters) -" et non pas la véracité.
    """

    AGENCY_PATTERNS = [
        # Préfixes d'agences de presse
        r'^[A-Z][A-Z\s/,\.]{2,40}\s*\(Reuters\)\s*[-–—]\s*',
        r'^[A-Z][A-Z\s/,\.]{2,40}\s*\(AP\)\s*[-–—]\s*',
        r'^[A-Z][A-Z\s/,\.]{2,40}\s*\(AFP\)\s*[-–—]\s*',
        r'^[A-Z][A-Z\s/,\.]{2,40}\s*[-–—]\s*(?=[A-Z])',
        # Attributions dans le corps du texte
        r'\(Reuters\)',
        r'\(AP\)',
        r'\(AFP\)',
        # Bylines et crédits en fin d'article
        r'Reporting by\s+.{5,80}?(?:;|$)',
        r'Editing by\s+.{5,80}?(?:;|$)',
        r'Additional reporting by\s+.{5,80}?(?:;|$)',
        r'Writing by\s+.{5,80}?(?:;|$)',
        r'\(Reporting by\s+.{5,80}?\)',
        r'\(Writing by\s+.{5,80}?\)',
        r'Our Standards:\s*The Thomson Reuters Trust Principles\.?',
        r'\|\s*Reuters\s*$',
        # Patterns d'agences françaises
        r'Avec AFP',
        r'Source\s*:\s*(AFP|Reuters|AP)',
        r'Rédaction de\s+.{5,80}?(?:;|$)',
        r'Édité par\s+.{5,80}?(?:;|$)',
    ]

    @classmethod
    def remove_agency_bias(cls, text: str) -> str:
        """Supprime tous les marqueurs d'agences de presse du texte."""
        if not isinstance(text, str):
            return ""
        for pattern in cls.AGENCY_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text.strip()

    @classmethod
    def clean_for_ml(cls, text: str) -> str:
        """Nettoyage ML : normalisation, URLs, mentions."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', ' ', text)
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'[^\w\sàâäéèêëïîôùûüÿçœæ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @classmethod
    def prepare_clean_dataset(
        cls,
        fake_path: str,
        true_path: str,
        remove_short: int = 20,
    ) -> pd.DataFrame:
        """
        Charge, nettoie (suppression biais Reuters) et retourne le dataset.

        Returns
        -------
        DataFrame avec colonnes: text_original, text_clean, label
        """
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)

        df_fake['label'] = 1
        df_true['label'] = 0

        df = pd.concat(
            [df_fake[['text', 'label']], df_true[['text', 'label']]],
            ignore_index=True,
        )

        df.rename(columns={'text': 'text_original'}, inplace=True)

        # Suppression du biais Reuters
        df['text_debiased'] = df['text_original'].apply(cls.remove_agency_bias)

        # Nettoyage ML
        df['text_clean'] = df['text_debiased'].apply(cls.clean_for_ml)

        # Suppression des textes trop courts après nettoyage
        df = df[df['text_clean'].str.split().str.len() >= remove_short]
        df = df.reset_index(drop=True)

        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(
            "Dataset nettoyé: %d articles | Distribution: %s",
            len(df),
            df['label'].value_counts().to_dict(),
        )
        return df

    @classmethod
    def prepare_bilingual_dataset(
        cls,
        fake_path: str,
        true_path: str,
        french_path: Optional[str] = None,
        kaggle_fr_dir: Optional[str] = None,
        fakenewsnet_dir: Optional[str] = None,
        constraint_dir: Optional[str] = None,
        credibility_dir: Optional[str] = None,
        remove_short: int = 20,
        social_remove_short: int = 5,
        french_oversample: int = 3,
        social_oversample: int = 2,
        en_subsample: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Crée un dataset bilingue en combinant les données anglaises nettoyées,
        le dataset français, et optionnellement des datasets de textes sociaux.

        Parameters
        ----------
        fake_path : Chemin vers Fake.csv (EN)
        true_path : Chemin vers True.csv (EN)
        french_path : Chemin vers french_fake_news.csv (FR) — fallback
        kaggle_fr_dir : Répertoire Kaggle FrenchFakeNewsDetector (~9 500 articles, prioritaire)
        fakenewsnet_dir : Répertoire FakeNewsNet (titres GossipCop + PolitiFact)
        constraint_dir : Répertoire CONSTRAINT 2021 (tweets COVID EN)
        credibility_dir : Répertoire Credibility Corpus (tweets FR+EN)
        remove_short : Longueur minimale en mots pour articles (défaut: 20)
        social_remove_short : Longueur minimale en mots pour textes sociaux (défaut: 5)
        french_oversample : Facteur d'oversampling pour les données FR (défaut: 3)
        social_oversample : Facteur d'oversampling pour les textes sociaux (défaut: 2)
        en_subsample : Si défini, sous-échantillonne les données EN

        Returns
        -------
        DataFrame avec colonnes: text_original, text_clean, label, language
        """
        # 1. Données anglaises (pipeline existant)
        df_en = cls.prepare_clean_dataset(fake_path, true_path, remove_short)
        df_en['language'] = 'en'

        if en_subsample and en_subsample < len(df_en):
            df_en = df_en.sample(n=en_subsample, random_state=42).reset_index(drop=True)

        # 2. Données françaises — Kaggle FR en priorité, fallback sur french_path
        if kaggle_fr_dir and os.path.isdir(kaggle_fr_dir):
            try:
                df_fr = cls.load_kaggle_french(kaggle_fr_dir, remove_short)
                logger.info("Données FR chargées depuis Kaggle FrenchFakeNewsDetector")
            except FileNotFoundError:
                logger.warning(
                    "Kaggle FR dir existe mais fichiers manquants, fallback sur french_path"
                )
                df_fr = None
        else:
            df_fr = None

        if df_fr is None and french_path:
            df_fr = pd.read_csv(french_path)
            df_fr.rename(columns={'text': 'text_original'}, inplace=True)
            df_fr['text_debiased'] = df_fr['text_original'].apply(cls.remove_agency_bias)
            df_fr['text_clean'] = df_fr['text_debiased'].apply(cls.clean_for_ml)
            df_fr = df_fr[df_fr['text_clean'].str.split().str.len() >= remove_short]
            logger.info("Données FR chargées depuis french_path (fallback)")

        if df_fr is None:
            raise ValueError(
                "Aucune source FR disponible. Fournissez kaggle_fr_dir ou french_path."
            )

        df_fr['language'] = 'fr'

        # Garder uniquement les colonnes alignées avec df_en
        cols = ['text_original', 'text_debiased', 'text_clean', 'label', 'language']
        df_fr = df_fr[[c for c in cols if c in df_fr.columns]]
        df_en = df_en[[c for c in cols if c in df_en.columns]]

        # 3. Oversampling FR
        if french_oversample > 1:
            df_fr = pd.concat(
                [df_fr] * french_oversample, ignore_index=True
            )

        # 4. Datasets sociaux (textes courts)
        social_parts = []

        if fakenewsnet_dir and os.path.isdir(fakenewsnet_dir):
            try:
                df_fnn = cls.load_fakenewsnet(fakenewsnet_dir, social_remove_short)
                df_fnn['language'] = 'en'
                social_parts.append(df_fnn)
                logger.info("FakeNewsNet intégré : %d titres", len(df_fnn))
            except FileNotFoundError as e:
                logger.warning("FakeNewsNet non chargé : %s", e)

        if constraint_dir and os.path.isdir(constraint_dir):
            try:
                df_cst = cls.load_constraint(constraint_dir, social_remove_short)
                df_cst['language'] = 'en'
                social_parts.append(df_cst)
                logger.info("CONSTRAINT intégré : %d tweets", len(df_cst))
            except FileNotFoundError as e:
                logger.warning("CONSTRAINT non chargé : %s", e)

        if credibility_dir and os.path.isdir(credibility_dir):
            try:
                df_cc = cls.load_credibility_corpus(credibility_dir, social_remove_short)
                # language déjà définie dans le loader
                social_parts.append(df_cc)
                logger.info("Credibility Corpus intégré : %d tweets", len(df_cc))
            except FileNotFoundError as e:
                logger.warning("Credibility Corpus non chargé : %s", e)

        # Concat social + oversampling
        df_social = None
        if social_parts:
            df_social = pd.concat(social_parts, ignore_index=True)
            # Aligner les colonnes
            df_social = df_social[[c for c in cols if c in df_social.columns]]
            if social_oversample > 1:
                df_social = pd.concat(
                    [df_social] * social_oversample, ignore_index=True
                )
            logger.info(
                "Datasets sociaux combinés : %d textes (x%d oversample) | EN=%d, FR=%d",
                len(df_social),
                social_oversample,
                (df_social['language'] == 'en').sum(),
                (df_social['language'] == 'fr').sum(),
            )

        # 5. Concat final + shuffle
        parts = [df_en, df_fr]
        if df_social is not None:
            parts.append(df_social)
        df = pd.concat(parts, ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(
            "Dataset bilingue: %d articles | EN=%d, FR=%d | Distribution labels: %s",
            len(df),
            (df['language'] == 'en').sum(),
            (df['language'] == 'fr').sum(),
            df['label'].value_counts().to_dict(),
        )
        return df

    @classmethod
    def generate_fr_short_augmentation(cls, df_fr: pd.DataFrame) -> pd.DataFrame:
        """
        Génère des textes courts FR à partir d'articles longs FR.

        Stratégies :
        1. Extraction de la première phrase de chaque article
        2. Extraction d'un titre synthétique (premiers 8-15 mots)

        Cela comble le manque de données FR courtes (type Bluesky/Twitter).

        Parameters
        ----------
        df_fr : DataFrame FR avec colonnes text_original, text_clean, label

        Returns
        -------
        DataFrame de textes courts FR générés
        """
        short_rows = []

        for _, row in df_fr.iterrows():
            text = str(row['text_original'])
            label = row['label']

            # Stratégie 1 : première phrase
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if sentences:
                first_sent = sentences[0].strip()
                words_first = first_sent.split()
                if 3 <= len(words_first) <= 25:
                    clean = cls.clean_for_ml(first_sent)
                    if len(clean.split()) >= 3:
                        short_rows.append({
                            'text_original': first_sent,
                            'text_debiased': first_sent,
                            'text_clean': clean,
                            'label': label,
                            'language': 'fr',
                        })

            # Stratégie 2 : titre synthétique (premiers 8-12 mots)
            words = text.split()
            if len(words) > 15:
                n = min(12, max(8, len(words) // 10))
                title = ' '.join(words[:n])
                clean_title = cls.clean_for_ml(title)
                if len(clean_title.split()) >= 5:
                    short_rows.append({
                        'text_original': title,
                        'text_debiased': title,
                        'text_clean': clean_title,
                        'label': label,
                        'language': 'fr',
                    })

        df_short = pd.DataFrame(short_rows)
        logger.info(
            "Augmentation FR courte : %d textes générés (< 25 mots) | Distribution : %s",
            len(df_short),
            df_short['label'].value_counts().to_dict() if len(df_short) > 0 else {},
        )
        return df_short

    @classmethod
    def audit_reuters_leakage(cls, df_true: pd.DataFrame) -> Dict:
        """
        Quantifie le biais Reuters dans le dataset True.csv.

        Returns
        -------
        Dict avec statistiques de leakage.
        """
        texts = df_true['text'].astype(str)

        has_reuters = texts.str.contains(r'\(Reuters\)', case=False).sum()
        has_city_dash = texts.str.contains(
            r'^[A-Z]{2,}.*[-–—]', regex=True
        ).sum()
        has_byline = texts.str.contains(
            r'Reporting by|Editing by', case=False
        ).sum()

        total = len(texts)
        return {
            'total_articles': total,
            'has_reuters_marker': int(has_reuters),
            'has_reuters_pct': round(has_reuters / total * 100, 1),
            'has_city_prefix': int(has_city_dash),
            'has_city_prefix_pct': round(has_city_dash / total * 100, 1),
            'has_journalist_byline': int(has_byline),
            'has_byline_pct': round(has_byline / total * 100, 1),
        }

    @classmethod
    def load_kaggle_french(
        cls,
        kaggle_dir: str,
        remove_short: int = 20,
    ) -> pd.DataFrame:
        """
        Charge le dataset Kaggle FrenchFakeNewsDetector (~9 500 articles).

        Fichiers attendus dans kaggle_dir :
            - datafake_train.csv (~6 645 articles)
            - datafake_test.csv (~2 849 articles)
        Format : CSV séparateur ';', colonnes media, post, fake

        Parameters
        ----------
        kaggle_dir : Répertoire contenant les CSV Kaggle FR
        remove_short : Longueur minimale en mots après nettoyage

        Returns
        -------
        DataFrame avec colonnes : text_original, text_debiased, text_clean, label
        """
        train_path = os.path.join(kaggle_dir, 'datafake_train.csv')
        test_path = os.path.join(kaggle_dir, 'datafake_test.csv')

        dfs = []
        for path in [train_path, test_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Fichier Kaggle FR manquant : {path}\n"
                    "Téléchargez depuis https://www.kaggle.com/datasets/hgilles06/frenchfakenewsdetector/data"
                )
            df = pd.read_csv(path, sep=';')
            dfs.append(df)

        df_fr = pd.concat(dfs, ignore_index=True)

        # Renommage et nettoyage des colonnes
        df_fr = df_fr.rename(columns={'post': 'text_original', 'fake': 'label'})
        df_fr = df_fr.drop(columns=['media'], errors='ignore')

        # Suppression des lignes sans texte
        df_fr = df_fr.dropna(subset=['text_original'])
        df_fr = df_fr[df_fr['text_original'].str.strip().astype(bool)]

        # Labels binaires (vérification)
        df_fr['label'] = df_fr['label'].astype(int)

        # Pipeline de nettoyage (même que ISOT)
        df_fr['text_debiased'] = df_fr['text_original'].apply(cls.remove_agency_bias)
        df_fr['text_clean'] = df_fr['text_debiased'].apply(cls.clean_for_ml)

        # Suppression des textes trop courts
        df_fr = df_fr[df_fr['text_clean'].str.split().str.len() >= remove_short]
        df_fr = df_fr[['text_original', 'text_debiased', 'text_clean', 'label']]
        df_fr = df_fr.reset_index(drop=True)

        logger.info(
            "Kaggle FR chargé : %d articles | Distribution : %s",
            len(df_fr),
            df_fr['label'].value_counts().to_dict(),
        )
        return df_fr

    @classmethod
    def load_fakenewsnet(
        cls,
        data_dir: str,
        remove_short: int = 5,
    ) -> pd.DataFrame:
        """
        Charge les titres FakeNewsNet (GossipCop + PolitiFact) depuis le repo GitHub.

        Fichiers attendus dans data_dir :
            - gossipcop_fake.csv, gossipcop_real.csv
            - politifact_fake.csv, politifact_real.csv
        Format : colonnes id, news_url, title, tweet_ids

        Returns
        -------
        DataFrame avec colonnes : text_original, text_debiased, text_clean, label
        """
        dfs = []
        for source in ['gossipcop', 'politifact']:
            for label_name, label_val in [('fake', 1), ('real', 0)]:
                path = os.path.join(data_dir, f'{source}_{label_name}.csv')
                if not os.path.exists(path):
                    logger.warning("FakeNewsNet fichier manquant : %s", path)
                    continue
                df = pd.read_csv(path)
                if 'title' not in df.columns:
                    continue
                df = df[['title']].dropna(subset=['title'])
                df = df[df['title'].str.strip().astype(bool)]
                df = df.rename(columns={'title': 'text_original'})
                df['label'] = label_val
                dfs.append(df)

        if not dfs:
            raise FileNotFoundError(
                f"Aucun fichier FakeNewsNet trouvé dans {data_dir}"
            )

        df_fnn = pd.concat(dfs, ignore_index=True)
        df_fnn['text_debiased'] = df_fnn['text_original'].apply(cls.remove_agency_bias)
        df_fnn['text_clean'] = df_fnn['text_debiased'].apply(cls.clean_for_ml)
        df_fnn = df_fnn[df_fnn['text_clean'].str.split().str.len() >= remove_short]
        df_fnn = df_fnn[['text_original', 'text_debiased', 'text_clean', 'label']]
        df_fnn = df_fnn.reset_index(drop=True)

        logger.info(
            "FakeNewsNet chargé : %d titres | Distribution : %s",
            len(df_fnn),
            df_fnn['label'].value_counts().to_dict(),
        )
        return df_fnn

    @classmethod
    def load_constraint(
        cls,
        data_dir: str,
        remove_short: int = 5,
    ) -> pd.DataFrame:
        """
        Charge le dataset CONSTRAINT 2021 (COVID-19 fake news tweets).

        Fichiers attendus dans data_dir :
            - Constraint_Train.csv, Constraint_Val.csv, Constraint_Test.csv
        Format : colonnes id, tweet, label ("real"/"fake")

        Returns
        -------
        DataFrame avec colonnes : text_original, text_debiased, text_clean, label
        """
        dfs = []
        for fname in ['Constraint_Train.csv', 'Constraint_Val.csv', 'Constraint_Test.csv']:
            path = os.path.join(data_dir, fname)
            if not os.path.exists(path):
                logger.warning("CONSTRAINT fichier manquant : %s", path)
                continue
            df = pd.read_csv(path)
            dfs.append(df)

        if not dfs:
            raise FileNotFoundError(
                f"Aucun fichier CONSTRAINT trouvé dans {data_dir}"
            )

        df_cst = pd.concat(dfs, ignore_index=True)
        df_cst = df_cst.rename(columns={'tweet': 'text_original'})
        df_cst = df_cst.dropna(subset=['text_original'])

        # Label mapping: "real" → 0, "fake" → 1
        label_map = {'real': 0, 'fake': 1}
        df_cst['label'] = df_cst['label'].str.lower().map(label_map)
        df_cst = df_cst.dropna(subset=['label'])
        df_cst['label'] = df_cst['label'].astype(int)

        df_cst['text_debiased'] = df_cst['text_original']  # pas de biais agence
        df_cst['text_clean'] = df_cst['text_debiased'].apply(cls.clean_for_ml)
        df_cst = df_cst[df_cst['text_clean'].str.split().str.len() >= remove_short]
        df_cst = df_cst[['text_original', 'text_debiased', 'text_clean', 'label']]
        df_cst = df_cst.reset_index(drop=True)

        logger.info(
            "CONSTRAINT chargé : %d tweets | Distribution : %s",
            len(df_cst),
            df_cst['label'].value_counts().to_dict(),
        )
        return df_cst

    @classmethod
    def load_credibility_corpus(
        cls,
        data_dir: str,
        remove_short: int = 5,
    ) -> pd.DataFrame:
        """
        Charge le Credibility Corpus (tweets FR+EN rumeurs/crédibles).

        Arborescence attendue dans data_dir :
            CorpusRumorTwitter/CorpusRumorTwitter/  — tweets rumeur (hollande, lemon=FR ; pin, swine-flu=EN)
            CorpusRandomTwitter/CorpusRandomTwitter/ — tweets aléatoires (FR, crédibles)
            CorpusEventTwitter/CorpusEventTwitter/   — tweets événements (*Fr=FR, *En=EN, crédibles)

        Returns
        -------
        DataFrame avec colonnes : text_original, text_debiased, text_clean, label, language
        """
        dfs = []

        # --- Rumor Twitter (semicolon-separated: num_rumor;date;name;id;content;retweets;)
        rumor_dir = os.path.join(data_dir, 'CorpusRumorTwitter', 'CorpusRumorTwitter')
        rumor_lang = {
            'hollande.txt': 'fr', 'lemon.txt': 'fr',
            'pin.txt': 'en', 'swine-flu.txt': 'en',
        }
        for fname, lang in rumor_lang.items():
            path = os.path.join(rumor_dir, fname)
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, sep=';', header=0, on_bad_lines='skip')
                text_col = 'content' if 'content' in df.columns else df.columns[4]
                df = df[[text_col]].rename(columns={text_col: 'text_original'})
                df['label'] = 1  # rumor → suspect
                df['language'] = lang
                dfs.append(df)
            except Exception as e:
                logger.warning("Erreur lecture %s : %s", fname, e)

        # --- Random Twitter (R-style CSV: "","x")
        random_dir = os.path.join(data_dir, 'CorpusRandomTwitter', 'CorpusRandomTwitter')
        if os.path.isdir(random_dir):
            for fname in sorted(os.listdir(random_dir)):
                if not fname.endswith('.txt'):
                    continue
                path = os.path.join(random_dir, fname)
                try:
                    df = pd.read_csv(path, header=0, on_bad_lines='skip')
                    text_col = df.columns[-1]  # last column = tweet text
                    df = df[[text_col]].rename(columns={text_col: 'text_original'})
                    df['label'] = 0  # random → crédible
                    df['language'] = 'fr'
                    dfs.append(df)
                except Exception as e:
                    logger.warning("Erreur lecture %s : %s", fname, e)

        # --- Event Twitter (R-style CSV, *Fr=FR, *En=EN)
        event_dir = os.path.join(data_dir, 'CorpusEventTwitter', 'CorpusEventTwitter')
        if os.path.isdir(event_dir):
            for fname in sorted(os.listdir(event_dir)):
                if not fname.endswith('.txt'):
                    continue
                path = os.path.join(event_dir, fname)
                lang = 'fr' if 'Fr' in fname else 'en'
                try:
                    df = pd.read_csv(path, header=0, on_bad_lines='skip')
                    text_col = df.columns[-1]
                    df = df[[text_col]].rename(columns={text_col: 'text_original'})
                    df['label'] = 0  # event → crédible
                    df['language'] = lang
                    dfs.append(df)
                except Exception as e:
                    logger.warning("Erreur lecture %s : %s", fname, e)

        if not dfs:
            raise FileNotFoundError(
                f"Aucun fichier Credibility Corpus trouvé dans {data_dir}"
            )

        df_cc = pd.concat(dfs, ignore_index=True)
        df_cc = df_cc.dropna(subset=['text_original'])
        df_cc['text_original'] = df_cc['text_original'].astype(str)
        df_cc['text_debiased'] = df_cc['text_original']
        df_cc['text_clean'] = df_cc['text_debiased'].apply(cls.clean_for_ml)
        df_cc = df_cc[df_cc['text_clean'].str.split().str.len() >= remove_short]
        df_cc = df_cc[['text_original', 'text_debiased', 'text_clean', 'label', 'language']]
        df_cc = df_cc.reset_index(drop=True)

        logger.info(
            "Credibility Corpus chargé : %d tweets | FR=%d, EN=%d | Distribution : %s",
            len(df_cc),
            (df_cc['language'] == 'fr').sum(),
            (df_cc['language'] == 'en').sum(),
            df_cc['label'].value_counts().to_dict(),
        )
        return df_cc


# ================================================================
#  2. FEATURE ENGINEERING LINGUISTIQUE
# ================================================================

class LinguisticFeatureExtractor:
    """
    Extrait des signaux linguistiques indicatifs de désinformation.

    Ces features capturent des patterns structurels (ponctuation,
    majuscules, sensationnalisme) indépendants du contenu lexical.
    Complémentaires au TF-IDF.
    """

    SENSATIONALIST_EN = frozenset({
        'breaking', 'shocking', 'bombshell', 'exposed',
        'secret', 'conspiracy', 'banned', 'censored',
        'hoax', 'alert', 'exclusive', 'unbelievable',
        'cover-up', 'coverup', 'wake up', 'they dont want',
        'mainstream media', 'deep state', 'big pharma',
        'must watch', 'must read', 'you wont believe',
        'what they hide', 'truth about', 'exposed the truth',
        'share before deleted', 'deleted soon', 'viral',
    })

    SENSATIONALIST_FR = frozenset({
        # Termes originaux
        'scandale', 'exclusif', 'choc', 'censuré', 'complot',
        'mensonge', 'urgent', 'alerte', 'incroyable', 'on vous cache',
        'manipulé', 'propagande', 'dictature', 'résistance',
        'big pharma', 'nouvel ordre mondial', 'great reset',
        # Conspiration
        'cabale', 'complotisme', 'dissimulé', 'falsifié', 'oligarchie',
        'mondialisme', 'lobbies', 'collusion', 'corruption',
        'état profond', 'fraude électorale', 'illuminati',
        # Sensationnalisme
        'hallucinant', 'stupéfiant', 'révélation', 'bombe', 'explosif',
        'terrifiant', 'catastrophique', 'apocalyptique', 'effrayant',
        # Manipulation émotionnelle
        'réveillons-nous', 'ouvrez les yeux', 'on nous ment',
        'vérité cachée', 'faites tourner', 'partagez avant censure',
        'partagez avant', 'réveillez-vous', 'on nous cache',
        'on vous ment', 'faites vos propres recherches',
        'avant censure', 'partagez massivement', 'info censurée',
        # Social media FR (ajout V4)
        'à partager', 'diffusez', 'la preuve', 'preuve en image',
        'regardez cette vidéo', 'vidéo censurée', 'témoignage choc',
        'enfin la vérité', 'ce que les médias cachent', 'les médias mentent',
        'info interdite', 'plandémie', 'génocide', 'empoisonnement',
        'puces', 'micro-puces', 'nanoparticules', 'graphène',
        'pass sanitaire', 'soumission', 'résistez', 'insurrection',
        'traîtres', 'vendu', 'vendus', 'marionnettes',
    })

    FEATURE_NAMES = [
        'word_count',
        'caps_ratio',
        'exclamation_count',
        'question_count',
        'punct_density',
        'avg_word_length',
        'sensationalism_score',
        'has_url',
        'numeric_density',
        'lexical_diversity',
        'sentence_count',
        'avg_sentence_length',
        # V4 : features texte court
        'all_caps_words_ratio',
        'interpellation_score',
        'is_short_text',
    ]

    # Patterns d'interpellation directe (manipulation sociale FR+EN)
    INTERPELLATION_PATTERNS_FR = [
        r'\b(réveillez[ -]vous|réveillons[ -]nous)\b',
        r'\b(ouvrez les yeux|ouvrons les yeux)\b',
        r'\b(faites tourner|partagez|diffusez|rt svp)\b',
        r'\b(on nous ment|on vous ment|ils nous mentent)\b',
        r'\b(ne soyez pas dupes?|ne soyez pas naï[fv]s?)\b',
        r'\b(dites non|boycott|refusez)\b',
        r'\b(attention danger|alerte rouge|alerte info)\b',
    ]
    INTERPELLATION_PATTERNS_EN = [
        r'\b(wake up|open your eyes)\b',
        r'\b(share before|retweet|spread the word)\b',
        r'\b(they are lying|they lied|dont be fooled)\b',
        r'\b(say no|boycott|fight back|resist)\b',
        r'\b(red alert|warning|danger)\b',
    ]

    @classmethod
    def extract(cls, texts: pd.Series) -> np.ndarray:
        """Retourne une matrice (n_samples, n_features) de features linguistiques."""
        results = np.zeros((len(texts), len(cls.FEATURE_NAMES)), dtype=np.float64)

        for i, text in enumerate(texts):
            text = str(text)
            words = text.split()
            n_words = len(words) if words else 1
            n_chars = len(text) if text else 1

            # Longueur
            results[i, 0] = n_words

            # Ratio majuscules (sur le texte original avant lower)
            alpha_chars = sum(c.isalpha() for c in text)
            results[i, 1] = (
                sum(c.isupper() for c in text) / alpha_chars
                if alpha_chars > 0
                else 0
            )

            # Ponctuation émotionnelle
            results[i, 2] = text.count('!')
            results[i, 3] = text.count('?')
            results[i, 4] = (
                sum(c in '!?.,;:…' for c in text) / n_chars
            )

            # Longueur moyenne des mots
            results[i, 5] = np.mean([len(w) for w in words]) if words else 0

            # Sensationnalisme (word-boundary aware for both single & multi-word)
            text_lower = text.lower()
            score = 0
            for w in cls.SENSATIONALIST_EN | cls.SENSATIONALIST_FR:
                # Use regex word boundaries to avoid partial matches
                # and correctly match multi-word expressions
                if re.search(r'(?:^|\b|\s)' + re.escape(w) + r'(?:\b|\s|$)', text_lower):
                    score += 1
            results[i, 6] = score

            # Présence d'URL
            results[i, 7] = 1.0 if re.search(r'http|www\.', text) else 0.0

            # Densité numérique
            results[i, 8] = sum(c.isdigit() for c in text) / n_chars

            # Diversité lexicale (TTR)
            results[i, 9] = len(set(words)) / n_words if n_words > 0 else 0

            # Phrases
            sentences = re.split(r'[.!?]+', text)
            sentences = [s for s in sentences if s.strip()]
            results[i, 10] = len(sentences)
            results[i, 11] = n_words / len(sentences) if sentences else n_words

            # V4 : Ratio de mots entièrement en majuscules (signal fort pour posts courts)
            caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
            results[i, 12] = caps_words / n_words if n_words > 0 else 0

            # V4 : Score d'interpellation (manipulation sociale directe)
            interp_score = 0
            for pat in cls.INTERPELLATION_PATTERNS_FR + cls.INTERPELLATION_PATTERNS_EN:
                if re.search(pat, text_lower):
                    interp_score += 1
            results[i, 13] = interp_score

            # V4 : Indicateur texte court (< 20 mots) — permet au modèle d'apprendre
            # des patterns spécifiques aux textes courts
            results[i, 14] = 1.0 if n_words < 20 else 0.0

        return results


# ================================================================
#  2b. FEATURES ÉMOTIONNELLES (MLP PyTorch bilingue)
# ================================================================

class _EmotionMLP(nn.Module):
    """Architecture MLP identique au notebook 02."""
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, 48)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(48, 24)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(24, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = torch.relu(self.fc1(x))
        x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)


class EmotionFeatureExtractor:
    """
    Charge le modèle émotions PyTorch bilingue et expose get_emotion_features().

    Contrat d'interface :
        get_emotion_features(texts) -> np.ndarray shape (n_texts, 7)
        Chaque colonne = probabilité d'une émotion :
        [colere, degout, joie, neutre, peur, surprise, tristesse]
    """

    VOCAB_SIZE = 25000
    MAX_LENGTH = 100
    EMBED_DIM = 64
    NUM_CLASSES = 7

    FEATURE_NAMES = [
        'emo_colere', 'emo_degout', 'emo_joie', 'emo_neutre',
        'emo_peur', 'emo_surprise', 'emo_tristesse',
    ]

    def __init__(self, model_dir: str = '../models'):
        self.model_dir = model_dir
        self.model = None
        self.vocab = None
        self.label_encoder = None
        self.device = torch.device('cpu')  # CPU pour inference en production
        self._loaded = False

    def load(self) -> bool:
        """Charge le modèle émotions. Retourne True si OK, False si fichiers absents."""
        pt_path = os.path.join(self.model_dir, 'emotion_bilingual.pt')
        vocab_path = os.path.join(self.model_dir, 'emotion_vocab_bilingual.pickle')
        le_path = os.path.join(self.model_dir, 'emotion_label_encoder_bilingual.pickle')

        if not all(os.path.exists(p) for p in [pt_path, vocab_path, le_path]):
            logger.warning("Modèle émotions non trouvé dans %s", self.model_dir)
            return False

        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        with open(le_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        cp = torch.load(pt_path, map_location=self.device, weights_only=True)
        if isinstance(cp, dict) and 'model_state_dict' in cp:
            sd = cp['model_state_dict']
            self.MAX_LENGTH = cp.get('max_len', 100)
        else:
            sd = cp
        vs = sd['embedding.weight'].shape[0]
        ed = sd['embedding.weight'].shape[1]
        nc = sd['fc3.weight'].shape[0]
        self.model = _EmotionMLP(vs, ed, nc).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()
        self._loaded = True
        logger.info("Modèle émotions chargé : %s", pt_path)
        return True

    def get_emotion_features(self, texts) -> np.ndarray:
        """
        Retourne les 7 probabilités d'émotion pour chaque texte.

        Parameters
        ----------
        texts : array-like de textes bruts

        Returns
        -------
        np.ndarray de shape (n_texts, 7)
        """
        if not self._loaded:
            raise RuntimeError("Modèle émotions non chargé. Appelez load() d'abord.")

        oov_idx = self.vocab.get('<OOV>', self.vocab.get('<UNK>', 1))
        sequences = []
        for text in texts:
            tokens = str(text).lower().split()
            seq = [self.vocab.get(t, oov_idx) for t in tokens[:self.MAX_LENGTH]]
            seq = seq + [0] * (self.MAX_LENGTH - len(seq))
            sequences.append(seq)

        X = torch.tensor(sequences, dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(X)
            probas = torch.softmax(logits, dim=1).cpu().numpy()

        return probas


# ================================================================
#  3. DÉTECTION DE LANGUE
# ================================================================

class LanguageRouter:
    """Détecte la langue de chaque post et route vers le traitement adapté."""

    @staticmethod
    def detect_language(text: str) -> str:
        """Retourne 'fr', 'en', ou 'other'."""
        if not LANGDETECT_AVAILABLE:
            return 'en'
        try:
            lang = detect(str(text)[:500])
            if lang == 'fr':
                return 'fr'
            if lang == 'en':
                return 'en'
            return 'other'
        except Exception:
            return 'en'

    @classmethod
    def detect_batch(cls, texts: pd.Series) -> pd.Series:
        """Détecte la langue pour une série de textes."""
        return texts.apply(cls.detect_language)


# ================================================================
#  4. DÉTECTEUR EXPERT
# ================================================================

class ExpertFakeNewsDetector:
    """
    Détecteur de fake news expert avec support bilingue FR/EN.

    Combine :
    - TF-IDF optimisé (20k features, tri-grams, sublinear TF)
    - 12 features linguistiques
    - 7 features émotionnelles (optionnel, via EmotionFeatureExtractor)
    - Classifieur calibré (LogReg / SVM / Ensemble)
    - Validation croisée stratifiée
    - Monitoring CodeCarbon
    """

    # Reference test cases for health_check():
    #   (text, expected_label, score_min, score_max)
    # Score ranges calibrated for V3 model (trained with corrected linguistic
    # features using original text instead of cleaned text).
    HEALTH_CHECK_CASES = [
        ("New study published in Nature confirms vaccine effectiveness.", 0, 0.55, 1.0),
        ("EXPOSED: Secret labs use 5G for mind control! Share before deleted!!!", 1, 0.0, 0.40),
        ("Le CNRS publie une etude confirmant l'efficacite des traitements.", 0, 0.85, 1.0),
        ("SCANDALE: le gouvernement cache la VERITE! Partagez avant censure!!!", 1, 0.0, 0.30),
        ("The weather is nice today.", 0, 0.70, 1.0),
    ]

    def __init__(self, model_dir: str = '../models', use_emotions: bool = False,
                 threshold: float = 0.44,
                 threshold_fr: Optional[float] = None,
                 threshold_en: Optional[float] = None):
        self.model_dir = model_dir
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model = None
        self.is_trained = False
        self.training_metrics: Dict = {}
        self.use_emotions = use_emotions
        self.emotion_extractor: Optional[EmotionFeatureExtractor] = None
        self.threshold = threshold
        # Per-language thresholds (P3 — seuils adaptatifs par langue)
        # When set, predict() uses these instead of the single threshold
        # for the corresponding language.  Falls back to self.threshold
        # for unrecognised languages or when the value is None.
        self.threshold_fr = threshold_fr
        self.threshold_en = threshold_en

        if use_emotions:
            self.emotion_extractor = EmotionFeatureExtractor(model_dir)
            if not self.emotion_extractor.load():
                logger.warning("Modèle émotions indisponible, use_emotions désactivé")
                self.use_emotions = False
                self.emotion_extractor = None

    # ---- Construction des features ----

    def _build_features(
        self,
        texts_clean: np.ndarray,
        texts_original: Optional[np.ndarray] = None,
        fit: bool = False,
    ) -> "scipy.sparse.csr_matrix":
        """
        Construit la matrice de features combinée.

        TF-IDF + 12 linguistiques [+ 7 émotionnelles si use_emotions].

        Parameters
        ----------
        texts_clean : Textes nettoyés (pour TF-IDF + linguistique)
        texts_original : Textes originaux (pour émotions, plus riches en signal).
                         Si None, utilise texts_clean.
        fit : True pour fit_transform, False pour transform
        """
        if fit:
            X_tfidf = self.vectorizer.fit_transform(texts_clean)
        else:
            X_tfidf = self.vectorizer.transform(texts_clean)

        # Linguistic features need ORIGINAL text (caps, punctuation, sentence boundaries)
        ling_texts = texts_original if texts_original is not None else texts_clean
        X_ling = LinguisticFeatureExtractor.extract(pd.Series(ling_texts))

        parts = [X_tfidf, X_ling]

        if self.use_emotions and self.emotion_extractor is not None:
            emo_texts = texts_original if texts_original is not None else texts_clean
            X_emo = self.emotion_extractor.get_emotion_features(emo_texts)
            parts.append(X_emo)

        return hstack(parts).tocsr()

    # ---- Entraînement ----

    def train(
        self,
        df: pd.DataFrame,
        model_type: str = 'logreg',
        n_folds: int = 5,
        track_emissions: bool = True,
        emissions_dir: Optional[str] = None,
    ) -> Dict:
        """
        Entraîne avec validation croisée stratifiée.

        Parameters
        ----------
        df : DataFrame avec colonnes 'text_clean' et 'label'
             Optionnel : colonne 'language' pour pondération bilingue
        model_type : 'logreg', 'svm', ou 'ensemble'
        n_folds : Nombre de folds CV
        track_emissions : Monitoring CodeCarbon

        Returns
        -------
        Dict de métriques CV (accuracy, f1, precision, recall, roc_auc)
        """
        tracker = None
        if track_emissions and CODECARBON_AVAILABLE:
            out_dir = emissions_dir or os.path.dirname(self.model_dir) or '.'
            tracker = EmissionsTracker(
                project_name=f"Thumalien_Expert_{model_type}",
                output_dir=out_dir,
            )
            tracker.start()

        try:
            X_text = df['text_clean'].values
            X_text_original = (
                df['text_original'].values
                if 'text_original' in df.columns else None
            )
            y = df['label'].values

            # Détection du mode bilingue
            bilingual = 'language' in df.columns
            sample_weights = None

            if bilingual:
                lang_counts = df['language'].value_counts()
                total = len(df)
                n_langs = len(lang_counts)
                lang_weight_map = {
                    lang: total / (n_langs * count)
                    for lang, count in lang_counts.items()
                }
                sample_weights = df['language'].map(lang_weight_map).values

            # TF-IDF optimisé (paramètres adaptés en mode bilingue)
            max_features = 30000 if bilingual else 20000
            min_df = 3 if bilingual else 3
            # En mode bilingue, conserver les accents FR (sémantiques : "ou"/"où", "a"/"à")
            strip = None if bilingual else 'unicode'

            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 3),
                min_df=min_df,
                max_df=0.95,
                sublinear_tf=True,
                strip_accents=strip,
                token_pattern=r'(?u)\b\w+\b',
            )

            X = self._build_features(X_text, texts_original=X_text_original, fit=True)

            # Cross-validation stratifiée
            cv = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=42
            )

            if bilingual and sample_weights is not None:
                # CV manuelle pour passer sample_weight à fit()
                cv_scores = {
                    'test_accuracy': [], 'test_f1': [], 'test_precision': [],
                    'test_recall': [], 'test_roc_auc': [], 'train_accuracy': [],
                }
                for train_idx, test_idx in cv.split(X, y):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    w_train = sample_weights[train_idx]

                    fold_model = self._get_model(model_type)
                    fold_model.fit(X_train, y_train, sample_weight=w_train)

                    y_pred = fold_model.predict(X_test)
                    cv_scores['test_accuracy'].append(accuracy_score(y_test, y_pred))
                    cv_scores['test_f1'].append(f1_score(y_test, y_pred))
                    cv_scores['test_precision'].append(precision_score(y_test, y_pred))
                    cv_scores['test_recall'].append(recall_score(y_test, y_pred))
                    if hasattr(fold_model, 'predict_proba'):
                        y_proba = fold_model.predict_proba(X_test)[:, 1]
                        cv_scores['test_roc_auc'].append(roc_auc_score(y_test, y_proba))
                    else:
                        cv_scores['test_roc_auc'].append(0.0)

                    y_train_pred = fold_model.predict(X_train)
                    cv_scores['train_accuracy'].append(accuracy_score(y_train, y_train_pred))

                # Convertir en arrays numpy
                cv_results = {k: np.array(v) for k, v in cv_scores.items()}
            else:
                # CV classique (mode monolingue)
                scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
                base_model = self._get_model(model_type)
                cv_results = cross_validate(
                    base_model,
                    X,
                    y,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=True,
                    n_jobs=-1,
                )

            # Entraînement final sur tout le dataset
            self.model = self._get_model(model_type)
            if sample_weights is not None:
                self.model.fit(X, y, sample_weight=sample_weights)
            else:
                self.model.fit(X, y)
            self.is_trained = True

            # Métriques
            self.training_metrics = {
                'model_type': model_type,
                'n_samples': len(y),
                'n_features_tfidf': self.vectorizer.max_features,
                'n_features_linguistic': len(LinguisticFeatureExtractor.FEATURE_NAMES),
                'n_features_emotion': len(EmotionFeatureExtractor.FEATURE_NAMES) if self.use_emotions else 0,
                'use_emotions': self.use_emotions,
                'n_folds': n_folds,
                'cv_accuracy_mean': round(float(np.mean(cv_results['test_accuracy'])), 4),
                'cv_accuracy_std': round(float(np.std(cv_results['test_accuracy'])), 4),
                'cv_f1_mean': round(float(np.mean(cv_results['test_f1'])), 4),
                'cv_f1_std': round(float(np.std(cv_results['test_f1'])), 4),
                'cv_precision_mean': round(float(np.mean(cv_results['test_precision'])), 4),
                'cv_recall_mean': round(float(np.mean(cv_results['test_recall'])), 4),
                'cv_roc_auc_mean': round(float(np.mean(cv_results['test_roc_auc'])), 4),
                'train_accuracy_mean': round(float(np.mean(cv_results['train_accuracy'])), 4),
                'cv_accuracy_per_fold': [
                    round(float(x), 4)
                    for x in cv_results['test_accuracy']
                ],
                'cv_f1_per_fold': [
                    round(float(x), 4)
                    for x in cv_results['test_f1']
                ],
            }

            # Métriques bilingues
            if bilingual:
                self.training_metrics['bilingual'] = True
                self.training_metrics['language_distribution'] = (
                    df['language'].value_counts().to_dict()
                )
                self.training_metrics['language_weights'] = lang_weight_map

            return self.training_metrics

        finally:
            if tracker:
                emissions = tracker.stop()
                self.training_metrics['co2_emissions_kg'] = float(emissions)
                self.training_metrics['energy_kwh'] = float(
                    tracker.final_emissions_data.energy_consumed
                )

    @staticmethod
    def _get_model(model_type: str):
        if model_type == 'logreg':
            return LogisticRegression(
                C=1.0,
                max_iter=10000,
                solver='lbfgs',
                class_weight='balanced',
                random_state=42,
            )
        if model_type == 'svm':
            return CalibratedClassifierCV(
                LinearSVC(
                    C=0.5,
                    max_iter=2000,
                    class_weight='balanced',
                    random_state=42,
                ),
                cv=3,
            )
        if model_type == 'ensemble':
            return VotingClassifier(
                estimators=[
                    (
                        'lr',
                        LogisticRegression(
                            C=1.0,
                            max_iter=2000,
                            solver='lbfgs',
                            class_weight='balanced',
                            random_state=42,
                        ),
                    ),
                    (
                        'svm',
                        CalibratedClassifierCV(
                            LinearSVC(
                                C=0.5,
                                max_iter=2000,
                                class_weight='balanced',
                                random_state=42,
                            ),
                            cv=3,
                        ),
                    ),
                ],
                voting='soft',
            )
        raise ValueError(f"model_type inconnu : {model_type}")

    # ---- Évaluation ----

    def evaluate_holdout(self, df: pd.DataFrame) -> Dict:
        """
        Évaluation complète sur un jeu de test holdout.

        Returns
        -------
        Dict avec accuracy, f1, classification_report, confusion_matrix, etc.
        """
        if not self.is_trained:
            raise RuntimeError("Modèle non entraîné.")

        X_text = df['text_clean'].values
        X_text_original = (
            df['text_original'].values
            if 'text_original' in df.columns else None
        )
        y_true = df['label'].values

        X = self._build_features(X_text, texts_original=X_text_original, fit=False)

        y_pred = self.model.predict(X)

        results = {
            'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
            'f1': round(float(f1_score(y_true, y_pred)), 4),
            'precision': round(float(precision_score(y_true, y_pred)), 4),
            'recall': round(float(recall_score(y_true, y_pred)), 4),
            'report': classification_report(
                y_true, y_pred, target_names=['VRAI', 'FAKE'], output_dict=True,
            ),
            'report_str': classification_report(
                y_true, y_pred, target_names=['VRAI', 'FAKE'],
            ),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'y_true': y_true,
            'y_pred': y_pred,
        }

        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X)[:, 1]
            results['roc_auc'] = round(
                float(roc_auc_score(y_true, y_proba)), 4
            )
            results['y_proba'] = y_proba

        return results

    # ---- Prédiction (production) ----

    def predict(self, texts: pd.Series, track_emissions: bool = False) -> pd.DataFrame:
        """
        Prédiction sur de nouveaux textes (posts Bluesky).

        Parameters
        ----------
        texts : pd.Series
            Textes bruts à analyser.
        track_emissions : bool, default False
            Si True, mesure l'empreinte carbone de l'inférence via CodeCarbon.
            Les résultats sont ajoutés au fichier ``emissions.csv`` du projet.

        Returns
        -------
        DataFrame : language, prediction_label, ai_score_credibility,
                    ai_analysis_log
        """
        if not self.is_trained:
            raise RuntimeError("Modèle non entraîné.")

        # --- Optionally start CodeCarbon tracker ---
        tracker = None
        if track_emissions and CODECARBON_AVAILABLE:
            emissions_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', '..', 'emissions.csv'
            )
            tracker = EmissionsTracker(
                project_name="Thumalien_Inference",
                output_dir=os.path.dirname(emissions_path),
                output_file="emissions.csv",
                log_level="warning",
            )
            tracker.start()
        elif track_emissions and not CODECARBON_AVAILABLE:
            logger.warning("track_emissions=True mais codecarbon n'est pas installé.")

        try:
            results = pd.DataFrame()
            results['text'] = texts.values

            # Détection de langue
            results['language'] = LanguageRouter.detect_batch(texts)

            # Nettoyage
            texts_clean = texts.apply(DatasetCleaner.clean_for_ml)

            # Features (textes originaux pour émotions, nettoyés pour TF-IDF)
            X = self._build_features(
                texts_clean.values,
                texts_original=texts.values,
                fit=False,
            )

            # Prédiction avec seuil ajustable (défaut: 0.44)
            # P3 : seuils adaptatifs par langue (FR/EN) si définis
            y_proba = self.model.predict_proba(X)
            scores = y_proba[:, 0]  # P(Fiable)

            if self.threshold_fr is not None or self.threshold_en is not None:
                # Seuils adaptatifs par langue
                lang_array = results['language'].values
                th_array = np.full(len(scores), self.threshold)
                if self.threshold_fr is not None:
                    th_array[lang_array == 'fr'] = self.threshold_fr
                if self.threshold_en is not None:
                    th_array[lang_array == 'en'] = self.threshold_en
                y_pred = (scores < th_array).astype(int)
            else:
                y_pred = (scores < self.threshold).astype(int)  # SUSPECT si P(Fiable) < seuil

            results['prediction_label'] = y_pred
            results['ai_score_credibility'] = np.round(scores, 4)

            results['ai_analysis_log'] = results.apply(
                lambda r: self._make_log(r), axis=1
            )

            return results
        finally:
            if tracker is not None:
                emissions_kg = tracker.stop()
                if emissions_kg is not None:
                    logger.info(
                        "Inference carbon footprint: %.6f kg CO2eq (%.4f g)",
                        emissions_kg,
                        emissions_kg * 1000,
                    )

    @staticmethod
    def _make_log(row) -> str:
        lang_names = {'fr': 'FR', 'en': 'EN', 'other': '??'}
        lang = lang_names.get(row.get('language', 'en'), '??')
        score = row.get('ai_score_credibility', 0.5)
        label = row.get('prediction_label', 0)
        if label == 1:
            return f"[{lang}] Suspect (crédibilité: {score:.0%})"
        return f"[{lang}] Fiable (crédibilité: {score:.0%})"

    # ---- Prédiction adaptative ----

    def predict_adaptive(
        self, texts: pd.Series, track_emissions: bool = False
    ) -> pd.DataFrame:
        """
        Prédiction avec seuils adaptatifs selon la longueur du texte.

        Les textes courts contiennent moins de signal statistique, donc un
        seuil plus conservateur (plus élevé) réduit les faux positifs.

        Seuils :
            - < 15 mots  : 0.54 (conservateur)
            - 15-30 mots : 0.49 (modéré)
            - > 30 mots  : 0.44 (standard)

        Parameters
        ----------
        texts : pd.Series
            Textes bruts à analyser.
        track_emissions : bool, default False
            Si True, mesure l'empreinte carbone via CodeCarbon.

        Returns
        -------
        DataFrame : language, prediction_label, ai_score_credibility,
                    ai_analysis_log, adaptive_threshold
        """
        if not self.is_trained:
            raise RuntimeError("Modèle non entraîné.")

        # --- Optionally start CodeCarbon tracker ---
        tracker = None
        if track_emissions and CODECARBON_AVAILABLE:
            emissions_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', '..', 'emissions.csv'
            )
            tracker = EmissionsTracker(
                project_name="Thumalien_Inference_Adaptive",
                output_dir=os.path.dirname(emissions_path),
                output_file="emissions.csv",
                log_level="warning",
            )
            tracker.start()
        elif track_emissions and not CODECARBON_AVAILABLE:
            logger.warning("track_emissions=True mais codecarbon n'est pas installé.")

        try:
            results = pd.DataFrame()
            results['text'] = texts.values

            # Détection de langue
            results['language'] = LanguageRouter.detect_batch(texts)

            # Nettoyage
            texts_clean = texts.apply(DatasetCleaner.clean_for_ml)

            # Features (textes originaux pour linguistique + émotions)
            X = self._build_features(
                texts_clean.values,
                texts_original=texts.values,
                fit=False,
            )

            # Prédiction avec seuils adaptatifs par longueur de texte
            y_proba = self.model.predict_proba(X)
            scores = y_proba[:, 0]  # P(Fiable)

            word_counts = texts.apply(lambda t: len(str(t).split()))
            thresholds = word_counts.apply(
                lambda n: 0.54 if n < 15 else (0.49 if n <= 30 else 0.44)
            )

            y_pred = (scores < thresholds.values).astype(int)

            results['prediction_label'] = y_pred
            results['ai_score_credibility'] = np.round(scores, 4)
            results['adaptive_threshold'] = thresholds.values

            results['ai_analysis_log'] = results.apply(
                lambda r: self._make_log(r), axis=1
            )

            return results
        finally:
            if tracker is not None:
                emissions_kg = tracker.stop()
                if emissions_kg is not None:
                    logger.info(
                        "Inference (adaptive) carbon footprint: %.6f kg CO2eq (%.4f g)",
                        emissions_kg,
                        emissions_kg * 1000,
                    )

    # ---- Explainability ----

    def explain_prediction(self, text: str, top_n: int = 10) -> Dict:
        """
        Explication per-instance basée sur les coefficients LogReg.

        Calcule la contribution exacte de chaque feature (TF-IDF + linguistique)
        à la décision du modèle. Fonctionne uniquement avec les modèles linéaires
        exposant ``coef_``.

        Parameters
        ----------
        text : Texte brut à expliquer
        top_n : Nombre de mots à retourner par direction (suspect / fiable)

        Returns
        -------
        Dict avec clés : explainable, language, prediction_label, score_credibility,
        top_suspect_words, top_fiable_words, linguistic_signals,
        sensationalist_words, summary
        """
        if not self.is_trained:
            raise RuntimeError("Modèle non entraîné.")

        if not hasattr(self.model, 'coef_'):
            return {
                'explainable': False,
                'reason': 'Le type de modèle ne permet pas l\'explication par coefficients.',
            }

        # --- Pipeline identique à predict() pour un seul texte ---
        lang = LanguageRouter.detect_language(text)
        text_clean = DatasetCleaner.clean_for_ml(text)

        X_tfidf = self.vectorizer.transform([text_clean])
        # Use original text for linguistic features (caps, punctuation, sentences)
        X_ling = LinguisticFeatureExtractor.extract(pd.Series([text]))

        parts = [X_tfidf, X_ling]
        X_emo = None
        if self.use_emotions and self.emotion_extractor is not None:
            X_emo = self.emotion_extractor.get_emotion_features([text])
            parts.append(X_emo)

        X = hstack(parts).tocsr()

        # Prédiction (P3 : seuil adaptatif par langue si défini)
        y_proba = self.model.predict_proba(X)
        score_fiable = float(y_proba[0, 0])
        effective_threshold = self.threshold
        if lang == 'fr' and self.threshold_fr is not None:
            effective_threshold = self.threshold_fr
        elif lang == 'en' and self.threshold_en is not None:
            effective_threshold = self.threshold_en
        pred_label = 1 if score_fiable < effective_threshold else 0

        # --- Contributions exactes : coef_i * feature_value_i ---
        coef = self.model.coef_[0]
        n_tfidf = X_tfidf.shape[1]
        n_ling = len(LinguisticFeatureExtractor.FEATURE_NAMES)

        # TF-IDF : itérer uniquement les indices non-zero (sparse-efficient)
        tfidf_coef = coef[:n_tfidf]
        tfidf_names = self.vectorizer.get_feature_names_out()
        nonzero_idx = X_tfidf.nonzero()[1]
        tfidf_vals = X_tfidf.toarray()[0]

        word_contribs = []
        for i in nonzero_idx:
            c = float(tfidf_coef[i] * tfidf_vals[i])
            if c != 0:
                word_contribs.append((str(tfidf_names[i]), c))

        word_contribs.sort(key=lambda x: x[1], reverse=True)
        top_suspect_words = [(w, c) for w, c in word_contribs if c > 0][:top_n]
        top_fiable_words = [(w, c) for w, c in word_contribs if c < 0]
        top_fiable_words.sort(key=lambda x: x[1])
        top_fiable_words = top_fiable_words[:top_n]

        # Linguistique : 12 features nommées
        ling_names = LinguisticFeatureExtractor.FEATURE_NAMES
        ling_vals = X_ling[0]
        ling_coef = coef[n_tfidf:n_tfidf + n_ling]
        ling_detail = []
        for j, name in enumerate(ling_names):
            c = float(ling_coef[j] * ling_vals[j])
            ling_detail.append({
                'feature': name,
                'value': float(ling_vals[j]),
                'contribution': c,
                'direction': 'SUSPECT' if c > 0 else 'FIABLE',
            })

        # Émotions (si actives)
        emo_detail = []
        if self.use_emotions and X_emo is not None:
            emo_names = EmotionFeatureExtractor.FEATURE_NAMES
            emo_vals = X_emo[0]
            emo_coef = coef[n_tfidf + n_ling:]
            for j, name in enumerate(emo_names):
                if j < len(emo_coef):
                    c = float(emo_coef[j] * emo_vals[j])
                    emo_detail.append({
                        'feature': name,
                        'value': float(emo_vals[j]),
                        'contribution': c,
                        'direction': 'SUSPECT' if c > 0 else 'FIABLE',
                    })

        # Mots sensationnalistes détectés
        text_lower = text.lower()
        found_sensationalist = []
        for word_set, lang_label in [
            (LinguisticFeatureExtractor.SENSATIONALIST_EN, 'EN'),
            (LinguisticFeatureExtractor.SENSATIONALIST_FR, 'FR'),
        ]:
            for w in word_set:
                if w in text_lower:
                    found_sensationalist.append({'word': w, 'language': lang_label})

        # --- Résumé textuel ---
        verdict = "SUSPECT" if pred_label == 1 else "FIABLE"
        summary_parts = [f"Verdict : {verdict} (crédibilité : {score_fiable:.0%})"]

        if top_suspect_words:
            top3 = ', '.join(f'"{w}"' for w, _ in top_suspect_words[:3])
            summary_parts.append(f"Mots suspects : {top3}")
        if top_fiable_words:
            top3 = ', '.join(f'"{w}"' for w, _ in top_fiable_words[:3])
            summary_parts.append(f"Mots fiables : {top3}")
        if found_sensationalist:
            sens = ', '.join(f'"{s["word"]}"' for s in found_sensationalist[:5])
            summary_parts.append(f"Sensationnalisme : {sens}")

        notable_ling = sorted(ling_detail, key=lambda x: abs(x['contribution']), reverse=True)[:3]
        if notable_ling:
            ling_strs = [
                f"{f['feature']}={f['value']:.2f} ({f['direction']})"
                for f in notable_ling
            ]
            summary_parts.append(f"Signaux : {', '.join(ling_strs)}")

        return {
            'explainable': True,
            'language': lang,
            'prediction_label': pred_label,
            'score_credibility': score_fiable,
            'top_suspect_words': [{'word': w, 'contribution': c} for w, c in top_suspect_words],
            'top_fiable_words': [{'word': w, 'contribution': c} for w, c in top_fiable_words],
            'linguistic_signals': ling_detail,
            'emotion_signals': emo_detail,
            'sensationalist_words': found_sensationalist,
            'summary': ' | '.join(summary_parts),
        }

    # ---- Persistance ----

    def save(self, suffix: str = 'expert'):
        """Sauvegarde modèle + vectorizer + métriques."""
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(
            self.model,
            os.path.join(self.model_dir, f'model_{suffix}.pkl'),
        )
        joblib.dump(
            self.vectorizer,
            os.path.join(self.model_dir, f'tfidf_{suffix}.pkl'),
        )
        joblib.dump(
            self.training_metrics,
            os.path.join(self.model_dir, f'metrics_{suffix}.pkl'),
        )
        logger.info("Modèle sauvegardé: %s (suffix=%s)", self.model_dir, suffix)

    def load(self, suffix: str = 'expert'):
        """Charge un modèle sauvegardé."""
        self.model = joblib.load(
            os.path.join(self.model_dir, f'model_{suffix}.pkl')
        )
        self.vectorizer = joblib.load(
            os.path.join(self.model_dir, f'tfidf_{suffix}.pkl')
        )
        metrics_path = os.path.join(self.model_dir, f'metrics_{suffix}.pkl')
        if os.path.exists(metrics_path):
            self.training_metrics = joblib.load(metrics_path)
        # Restaurer use_emotions depuis les métriques sauvegardées
        saved_emotions = self.training_metrics.get('use_emotions', False)
        if saved_emotions and self.emotion_extractor is None:
            self.emotion_extractor = EmotionFeatureExtractor(self.model_dir)
            if self.emotion_extractor.load():
                self.use_emotions = True
            else:
                self.use_emotions = False
                self.emotion_extractor = None
        elif not saved_emotions:
            self.use_emotions = False
        self.is_trained = True
        logger.info("Modèle chargé depuis %s (suffix=%s)", self.model_dir, suffix)

    # ---- Health check ----

    def health_check(self) -> Dict:
        """
        Run reference test cases through predict() and verify scores
        fall within expected ranges.

        Returns
        -------
        dict with keys:
            healthy : bool — True if all cases pass
            details : list[dict] — per-case results with pass/fail info
        """
        if not self.is_trained:
            return {'healthy': False, 'details': [{'error': 'Model not loaded.'}]}

        texts = pd.Series([t for t, _, _, _ in self.HEALTH_CHECK_CASES])
        results = self.predict(texts)

        details = []
        all_ok = True

        for i, (text, expected_label, score_min, score_max) in enumerate(self.HEALTH_CHECK_CASES):
            pred_label = int(results['prediction_label'].iloc[i])
            score = float(results['ai_score_credibility'].iloc[i])
            label_ok = pred_label == expected_label
            score_ok = score_min <= score <= score_max
            passed = label_ok and score_ok

            if not passed:
                all_ok = False

            details.append({
                'text': text[:80],
                'expected_label': expected_label,
                'predicted_label': pred_label,
                'label_ok': label_ok,
                'score': round(score, 4),
                'expected_range': [score_min, score_max],
                'score_ok': score_ok,
                'passed': passed,
            })

        return {'healthy': all_ok, 'details': details}
