"""
ThumaCheck — Dataset Cleaner
============================

Suppression du biais Reuters dans les datasets d'entraînement.
Gestion multi-sources : Kaggle, FakeNewsNet, CONSTRAINT, CredibilityCorpus.

Auteur : Niamato Consulting (pour Thumalien)
"""

import logging
import os
import re

import pandas as pd

logger = logging.getLogger(__name__)

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
        french_path: str | None = None,
        kaggle_fr_dir: str | None = None,
        fakenewsnet_dir: str | None = None,
        constraint_dir: str | None = None,
        credibility_dir: str | None = None,
        remove_short: int = 20,
        social_remove_short: int = 5,
        french_oversample: int = 3,
        social_oversample: int = 2,
        en_subsample: int | None = None,
        **kwargs,
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
    def audit_reuters_leakage(cls, df_true: pd.DataFrame) -> dict:
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


