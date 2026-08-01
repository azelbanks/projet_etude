"""
ThumaCheck — Expert Fake News Detector
=======================================

Main detector class orchestrating the full V9 cascade pipeline:
1. Dataset cleaning (DatasetCleaner)
2. Linguistic feature extraction (LinguisticFeatureExtractor)
3. Emotion features (EmotionFeatureExtractor)
4. Language routing (LanguageRouter)
5. TF-IDF + meta-learner ensemble
6. CamemBERT (FR) / RoBERTa (EN) cascade

Auteur : Niamato Consulting (pour Thumalien)
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Optional, List, Tuple

import torch

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

from .dataset_cleaner import DatasetCleaner
from .linguistic_features import LinguisticFeatureExtractor
from .emotion_classifier import EmotionFeatureExtractor, _EmotionMLP
from .language_router import LanguageRouter

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)

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

    def __init__(self, model_dir: str = '../models', use_emotions: bool = False,  # -> None
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

        # V9 cascade: optional RoBERTa EN for short English texts
        self._roberta_en = None

        if use_emotions:
            self.emotion_extractor = EmotionFeatureExtractor(model_dir)
            if not self.emotion_extractor.load():
                logger.warning("Modèle émotions indisponible, use_emotions désactivé")
                self.use_emotions = False
                self.emotion_extractor = None

    def _trim_ling_features(self, X_ling: np.ndarray, n_tfidf: int) -> np.ndarray:
        """Trim linguistic features if model was trained with fewer (backward compat)."""
        if getattr(self, 'model', None) is None:
            return X_ling
        try:
            expected = getattr(self.model, 'n_features_in_', None)
            if not isinstance(expected, int):
                return X_ling
            n_ling_expected = expected - n_tfidf
            if self.use_emotions and self.emotion_extractor is not None:
                n_ling_expected -= len(EmotionFeatureExtractor.FEATURE_NAMES)
            if X_ling.shape[1] > n_ling_expected > 0:
                return X_ling[:, :n_ling_expected]
        except (TypeError, ValueError):
            pass
        return X_ling

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

        X_ling = self._trim_ling_features(X_ling, X_tfidf.shape[1])

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
                project_name=f"ThumaCheck_Expert_{model_type}",
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
    def _get_model(model_type: str) -> object:
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
                project_name="ThumaCheck_Inference",
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
    def _make_log(row: pd.Series) -> str:
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
                project_name="ThumaCheck_Inference_Adaptive",
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

            # V9 cascade: blend RoBERTa EN scores for short English texts
            if getattr(self, '_roberta_en', None) is not None:
                en_short_mask = (
                    (results['language'] == 'en') & (word_counts < 30)
                ).values
                if en_short_mask.any():
                    en_short_texts = texts[en_short_mask].tolist()
                    try:
                        roberta_scores = self._roberta_en.predict_credibility_scores(en_short_texts)
                        # Blend: 60% RoBERTa + 40% TF-IDF for short EN
                        scores[en_short_mask] = 0.6 * roberta_scores + 0.4 * scores[en_short_mask]
                        logger.info("V9 cascade: blended %d short EN texts with RoBERTa", en_short_mask.sum())
                    except Exception as e:
                        logger.warning("V9 cascade RoBERTa EN failed: %s", e)

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
        X_ling = self._trim_ling_features(X_ling, X_tfidf.shape[1])

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
        n_ling = X_ling.shape[1]  # May be trimmed for backward compat

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

        # Linguistique features (may be trimmed for backward compat)
        ling_names = LinguisticFeatureExtractor.FEATURE_NAMES[:n_ling]
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

    def save(self, suffix: str = 'expert') -> None:
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

    def load(self, suffix: str = 'expert') -> bool:
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

        # V9 cascade: try loading RoBERTa EN for short English texts
        try:
            from pipeline.roberta_en_classifier import RoBERTaENClassifier
            roberta = RoBERTaENClassifier(model_dir=self.model_dir)
            if roberta.load():
                self._roberta_en = roberta
                logger.info("V9 cascade: RoBERTa EN chargé pour textes courts EN")
            else:
                self._roberta_en = None
        except Exception:
            self._roberta_en = None

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
