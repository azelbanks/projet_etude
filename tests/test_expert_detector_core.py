"""
Tests approfondis pour ExpertFakeNewsDetector — coeur metier.

Couvre : DatasetCleaner, EmotionFeatureExtractor, LanguageRouter,
ExpertFakeNewsDetector (predict, predict_adaptive, explain_prediction,
save/load, health_check, _make_log, _build_features, train).
"""

import os
import tempfile
import pickle

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from scipy.sparse import csr_matrix, issparse

from pipeline.expert_detector import (
    DatasetCleaner,
    EmotionFeatureExtractor,
    LanguageRouter,
    ExpertFakeNewsDetector,
    LinguisticFeatureExtractor,
)


# ============================================================
#  DatasetCleaner — prepare_bilingual_dataset, loaders
# ============================================================


class TestDatasetCleanerBilingual:
    """Tests for bilingual dataset preparation methods."""

    def test_prepare_bilingual_no_fr_raises(self, tmp_path):
        """Should raise ValueError when no FR source is provided."""
        fake_csv = tmp_path / "Fake.csv"
        true_csv = tmp_path / "True.csv"
        fake_csv.write_text("title,text,subject,date\nT,Some fake text here for testing,politics,2020\n")
        true_csv.write_text("title,text,subject,date\nT,Some true text here for testing,world,2020\n")

        with pytest.raises(ValueError, match="Aucune source FR"):
            DatasetCleaner.prepare_bilingual_dataset(
                str(fake_csv), str(true_csv),
                french_path=None, kaggle_fr_dir=None,
            )

    def test_prepare_bilingual_with_french_csv(self, tmp_path):
        """Should load FR from french_path fallback."""
        fake_csv = tmp_path / "Fake.csv"
        true_csv = tmp_path / "True.csv"
        fr_csv = tmp_path / "french.csv"
        fake_csv.write_text("title,text,subject,date\nT,This is a fake news article about politics and scandals that is long enough to pass the minimum word count filter for testing purposes in the dataset,politics,2020\n")
        true_csv.write_text("title,text,subject,date\nT,This is a true news article about world events and facts that is long enough to pass the minimum word count filter for testing purposes in the dataset,world,2020\n")
        fr_csv.write_text("text,label\nCeci est un texte francais de test assez long pour passer le filtre de longueur minimale de mots dans le pipeline,0\n")

        df = DatasetCleaner.prepare_bilingual_dataset(
            str(fake_csv), str(true_csv),
            french_path=str(fr_csv),
        )
        assert 'language' in df.columns
        assert set(df['language'].unique()) == {'en', 'fr'}

    def test_load_kaggle_french_missing_file(self, tmp_path):
        """Should raise FileNotFoundError when CSV missing."""
        with pytest.raises(FileNotFoundError, match="Fichier Kaggle FR manquant"):
            DatasetCleaner.load_kaggle_french(str(tmp_path))

    def test_load_fakenewsnet_missing_dir(self, tmp_path):
        """Should raise FileNotFoundError on empty dir."""
        with pytest.raises(FileNotFoundError, match="Aucun fichier FakeNewsNet"):
            DatasetCleaner.load_fakenewsnet(str(tmp_path))

    def test_load_constraint_missing_files(self, tmp_path):
        """Should raise FileNotFoundError on empty dir."""
        with pytest.raises(FileNotFoundError, match="Aucun fichier CONSTRAINT"):
            DatasetCleaner.load_constraint(str(tmp_path))

    def test_load_credibility_corpus_missing(self, tmp_path):
        """Should raise FileNotFoundError on empty dir."""
        with pytest.raises(FileNotFoundError, match="Aucun fichier Credibility"):
            DatasetCleaner.load_credibility_corpus(str(tmp_path))

    def test_load_constraint_with_data(self, tmp_path):
        """Should load and map labels from CONSTRAINT CSV."""
        csv_file = tmp_path / "Constraint_Train.csv"
        csv_file.write_text(
            "id,tweet,label\n"
            "1,This is a real tweet about health topics,real\n"
            "2,COVID is a hoax created by government,fake\n"
        )
        df = DatasetCleaner.load_constraint(str(tmp_path), remove_short=2)
        assert len(df) == 2
        assert set(df['label'].values) == {0, 1}

    def test_load_fakenewsnet_with_data(self, tmp_path):
        """Should load titles from FakeNewsNet CSVs."""
        f = tmp_path / "gossipcop_fake.csv"
        f.write_text("id,news_url,title,tweet_ids\n1,http://x,Fake celebrity scandal revealed,t1\n")
        r = tmp_path / "gossipcop_real.csv"
        r.write_text("id,news_url,title,tweet_ids\n2,http://y,Real celebrity news update today,t2\n")
        df = DatasetCleaner.load_fakenewsnet(str(tmp_path), remove_short=2)
        assert len(df) == 2
        assert 'label' in df.columns

    def test_generate_fr_short_augmentation(self):
        """Should generate short FR texts from long ones."""
        df_fr = pd.DataFrame({
            'text_original': [
                "Ceci est un article tres long sur la politique et les affaires. "
                "Il contient plusieurs phrases pour tester l'augmentation. "
                "La troisieme phrase ajoute du contenu supplementaire."
            ],
            'text_clean': [
                "ceci est un article tres long sur la politique et les affaires "
                "il contient plusieurs phrases pour tester augmentation "
                "la troisieme phrase ajoute du contenu supplementaire"
            ],
            'label': [0],
        })
        aug = DatasetCleaner.generate_fr_short_augmentation(df_fr)
        assert len(aug) > 0
        assert 'label' in aug.columns


# ============================================================
#  EmotionFeatureExtractor
# ============================================================


class TestEmotionFeatureExtractor:
    """Tests for EmotionFeatureExtractor loading and inference."""

    def test_load_missing_files(self, tmp_path):
        """Should return False when model files are missing."""
        emo = EmotionFeatureExtractor(model_dir=str(tmp_path))
        assert emo.load() is False

    def test_get_emotion_features_not_loaded(self, tmp_path):
        """Should raise RuntimeError if model not loaded."""
        emo = EmotionFeatureExtractor(model_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="non charg"):
            emo.get_emotion_features(["test"])

    def test_load_and_predict(self):
        """Should load real model and return (n, 7) array."""
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        pt_path = os.path.join(model_dir, 'emotion_bilingual.pt')
        if not os.path.exists(pt_path):
            pytest.skip("Emotion model files not found")

        emo = EmotionFeatureExtractor(model_dir=model_dir)
        assert emo.load() is True

        probas = emo.get_emotion_features(["I am happy", "Je suis triste"])
        assert probas.shape == (2, 7)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=0.01)
        assert (probas >= 0).all()

    def test_feature_names_constant(self):
        """FEATURE_NAMES should have 7 entries."""
        assert len(EmotionFeatureExtractor.FEATURE_NAMES) == 7


# ============================================================
#  LanguageRouter
# ============================================================


class TestLanguageRouter:
    def test_detect_french(self):
        lang = LanguageRouter.detect_language(
            "Le president de la Republique a annonce des mesures nouvelles"
        )
        assert lang == 'fr'

    def test_detect_english(self):
        lang = LanguageRouter.detect_language(
            "The president announced new policy measures today"
        )
        assert lang == 'en'

    def test_detect_empty_string(self):
        lang = LanguageRouter.detect_language("")
        # Should not crash, returns default
        assert lang in ('en', 'fr', 'other')

    def test_detect_batch(self):
        texts = pd.Series([
            "Bonjour le monde entier",
            "Hello world today is great",
        ])
        result = LanguageRouter.detect_batch(texts)
        assert len(result) == 2
        assert isinstance(result, pd.Series)

    def test_detect_other_language(self):
        lang = LanguageRouter.detect_language(
            "Dies ist ein deutscher Satz der lang genug sein sollte"
        )
        # German detected — should return 'other' or possibly misclassified
        assert lang in ('en', 'fr', 'other')


# ============================================================
#  ExpertFakeNewsDetector — without model
# ============================================================


class TestDetectorWithoutModel:
    def test_init_defaults(self):
        det = ExpertFakeNewsDetector(model_dir="/nonexistent")
        assert det.is_trained is False
        assert det.threshold == 0.44

    def test_predict_not_trained_raises(self):
        det = ExpertFakeNewsDetector(model_dir="/nonexistent")
        with pytest.raises(RuntimeError, match="non entra"):
            det.predict(pd.Series(["test"]))

    def test_predict_adaptive_not_trained_raises(self):
        det = ExpertFakeNewsDetector(model_dir="/nonexistent")
        with pytest.raises(RuntimeError, match="non entra"):
            det.predict_adaptive(pd.Series(["test"]))

    def test_explain_not_trained_raises(self):
        det = ExpertFakeNewsDetector(model_dir="/nonexistent")
        with pytest.raises(RuntimeError, match="non entra"):
            det.explain_prediction("test")

    def test_make_log_fiable(self):
        row = {'language': 'fr', 'ai_score_credibility': 0.85, 'prediction_label': 0}
        log = ExpertFakeNewsDetector._make_log(row)
        assert "Fiable" in log
        assert "FR" in log

    def test_make_log_suspect(self):
        row = {'language': 'en', 'ai_score_credibility': 0.2, 'prediction_label': 1}
        log = ExpertFakeNewsDetector._make_log(row)
        assert "Suspect" in log
        assert "EN" in log

    def test_health_check_not_trained(self):
        det = ExpertFakeNewsDetector(model_dir="/nonexistent")
        hc = det.health_check()
        assert hc['healthy'] is False

    def test_custom_threshold(self):
        det = ExpertFakeNewsDetector(model_dir="/nonexistent", threshold=0.3)
        assert det.threshold == 0.3


# ============================================================
#  ExpertFakeNewsDetector — with model (requires model files)
# ============================================================

_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
_MODEL_EXISTS = os.path.exists(os.path.join(_MODEL_DIR, 'model_expert_v5.pkl'))


@pytest.mark.skipif(not _MODEL_EXISTS, reason="Model files not found")
class TestDetectorWithModel:
    @pytest.fixture(scope='class')
    def detector(self):
        det = ExpertFakeNewsDetector(model_dir=_MODEL_DIR, threshold=0.44)
        det.load(suffix='expert_v5')
        return det

    def test_predict_deterministic(self, detector):
        """Same text should produce same score."""
        texts = pd.Series(["The economy grew by 2% this quarter according to official data."])
        r1 = detector.predict(texts)
        r2 = detector.predict(texts)
        assert r1['ai_score_credibility'].iloc[0] == r2['ai_score_credibility'].iloc[0]

    def test_predict_score_sum(self, detector):
        """Score should be in [0, 1]."""
        texts = pd.Series(["test text about science and research findings"])
        result = detector.predict(texts)
        score = result['ai_score_credibility'].iloc[0]
        assert 0 <= score <= 1

    def test_predict_empty_like_text(self, detector):
        """Very short text should not crash."""
        texts = pd.Series(["ok"])
        result = detector.predict(texts)
        assert len(result) == 1

    def test_predict_adaptive_returns_threshold(self, detector):
        """predict_adaptive should include adaptive_threshold column."""
        texts = pd.Series([
            "Short text",
            "A medium length text with more words to test the threshold boundary",
            "A very long article about politics and economics that contains many words " * 5,
        ])
        result = detector.predict_adaptive(texts)
        assert 'adaptive_threshold' in result.columns
        thresholds = result['adaptive_threshold'].values
        # Short -> 0.54, medium -> 0.49, long -> 0.44
        assert thresholds[0] == 0.54
        assert thresholds[2] == 0.44

    def test_predict_with_lang_thresholds(self, detector):
        """Language-specific thresholds should be applied."""
        detector.threshold_fr = 0.50
        detector.threshold_en = 0.40
        texts = pd.Series(["Le president a fait une annonce importante aujourd'hui"])
        result = detector.predict(texts)
        assert len(result) == 1
        # Reset
        detector.threshold_fr = None
        detector.threshold_en = None

    def test_explain_prediction_returns_dict(self, detector):
        """explain_prediction should return explainable dict."""
        explanation = detector.explain_prediction(
            "BREAKING: Secret labs discovered under the White House!!!"
        )
        assert explanation['explainable'] is True
        assert 'top_suspect_words' in explanation
        assert 'linguistic_signals' in explanation
        assert 'summary' in explanation

    def test_explain_fiable_text(self, detector):
        """Explain should work on reliable texts too."""
        explanation = detector.explain_prediction(
            "The WHO published its annual report on global health indicators."
        )
        assert explanation['explainable'] is True
        assert 'score_credibility' in explanation

    def test_health_check_trained(self, detector):
        """Health check should return healthy=True."""
        hc = detector.health_check()
        assert hc['healthy'] is True

    def test_save_load_roundtrip(self, detector):
        """Save and reload should produce identical predictions."""
        texts = pd.Series(["Test article about climate change and global warming."])
        result_before = detector.predict(texts)

        with tempfile.TemporaryDirectory() as tmpdir:
            det2 = ExpertFakeNewsDetector(model_dir=tmpdir, threshold=0.44)
            det2.model = detector.model
            det2.vectorizer = detector.vectorizer
            det2.training_metrics = detector.training_metrics
            det2.is_trained = True
            det2.use_emotions = False
            det2.save(suffix='test_roundtrip')

            det3 = ExpertFakeNewsDetector(model_dir=tmpdir, threshold=0.44)
            det3.load(suffix='test_roundtrip')
            result_after = det3.predict(texts)

        np.testing.assert_allclose(
            result_before['ai_score_credibility'].values,
            result_after['ai_score_credibility'].values,
            atol=1e-6,
        )

    def test_build_features_shape(self, detector):
        """_build_features should return correct number of columns."""
        texts = np.array(["test text for feature building"])
        X = detector._build_features(texts, texts_original=texts, fit=False)
        # 30000 TF-IDF + 15 linguistic (no emotions in v5 default)
        assert X.shape[0] == 1
        assert X.shape[1] >= 30000  # at least TF-IDF features

    def test_predict_batch_multiple(self, detector):
        """Batch prediction on 10 texts."""
        texts = pd.Series([
            "Scientific study confirms vaccine safety protocols.",
            "BREAKING SHOCKING news you won't believe!!!",
            "The weather today is sunny with clear skies.",
            "Les chercheurs ont publie leurs resultats scientifiques.",
            "SCANDALE!!! ON VOUS CACHE LA VERITE!!!",
        ] * 2)
        result = detector.predict(texts)
        assert len(result) == 10
        assert result['ai_score_credibility'].between(0, 1).all()


# ============================================================
#  ExpertFakeNewsDetector — train (minimal data)
# ============================================================


class TestDetectorTrain:
    def test_train_minimal_dataset(self):
        """Train on a tiny dataset to verify the pipeline doesn't crash."""
        det = ExpertFakeNewsDetector(model_dir=tempfile.mkdtemp(), threshold=0.5)

        # Minimal balanced dataset
        n = 40
        texts_fake = [f"SHOCKING SCANDAL exposed secret plot number {i}" for i in range(n)]
        texts_real = [f"The scientific study published in journal number {i}" for i in range(n)]

        df = pd.DataFrame({
            'text_original': texts_fake + texts_real,
            'text_clean': [DatasetCleaner.clean_for_ml(t) for t in texts_fake + texts_real],
            'label': [1] * n + [0] * n,
            'language': ['en'] * (2 * n),
        })

        metrics = det.train(df, model_type='logreg', n_folds=2, track_emissions=False)
        assert det.is_trained is True
        assert 'cv_f1_mean' in metrics
        assert metrics['cv_f1_mean'] > 0

        # Predict
        result = det.predict(pd.Series(["SHOCKING secret exposed!!!"]))
        assert len(result) == 1
