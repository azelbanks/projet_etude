"""
Extended tests for ExpertFakeNewsDetector and related classes.

Goal: increase code coverage of src/pipeline/expert_detector.py
from ~37% to significantly higher by testing uncovered methods
with mocks for heavy dependencies.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, PropertyMock
from scipy.sparse import csr_matrix, hstack

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pipeline.expert_detector import (
    ExpertFakeNewsDetector,
    LinguisticFeatureExtractor,
    DatasetCleaner,
    LanguageRouter,
    EmotionFeatureExtractor,
)


# ================================================================
#  Fixtures
# ================================================================

@pytest.fixture
def trained_detector():
    """Create a detector with mocked model and vectorizer (simulates trained state)."""
    d = ExpertFakeNewsDetector.__new__(ExpertFakeNewsDetector)
    d.model_dir = '/nonexistent'
    d.is_trained = True
    d.model = MagicMock()
    d.model.predict_proba.return_value = np.array([[0.7, 0.3], [0.3, 0.7]])
    d.model.predict.return_value = np.array([0, 1])
    d.model.coef_ = np.random.randn(1, 20015)  # TF-IDF features + 15 linguistic
    d.vectorizer = MagicMock()
    d.vectorizer.transform.return_value = csr_matrix(np.random.rand(2, 20000))
    d.vectorizer.get_feature_names_out.return_value = np.array(
        [f'word_{i}' for i in range(20000)]
    )
    d.vectorizer.max_features = 20000
    d.threshold = 0.44
    d.threshold_fr = None
    d.threshold_en = None
    d.use_emotions = False
    d.emotion_extractor = None
    d.training_metrics = {}
    return d


@pytest.fixture
def trained_detector_lang_thresholds(trained_detector):
    """Detector with per-language thresholds set."""
    trained_detector.threshold_fr = 0.50
    trained_detector.threshold_en = 0.40
    return trained_detector


@pytest.fixture
def sample_texts():
    """Two sample texts for predict tests."""
    return pd.Series([
        "This is a normal news article about weather.",
        "SHOCKING: secret conspiracy revealed! Share before deleted!!!",
    ])


# ================================================================
#  1. _get_model()
# ================================================================

class TestGetModel:
    def test_logreg(self):
        model = ExpertFakeNewsDetector._get_model('logreg')
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_svm(self):
        model = ExpertFakeNewsDetector._get_model('svm')
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_ensemble(self):
        model = ExpertFakeNewsDetector._get_model('ensemble')
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="model_type inconnu"):
            ExpertFakeNewsDetector._get_model('unknown')

    def test_logreg_returns_logistic_regression(self):
        from sklearn.linear_model import LogisticRegression
        model = ExpertFakeNewsDetector._get_model('logreg')
        assert isinstance(model, LogisticRegression)

    def test_svm_returns_calibrated(self):
        from sklearn.calibration import CalibratedClassifierCV
        model = ExpertFakeNewsDetector._get_model('svm')
        assert isinstance(model, CalibratedClassifierCV)

    def test_ensemble_returns_voting(self):
        from sklearn.ensemble import VotingClassifier
        model = ExpertFakeNewsDetector._get_model('ensemble')
        assert isinstance(model, VotingClassifier)


# ================================================================
#  2. _make_log()
# ================================================================

class TestMakeLog:
    def test_suspect_fr(self):
        row = {'language': 'fr', 'ai_score_credibility': 0.3, 'prediction_label': 1}
        result = ExpertFakeNewsDetector._make_log(row)
        assert 'Suspect' in result
        assert 'FR' in result

    def test_fiable_en(self):
        row = {'language': 'en', 'ai_score_credibility': 0.8, 'prediction_label': 0}
        result = ExpertFakeNewsDetector._make_log(row)
        assert 'Fiable' in result
        assert 'EN' in result

    def test_other_language(self):
        row = {'language': 'other', 'ai_score_credibility': 0.5, 'prediction_label': 0}
        result = ExpertFakeNewsDetector._make_log(row)
        assert '??' in result

    def test_unknown_language_defaults_to_question_marks(self):
        row = {'language': 'de', 'ai_score_credibility': 0.6, 'prediction_label': 0}
        result = ExpertFakeNewsDetector._make_log(row)
        assert '??' in result

    def test_suspect_label_1(self):
        row = {'language': 'en', 'ai_score_credibility': 0.2, 'prediction_label': 1}
        result = ExpertFakeNewsDetector._make_log(row)
        assert 'Suspect' in result

    def test_missing_language_defaults(self):
        row = {'ai_score_credibility': 0.5, 'prediction_label': 0}
        result = ExpertFakeNewsDetector._make_log(row)
        # defaults to 'en' -> 'EN'
        assert 'EN' in result


# ================================================================
#  3. predict()
# ================================================================

class TestPredict:
    def test_predict_returns_dataframe(self, trained_detector, sample_texts):
        result = trained_detector.predict(sample_texts, track_emissions=False)
        assert isinstance(result, pd.DataFrame)
        assert 'prediction_label' in result.columns
        assert 'ai_score_credibility' in result.columns
        assert 'ai_analysis_log' in result.columns
        assert 'language' in result.columns

    def test_predict_correct_length(self, trained_detector, sample_texts):
        result = trained_detector.predict(sample_texts, track_emissions=False)
        assert len(result) == len(sample_texts)

    def test_predict_not_trained_raises(self, sample_texts):
        d = ExpertFakeNewsDetector.__new__(ExpertFakeNewsDetector)
        d.is_trained = False
        with pytest.raises(RuntimeError, match="non entraîné"):
            d.predict(sample_texts, track_emissions=False)

    def test_predict_with_lang_thresholds(self, trained_detector_lang_thresholds, sample_texts):
        result = trained_detector_lang_thresholds.predict(
            sample_texts, track_emissions=False
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_predict_scores_are_rounded(self, trained_detector, sample_texts):
        result = trained_detector.predict(sample_texts, track_emissions=False)
        for score in result['ai_score_credibility']:
            # Check that scores are rounded to 4 decimals
            assert score == round(score, 4)

    def test_predict_labels_are_binary(self, trained_detector, sample_texts):
        result = trained_detector.predict(sample_texts, track_emissions=False)
        assert set(result['prediction_label'].unique()).issubset({0, 1})


# ================================================================
#  4. predict_adaptive()
# ================================================================

class TestPredictAdaptive:
    def test_predict_adaptive_returns_dataframe(self, trained_detector, sample_texts):
        result = trained_detector.predict_adaptive(sample_texts, track_emissions=False)
        assert isinstance(result, pd.DataFrame)
        assert 'adaptive_threshold' in result.columns

    def test_predict_adaptive_not_trained_raises(self, sample_texts):
        d = ExpertFakeNewsDetector.__new__(ExpertFakeNewsDetector)
        d.is_trained = False
        with pytest.raises(RuntimeError, match="non entraîné"):
            d.predict_adaptive(sample_texts, track_emissions=False)

    def test_predict_adaptive_short_text_threshold(self, trained_detector):
        """Short texts (<15 words) should get threshold 0.54."""
        short = pd.Series(["Short text here"])
        # Need single-row predict_proba
        trained_detector.model.predict_proba.return_value = np.array([[0.5, 0.5]])
        trained_detector.vectorizer.transform.return_value = csr_matrix(
            np.random.rand(1, 20000)
        )
        result = trained_detector.predict_adaptive(short, track_emissions=False)
        assert result['adaptive_threshold'].iloc[0] == 0.54

    def test_predict_adaptive_medium_text_threshold(self, trained_detector):
        """Medium texts (15-30 words) should get threshold 0.49."""
        medium = pd.Series([" ".join(["word"] * 20)])
        trained_detector.model.predict_proba.return_value = np.array([[0.5, 0.5]])
        trained_detector.vectorizer.transform.return_value = csr_matrix(
            np.random.rand(1, 20000)
        )
        result = trained_detector.predict_adaptive(medium, track_emissions=False)
        assert result['adaptive_threshold'].iloc[0] == 0.49

    def test_predict_adaptive_long_text_threshold(self, trained_detector):
        """Long texts (>30 words) should get threshold 0.44."""
        long_text = pd.Series([" ".join(["word"] * 50)])
        trained_detector.model.predict_proba.return_value = np.array([[0.5, 0.5]])
        trained_detector.vectorizer.transform.return_value = csr_matrix(
            np.random.rand(1, 20000)
        )
        result = trained_detector.predict_adaptive(long_text, track_emissions=False)
        assert result['adaptive_threshold'].iloc[0] == 0.44


# ================================================================
#  5. save() and load()
# ================================================================

class TestSaveLoad:
    def test_save_creates_files(self, trained_detector, tmp_path):
        """Mock joblib.dump to verify save() calls it with correct paths."""
        trained_detector.model_dir = str(tmp_path / "models")
        trained_detector.training_metrics = {'cv_f1_mean': 0.95}

        with patch('pipeline.expert_detector.joblib.dump') as mock_dump:
            trained_detector.save(suffix='test')

        assert mock_dump.call_count == 3
        # Verify correct file paths used
        calls = [str(c) for c in mock_dump.call_args_list]
        assert any('model_test.pkl' in c for c in calls)
        assert any('tfidf_test.pkl' in c for c in calls)
        assert any('metrics_test.pkl' in c for c in calls)

    def test_save_creates_directory(self, tmp_path):
        """save() should create model_dir if it does not exist."""
        d = ExpertFakeNewsDetector.__new__(ExpertFakeNewsDetector)
        d.model_dir = str(tmp_path / "new_dir" / "models")
        d.model = "fake_model"
        d.vectorizer = "fake_vec"
        d.training_metrics = {}

        with patch('pipeline.expert_detector.joblib.dump'):
            d.save(suffix='test')

        assert os.path.isdir(d.model_dir)

    def test_load_restores_state(self, tmp_path):
        """Use real joblib with picklable objects to test load()."""
        import joblib

        model_dir = str(tmp_path / "models")
        os.makedirs(model_dir, exist_ok=True)

        # Use real picklable objects instead of MagicMock
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer

        real_model = LogisticRegression()
        real_vectorizer = TfidfVectorizer()
        metrics = {'cv_f1_mean': 0.95, 'use_emotions': False}

        joblib.dump(real_model, os.path.join(model_dir, 'model_expert.pkl'))
        joblib.dump(real_vectorizer, os.path.join(model_dir, 'tfidf_expert.pkl'))
        joblib.dump(metrics, os.path.join(model_dir, 'metrics_expert.pkl'))

        d = ExpertFakeNewsDetector.__new__(ExpertFakeNewsDetector)
        d.model_dir = model_dir
        d.is_trained = False
        d.training_metrics = {}
        d.use_emotions = False
        d.emotion_extractor = None
        d.load(suffix='expert')

        assert d.is_trained is True
        assert d.training_metrics['cv_f1_mean'] == 0.95

    def test_load_without_metrics_file(self, tmp_path):
        """Load works even if metrics file is missing."""
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer

        model_dir = str(tmp_path / "models")
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(LogisticRegression(), os.path.join(model_dir, 'model_expert.pkl'))
        joblib.dump(TfidfVectorizer(), os.path.join(model_dir, 'tfidf_expert.pkl'))
        # No metrics file

        d = ExpertFakeNewsDetector.__new__(ExpertFakeNewsDetector)
        d.model_dir = model_dir
        d.is_trained = False
        d.training_metrics = {}
        d.use_emotions = False
        d.emotion_extractor = None
        d.load(suffix='expert')

        assert d.is_trained is True


# ================================================================
#  6. health_check()
# ================================================================

class TestHealthCheck:
    def test_health_check_not_trained(self):
        d = ExpertFakeNewsDetector.__new__(ExpertFakeNewsDetector)
        d.is_trained = False
        result = d.health_check()
        assert result['healthy'] is False
        assert 'error' in result['details'][0]

    def test_health_check_trained_all_pass(self, trained_detector):
        """Mock predict to return expected values for all health check cases."""
        n_cases = len(ExpertFakeNewsDetector.HEALTH_CHECK_CASES)

        # Build mock predict return that satisfies all cases
        labels = []
        scores = []
        for text, expected_label, score_min, score_max in ExpertFakeNewsDetector.HEALTH_CHECK_CASES:
            labels.append(expected_label)
            scores.append((score_min + score_max) / 2)

        mock_result = pd.DataFrame({
            'prediction_label': labels,
            'ai_score_credibility': scores,
            'language': ['en'] * n_cases,
            'ai_analysis_log': ['log'] * n_cases,
            'text': [c[0] for c in ExpertFakeNewsDetector.HEALTH_CHECK_CASES],
        })

        trained_detector.predict = MagicMock(return_value=mock_result)
        result = trained_detector.health_check()
        assert result['healthy'] is True
        assert len(result['details']) == n_cases

    def test_health_check_trained_some_fail(self, trained_detector):
        """Mock predict with bad predictions to test failure reporting."""
        n_cases = len(ExpertFakeNewsDetector.HEALTH_CHECK_CASES)

        mock_result = pd.DataFrame({
            'prediction_label': [1] * n_cases,  # all suspect -> some will fail
            'ai_score_credibility': [0.1] * n_cases,
            'language': ['en'] * n_cases,
            'ai_analysis_log': ['log'] * n_cases,
            'text': [c[0] for c in ExpertFakeNewsDetector.HEALTH_CHECK_CASES],
        })

        trained_detector.predict = MagicMock(return_value=mock_result)
        result = trained_detector.health_check()
        assert result['healthy'] is False


# ================================================================
#  7. evaluate_holdout()
# ================================================================

class TestEvaluateHoldout:
    def test_evaluate_holdout_returns_metrics(self, trained_detector):
        """Test evaluation with mocked model."""
        df = pd.DataFrame({
            'text_clean': ['clean text one', 'clean text two'],
            'text_original': ['Original text one.', 'Original text two!'],
            'label': [0, 1],
        })

        trained_detector.model.predict.return_value = np.array([0, 1])
        trained_detector.model.predict_proba.return_value = np.array(
            [[0.8, 0.2], [0.3, 0.7]]
        )

        result = trained_detector.evaluate_holdout(df)
        assert 'accuracy' in result
        assert 'f1' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'confusion_matrix' in result
        assert 'report' in result

    def test_evaluate_holdout_not_trained_raises(self):
        d = ExpertFakeNewsDetector.__new__(ExpertFakeNewsDetector)
        d.is_trained = False
        df = pd.DataFrame({'text_clean': ['x'], 'label': [0]})
        with pytest.raises(RuntimeError, match="non entraîné"):
            d.evaluate_holdout(df)

    def test_evaluate_holdout_has_roc_auc(self, trained_detector):
        """Models with predict_proba should produce roc_auc."""
        df = pd.DataFrame({
            'text_clean': ['text a', 'text b'],
            'text_original': ['Text A.', 'Text B!'],
            'label': [0, 1],
        })
        trained_detector.model.predict.return_value = np.array([0, 1])
        trained_detector.model.predict_proba.return_value = np.array(
            [[0.9, 0.1], [0.2, 0.8]]
        )
        result = trained_detector.evaluate_holdout(df)
        assert 'roc_auc' in result

    def test_evaluate_holdout_without_text_original(self, trained_detector):
        """Evaluation should work even without text_original column."""
        df = pd.DataFrame({
            'text_clean': ['text a', 'text b'],
            'label': [0, 1],
        })
        trained_detector.model.predict.return_value = np.array([0, 1])
        trained_detector.model.predict_proba.return_value = np.array(
            [[0.9, 0.1], [0.2, 0.8]]
        )
        result = trained_detector.evaluate_holdout(df)
        assert result['accuracy'] == 1.0


# ================================================================
#  8. explain_prediction()
# ================================================================

class TestExplainPrediction:
    def test_explain_returns_explainable(self, trained_detector):
        """With coef_ attribute, explanation should be explainable."""
        # Single text prediction
        trained_detector.model.predict_proba.return_value = np.array([[0.6, 0.4]])
        trained_detector.vectorizer.transform.return_value = csr_matrix(
            np.random.rand(1, 20000)
        )
        trained_detector.vectorizer.get_feature_names_out.return_value = np.array(
            [f'word_{i}' for i in range(20000)]
        )

        result = trained_detector.explain_prediction("This is a test article about science.")
        assert result['explainable'] is True
        assert 'prediction_label' in result
        assert 'score_credibility' in result
        assert 'linguistic_signals' in result
        assert 'summary' in result

    def test_explain_not_explainable_without_coef(self, trained_detector):
        """Model without coef_ should return explainable=False."""
        del trained_detector.model.coef_
        result = trained_detector.explain_prediction("Test text")
        assert result['explainable'] is False
        assert 'reason' in result

    def test_explain_not_trained_raises(self):
        d = ExpertFakeNewsDetector.__new__(ExpertFakeNewsDetector)
        d.is_trained = False
        with pytest.raises(RuntimeError, match="non entraîné"):
            d.explain_prediction("some text")

    def test_explain_has_sensationalist_words(self, trained_detector):
        """Text with sensationalist words should be detected."""
        trained_detector.model.predict_proba.return_value = np.array([[0.3, 0.7]])
        trained_detector.vectorizer.transform.return_value = csr_matrix(
            np.random.rand(1, 20000)
        )
        trained_detector.vectorizer.get_feature_names_out.return_value = np.array(
            [f'word_{i}' for i in range(20000)]
        )

        result = trained_detector.explain_prediction(
            "SHOCKING conspiracy exposed! Wake up people!"
        )
        assert result['explainable'] is True
        assert len(result['sensationalist_words']) > 0

    def test_explain_with_lang_thresholds(self, trained_detector_lang_thresholds):
        """Explain should use language-specific thresholds when set."""
        d = trained_detector_lang_thresholds
        d.model.predict_proba.return_value = np.array([[0.45, 0.55]])
        d.vectorizer.transform.return_value = csr_matrix(
            np.random.rand(1, 20000)
        )
        d.vectorizer.get_feature_names_out.return_value = np.array(
            [f'word_{i}' for i in range(20000)]
        )

        # With threshold_en=0.40, score 0.45 > 0.40 -> label=0 (Fiable)
        with patch.object(LanguageRouter, 'detect_language', return_value='en'):
            result = d.explain_prediction("Normal English text about research.")
        assert result['prediction_label'] == 0

    def test_explain_linguistic_signals_structure(self, trained_detector):
        """Each linguistic signal should have feature/value/contribution/direction."""
        trained_detector.model.predict_proba.return_value = np.array([[0.6, 0.4]])
        trained_detector.vectorizer.transform.return_value = csr_matrix(
            np.random.rand(1, 20000)
        )
        trained_detector.vectorizer.get_feature_names_out.return_value = np.array(
            [f'word_{i}' for i in range(20000)]
        )

        result = trained_detector.explain_prediction("Test article text.")
        for signal in result['linguistic_signals']:
            assert 'feature' in signal
            assert 'value' in signal
            assert 'contribution' in signal
            assert signal['direction'] in ('SUSPECT', 'FIABLE')


# ================================================================
#  9. _build_features()
# ================================================================

class TestBuildFeatures:
    def test_build_features_fit(self, trained_detector):
        """Test feature building in fit mode."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        trained_detector.vectorizer = TfidfVectorizer(max_features=100)
        texts_clean = np.array(["hello world test", "another sample text here"])

        X = trained_detector._build_features(texts_clean, fit=True)
        # Should have TF-IDF cols + 15 linguistic cols
        assert X.shape[0] == 2
        assert X.shape[1] > 15  # at least linguistic features

    def test_build_features_transform(self, trained_detector):
        """Test feature building in transform (predict) mode."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        trained_detector.vectorizer = TfidfVectorizer(max_features=100)
        texts = np.array(["hello world test", "another sample text here"])
        # First fit
        trained_detector._build_features(texts, fit=True)
        # Then transform
        X = trained_detector._build_features(texts, fit=False)
        assert X.shape[0] == 2

    def test_build_features_with_original_texts(self, trained_detector):
        """When texts_original is provided, linguistic features use it."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        trained_detector.vectorizer = TfidfVectorizer(max_features=100)
        texts_clean = np.array(["hello world test", "another sample"])
        texts_original = np.array(["Hello WORLD Test!", "Another SAMPLE!!"])

        X = trained_detector._build_features(
            texts_clean, texts_original=texts_original, fit=True
        )
        assert X.shape[0] == 2

    def test_build_features_with_emotions(self):
        """When use_emotions=True, emotion features are appended."""
        d = ExpertFakeNewsDetector.__new__(ExpertFakeNewsDetector)
        d.use_emotions = True
        d.emotion_extractor = MagicMock()
        d.emotion_extractor.get_emotion_features.return_value = np.random.rand(2, 7)

        from sklearn.feature_extraction.text import TfidfVectorizer
        d.vectorizer = TfidfVectorizer(max_features=100)

        texts_clean = np.array(["hello world test", "another sample text here"])
        X = d._build_features(texts_clean, fit=True)
        n_ling = len(LinguisticFeatureExtractor.FEATURE_NAMES)
        # total features = tfidf + linguistic + 7 emotions
        assert X.shape[1] > n_ling + 7


# ================================================================
#  10. DatasetCleaner additional methods
# ================================================================

class TestDatasetCleaner:
    def test_remove_agency_bias_reuters(self):
        text = "WASHINGTON (Reuters) - The president announced something."
        cleaned = DatasetCleaner.remove_agency_bias(text)
        assert "(Reuters)" not in cleaned

    def test_remove_agency_bias_non_string(self):
        result = DatasetCleaner.remove_agency_bias(None)
        assert result == ""

    def test_clean_for_ml_removes_urls(self):
        text = "Check http://example.com for more"
        cleaned = DatasetCleaner.clean_for_ml(text)
        assert "http" not in cleaned

    def test_clean_for_ml_non_string(self):
        result = DatasetCleaner.clean_for_ml(42)
        assert result == ""

    def test_clean_for_ml_lowercases(self):
        result = DatasetCleaner.clean_for_ml("HELLO World")
        assert result == "hello world"

    def test_prepare_clean_dataset(self, tmp_path):
        """Test prepare_clean_dataset with CSV files."""
        fake_csv = tmp_path / "Fake.csv"
        true_csv = tmp_path / "True.csv"

        # Create minimal CSV data with enough words (>20)
        long_text = " ".join(["word"] * 25)
        pd.DataFrame({'text': [long_text, long_text]}).to_csv(fake_csv, index=False)
        pd.DataFrame({'text': [long_text, long_text]}).to_csv(true_csv, index=False)

        df = DatasetCleaner.prepare_clean_dataset(str(fake_csv), str(true_csv))
        assert 'text_clean' in df.columns
        assert 'label' in df.columns
        assert set(df['label'].unique()) == {0, 1}

    def test_generate_fr_short_augmentation(self):
        """Test short FR augmentation with a simple DataFrame."""
        long_text = "Ceci est une premiere phrase. " + " ".join(["mot"] * 20)
        df_fr = pd.DataFrame({
            'text_original': [long_text],
            'text_clean': [long_text.lower()],
            'label': [1],
        })
        result = DatasetCleaner.generate_fr_short_augmentation(df_fr)
        assert isinstance(result, pd.DataFrame)
        assert 'language' in result.columns
        if len(result) > 0:
            assert all(result['language'] == 'fr')

    def test_audit_reuters_leakage(self):
        df_true = pd.DataFrame({
            'text': [
                "WASHINGTON (Reuters) - Some text here",
                "No agency markers here",
                "Reporting by John Smith; Editing by Jane Doe",
            ]
        })
        audit = DatasetCleaner.audit_reuters_leakage(df_true)
        assert audit['total_articles'] == 3
        assert audit['has_reuters_marker'] >= 1


# ================================================================
#  11. LinguisticFeatureExtractor
# ================================================================

class TestLinguisticFeatureExtractor:
    def test_extract_shape(self):
        texts = pd.Series(["Hello world", "Test article"])
        result = LinguisticFeatureExtractor.extract(texts)
        assert result.shape == (2, len(LinguisticFeatureExtractor.FEATURE_NAMES))

    def test_feature_names_count(self):
        assert len(LinguisticFeatureExtractor.FEATURE_NAMES) == 15

    def test_sensationalist_detection(self):
        texts = pd.Series(["SHOCKING scandal exposed conspiracy!"])
        result = LinguisticFeatureExtractor.extract(texts)
        # sensationalism_score is index 6
        assert result[0, 6] > 0

    def test_short_text_indicator(self):
        short = pd.Series(["Short text"])
        result = LinguisticFeatureExtractor.extract(short)
        # is_short_text is index 14
        assert result[0, 14] == 1.0

    def test_long_text_indicator(self):
        long_text = pd.Series([" ".join(["word"] * 30)])
        result = LinguisticFeatureExtractor.extract(long_text)
        assert result[0, 14] == 0.0


# ================================================================
#  12. LanguageRouter
# ================================================================

class TestLanguageRouter:
    def test_detect_batch(self):
        texts = pd.Series(["Hello world", "Bonjour le monde"])
        result = LanguageRouter.detect_batch(texts)
        assert len(result) == 2
        assert all(lang in ('fr', 'en', 'other') for lang in result)


# ================================================================
#  13. EmotionFeatureExtractor
# ================================================================

class TestEmotionFeatureExtractor:
    def test_init(self):
        extractor = EmotionFeatureExtractor(model_dir='/nonexistent')
        assert extractor._loaded is False

    def test_load_missing_files_returns_false(self):
        extractor = EmotionFeatureExtractor(model_dir='/nonexistent')
        result = extractor.load()
        assert result is False

    def test_get_emotion_features_not_loaded_raises(self):
        extractor = EmotionFeatureExtractor(model_dir='/nonexistent')
        with pytest.raises(RuntimeError, match="non chargé"):
            extractor.get_emotion_features(["test"])

    def test_feature_names(self):
        assert len(EmotionFeatureExtractor.FEATURE_NAMES) == 7
        assert 'emo_colere' in EmotionFeatureExtractor.FEATURE_NAMES


# ================================================================
#  14. ExpertFakeNewsDetector __init__
# ================================================================

class TestDetectorInit:
    def test_init_default_params(self):
        with patch.object(EmotionFeatureExtractor, 'load', return_value=False):
            d = ExpertFakeNewsDetector(model_dir='/tmp/test', use_emotions=False)
        assert d.is_trained is False
        assert d.threshold == 0.44
        assert d.threshold_fr is None
        assert d.threshold_en is None

    def test_init_with_custom_thresholds(self):
        d = ExpertFakeNewsDetector(
            model_dir='/tmp/test',
            use_emotions=False,
            threshold=0.5,
            threshold_fr=0.55,
            threshold_en=0.45,
        )
        assert d.threshold == 0.5
        assert d.threshold_fr == 0.55
        assert d.threshold_en == 0.45

    def test_init_emotions_unavailable_fallback(self):
        """When emotion model files are missing, use_emotions should be disabled."""
        d = ExpertFakeNewsDetector(
            model_dir='/nonexistent_path',
            use_emotions=True,
        )
        # Should fallback to False since model files don't exist
        assert d.use_emotions is False
        assert d.emotion_extractor is None
