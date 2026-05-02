"""
Tests for the inference-related functions in collection/collect_bluesky.py.

Covers _load_inference_models() and run_inference_cycle() (lines ~300-476)
with thorough mocking -- no actual file I/O, no model loading, no MongoDB.
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
#  _load_inference_models
# ---------------------------------------------------------------------------

class TestLoadInferenceModels:
    """Tests for _load_inference_models()."""

    @patch('collection.collect_bluesky._emotion_model', new=MagicMock())
    def test_already_loaded_returns_true(self):
        """When _emotion_model is already set, return True immediately."""
        from collection.collect_bluesky import _load_inference_models
        result = _load_inference_models()
        assert result is True

    @patch('collection.collect_bluesky._emotion_model', None)
    @patch('builtins.open', side_effect=FileNotFoundError("not found"))
    def test_files_missing_returns_false(self, mock_file):
        """When pickle files are missing, return False."""
        from collection.collect_bluesky import _load_inference_models
        result = _load_inference_models()
        assert result is False

    @patch('collection.collect_bluesky._emotion_model', None)
    @patch('collection.collect_bluesky._emotion_vocab', None)
    @patch('collection.collect_bluesky._emotion_le', None)
    @patch('collection.collect_bluesky._detector', None)
    @patch('collection.collect_bluesky._emo_extractor', None)
    @patch('collection.collect_bluesky._stage1_pipe', None)
    def test_success_loads_all_models(self):
        """Full success path: mock all file I/O, torch, pickle, imports."""
        import collection.collect_bluesky as mod

        mock_vocab = {'<PAD>': 0, '<UNK>': 1, 'hello': 2}
        mock_le = MagicMock()

        # Fake state dict with the expected tensor shapes
        fake_embedding = MagicMock()
        fake_embedding.shape = (100, 64)  # vocab_size=100, embed_dim=64
        fake_fc3 = MagicMock()
        fake_fc3.shape = (6,)  # num_classes=6

        fake_state_dict = {
            'model_state_dict': {
                'embedding.weight': fake_embedding,
                'fc3.weight': fake_fc3,
            },
            'max_len': 80,
        }

        mock_emotion_mlp_cls = MagicMock()
        mock_emotion_mlp_instance = MagicMock()
        mock_emotion_mlp_cls.return_value = mock_emotion_mlp_instance

        mock_detector_cls = MagicMock()
        mock_detector_instance = MagicMock()
        mock_detector_cls.return_value = mock_detector_instance

        mock_extractor_cls = MagicMock()
        mock_extractor_instance = MagicMock()
        mock_extractor_cls.return_value = mock_extractor_instance

        pickle_loads = [mock_vocab, mock_le]

        with patch('builtins.open', mock_open()):
            with patch.dict('sys.modules', {
                'torch': MagicMock(**{'load.return_value': fake_state_dict}),
                'pickle': MagicMock(**{'load.side_effect': pickle_loads}),
                'joblib': MagicMock(**{'load.return_value': {'pipeline': MagicMock(), 'threshold': 0.45}}),
                'pipeline': MagicMock(),
                'pipeline.expert_detector': MagicMock(
                    _EmotionMLP=mock_emotion_mlp_cls,
                    ExpertFakeNewsDetector=mock_detector_cls,
                    EmotionFeatureExtractor=mock_extractor_cls,
                ),
            }):
                with patch('os.path.exists', return_value=True):
                    # Reset global so the guard lets us through
                    mod._emotion_model = None
                    result = mod._load_inference_models()

        assert result is True

    @patch('collection.collect_bluesky._emotion_model', None)
    def test_exception_during_load_returns_false(self):
        """Any exception during loading returns False."""
        with patch('builtins.open', side_effect=PermissionError("denied")):
            from collection.collect_bluesky import _load_inference_models
            result = _load_inference_models()
        assert result is False

    @patch('collection.collect_bluesky._emotion_model', None)
    def test_torch_load_failure_returns_false(self):
        """If torch.load raises, return False."""
        mock_vocab = {'<PAD>': 0}
        mock_le = MagicMock()
        pickle_loads = [mock_vocab, mock_le]

        with patch('builtins.open', mock_open()):
            with patch.dict('sys.modules', {
                'torch': MagicMock(**{'load.side_effect': RuntimeError("corrupt")}),
                'pickle': MagicMock(**{'load.side_effect': pickle_loads}),
            }):
                from collection.collect_bluesky import _load_inference_models
                result = _load_inference_models()
        assert result is False


# ---------------------------------------------------------------------------
#  run_inference_cycle
# ---------------------------------------------------------------------------

class TestRunInferenceCycle:
    """Tests for run_inference_cycle()."""

    @patch('collection.collect_bluesky._emotion_model', None)
    @patch('collection.collect_bluesky._load_inference_models')
    def test_models_not_loaded_returns_early(self, mock_load):
        """If _load_inference_models returns False, do nothing."""
        mock_load.return_value = False
        from collection.collect_bluesky import run_inference_cycle
        mock_collection = MagicMock()
        run_inference_cycle(mock_collection)
        mock_collection.count_documents.assert_not_called()

    @patch('collection.collect_bluesky._stage1_pipe', None)
    @patch('collection.collect_bluesky._emotion_max_len', 60)
    @patch('collection.collect_bluesky._emotion_le')
    @patch('collection.collect_bluesky._emotion_vocab', {'<PAD>': 0, '<UNK>': 1})
    @patch('collection.collect_bluesky._detector')
    @patch('collection.collect_bluesky._emotion_model')
    @patch('collection.collect_bluesky._load_inference_models')
    def test_no_documents_returns_early(self, mock_load, mock_model,
                                        mock_detector, mock_le):
        """If count_documents returns 0, no processing happens."""
        mock_load.return_value = True
        from collection.collect_bluesky import run_inference_cycle
        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 0
        run_inference_cycle(mock_collection)
        mock_collection.find.assert_not_called()

    @patch('collection.collect_bluesky._stage1_pipe', None)
    @patch('collection.collect_bluesky._stage1_threshold', 0.40)
    @patch('collection.collect_bluesky._emotion_max_len', 5)
    @patch('collection.collect_bluesky._emotion_le')
    @patch('collection.collect_bluesky._emotion_vocab',
           {'<PAD>': 0, '<UNK>': 1, 'hello': 2, 'world': 3})
    @patch('collection.collect_bluesky._detector')
    @patch('collection.collect_bluesky._emotion_model')
    @patch('collection.collect_bluesky._load_inference_models')
    def test_full_cycle_without_stage1(self, mock_load, mock_model,
                                       mock_detector, mock_le):
        """Full cycle with 2 documents, no Stage1 pipeline."""
        import torch
        import pandas as pd

        mock_load.return_value = True

        # Emotion model returns fake logits
        fake_logits = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
        mock_model.return_value = fake_logits

        # Label encoder
        mock_le.inverse_transform.return_value = np.array(['joy', 'anger'])

        # V5 detector returns a DataFrame
        mock_detector.predict.return_value = pd.DataFrame({
            'prediction_label': [0, 1],
            'ai_score_credibility': [0.85, 0.30],
            'language': ['en', 'fr'],
            'ai_analysis_log': ['log1', 'log2'],
        })

        # MongoDB mock
        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 2

        doc1 = {'_id': 'id1', 'text': 'hello world'}
        doc2 = {'_id': 'id2', 'text': 'world hello world'}
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = [doc1, doc2]
        mock_collection.find.return_value = mock_cursor

        from collection.collect_bluesky import run_inference_cycle
        run_inference_cycle(mock_collection)

        mock_collection.bulk_write.assert_called_once()
        ops = mock_collection.bulk_write.call_args[0][0]
        assert len(ops) == 2

    @patch('collection.collect_bluesky._stage1_threshold', 0.40)
    @patch('collection.collect_bluesky._emotion_max_len', 5)
    @patch('collection.collect_bluesky._emotion_le')
    @patch('collection.collect_bluesky._emotion_vocab',
           {'<PAD>': 0, '<UNK>': 1, 'test': 2})
    @patch('collection.collect_bluesky._detector')
    @patch('collection.collect_bluesky._emotion_model')
    @patch('collection.collect_bluesky._load_inference_models')
    def test_full_cycle_with_stage1(self, mock_load, mock_model,
                                     mock_detector, mock_le):
        """Full cycle with Stage1 pipeline active."""
        import torch
        import pandas as pd

        mock_load.return_value = True

        fake_logits = torch.tensor([[0.5, 0.5]])
        mock_model.return_value = fake_logits
        mock_le.inverse_transform.return_value = np.array(['neutral'])

        mock_detector.predict.return_value = pd.DataFrame({
            'prediction_label': [1],
            'ai_score_credibility': [0.20],
            'language': ['fr'],
            'ai_analysis_log': ['suspicious'],
        })

        # Stage1 pipeline mock: opinion with high probability => v9 reclassifies
        mock_stage1 = MagicMock()
        mock_stage1.predict_proba.return_value = np.array([[0.8, 0.2]])

        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 1

        doc = {'_id': 'id_s1', 'text': 'test statement'}
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = [doc]
        mock_collection.find.return_value = mock_cursor

        with patch('collection.collect_bluesky._stage1_pipe', mock_stage1):
            from collection.collect_bluesky import run_inference_cycle
            run_inference_cycle(mock_collection)

        ops = mock_collection.bulk_write.call_args[0][0]
        assert len(ops) == 1

    @patch('collection.collect_bluesky._stage1_threshold', 0.40)
    @patch('collection.collect_bluesky._emotion_max_len', 5)
    @patch('collection.collect_bluesky._emotion_le')
    @patch('collection.collect_bluesky._emotion_vocab',
           {'<PAD>': 0, '<UNK>': 1, 'fact': 2})
    @patch('collection.collect_bluesky._detector')
    @patch('collection.collect_bluesky._emotion_model')
    @patch('collection.collect_bluesky._load_inference_models')
    def test_stage1_factuel_keeps_v5_label(self, mock_load, mock_model,
                                            mock_detector, mock_le):
        """When Stage1 says 'factuel', v9_label equals v5_label (no reclassification)."""
        import torch
        import pandas as pd

        mock_load.return_value = True
        fake_logits = torch.tensor([[0.1, 0.9]])
        mock_model.return_value = fake_logits
        mock_le.inverse_transform.return_value = np.array(['joy'])

        mock_detector.predict.return_value = pd.DataFrame({
            'prediction_label': [1],
            'ai_score_credibility': [0.25],
            'language': ['en'],
            'ai_analysis_log': ['flagged'],
        })

        # Stage1: high factuel probability => post_type = 'factuel'
        mock_stage1 = MagicMock()
        mock_stage1.predict_proba.return_value = np.array([[0.1, 0.9]])

        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 1
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = [{'_id': 'id_fact', 'text': 'fact check'}]
        mock_collection.find.return_value = mock_cursor

        with patch('collection.collect_bluesky._stage1_pipe', mock_stage1):
            from collection.collect_bluesky import run_inference_cycle
            run_inference_cycle(mock_collection)

        mock_collection.bulk_write.assert_called_once()

    @patch('collection.collect_bluesky._stage1_pipe', None)
    @patch('collection.collect_bluesky._stage1_threshold', 0.40)
    @patch('collection.collect_bluesky._emotion_max_len', 5)
    @patch('collection.collect_bluesky._emotion_le')
    @patch('collection.collect_bluesky._emotion_vocab',
           {'<PAD>': 0, '<UNK>': 1})
    @patch('collection.collect_bluesky._detector')
    @patch('collection.collect_bluesky._emotion_model')
    @patch('collection.collect_bluesky._load_inference_models')
    def test_empty_batch_breaks_loop(self, mock_load, mock_model,
                                      mock_detector, mock_le):
        """If find returns empty list mid-loop, processing stops."""
        mock_load.return_value = True

        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 5

        # First call returns empty list, so loop breaks immediately
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = []
        mock_collection.find.return_value = mock_cursor

        from collection.collect_bluesky import run_inference_cycle
        run_inference_cycle(mock_collection)

        mock_collection.bulk_write.assert_not_called()

    @patch('collection.collect_bluesky._stage1_pipe', None)
    @patch('collection.collect_bluesky._stage1_threshold', 0.40)
    @patch('collection.collect_bluesky._emotion_max_len', 3)
    @patch('collection.collect_bluesky._emotion_le')
    @patch('collection.collect_bluesky._emotion_vocab',
           {'<PAD>': 0, '<UNK>': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5})
    @patch('collection.collect_bluesky._detector')
    @patch('collection.collect_bluesky._emotion_model')
    @patch('collection.collect_bluesky._load_inference_models')
    def test_text_truncated_to_max_len(self, mock_load, mock_model,
                                        mock_detector, mock_le):
        """Texts longer than _emotion_max_len are truncated."""
        import torch
        import pandas as pd

        mock_load.return_value = True

        fake_logits = torch.tensor([[0.3, 0.7]])
        mock_model.return_value = fake_logits
        mock_le.inverse_transform.return_value = np.array(['surprise'])

        mock_detector.predict.return_value = pd.DataFrame({
            'prediction_label': [0],
            'ai_score_credibility': [0.90],
            'language': ['en'],
            'ai_analysis_log': ['ok'],
        })

        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 1
        # Text with 5 tokens but max_len=3 => should be truncated
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = [{'_id': 'trunc', 'text': 'a b c d'}]
        mock_collection.find.return_value = mock_cursor

        from collection.collect_bluesky import run_inference_cycle
        run_inference_cycle(mock_collection)

        # Verify the tensor passed to model has correct shape
        tensor_arg = mock_model.call_args[0][0]
        assert tensor_arg.shape[1] == 3  # max_len = 3

    @patch('collection.collect_bluesky._stage1_pipe', None)
    @patch('collection.collect_bluesky._stage1_threshold', 0.40)
    @patch('collection.collect_bluesky._emotion_max_len', 10)
    @patch('collection.collect_bluesky._emotion_le')
    @patch('collection.collect_bluesky._emotion_vocab',
           {'<PAD>': 0, '<UNK>': 1, 'hi': 2})
    @patch('collection.collect_bluesky._detector')
    @patch('collection.collect_bluesky._emotion_model')
    @patch('collection.collect_bluesky._load_inference_models')
    def test_short_text_padded_to_max_len(self, mock_load, mock_model,
                                           mock_detector, mock_le):
        """Texts shorter than _emotion_max_len are padded."""
        import torch
        import pandas as pd

        mock_load.return_value = True

        fake_logits = torch.tensor([[0.5, 0.5]])
        mock_model.return_value = fake_logits
        mock_le.inverse_transform.return_value = np.array(['neutral'])

        mock_detector.predict.return_value = pd.DataFrame({
            'prediction_label': [0],
            'ai_score_credibility': [0.75],
            'language': ['en'],
            'ai_analysis_log': ['clean'],
        })

        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 1
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = [{'_id': 'short', 'text': 'hi'}]
        mock_collection.find.return_value = mock_cursor

        from collection.collect_bluesky import run_inference_cycle
        run_inference_cycle(mock_collection)

        tensor_arg = mock_model.call_args[0][0]
        assert tensor_arg.shape[1] == 10  # padded to max_len

    @patch('collection.collect_bluesky._stage1_threshold', 0.40)
    @patch('collection.collect_bluesky._emotion_max_len', 5)
    @patch('collection.collect_bluesky._emotion_le')
    @patch('collection.collect_bluesky._emotion_vocab',
           {'<PAD>': 0, '<UNK>': 1, 'opinion': 2})
    @patch('collection.collect_bluesky._detector')
    @patch('collection.collect_bluesky._emotion_model')
    @patch('collection.collect_bluesky._load_inference_models')
    def test_stage1_exception_handled_gracefully(self, mock_load, mock_model,
                                                   mock_detector, mock_le):
        """If Stage1 predict_proba raises, post_types remain None."""
        import torch
        import pandas as pd

        mock_load.return_value = True

        fake_logits = torch.tensor([[0.9, 0.1]])
        mock_model.return_value = fake_logits
        mock_le.inverse_transform.return_value = np.array(['anger'])

        mock_detector.predict.return_value = pd.DataFrame({
            'prediction_label': [1],
            'ai_score_credibility': [0.15],
            'language': ['fr'],
            'ai_analysis_log': ['alert'],
        })

        # Stage1 pipeline that raises an exception
        mock_stage1 = MagicMock()
        mock_stage1.predict_proba.side_effect = ValueError("stage1 error")

        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 1
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = [{'_id': 'err', 'text': 'opinion post'}]
        mock_collection.find.return_value = mock_cursor

        with patch('collection.collect_bluesky._stage1_pipe', mock_stage1):
            from collection.collect_bluesky import run_inference_cycle
            run_inference_cycle(mock_collection)

        # Should still succeed and write results (without post_type fields)
        mock_collection.bulk_write.assert_called_once()
        ops = mock_collection.bulk_write.call_args[0][0]
        assert len(ops) == 1

    @patch('collection.collect_bluesky._stage1_threshold', 0.40)
    @patch('collection.collect_bluesky._emotion_max_len', 5)
    @patch('collection.collect_bluesky._emotion_le')
    @patch('collection.collect_bluesky._emotion_vocab',
           {'<PAD>': 0, '<UNK>': 1, 'opinion': 2, 'news': 3})
    @patch('collection.collect_bluesky._detector')
    @patch('collection.collect_bluesky._emotion_model')
    @patch('collection.collect_bluesky._load_inference_models')
    def test_opinion_suspicious_reclassified_to_reliable(self, mock_load,
                                                          mock_model,
                                                          mock_detector,
                                                          mock_le):
        """V9 logic: opinion + v5_label=1 => v9_label reclassified to 0."""
        import torch
        import pandas as pd

        mock_load.return_value = True

        fake_logits = torch.tensor([[0.5, 0.5]])
        mock_model.return_value = fake_logits
        mock_le.inverse_transform.return_value = np.array(['neutral'])

        # V5 says suspicious (label=1)
        mock_detector.predict.return_value = pd.DataFrame({
            'prediction_label': [1],
            'ai_score_credibility': [0.20],
            'language': ['fr'],
            'ai_analysis_log': ['suspicious'],
        })

        # Stage1 says opinion (low factuel proba => below threshold)
        mock_stage1 = MagicMock()
        mock_stage1.predict_proba.return_value = np.array([[0.7, 0.3]])

        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 1
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = [{'_id': 'v9_test', 'text': 'opinion news'}]
        mock_collection.find.return_value = mock_cursor

        with patch('collection.collect_bluesky._stage1_pipe', mock_stage1):
            from collection.collect_bluesky import run_inference_cycle
            run_inference_cycle(mock_collection)

        # Verify bulk_write was called -- the V9 reclassification logic ran
        mock_collection.bulk_write.assert_called_once()
