"""
Extended tests for pipeline/camembert_classifier.py -- CamemBERT components.

Targets uncovered lines to increase code coverage beyond the 29% baseline.
All heavy dependencies (transformers models, torch downloads) are mocked
so that tests run without GPU or network access.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

torch = pytest.importorskip("torch", reason="PyTorch required")

from unittest.mock import patch, MagicMock, PropertyMock
import torch.nn as nn

from pipeline.camembert_classifier import (
    CamemBERTHead,
    CamemBERTClassifier,
    TextDataset,
    TRANSFORMERS_AVAILABLE,
)


# -----------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------

def _make_mock_base_model(hidden_size=768):
    """Create a mock CamemBERT base model with realistic attributes."""
    mock_base = MagicMock()
    mock_base.config.hidden_size = hidden_size
    mock_base.to.return_value = mock_base

    # Simulate named_parameters with encoder layers and embeddings
    params = []
    for i in range(12):
        p = torch.randn(10, requires_grad=True)
        params.append((f'encoder.layer.{i}.attention.weight', p))
    emb_param = torch.randn(10, requires_grad=True)
    params.append(('embeddings.word_embeddings.weight', emb_param))

    mock_base.named_parameters.return_value = params
    # parameters() returns just the param tensors
    mock_base.parameters.return_value = [p for _, p in params]

    return mock_base


def _make_mock_model_output(batch_size, hidden_size=768):
    """Create a mock transformer output with last_hidden_state."""
    output = MagicMock()
    # Shape: (batch_size, seq_len, hidden_size) -- seq_len=128
    output.last_hidden_state = torch.randn(batch_size, 128, hidden_size)
    return output


def _build_loaded_classifier(tmp_path, hidden_size=768):
    """Build a CamemBERTClassifier with real head and mocked base model, marked as loaded."""
    clf = CamemBERTClassifier(model_dir=str(tmp_path))

    mock_tokenizer = MagicMock()
    # Tokenizer returns dict with tensors
    def _tokenize(text, **kwargs):
        max_length = kwargs.get('max_length', 128)
        return {
            'input_ids': torch.ones(1, max_length, dtype=torch.long),
            'attention_mask': torch.ones(1, max_length, dtype=torch.long),
        }
    mock_tokenizer.side_effect = _tokenize
    clf.tokenizer = mock_tokenizer

    # Use a real head so forward pass works
    clf.head = CamemBERTHead(hidden_size=hidden_size, num_classes=2)

    # Mock base_model that returns proper tensor outputs
    mock_base = MagicMock()
    mock_base.eval.return_value = None
    mock_base.train.return_value = None

    def _forward(**kwargs):
        batch_size = kwargs['input_ids'].shape[0]
        return _make_mock_model_output(batch_size, hidden_size)

    mock_base.side_effect = _forward
    mock_base.__call__ = _forward
    clf.base_model = mock_base

    clf.device = torch.device('cpu')
    clf._loaded = True
    return clf


# -----------------------------------------------------------------------
#  CamemBERTHead -- additional coverage
# -----------------------------------------------------------------------

class TestCamemBERTHeadExtended:

    def test_head_output_shape_batch_8(self):
        head = CamemBERTHead(hidden_size=768, num_classes=2)
        x = torch.randn(8, 768)
        out = head(x)
        assert out.shape == (8, 2)

    def test_head_different_hidden_sizes(self):
        for hs in [256, 512, 1024]:
            head = CamemBERTHead(hidden_size=hs, num_classes=2)
            x = torch.randn(3, hs)
            out = head(x)
            assert out.shape == (3, 2), f"Failed for hidden_size={hs}"

    def test_head_five_classes(self):
        head = CamemBERTHead(hidden_size=768, num_classes=5)
        x = torch.randn(4, 768)
        out = head(x)
        assert out.shape == (4, 5)

    def test_head_parameters_count(self):
        head = CamemBERTHead(hidden_size=768, num_classes=2)
        # Linear(768,256) + bias + Linear(256,2) + bias = 768*256+256 + 256*2+2
        total = sum(p.numel() for p in head.parameters())
        expected = 768 * 256 + 256 + 256 * 2 + 2
        assert total == expected


# -----------------------------------------------------------------------
#  TextDataset -- with mock tokenizer
# -----------------------------------------------------------------------

class TestTextDatasetMocked:
    """Tests TextDataset using a mock tokenizer so no model download is needed."""

    @pytest.fixture
    def mock_tokenizer(self):
        tok = MagicMock()

        def _tokenize(text, **kwargs):
            ml = kwargs.get('max_length', 128)
            return {
                'input_ids': torch.ones(1, ml, dtype=torch.long),
                'attention_mask': torch.ones(1, ml, dtype=torch.long),
            }
        tok.side_effect = _tokenize
        return tok

    def test_dataset_len(self, mock_tokenizer):
        ds = TextDataset(["a", "b"], [0, 1], mock_tokenizer)
        assert len(ds) == 2

    def test_dataset_getitem_keys(self, mock_tokenizer):
        ds = TextDataset(["hello"], [0], mock_tokenizer, max_length=64)
        item = ds[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'label' in item
        assert 'weight' not in item

    def test_dataset_with_sample_weights(self, mock_tokenizer):
        ds = TextDataset(
            ["hello", "world"], [0, 1], mock_tokenizer,
            max_length=64, sample_weights=[0.5, 1.5],
        )
        item = ds[0]
        assert 'weight' in item
        assert item['weight'].item() == pytest.approx(0.5)

        item1 = ds[1]
        assert item1['weight'].item() == pytest.approx(1.5)

    def test_dataset_label_dtype(self, mock_tokenizer):
        ds = TextDataset(["text"], [1], mock_tokenizer)
        item = ds[0]
        assert item['label'].dtype == torch.long

    def test_dataset_input_ids_shape(self, mock_tokenizer):
        ds = TextDataset(["text"], [0], mock_tokenizer, max_length=32)
        item = ds[0]
        assert item['input_ids'].shape == (32,)
        assert item['attention_mask'].shape == (32,)


# -----------------------------------------------------------------------
#  CamemBERTClassifier -- init
# -----------------------------------------------------------------------

class TestCamemBERTClassifierInit:

    def test_init_defaults(self):
        clf = CamemBERTClassifier()
        assert clf.model_dir == 'models'
        assert clf.tokenizer is None
        assert clf.base_model is None
        assert clf.head is None
        assert clf._loaded is False
        assert clf.training_metrics == {}

    def test_init_custom_model_dir(self):
        clf = CamemBERTClassifier(model_dir='/tmp/custom_models')
        assert clf.model_dir == '/tmp/custom_models'

    def test_device_selection(self):
        clf = CamemBERTClassifier()
        assert clf.device.type in ('cpu', 'mps', 'cuda')

    def test_max_length_constant(self):
        assert CamemBERTClassifier.MAX_LENGTH == 128

    def test_model_name_constant(self):
        assert CamemBERTClassifier.MODEL_NAME == 'camembert-base'


# -----------------------------------------------------------------------
#  CamemBERTClassifier._init_model -- mocked
# -----------------------------------------------------------------------

class TestInitModel:

    @patch('pipeline.camembert_classifier.AutoModel')
    @patch('pipeline.camembert_classifier.AutoTokenizer')
    def test_init_model_creates_components(self, mock_tok_cls, mock_model_cls):
        mock_base = _make_mock_base_model(hidden_size=768)
        mock_model_cls.from_pretrained.return_value = mock_base
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        clf = CamemBERTClassifier()
        clf.device = torch.device('cpu')
        clf._init_model()

        assert clf.tokenizer is not None
        assert clf.head is not None
        assert clf.base_model is not None
        mock_tok_cls.from_pretrained.assert_called_once_with('camembert-base')
        mock_model_cls.from_pretrained.assert_called_once_with('camembert-base')

    @patch('pipeline.camembert_classifier.AutoModel')
    @patch('pipeline.camembert_classifier.AutoTokenizer')
    def test_init_model_freezes_low_layers(self, mock_tok_cls, mock_model_cls):
        mock_base = _make_mock_base_model(hidden_size=768)
        mock_model_cls.from_pretrained.return_value = mock_base
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        clf = CamemBERTClassifier()
        clf.device = torch.device('cpu')
        clf._init_model()

        # Layers 0-8 and embeddings should have requires_grad=False
        for name, param in mock_base.named_parameters():
            if 'encoder.layer' in name:
                layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                if layer_num < 9:
                    assert param.requires_grad is False, f"Layer {layer_num} should be frozen"
                else:
                    assert param.requires_grad is True, f"Layer {layer_num} should be trainable"
            elif 'embeddings' in name:
                assert param.requires_grad is False, "Embeddings should be frozen"

    @patch('pipeline.camembert_classifier.AutoModel')
    @patch('pipeline.camembert_classifier.AutoTokenizer')
    def test_init_model_head_hidden_size(self, mock_tok_cls, mock_model_cls):
        mock_base = _make_mock_base_model(hidden_size=512)
        mock_model_cls.from_pretrained.return_value = mock_base
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        clf = CamemBERTClassifier()
        clf.device = torch.device('cpu')
        clf._init_model()

        # Head's first linear layer should accept 512-dim input
        first_linear = clf.head.classifier[0]
        assert first_linear.in_features == 512

    @patch('pipeline.camembert_classifier.TRANSFORMERS_AVAILABLE', False)
    def test_init_model_no_transformers_raises(self):
        clf = CamemBERTClassifier()
        with pytest.raises(RuntimeError, match="transformers non installe"):
            clf._init_model()


# -----------------------------------------------------------------------
#  predict() and predict_credibility_scores()
# -----------------------------------------------------------------------

class TestPredict:

    def test_predict_not_loaded_raises(self):
        clf = CamemBERTClassifier()
        with pytest.raises(RuntimeError, match="non charge"):
            clf.predict(["test"])

    def test_predict_returns_correct_keys(self, tmp_path):
        clf = _build_loaded_classifier(tmp_path)
        result = clf.predict(["Ceci est un test"])
        assert 'predictions' in result
        assert 'probabilities' in result
        assert 'labels' in result

    def test_predict_output_types(self, tmp_path):
        clf = _build_loaded_classifier(tmp_path)
        result = clf.predict(["texte un", "texte deux"])
        assert isinstance(result['predictions'], np.ndarray)
        assert isinstance(result['probabilities'], np.ndarray)
        assert isinstance(result['labels'], list)

    def test_predict_output_lengths(self, tmp_path):
        clf = _build_loaded_classifier(tmp_path)
        texts = ["a", "b", "c"]
        result = clf.predict(texts)
        assert len(result['predictions']) == 3
        assert len(result['probabilities']) == 3
        assert len(result['labels']) == 3

    def test_predict_labels_values(self, tmp_path):
        clf = _build_loaded_classifier(tmp_path)
        result = clf.predict(["test"])
        # Labels should be FIABLE or SUSPECT
        for label in result['labels']:
            assert label in ('FIABLE', 'SUSPECT')

    def test_predict_predictions_binary(self, tmp_path):
        clf = _build_loaded_classifier(tmp_path)
        result = clf.predict(["test"])
        for p in result['predictions']:
            assert p in (0, 1)

    def test_predict_credibility_scores(self, tmp_path):
        clf = _build_loaded_classifier(tmp_path)
        scores = clf.predict_credibility_scores(["texte un", "texte deux"])
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2

    def test_predict_credibility_scores_not_loaded_raises(self):
        clf = CamemBERTClassifier()
        with pytest.raises(RuntimeError, match="non charge"):
            clf.predict_credibility_scores(["test"])


# -----------------------------------------------------------------------
#  _evaluate() -- mocked
# -----------------------------------------------------------------------

class TestEvaluate:

    def test_evaluate_returns_metrics(self, tmp_path):
        clf = _build_loaded_classifier(tmp_path)

        # Build a small DataLoader with mock tokenizer
        ds = TextDataset(
            ["text1", "text2", "text3", "text4"],
            [0, 1, 0, 1],
            clf.tokenizer,
            max_length=128,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=2)

        result = clf._evaluate(loader)

        assert 'accuracy' in result
        assert 'f1' in result
        assert 'precision' in result
        assert 'recall' in result

    def test_evaluate_metric_ranges(self, tmp_path):
        clf = _build_loaded_classifier(tmp_path)
        ds = TextDataset(
            ["a", "b", "c", "d"],
            [0, 1, 0, 1],
            clf.tokenizer,
            max_length=128,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        result = clf._evaluate(loader)

        for key in ('accuracy', 'f1', 'precision', 'recall'):
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"


# -----------------------------------------------------------------------
#  _save_checkpoint / _load_checkpoint
# -----------------------------------------------------------------------

class TestCheckpoints:

    def test_save_and_load_checkpoint(self, tmp_path):
        clf = CamemBERTClassifier(model_dir=str(tmp_path))
        clf.device = torch.device('cpu')

        # Create minimal real models
        clf.base_model = nn.Linear(10, 10)
        clf.head = CamemBERTHead(hidden_size=768, num_classes=2)

        clf._save_checkpoint('test_ckpt')

        expected_path = os.path.join(str(tmp_path), 'camembert_test_ckpt.pt')
        assert os.path.exists(expected_path)

        # Modify head weights, then reload
        with torch.no_grad():
            for p in clf.head.parameters():
                p.fill_(999.0)

        result = clf._load_checkpoint('test_ckpt')
        assert result is True

        # Weights should be restored (not all 999.0)
        first_param = next(clf.head.parameters())
        assert not torch.all(first_param == 999.0)

    def test_load_checkpoint_missing_file(self, tmp_path):
        clf = CamemBERTClassifier(model_dir=str(tmp_path))
        clf.device = torch.device('cpu')
        result = clf._load_checkpoint('nonexistent')
        assert result is False

    def test_save_checkpoint_default_name(self, tmp_path):
        clf = CamemBERTClassifier(model_dir=str(tmp_path))
        clf.base_model = nn.Linear(10, 10)
        clf.head = CamemBERTHead(hidden_size=768, num_classes=2)

        clf._save_checkpoint()  # default name='best'
        assert os.path.exists(os.path.join(str(tmp_path), 'camembert_best.pt'))


# -----------------------------------------------------------------------
#  save() and load() -- full pipeline with mocks
# -----------------------------------------------------------------------

class TestSaveLoad:

    def test_save_creates_file(self, tmp_path):
        clf = CamemBERTClassifier(model_dir=str(tmp_path))
        clf.device = torch.device('cpu')
        clf.training_metrics = {'best_val_f1': 0.95}

        # Use real small models
        mock_base = MagicMock()
        mock_base.state_dict.return_value = {'dummy': torch.tensor([1.0])}
        mock_base.config.hidden_size = 768
        clf.base_model = mock_base
        clf.head = CamemBERTHead(hidden_size=768, num_classes=2)

        clf.save(suffix='test_model')
        expected_path = os.path.join(str(tmp_path), 'test_model.pt')
        assert os.path.exists(expected_path)

    def test_save_file_content(self, tmp_path):
        clf = CamemBERTClassifier(model_dir=str(tmp_path))
        clf.device = torch.device('cpu')
        clf.training_metrics = {'best_val_f1': 0.92}

        mock_base = MagicMock()
        mock_base.state_dict.return_value = {'layer.weight': torch.tensor([1.0, 2.0])}
        mock_base.config.hidden_size = 768
        clf.base_model = mock_base
        clf.head = CamemBERTHead(hidden_size=768, num_classes=2)

        clf.save(suffix='content_test')
        path = os.path.join(str(tmp_path), 'content_test.pt')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)

        assert 'base_model_state' in checkpoint
        assert 'head_state' in checkpoint
        assert 'config' in checkpoint
        assert 'metrics' in checkpoint
        assert checkpoint['config']['model_name'] == 'camembert-base'
        assert checkpoint['config']['max_length'] == 128
        assert checkpoint['config']['hidden_size'] == 768

    def test_load_missing_file_returns_false(self, tmp_path):
        clf = CamemBERTClassifier(model_dir=str(tmp_path))
        if TRANSFORMERS_AVAILABLE:
            result = clf.load(suffix='nonexistent')
            assert result is False

    @patch('pipeline.camembert_classifier.TRANSFORMERS_AVAILABLE', False)
    def test_load_no_transformers_returns_false(self, tmp_path):
        clf = CamemBERTClassifier(model_dir=str(tmp_path))
        result = clf.load(suffix='anything')
        assert result is False

    @patch('pipeline.camembert_classifier.AutoModel')
    @patch('pipeline.camembert_classifier.AutoTokenizer')
    def test_load_full_roundtrip(self, mock_tok_cls, mock_model_cls, tmp_path):
        """Save then load -- verify _loaded is True after load."""
        clf = CamemBERTClassifier(model_dir=str(tmp_path))
        clf.device = torch.device('cpu')
        clf.training_metrics = {'best_val_f1': 0.88}

        # Build real head + mock base for save
        head = CamemBERTHead(hidden_size=768, num_classes=2)
        clf.head = head

        mock_base = MagicMock()
        mock_base.state_dict.return_value = {'w': torch.tensor([1.0])}
        mock_base.config.hidden_size = 768
        clf.base_model = mock_base

        clf.save(suffix='roundtrip')

        # Now load into a fresh classifier
        # Mock the transformers calls during load
        mock_base_loaded = MagicMock()
        mock_base_loaded.to.return_value = mock_base_loaded
        mock_base_loaded.eval.return_value = None
        mock_base_loaded.load_state_dict.return_value = None
        mock_model_cls.from_pretrained.return_value = mock_base_loaded
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        clf2 = CamemBERTClassifier(model_dir=str(tmp_path))
        clf2.device = torch.device('cpu')

        # Patch torch.load to return compatible state dict for head
        path = os.path.join(str(tmp_path), 'roundtrip.pt')
        result = clf2.load(suffix='roundtrip')

        assert result is True
        assert clf2._loaded is True
        assert clf2.tokenizer is not None
        assert clf2.head is not None


# -----------------------------------------------------------------------
#  fine_tune() -- heavily mocked to avoid real training
# -----------------------------------------------------------------------

class TestFineTune:

    @patch('pipeline.camembert_classifier.AutoModel')
    @patch('pipeline.camembert_classifier.AutoTokenizer')
    def test_fine_tune_runs(self, mock_tok_cls, mock_model_cls, tmp_path):
        """Verify fine_tune completes and returns metrics dict."""
        import pandas as pd

        # Setup mock tokenizer
        mock_tok = MagicMock()
        def _tokenize(text, **kwargs):
            ml = kwargs.get('max_length', 128)
            return {
                'input_ids': torch.ones(1, ml, dtype=torch.long),
                'attention_mask': torch.ones(1, ml, dtype=torch.long),
            }
        mock_tok.side_effect = _tokenize
        mock_tok_cls.from_pretrained.return_value = mock_tok

        # Setup mock base model
        hidden_size = 768
        real_linear = nn.Linear(hidden_size, hidden_size)  # for real gradient flow
        mock_base = MagicMock()
        mock_base.config.hidden_size = hidden_size
        mock_base.to.return_value = mock_base
        mock_base.named_parameters.return_value = []
        mock_base.parameters.return_value = list(real_linear.parameters())
        mock_base.train.return_value = None
        mock_base.eval.return_value = None
        mock_base.state_dict.return_value = {'w': torch.tensor([1.0])}

        def _forward(*args, **kwargs):
            bs = kwargs.get('input_ids', args[0] if args else None).shape[0]
            out = MagicMock()
            # Return tensor that supports autograd
            out.last_hidden_state = torch.randn(bs, 128, hidden_size, requires_grad=True)
            return out
        mock_base.__call__ = _forward
        mock_base.side_effect = _forward
        mock_model_cls.from_pretrained.return_value = mock_base

        # Build minimal DataFrame -- need at least 20 samples for stratified split
        n = 40
        texts = [f"texte numero {i}" for i in range(n)]
        labels = [0] * 20 + [1] * 20
        df = pd.DataFrame({'text_original': texts, 'label': labels})

        clf = CamemBERTClassifier(model_dir=str(tmp_path))
        clf.device = torch.device('cpu')

        metrics = clf.fine_tune(
            df, epochs=1, batch_size=8, lr=1e-4,
            short_text_weight=2.0, track_emissions=False,
        )

        assert isinstance(metrics, dict)
        assert 'epochs' in metrics
        assert 'best_val_f1' in metrics
        assert 'history' in metrics
        assert 'n_train' in metrics
        assert 'n_val' in metrics
        assert clf._loaded is True
