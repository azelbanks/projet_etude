"""
Tests for pipeline/camembert_classifier.py — CamemBERT components.

Tests the TextDataset, CamemBERTHead neural network architecture,
and CamemBERTClassifier initialization without requiring a GPU
or a pre-trained model on disk.
"""

import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

torch = pytest.importorskip("torch", reason="PyTorch required")

from pipeline.camembert_classifier import (
    CamemBERTHead,
    CamemBERTClassifier,
    TRANSFORMERS_AVAILABLE,
)


# -----------------------------------------------------------------------
#  CamemBERTHead
# -----------------------------------------------------------------------

class TestCamemBERTHead:
    """Tests for the CamemBERTHead classification module."""

    def test_output_shape(self):
        head = CamemBERTHead(hidden_size=768, num_classes=2)
        x = torch.randn(4, 768)  # batch=4, hidden=768
        out = head(x)
        assert out.shape == (4, 2)

    def test_custom_hidden_size(self):
        head = CamemBERTHead(hidden_size=512, num_classes=3)
        x = torch.randn(2, 512)
        out = head(x)
        assert out.shape == (2, 3)

    def test_single_sample(self):
        head = CamemBERTHead(hidden_size=768, num_classes=2)
        x = torch.randn(1, 768)
        out = head(x)
        assert out.shape == (1, 2)

    def test_output_is_differentiable(self):
        head = CamemBERTHead(hidden_size=768, num_classes=2)
        x = torch.randn(2, 768, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# -----------------------------------------------------------------------
#  CamemBERTClassifier
# -----------------------------------------------------------------------

class TestCamemBERTClassifier:
    """Tests for CamemBERTClassifier initialization and error handling."""

    def test_init_default_values(self):
        clf = CamemBERTClassifier(model_dir='/tmp/test_models')
        assert clf.model_dir == '/tmp/test_models'
        assert clf._loaded is False
        assert clf.tokenizer is None

    def test_predict_raises_when_not_loaded(self):
        clf = CamemBERTClassifier()
        with pytest.raises(RuntimeError, match="non charge"):
            clf.predict(["test text"])

    def test_load_returns_false_when_file_missing(self):
        clf = CamemBERTClassifier(model_dir='/tmp/nonexistent_dir')
        if TRANSFORMERS_AVAILABLE:
            result = clf.load(suffix='nonexistent_model')
            assert result is False

    def test_max_length_constant(self):
        assert CamemBERTClassifier.MAX_LENGTH == 128

    def test_model_name_constant(self):
        assert CamemBERTClassifier.MODEL_NAME == 'camembert-base'


# -----------------------------------------------------------------------
#  TextDataset (requires transformers)
# -----------------------------------------------------------------------

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestTextDataset:
    """Tests for the TextDataset PyTorch Dataset."""

    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained('camembert-base')

    def test_dataset_length(self, tokenizer):
        from pipeline.camembert_classifier import TextDataset
        ds = TextDataset(["text1", "text2", "text3"], [0, 1, 0], tokenizer)
        assert len(ds) == 3

    def test_dataset_item_keys(self, tokenizer):
        from pipeline.camembert_classifier import TextDataset
        ds = TextDataset(["Ceci est un test"], [1], tokenizer)
        item = ds[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'label' in item

    def test_dataset_item_shapes(self, tokenizer):
        from pipeline.camembert_classifier import TextDataset
        ds = TextDataset(["Ceci est un test"], [1], tokenizer, max_length=64)
        item = ds[0]
        assert item['input_ids'].shape == (64,)
        assert item['attention_mask'].shape == (64,)
        assert item['label'].shape == ()

    def test_label_value(self, tokenizer):
        from pipeline.camembert_classifier import TextDataset
        ds = TextDataset(["text"], [1], tokenizer)
        item = ds[0]
        assert item['label'].item() == 1
