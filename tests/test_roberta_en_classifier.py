"""
Tests du classifieur RoBERTa EN.

Ce module n'avait aucun test (23 % de couverture). Les poids `roberta_en.pt`
pesant plus de 100 Mo ne sont pas versionnes, ces tests couvrent donc la
surface testable sans modele : dataset, tete de classification, gestion du
mode degrade et garde-fous d'etat.
"""

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline.roberta_en_classifier import (
    RoBERTaENClassifier,
    RoBERTaHead,
    TextDataset,
)


class _FakeTokenizer:
    """Tokenizer minimal : renvoie la forme attendue par TextDataset."""

    def __call__(self, text, truncation=None, padding=None, max_length=128, return_tensors=None):
        return {
            "input_ids": torch.ones((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


class TestTextDataset:
    def test_len_matches_texts(self):
        ds = TextDataset(["a", "b", "c"], [0, 1, 0], _FakeTokenizer())
        assert len(ds) == 3

    def test_getitem_returns_expected_keys(self):
        ds = TextDataset(["texte"], [1], _FakeTokenizer())
        item = ds[0]
        assert set(item) == {"input_ids", "attention_mask", "label"}

    def test_label_is_long_tensor(self):
        ds = TextDataset(["texte"], [1], _FakeTokenizer())
        assert ds[0]["label"].dtype == torch.long
        assert int(ds[0]["label"]) == 1

    def test_sample_weights_add_weight_key(self):
        ds = TextDataset(["a"], [0], _FakeTokenizer(), sample_weights=[2.5])
        item = ds[0]
        assert "weight" in item
        assert pytest.approx(float(item["weight"]), abs=1e-6) == 2.5

    def test_no_weights_means_no_weight_key(self):
        ds = TextDataset(["a"], [0], _FakeTokenizer())
        assert "weight" not in ds[0]

    def test_non_string_text_is_coerced(self):
        """__getitem__ applique str() : un entier ne doit pas lever."""
        ds = TextDataset([12345], [0], _FakeTokenizer())
        assert ds[0]["input_ids"].shape[0] == 128

    def test_max_length_is_honoured(self):
        ds = TextDataset(["a"], [0], _FakeTokenizer(), max_length=64)
        assert ds[0]["input_ids"].shape[0] == 64


class TestRoBERTaHead:
    def test_output_shape_matches_num_classes(self):
        head = RoBERTaHead(hidden_size=32, num_classes=2)
        out = head(torch.randn(4, 32))
        assert out.shape == (4, 2)

    def test_custom_num_classes(self):
        head = RoBERTaHead(hidden_size=16, num_classes=5)
        assert head(torch.randn(2, 16)).shape == (2, 5)

    def test_is_deterministic_in_eval_mode(self):
        head = RoBERTaHead(hidden_size=16)
        head.eval()
        x = torch.randn(3, 16)
        with torch.no_grad():
            assert torch.allclose(head(x), head(x))

    def test_dropout_makes_train_mode_stochastic(self):
        head = RoBERTaHead(hidden_size=64, dropout=0.9)
        head.train()
        x = torch.randn(8, 64)
        assert not torch.allclose(head(x), head(x))


class TestClassifierState:
    def test_init_starts_unloaded(self):
        clf = RoBERTaENClassifier(model_dir="/nonexistent")
        assert clf._loaded is False
        assert clf.tokenizer is None
        assert clf.base_model is None
        assert clf.head is None

    def test_model_dir_is_stored(self):
        assert RoBERTaENClassifier(model_dir="/tmp/modeles").model_dir == "/tmp/modeles"

    def test_predict_before_load_raises(self):
        clf = RoBERTaENClassifier(model_dir="/nonexistent")
        with pytest.raises(RuntimeError, match="non charge"):
            clf.predict(["hello"])

    def test_credibility_scores_before_load_raises(self):
        clf = RoBERTaENClassifier(model_dir="/nonexistent")
        with pytest.raises(RuntimeError):
            clf.predict_credibility_scores(["hello"])

    def test_device_is_a_torch_device(self):
        assert isinstance(RoBERTaENClassifier().device, torch.device)


class TestDegradedMode:
    """Poids absents : load() doit rendre False, jamais lever."""

    def test_load_returns_false_when_file_missing(self, tmp_path):
        clf = RoBERTaENClassifier(model_dir=str(tmp_path))
        assert clf.load() is False

    def test_load_leaves_classifier_unloaded(self, tmp_path):
        clf = RoBERTaENClassifier(model_dir=str(tmp_path))
        clf.load()
        assert clf._loaded is False

    def test_load_honours_custom_suffix(self, tmp_path):
        clf = RoBERTaENClassifier(model_dir=str(tmp_path))
        assert clf.load(suffix="inexistant_v42") is False

    def test_predict_credibility_delegates_to_predict(self):
        """predict_credibility_scores extrait la cle 'probabilities'."""
        clf = RoBERTaENClassifier()
        clf.predict = MagicMock(return_value={"probabilities": np.array([0.1, 0.9])})
        scores = clf.predict_credibility_scores(["a", "b"])
        clf.predict.assert_called_once_with(["a", "b"])
        assert np.allclose(scores, [0.1, 0.9])
