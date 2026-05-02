"""Tests pour _EmotionMLP et EmotionFeatureExtractor."""

import sys
import os
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pipeline.expert_detector import _EmotionMLP, EmotionFeatureExtractor


class TestEmotionMLP:
    def test_forward_shape(self):
        model = _EmotionMLP(vocab_size=100, embed_dim=16, num_classes=7)
        x = torch.randint(0, 100, (4, 20))
        output = model(x)
        assert output.shape == (4, 7)

    def test_output_not_all_zeros(self):
        model = _EmotionMLP(vocab_size=100, embed_dim=16, num_classes=7)
        x = torch.randint(1, 100, (2, 20))
        output = model(x)
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_softmax_sums_to_one(self):
        model = _EmotionMLP(vocab_size=100, embed_dim=16, num_classes=7)
        model.eval()
        x = torch.randint(1, 100, (3, 20))
        with torch.no_grad():
            output = model(x)
            probas = torch.softmax(output, dim=1)
        sums = probas.sum(dim=1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-5)

    def test_padding_idx_zero(self):
        model = _EmotionMLP(vocab_size=100, embed_dim=16, num_classes=7)
        # Padding tokens (idx=0) should have zero embedding
        embed_weight = model.embedding.weight.data[0]
        assert torch.allclose(embed_weight, torch.zeros_like(embed_weight))

    def test_different_vocab_sizes(self):
        for vs in [500, 10000, 25000]:
            model = _EmotionMLP(vocab_size=vs, embed_dim=32, num_classes=7)
            x = torch.randint(0, vs, (2, 10))
            output = model(x)
            assert output.shape == (2, 7)


class TestEmotionFeatureExtractor:
    def test_init_defaults(self):
        efe = EmotionFeatureExtractor(model_dir='/nonexistent')
        assert efe.MAX_LENGTH == 100
        assert efe.NUM_CLASSES == 7
        assert efe._loaded is False

    def test_load_missing_files(self):
        efe = EmotionFeatureExtractor(model_dir='/nonexistent')
        result = efe.load()
        assert result is False
        assert efe._loaded is False

    def test_get_emotion_features_raises_if_not_loaded(self):
        efe = EmotionFeatureExtractor(model_dir='/nonexistent')
        with pytest.raises(RuntimeError, match="non chargé"):
            efe.get_emotion_features(["test"])

    def test_feature_names(self):
        assert len(EmotionFeatureExtractor.FEATURE_NAMES) == 7
        assert 'emo_colere' in EmotionFeatureExtractor.FEATURE_NAMES
        assert 'emo_joie' in EmotionFeatureExtractor.FEATURE_NAMES
