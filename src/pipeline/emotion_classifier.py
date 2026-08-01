"""
ThumaCheck — Emotion Classifier
================================

MLP-based emotion classification across 7 categories:
Anger, Disgust, Joy, Neutral, Fear, Surprise, Sadness.

Design choice: MLP over transformer for latency (<0.5ms/text),
keeping total pipeline under 1.5ms for real-time requirements.

Auteur : Niamato Consulting (pour Thumalien)
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

class _EmotionMLP(nn.Module):
    """Architecture MLP identique au notebook 02."""
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, 48)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(48, 24)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(24, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    Métriques de validation (notebook 02_Analyse_Emotions_MLP.ipynb) :
        - F1 macro global : 0.62 (objectif >= 0.60 — ATTEINT)
        - F1 par classe : joie 0.78, neutre 0.71, colere 0.65,
          tristesse 0.58, peur 0.54, surprise 0.48, degout 0.42
        - Train : 25 800 samples (EN 16K + FR 9.8K)
        - Test : 4 100 samples
        - Limites connues : degout et neutre peu fiables en anglais
          (donnees d'entrainement EN limitees pour ces classes)
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

    def get_emotion_features(self, texts: List[str]) -> np.ndarray:
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

    def explain_emotions(self, texts: List[str], n_background: int = 50) -> Optional[Dict]:
        """
        SHAP explainability pour le modele emotion.

        Utilise KernelExplainer (model-agnostic) pour expliquer pourquoi
        une emotion est predite.

        Parameters
        ----------
        texts : list of str — textes a expliquer
        n_background : int — nombre de samples background pour SHAP

        Returns
        -------
        Dict avec shap_values (n_texts, 7), feature_words, emotion_names
        ou None si SHAP non disponible
        """
        if not self._loaded:
            raise RuntimeError("Modele emotions non charge.")

        try:
            import shap
        except ImportError:
            logger.warning("shap non installe, explain_emotions indisponible")
            return None

        oov_idx = self.vocab.get('<OOV>', self.vocab.get('<UNK>', 1))

        def _texts_to_matrix(text_list):
            sequences = []
            for text in text_list:
                tokens = str(text).lower().split()
                seq = [self.vocab.get(t, oov_idx) for t in tokens[:self.MAX_LENGTH]]
                seq = seq + [0] * (self.MAX_LENGTH - len(seq))
                sequences.append(seq)
            return np.array(sequences, dtype=np.int64)

        def _predict_fn(X_np):
            X_t = torch.tensor(X_np, dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.model(X_t)
                return torch.softmax(logits, dim=1).cpu().numpy()

        # Build background from input texts (or subset)
        X_input = _texts_to_matrix(texts)
        bg_size = min(n_background, len(texts))
        background = X_input[:bg_size]

        explainer = shap.KernelExplainer(_predict_fn, background)
        shap_values = explainer.shap_values(X_input, nsamples=100)

        # shap_values is list of 7 arrays, each (n_texts, MAX_LENGTH)
        # Aggregate per-text: sum abs SHAP across token positions per emotion
        emotion_names = self.FEATURE_NAMES

        # Extract word tokens for each text for readability
        feature_words = []
        for text in texts:
            tokens = str(text).lower().split()[:self.MAX_LENGTH]
            tokens += ['[PAD]'] * (self.MAX_LENGTH - len(tokens))
            feature_words.append(tokens)

        return {
            'shap_values': shap_values,  # list of 7 arrays (n, MAX_LENGTH)
            'feature_words': feature_words,
            'emotion_names': emotion_names,
        }


