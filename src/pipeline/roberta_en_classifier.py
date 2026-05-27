"""
ThumaCheck — RoBERTa Fine-tuned Classifier pour textes courts EN
================================================================

Module complementaire au pipeline TF-IDF+LogReg pour ameliorer la
detection de fake news sur les textes courts en anglais (< 30 mots).

Architecture :
    RoBERTa-base -> Linear(768, 256) -> ReLU -> Dropout -> Linear(256, 2)

Le modele est fine-tune sur les donnees EN du dataset Thumalien (client)
avec un focus sur les textes courts (< 30 mots).

Usage :
    classifier = RoBERTaENClassifier(model_dir='models')
    classifier.fine_tune(df_train_en)  # DataFrame avec text_original, label
    classifier.save()

    # Inference
    classifier.load()
    results = classifier.predict(["EXPOSED: Secret mind control program!!!"])

Auteur : Niamato Consulting (pour Thumalien)
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers non disponible. RoBERTa EN desactive.")

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


# ================================================================
#  Dataset PyTorch
# ================================================================

class TextDataset(Dataset):
    """Dataset PyTorch pour le fine-tuning RoBERTa."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128,
                 sample_weights: Optional[List[float]] = None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt',
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.sample_weights is not None:
            item['weight'] = torch.tensor(self.sample_weights[idx], dtype=torch.float)
        return item


# ================================================================
#  Classification head
# ================================================================

class RoBERTaHead(nn.Module):
    """Classification head pour RoBERTa."""
    def __init__(self, hidden_size: int = 768, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


# ================================================================
#  RoBERTa EN Classifier
# ================================================================

class RoBERTaENClassifier:
    """
    Fine-tuned RoBERTa pour detection de fake news EN courtes.

    Strategies d'entrainement :
    - Freeze des couches basses de RoBERTa (layers 0-8)
    - Fine-tune uniquement les couches hautes (9-11) + head
    - Focus sur textes courts (< 30 mots) avec surpoids
    - max_length=128 tokens (suffisant pour textes courts)
    """

    MODEL_NAME = 'roberta-base'
    MAX_LENGTH = 128

    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.tokenizer = None
        self.base_model = None
        self.head = None
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self._loaded = False
        self.training_metrics: Dict = {}

    def _init_model(self):
        """Initialise RoBERTa + classification head."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers non installe. pip install transformers")

        logger.info("Chargement de %s...", self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.base_model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.head = RoBERTaHead(
            hidden_size=self.base_model.config.hidden_size,
            num_classes=2,
        ).to(self.device)

        # Freeze couches basses (0-8) — ne fine-tune que les couches hautes
        for name, param in self.base_model.named_parameters():
            if 'encoder.layer' in name:
                layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                if layer_num < 9:
                    param.requires_grad = False
            elif 'embeddings' in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.base_model.parameters())
        logger.info(
            "RoBERTa: %d/%d parametres entrainables (%.1f%%)",
            trainable, total, 100 * trainable / total,
        )

    def fine_tune(
        self,
        df: pd.DataFrame,
        epochs: int = 3,
        batch_size: int = 32,
        lr: float = 2e-5,
        short_text_weight: float = 2.0,
        track_emissions: bool = True,
    ) -> Dict:
        """
        Fine-tune RoBERTa sur les donnees EN.

        Parameters
        ----------
        df : DataFrame avec colonnes 'text_original' et 'label'
        epochs : Nombre d'epochs
        batch_size : Taille de batch
        lr : Learning rate
        short_text_weight : Poids supplementaire pour les textes courts (< 30 mots)
        track_emissions : Tracking CodeCarbon
        """
        tracker = None
        if track_emissions and CODECARBON_AVAILABLE:
            tracker = EmissionsTracker(
                project_name='ThumaCheck_RoBERTa_EN',
                output_dir=self.model_dir,
                output_file='roberta_en_emissions.csv',
                log_level='error',
            )
            tracker.start()

        self._init_model()

        texts = df['text_original'].tolist()
        labels = df['label'].tolist()

        # Surpoids pour les textes courts
        weights = []
        for t in texts:
            n_words = len(str(t).split())
            weights.append(short_text_weight if n_words < 30 else 1.0)

        # Train/val split (80/20)
        from sklearn.model_selection import train_test_split
        idx_train, idx_val = train_test_split(
            range(len(texts)), test_size=0.2, stratify=labels, random_state=42,
        )

        train_ds = TextDataset(
            [texts[i] for i in idx_train],
            [labels[i] for i in idx_train],
            self.tokenizer, self.MAX_LENGTH,
            sample_weights=[weights[i] for i in idx_train],
        )
        val_ds = TextDataset(
            [texts[i] for i in idx_val],
            [labels[i] for i in idx_val],
            self.tokenizer, self.MAX_LENGTH,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        optimizer = torch.optim.AdamW(
            [p for p in list(self.base_model.parameters()) + list(self.head.parameters()) if p.requires_grad],
            lr=lr, weight_decay=0.01,
        )
        criterion = nn.CrossEntropyLoss(reduction='none')

        best_val_f1 = 0.0
        history = []

        for epoch in range(epochs):
            self.base_model.train()
            self.head.train()
            total_loss = 0
            n_batches = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['label'].to(self.device)
                batch_weights = batch.get('weight', torch.ones(len(batch_labels))).to(self.device)

                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                cls_output = outputs.last_hidden_state[:, 0, :]
                logits = self.head(cls_output)

                loss_per_sample = criterion(logits, batch_labels)
                loss = (loss_per_sample * batch_weights).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.base_model.parameters()) + list(self.head.parameters()), 1.0,
                )
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            val_metrics = self._evaluate(val_loader)
            avg_loss = total_loss / max(n_batches, 1)
            logger.info(
                "RoBERTa EN Epoch %d/%d — loss=%.4f, val_f1=%.4f, val_acc=%.4f",
                epoch + 1, epochs, avg_loss, val_metrics['f1'], val_metrics['accuracy'],
            )

            history.append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                **{f'val_{k}': v for k, v in val_metrics.items()},
            })

            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self._save_checkpoint('best')

        self._load_checkpoint('best')
        self._loaded = True

        self.training_metrics = {
            'best_val_f1': best_val_f1,
            'history': history,
            'n_train': len(idx_train),
            'n_val': len(idx_val),
        }

        if tracker is not None:
            try:
                tracker.stop()
            except Exception:
                pass

        logger.info("RoBERTa EN fine-tune termine — best_val_f1=%.4f", best_val_f1)
        return self.training_metrics

    def _evaluate(self, loader: DataLoader) -> Dict:
        """Evaluation sur un DataLoader."""
        self.base_model.eval()
        self.head.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label']

                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                cls_output = outputs.last_hidden_state[:, 0, :]
                logits = self.head(cls_output)
                preds = logits.argmax(dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
        }

    def predict(self, texts: List[str]) -> Dict:
        """
        Prediction sur une liste de textes EN.

        Returns
        -------
        Dict avec 'predictions' (0/1), 'probabilities' (float), 'labels' (FIABLE/SUSPECT)
        """
        if not self._loaded:
            raise RuntimeError("Modele non charge. Appelez load() ou fine_tune() d'abord.")

        self.base_model.eval()
        self.head.eval()

        dataset = TextDataset(texts, [0] * len(texts), self.tokenizer, self.MAX_LENGTH)
        loader = DataLoader(dataset, batch_size=32)

        all_preds = []
        all_probas = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                cls_output = outputs.last_hidden_state[:, 0, :]
                logits = self.head(cls_output)

                probas = torch.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_probas.extend(probas[:, 0])  # P(fiable)

        return {
            'predictions': np.array(all_preds),
            'probabilities': np.array(all_probas),
            'labels': ['FIABLE' if p == 0 else 'SUSPECT' for p in all_preds],
        }

    def predict_credibility_scores(self, texts: List[str]) -> np.ndarray:
        """Retourne uniquement les scores de credibilite (0-1)."""
        result = self.predict(texts)
        return result['probabilities']

    def _save_checkpoint(self, name: str = 'best'):
        """Sauvegarde un checkpoint du modele."""
        path = os.path.join(self.model_dir, f'roberta_en_{name}.pt')
        torch.save({
            'base_model_state': self.base_model.state_dict(),
            'head_state': self.head.state_dict(),
        }, path)

    def _load_checkpoint(self, name: str = 'best'):
        """Charge un checkpoint."""
        path = os.path.join(self.model_dir, f'roberta_en_{name}.pt')
        if not os.path.exists(path):
            logger.warning("Checkpoint non trouve : %s", path)
            return False
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.base_model.load_state_dict(checkpoint['base_model_state'])
        self.head.load_state_dict(checkpoint['head_state'])
        return True

    def save(self, suffix: str = 'roberta_en'):
        """Sauvegarde le modele final."""
        base_path = os.path.join(self.model_dir, f'{suffix}.pt')
        torch.save({
            'base_model_state': self.base_model.state_dict(),
            'head_state': self.head.state_dict(),
            'config': {
                'model_name': self.MODEL_NAME,
                'max_length': self.MAX_LENGTH,
                'hidden_size': self.base_model.config.hidden_size,
            },
            'metrics': self.training_metrics,
        }, base_path)
        logger.info("RoBERTa EN sauvegarde : %s", base_path)

    def load(self, suffix: str = 'roberta_en') -> bool:
        """Charge le modele sauvegarde."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("transformers non disponible")
            return False

        path = os.path.join(self.model_dir, f'{suffix}.pt')
        if not os.path.exists(path):
            logger.warning("RoBERTa EN non trouve : %s", path)
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.base_model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)

        cfg = checkpoint.get('config') or {}
        hidden_size = cfg.get('hidden_size', self.base_model.config.hidden_size)
        self.head = RoBERTaHead(
            hidden_size=hidden_size,
            num_classes=2,
        ).to(self.device)

        self.base_model.load_state_dict(checkpoint['base_model_state'])
        self.head.load_state_dict(checkpoint['head_state'])
        self.base_model.eval()
        self.head.eval()
        self._loaded = True
        self.training_metrics = checkpoint.get('metrics', {})

        logger.info("RoBERTa EN charge : %s (F1=%.4f)", path, self.training_metrics.get('best_val_f1', 0))
        return True
