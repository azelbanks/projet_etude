"""
Thumalien — CamemBERT Fine-tuned Classifier pour textes courts FR
=================================================================

Module complementaire au pipeline TF-IDF+LogReg pour ameliorer la
detection de fake news sur les textes courts en francais (< 30 mots).

Architecture :
    CamemBERT-base -> Linear(768, 256) -> ReLU -> Dropout -> Linear(256, 2)

Le modele est fine-tune sur les donnees FR du dataset Thumalien
avec un focus sur les textes courts (< 30 mots).

Usage :
    classifier = CamemBERTClassifier(model_dir='models')
    classifier.fine_tune(df_train_fr)  # DataFrame avec text_original, label
    classifier.save()

    # Inference
    classifier.load()
    results = classifier.predict(["SCANDALE ! On nous ment !"])

Auteur : Thumalien Team
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
    logger.warning("transformers non disponible. CamemBERT desactive.")

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


# ================================================================
#  Dataset PyTorch
# ================================================================

class TextDataset(Dataset):
    """Dataset PyTorch pour le fine-tuning CamemBERT."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ================================================================
#  Classification Head
# ================================================================

class CamemBERTHead(nn.Module):
    """Classification head sur les embeddings CamemBERT [CLS]."""

    def __init__(self, hidden_size: int = 768, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


# ================================================================
#  CamemBERT Classifier
# ================================================================

class CamemBERTClassifier:
    """
    Fine-tuned CamemBERT pour detection de fake news FR courtes.

    Strategies d'entrainement :
    - Freeze des couches basses de CamemBERT (layers 0-8)
    - Fine-tune uniquement les couches hautes (9-11) + head
    - Focus sur textes courts (< 30 mots) avec surpoids
    - max_length=128 tokens (suffisant pour textes courts)
    """

    MODEL_NAME = 'camembert-base'
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
        """Initialise CamemBERT + classification head."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers non installe. pip install transformers")

        logger.info("Chargement de %s...", self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.base_model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.head = CamemBERTHead(
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
            "CamemBERT: %d/%d parametres entrainables (%.1f%%)",
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
        Fine-tune CamemBERT sur les donnees FR.

        Parameters
        ----------
        df : DataFrame avec colonnes 'text_original' et 'label'
        epochs : Nombre d'epochs
        batch_size : Taille de batch
        lr : Learning rate
        short_text_weight : Poids supplementaire pour les textes courts (< 30 mots)
        track_emissions : Tracking CodeCarbon

        Returns
        -------
        Dict avec metriques d'entrainement
        """
        tracker = None
        if track_emissions and CODECARBON_AVAILABLE:
            tracker = EmissionsTracker(
                project_name="Thumalien_CamemBERT",
                output_dir=os.path.dirname(self.model_dir) or '.',
            )
            tracker.start()

        try:
            self._init_model()

            texts = df['text_original'].astype(str).tolist()
            labels = df['label'].astype(int).tolist()

            # Poids par echantillon : surpoids pour textes courts
            word_counts = [len(t.split()) for t in texts]
            sample_weights = torch.tensor([
                short_text_weight if wc < 30 else 1.0
                for wc in word_counts
            ], dtype=torch.float32)

            # Split train/val (90/10)
            from sklearn.model_selection import train_test_split
            indices = list(range(len(texts)))
            train_idx, val_idx = train_test_split(
                indices, test_size=0.1, stratify=labels, random_state=42
            )

            train_dataset = TextDataset(
                [texts[i] for i in train_idx],
                [labels[i] for i in train_idx],
                self.tokenizer, self.MAX_LENGTH,
            )
            val_dataset = TextDataset(
                [texts[i] for i in val_idx],
                [labels[i] for i in val_idx],
                self.tokenizer, self.MAX_LENGTH,
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            train_weights = sample_weights[train_idx]

            # Optimizer (different LR pour base et head)
            optimizer = torch.optim.AdamW([
                {'params': [p for p in self.base_model.parameters() if p.requires_grad], 'lr': lr},
                {'params': self.head.parameters(), 'lr': lr * 10},
            ], weight_decay=0.01)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs * len(train_loader),
            )

            criterion = nn.CrossEntropyLoss()
            best_val_f1 = 0.0
            history = []

            for epoch in range(epochs):
                # --- Train ---
                self.base_model.train()
                self.head.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_idx, batch in enumerate(train_loader):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    batch_labels = batch['label'].to(self.device)

                    optimizer.zero_grad()

                    outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                    cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                    logits = self.head(cls_output)

                    # Weighted loss
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, len(train_weights))
                    if batch_end > batch_start and batch_end <= len(train_weights):
                        w = train_weights[batch_start:batch_end].to(self.device)
                        loss_per_sample = nn.functional.cross_entropy(
                            logits, batch_labels, reduction='none'
                        )
                        w_adjusted = w[:len(loss_per_sample)]
                        loss = (loss_per_sample * w_adjusted).mean()
                    else:
                        loss = criterion(logits, batch_labels)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.base_model.parameters()) + list(self.head.parameters()),
                        max_norm=1.0,
                    )
                    optimizer.step()
                    scheduler.step()

                    train_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    train_correct += (preds == batch_labels).sum().item()
                    train_total += len(batch_labels)

                    if (batch_idx + 1) % 50 == 0:
                        logger.info(
                            "Epoch %d/%d | Batch %d/%d | Loss: %.4f",
                            epoch + 1, epochs, batch_idx + 1, len(train_loader), loss.item(),
                        )

                # --- Validation ---
                val_metrics = self._evaluate(val_loader)

                epoch_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss / len(train_loader),
                    'train_accuracy': train_correct / train_total,
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                }
                history.append(epoch_metrics)

                print(
                    f"  Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {epoch_metrics['train_loss']:.4f} | "
                    f"Train Acc: {epoch_metrics['train_accuracy']:.4f} | "
                    f"Val F1: {val_metrics['f1']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

                # Save best model
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    self._save_checkpoint('best')

            self.training_metrics = {
                'epochs': epochs,
                'best_val_f1': best_val_f1,
                'history': history,
                'n_train': len(train_idx),
                'n_val': len(val_idx),
                'device': str(self.device),
            }

            # Restore best checkpoint
            self._load_checkpoint('best')
            self._loaded = True

            return self.training_metrics

        finally:
            if tracker:
                emissions = tracker.stop()
                self.training_metrics['co2_emissions_kg'] = float(emissions)

    def _evaluate(self, dataloader: DataLoader) -> Dict:
        """Evalue sur un DataLoader. Retourne accuracy, f1, precision, recall."""
        self.base_model.eval()
        self.head.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
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
        Prediction sur une liste de textes.

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
                all_probas.extend(probas[:, 0])  # Proba de label 0 (fiable)

        return {
            'predictions': np.array(all_preds),
            'probabilities': np.array(all_probas),  # Score credibilite (0=suspect, 1=fiable)
            'labels': ['FIABLE' if p == 0 else 'SUSPECT' for p in all_preds],
        }

    def predict_credibility_scores(self, texts: List[str]) -> np.ndarray:
        """
        Retourne uniquement les scores de credibilite (0-1).
        Compatible avec le pipeline ExpertFakeNewsDetector.
        """
        result = self.predict(texts)
        return result['probabilities']

    def _save_checkpoint(self, name: str = 'best'):
        """Sauvegarde un checkpoint du modele."""
        path = os.path.join(self.model_dir, f'camembert_{name}.pt')
        torch.save({
            'base_model_state': self.base_model.state_dict(),
            'head_state': self.head.state_dict(),
        }, path)

    def _load_checkpoint(self, name: str = 'best'):
        """Charge un checkpoint."""
        path = os.path.join(self.model_dir, f'camembert_{name}.pt')
        if not os.path.exists(path):
            logger.warning("Checkpoint non trouve : %s", path)
            return False
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.base_model.load_state_dict(checkpoint['base_model_state'])
        self.head.load_state_dict(checkpoint['head_state'])
        return True

    def save(self, suffix: str = 'camembert_fr'):
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
        logger.info("CamemBERT sauvegarde : %s", base_path)

    def load(self, suffix: str = 'camembert_fr') -> bool:
        """Charge le modele sauvegarde."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("transformers non disponible")
            return False

        path = os.path.join(self.model_dir, f'{suffix}.pt')
        if not os.path.exists(path):
            logger.warning("CamemBERT non trouve : %s", path)
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.base_model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.head = CamemBERTHead(
            hidden_size=checkpoint['config']['hidden_size'],
            num_classes=2,
        ).to(self.device)

        self.base_model.load_state_dict(checkpoint['base_model_state'])
        self.head.load_state_dict(checkpoint['head_state'])
        self.base_model.eval()
        self.head.eval()
        self._loaded = True
        self.training_metrics = checkpoint.get('metrics', {})

        logger.info("CamemBERT charge : %s (F1=%.4f)", path, self.training_metrics.get('best_val_f1', 0))
        return True
