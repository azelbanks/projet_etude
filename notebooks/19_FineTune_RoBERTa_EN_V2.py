#!/usr/bin/env python3
"""
19 — Fine-Tuning RoBERTa V2 pour textes courts EN + donnees sociales
=====================================================================

Contexte :
    RoBERTa V1 (notebook 18) atteignait F1 EN ultra-court = 0.838 (+8.2% vs
    V5 TF-IDF) mais le test rapide ne passait qu'a 6/10 — meme biais que
    CamemBERT V1 : textes courts neutres sur-classifies comme suspect.

    Solution : meme approche que P2 (CamemBERT V1 -> V2) — integrer les
    10K posts EN sociaux synthetiques generes par generate_en_social_dataset.py.
    CamemBERT V2 est passe de 3/6 a 9/10 avec cette methode.

Architecture :
    - Modele de base : roberta-base (125M params)
    - Fine-tuning : couches 9-11 + classification head (768 -> 256 -> 2)
    - Dataset : textes EN V5 + 10K EN social synthetique
    - Surpoids : x2 sur les textes courts (< 30 mots)

Auteur : Thumalien Team
"""

import sys
import os
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Setup paths ---
_proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_proj, 'src'))

from pipeline.expert_detector import DatasetCleaner
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("ERREUR: pip install transformers")
    sys.exit(1)

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA = os.path.join(_proj, 'data', 'training')
MODEL_DIR = os.path.join(_proj, 'models')

# ============================================================
#  PyTorch Dataset + Head (meme architecture que CamemBERT)
# ============================================================

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


# ============================================================
#  MAIN
# ============================================================
print("=" * 70)
print("FINE-TUNING RoBERTa EN V2 — Avec donnees EN sociales synthetiques")
print("=" * 70)

t0 = time.time()
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"  Device : {device}")

# ============================================================
#  1. CHARGEMENT DU DATASET (EN V5 + EN social synthetique)
# ============================================================
print("\n[1/6] Chargement du dataset V5 (extraction EN) + EN social...")

df_v5 = DatasetCleaner.prepare_bilingual_dataset(
    fake_path=os.path.join(DATA, 'Fake.csv'),
    true_path=os.path.join(DATA, 'True.csv'),
    kaggle_fr_dir=os.path.join(DATA, 'kaggle_fr'),
    fakenewsnet_dir=os.path.join(DATA, 'fakenewsnet'),
    constraint_dir=os.path.join(DATA, 'constraint'),
    credibility_dir=os.path.join(DATA, 'credibility_corpus'),
    french_oversample=5,
    social_oversample=2,
    fr_short_augment=True,
    fr_short_oversample=3,
    fr_social_path=os.path.join(DATA, 'fr_social_media_synthetic.csv'),
)

# Filtrer EN uniquement
df_en = df_v5[df_v5['language'] == 'en'].copy()
print(f"  Textes EN (V5 base) : {len(df_en)}")

# Charger les donnees EN sociales synthetiques
en_social_path = os.path.join(DATA, 'en_social_media_synthetic.csv')
df_social = pd.read_csv(en_social_path)
print(f"  Textes EN social synthetique : {len(df_social)}")
print(f"    Labels : {df_social['label'].value_counts().to_dict()}")

# Formatter pour correspondre au schema df_en
df_social_formatted = pd.DataFrame({
    'text_original': df_social['text'],
    'text_clean': df_social['text'].str.lower().str.strip(),
    'label': df_social['label'],
    'language': 'en',
    'source': 'synthetic_en_social',
})

# Combiner
df_en_v2 = pd.concat([df_en, df_social_formatted], ignore_index=True)
print(f"  Total EN V2 : {len(df_en_v2)} (V5: {len(df_en)} + social: {len(df_social)})")
print(f"  Labels : {df_en_v2['label'].value_counts().to_dict()}")

# Stats par longueur
df_en_v2['word_count'] = df_en_v2['text_original'].astype(str).str.split().str.len()
for cat, mask in [('<15 mots', df_en_v2.word_count < 15),
                  ('15-30 mots', (df_en_v2.word_count >= 15) & (df_en_v2.word_count < 30)),
                  ('30-100 mots', (df_en_v2.word_count >= 30) & (df_en_v2.word_count < 100)),
                  ('>100 mots', df_en_v2.word_count >= 100)]:
    print(f"  {cat} : {mask.sum()}")

# ============================================================
#  2. SPLIT TRAIN/TEST
# ============================================================
print("\n[2/6] Split train/test 80/20 stratifie...")

df_train, df_test = train_test_split(
    df_en_v2, test_size=0.2, stratify=df_en_v2['label'], random_state=42
)
print(f"  Train : {len(df_train)} | Test : {len(df_test)}")
print(f"  Train labels : {df_train['label'].value_counts().to_dict()}")
print(f"  Test labels  : {df_test['label'].value_counts().to_dict()}")

# Compter les textes sociaux dans train/test
social_train = (df_train.get('source', pd.Series()) == 'synthetic_en_social').sum()
social_test = (df_test.get('source', pd.Series()) == 'synthetic_en_social').sum()
print(f"  EN social dans train : {social_train} | dans test : {social_test}")

# ============================================================
#  3. FINE-TUNING RoBERTa V2
# ============================================================
print("\n[3/6] Fine-tuning RoBERTa EN V2...")

MODEL_NAME = 'roberta-base'
MAX_LENGTH = 128
EPOCHS = 3
BATCH_SIZE = 32
LR = 2e-5
SHORT_TEXT_WEIGHT = 2.0

print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR} | Short text weight: {SHORT_TEXT_WEIGHT}")

# Init model
logger.info("Chargement de %s...", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
head = ClassificationHead(
    hidden_size=base_model.config.hidden_size,
    num_classes=2,
).to(device)

# Freeze couches 0-8
for name, param in base_model.named_parameters():
    if 'encoder.layer' in name:
        layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
        if layer_num < 9:
            param.requires_grad = False
    elif 'embeddings' in name:
        param.requires_grad = False

trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in base_model.parameters())
print(f"  Couches gelees: 0-8 | Couches fine-tunees: 9-11 + head")
logger.info("RoBERTa: %d/%d parametres entrainables (%.1f%%)", trainable, total, 100 * trainable / total)

# Prepare data
texts_train = df_train['text_original'].astype(str).tolist()
labels_train = df_train['label'].astype(int).tolist()
texts_test = df_test['text_original'].astype(str).tolist()
labels_test = df_test['label'].astype(int).tolist()

# Sample weights
word_counts_train = [len(t.split()) for t in texts_train]
sample_weights = torch.tensor([
    SHORT_TEXT_WEIGHT if wc < 30 else 1.0
    for wc in word_counts_train
], dtype=torch.float32)

# Train/val split (90/10)
indices = list(range(len(texts_train)))
train_idx, val_idx = train_test_split(
    indices, test_size=0.1, stratify=labels_train, random_state=42
)

train_dataset = TextDataset(
    [texts_train[i] for i in train_idx],
    [labels_train[i] for i in train_idx],
    tokenizer, MAX_LENGTH,
)
val_dataset = TextDataset(
    [texts_train[i] for i in val_idx],
    [labels_train[i] for i in val_idx],
    tokenizer, MAX_LENGTH,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

train_weights = sample_weights[train_idx]

# Optimizer
optimizer = torch.optim.AdamW([
    {'params': [p for p in base_model.parameters() if p.requires_grad], 'lr': LR},
    {'params': head.parameters(), 'lr': LR * 10},
], weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS * len(train_loader),
)

criterion = nn.CrossEntropyLoss()

# Emissions tracking
tracker = None
if CODECARBON_AVAILABLE:
    tracker = EmissionsTracker(
        project_name="Thumalien_RoBERTa_EN_V2",
        output_dir=_proj,
    )
    tracker.start()

best_val_f1 = 0.0

for epoch in range(EPOCHS):
    # --- Train ---
    base_model.train()
    head.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = head(cls_output)

        # Weighted loss
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(train_weights))
        if end_idx > start_idx and end_idx <= len(train_weights):
            batch_weights = train_weights[start_idx:end_idx].to(device)
            per_sample_loss = nn.functional.cross_entropy(logits, batch_labels, reduction='none')
            if len(batch_weights) == len(per_sample_loss):
                loss = (per_sample_loss * batch_weights).mean()
            else:
                loss = criterion(logits, batch_labels)
        else:
            loss = criterion(logits, batch_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(base_model.parameters()) + list(head.parameters()), 1.0
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
                epoch + 1, EPOCHS, batch_idx + 1, len(train_loader), loss.item(),
            )

    train_acc = train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)

    # --- Validation ---
    base_model.eval()
    head.eval()
    val_preds, val_labels_list = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label']

            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            logits = head(cls_output)
            preds = logits.argmax(dim=1).cpu().numpy()

            val_preds.extend(preds)
            val_labels_list.extend(batch_labels.numpy())

    val_f1 = f1_score(val_labels_list, val_preds, zero_division=0)
    val_acc = accuracy_score(val_labels_list, val_preds)

    print(f"  Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        # Save best checkpoint
        torch.save({
            'base_model_state': base_model.state_dict(),
            'head_state': head.state_dict(),
            'config': {
                'model_name': MODEL_NAME,
                'max_length': MAX_LENGTH,
                'hidden_size': base_model.config.hidden_size,
                'version': 'v2',
                'en_social_data': True,
            },
            'metrics': {
                'best_val_f1': best_val_f1,
                'epoch': epoch + 1,
            },
        }, os.path.join(MODEL_DIR, 'roberta_en_v2.pt'))

emissions_kg = 0.0
if tracker:
    emissions_kg = tracker.stop()

print(f"\n  Best Val F1 : {best_val_f1:.4f}")
print(f"  Epochs entraines : {EPOCHS}")
print(f"  Train size : {len(train_dataset)} | Val size : {len(val_dataset)}")
print(f"  Device : {device}")
if emissions_kg:
    print(f"  Emissions CO2 : {emissions_kg:.6f} kg")

# Reload best checkpoint
checkpoint = torch.load(os.path.join(MODEL_DIR, 'roberta_en_v2.pt'), map_location=device, weights_only=True)
base_model.load_state_dict(checkpoint['base_model_state'])
head.load_state_dict(checkpoint['head_state'])
logger.info("RoBERTa V2 best checkpoint recharge (F1=%.4f)", best_val_f1)

# ============================================================
#  4. EVALUATION DETAILLEE SUR HOLDOUT
# ============================================================
print("\n[4/6] Evaluation detaillee sur holdout test...")

base_model.eval()
head.eval()

test_dataset = TextDataset(texts_test, labels_test, tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

all_preds = []
all_probas = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = head(cls_output)

        probas = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_probas.extend(probas[:, 0])  # P(FIABLE)

y_pred = np.array(all_preds)
y_proba = np.array(all_probas)
y_true = np.array(labels_test)

# Metriques globales
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, zero_division=0)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)

print(f"\n  Metriques globales RoBERTa V2:")
print(f"    Accuracy  : {acc:.4f}")
print(f"    F1        : {f1:.4f}")
print(f"    Precision : {prec:.4f}")
print(f"    Recall    : {rec:.4f}")

# Par longueur
df_test_eval = df_test.copy()
df_test_eval['y_pred'] = y_pred

def length_cat(wc):
    if wc < 15:
        return 'ultra_court (<15)'
    elif wc < 30:
        return 'court (15-30)'
    elif wc < 100:
        return 'moyen (30-100)'
    else:
        return 'long (>100)'

df_test_eval['length_cat'] = df_test_eval['word_count'].apply(length_cat)

print(f"\n  Metriques par categorie de longueur:")
print(f"  {'Longueur':<25} {'N':>6} {'Accuracy':>9} {'F1':>7} {'Precision':>10} {'Recall':>7}")
print(f"  {'-'*60}")

for cat in ['ultra_court (<15)', 'court (15-30)', 'moyen (30-100)', 'long (>100)']:
    mask = df_test_eval['length_cat'] == cat
    sub = df_test_eval[mask]
    if len(sub) < 10:
        continue
    yt = sub['label'].values
    yp = sub['y_pred'].values
    a = accuracy_score(yt, yp)
    f = f1_score(yt, yp, zero_division=0)
    p = precision_score(yt, yp, zero_division=0)
    r = recall_score(yt, yp, zero_division=0)
    print(f"  {cat:<25} {len(sub):>6} {a:>9.4f} {f:>7.4f} {p:>10.4f} {r:>7.4f}")

# ============================================================
#  5. TESTS COMPARATIFS (memes phrases que V1)
# ============================================================
print("\n[5/6] Tests comparatifs V1 vs V2...")

def predict_texts(texts):
    ds = TextDataset(texts, [0]*len(texts), tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=32)
    preds, probas = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            logits = head(cls_output)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            probas.extend(p[:, 0])
    return np.array(probas)

# Memes 10 phrases que V1 pour comparaison directe
test_en = [
    ("BREAKING: Government EXPOSED in massive cover-up!", "suspect"),
    ("A new study published in Nature examines climate change.", "fiable"),
    ("SHARE before they DELETE this!! The truth about 5G!", "suspect"),
    ("The city council approved the new budget.", "fiable"),
    ("WAKE UP PEOPLE!! They're putting chemicals in the water!", "suspect"),
    ("The weather forecast calls for rain tomorrow.", "fiable"),
    ("EXPOSED: Secret labs use 5G for mind control! Share before deleted!", "suspect"),
    ("NASA announced a new mission to Mars.", "fiable"),
    ("DON'T TRUST the mainstream media!! They lie about EVERYTHING!", "suspect"),
    ("The university published its annual research report.", "fiable"),
]

scores = predict_texts([t[0] for t in test_en])

print(f"\n  --- Test rapide EN V2 (10 phrases) ---")
correct = 0
for i, (text, expected) in enumerate(test_en):
    score = float(scores[i])
    label = "fiable" if score >= 0.5 else "suspect"
    ok = "OK" if label == expected else "FAIL"
    if ok == "OK":
        correct += 1
    print(f"  [{'OK' if ok == 'OK' else 'FAIL'}] {text[:60]:<60} score={score:.3f} -> {label} (attendu={expected})")

print(f"\n  Score test EN V2 : {correct}/{len(test_en)}")

# Tests supplementaires social media EN (comme CamemBERT V2)
test_social = [
    ("Just got back from the gym, feeling great!", "fiable"),
    ("The new coffee shop downtown is amazing", "fiable"),
    ("School board meeting tonight at 7pm", "fiable"),
    ("Traffic is terrible on the highway today", "fiable"),
    ("THEY ARE LYING TO YOU ABOUT EVERYTHING!! Wake up!!", "suspect"),
    ("Big Pharma doesn't want you to know about this cure!!", "suspect"),
    ("EXPOSED: the truth about what's really in your food!!", "suspect"),
    ("Share this before they take it down!! The government is hiding the truth", "suspect"),
]

scores_social = predict_texts([t[0] for t in test_social])

print(f"\n  --- Tests supplementaires social media EN ---")
correct_social = 0
for i, (text, expected) in enumerate(test_social):
    score = float(scores_social[i])
    label = "fiable" if score >= 0.5 else "suspect"
    ok = "OK" if label == expected else "FAIL"
    if ok == "OK":
        correct_social += 1
    print(f"  [{'OK' if ok == 'OK' else 'FAIL'}] {text[:60]:<60} score={score:.3f} -> {label} (attendu={expected})")

print(f"\n  Score social EN : {correct_social}/{len(test_social)}")
print(f"  TOTAL V2 : {correct + correct_social}/{len(test_en) + len(test_social)}")

# ============================================================
#  6. RESUME COMPARATIF V1 vs V2
# ============================================================
print(f"\n[6/6] Sauvegarde et resume...")
print(f"  Modele sauvegarde : {os.path.join(MODEL_DIR, 'roberta_en_v2.pt')}")

elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"Fine-tuning RoBERTa EN V2 termine en {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"  Dataset : {len(df_en_v2)} textes EN (V5: {len(df_en)} + social: {len(df_social)})")
print(f"  Test rapide : {correct}/{len(test_en)}")
print(f"  Test social : {correct_social}/{len(test_social)}")
print(f"  Total : {correct + correct_social}/{len(test_en) + len(test_social)}")
if emissions_kg:
    print(f"  Emissions CO2 : {emissions_kg:.6f} kg")
print(f"{'=' * 70}")
