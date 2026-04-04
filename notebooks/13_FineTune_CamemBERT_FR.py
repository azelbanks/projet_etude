#!/usr/bin/env python3
"""
13 — Fine-tuning CamemBERT pour textes courts FR
=================================================

Contexte :
    Le modele V4 (TF-IDF+LogReg) atteint F1=0.86 sur les textes FR courts.
    CamemBERT, pre-entraine sur 138 Go de texte francais, peut capturer des
    patterns semantiques que le TF-IDF ne voit pas (ironie, sous-entendus,
    formulations conspirationnistes).

Architecture :
    CamemBERT-base (couches 9-11 fine-tunees) + Head(768->256->2)
    Surpoids x2 sur les textes courts (< 30 mots)

Dataset : textes FR extraits du dataset bilingue V4
    - Kaggle FR articles + augmentation courte
    - Credibility Corpus FR tweets

Auteur : Thumalien Team
"""

import sys
import os
import time
import logging

# --- Setup paths ---
_proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_proj, 'src'))

from pipeline.expert_detector import DatasetCleaner
from pipeline.camembert_classifier import CamemBERTClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA = os.path.join(_proj, 'data', 'training')
MODEL_DIR = os.path.join(_proj, 'models')

# ============================================================
#  1. CHARGEMENT DATASET FR
# ============================================================
print("=" * 70)
print("FINE-TUNING CamemBERT — Textes courts FR")
print("=" * 70)

t0 = time.time()

print("\n[1/5] Chargement du dataset bilingue (extraction FR)...")

df_all = DatasetCleaner.prepare_bilingual_dataset(
    fake_path=os.path.join(DATA, 'Fake.csv'),
    true_path=os.path.join(DATA, 'True.csv'),
    kaggle_fr_dir=os.path.join(DATA, 'kaggle_fr'),
    fakenewsnet_dir=os.path.join(DATA, 'fakenewsnet'),
    constraint_dir=os.path.join(DATA, 'constraint'),
    credibility_dir=os.path.join(DATA, 'credibility_corpus'),
    french_oversample=1,        # Pas d'oversample, CamemBERT gere mieux
    social_oversample=1,
    fr_short_augment=True,      # Garder l'augmentation courte
    fr_short_oversample=1,      # Pas de duplication
)

# Filtrer uniquement les textes FR
df_fr = df_all[df_all['language'] == 'fr'].copy()
df_fr['word_count'] = df_fr['text_original'].str.split().str.len()

print(f"  Textes FR total : {len(df_fr)}")
print(f"  Labels : {df_fr.label.value_counts().to_dict()}")
print(f"  < 15 mots : {(df_fr.word_count < 15).sum()}")
print(f"  15-30 mots : {((df_fr.word_count >= 15) & (df_fr.word_count < 30)).sum()}")
print(f"  > 30 mots : {(df_fr.word_count >= 30).sum()}")

# ============================================================
#  2. SPLIT TRAIN/TEST
# ============================================================
print("\n[2/5] Split train/test 80/20 stratifie...")

df_train, df_test = train_test_split(
    df_fr, test_size=0.2, stratify=df_fr['label'], random_state=42
)
print(f"  Train : {len(df_train)} | Test : {len(df_test)}")

# ============================================================
#  3. FINE-TUNING
# ============================================================
print("\n[3/5] Fine-tuning CamemBERT-base...")
print("  Epochs: 3 | Batch: 32 | LR: 2e-5 | Short text weight: 2.0")
print("  Couches gelee: 0-8 | Couches fine-tunees: 9-11 + head")

classifier = CamemBERTClassifier(model_dir=MODEL_DIR)
metrics = classifier.fine_tune(
    df_train,
    epochs=3,
    batch_size=32,
    lr=2e-5,
    short_text_weight=2.0,
    track_emissions=True,
)

print(f"\n  Best Val F1 : {metrics['best_val_f1']:.4f}")

# ============================================================
#  4. EVALUATION DETAILLEE
# ============================================================
print("\n[4/5] Evaluation detaillee sur holdout test...")

# Predictions
test_texts = df_test['text_original'].astype(str).tolist()
test_labels = df_test['label'].astype(int).tolist()
results = classifier.predict(test_texts)

y_pred = results['predictions']
y_true = np.array(test_labels)
y_proba = results['probabilities']

# Global metrics
print(f"\n  Global:")
print(f"    Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
print(f"    F1        : {f1_score(y_true, y_pred):.4f}")
print(f"    Precision : {precision_score(y_true, y_pred):.4f}")
print(f"    Recall    : {recall_score(y_true, y_pred):.4f}")

# Par longueur
df_test_eval = df_test.copy()
df_test_eval['y_pred'] = y_pred
df_test_eval['y_proba'] = y_proba

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

print(f"\n  {'Longueur':<20} {'N':>6} {'Accuracy':>9} {'F1':>7} {'Precision':>10} {'Recall':>7}")
print(f"  {'-'*60}")

for cat in ['ultra_court (<15)', 'court (15-30)', 'moyen (30-100)', 'long (>100)']:
    mask = df_test_eval['length_cat'] == cat
    sub = df_test_eval[mask]
    if len(sub) < 5:
        continue
    yt = sub['label'].values
    yp = sub['y_pred'].values
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, zero_division=0)
    prec = precision_score(yt, yp, zero_division=0)
    rec = recall_score(yt, yp, zero_division=0)
    print(f"  {cat:<20} {len(sub):>6} {acc:>9.4f} {f1:>7.4f} {prec:>10.4f} {rec:>7.4f}")

# ============================================================
#  5. SAUVEGARDE
# ============================================================
print("\n[5/5] Sauvegarde CamemBERT fine-tune...")

classifier.save('camembert_fr')
print(f"  Sauvegarde dans {MODEL_DIR}/camembert_fr.pt")

# Test final
print("\n  --- Test rapide ---")
test_samples = [
    ("SCANDALE ! On nous cache la verite sur les vaccins !", "suspect"),
    ("Le CNRS publie une etude sur le climat.", "fiable"),
    ("URGENT: puces 5G dans les vaccins !!", "suspect"),
    ("La mairie organise une fete ce weekend.", "fiable"),
    ("Partagez avant censure !! Revelations choc", "suspect"),
    ("Les resultats du bac sont disponibles.", "fiable"),
]
sample_results = classifier.predict([t[0] for t in test_samples])
correct = 0
for i, (text, expected) in enumerate(test_samples):
    label = "suspect" if sample_results['predictions'][i] == 1 else "fiable"
    ok = "OK" if label == expected else "FAIL"
    if ok == "OK":
        correct += 1
    score = sample_results['probabilities'][i]
    print(f"  [{ok}] {text[:50]:<50} score={score:.3f} -> {label} (attendu={expected})")

print(f"\n  Score : {correct}/{len(test_samples)}")

elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"Fine-tuning CamemBERT termine en {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'=' * 70}")
