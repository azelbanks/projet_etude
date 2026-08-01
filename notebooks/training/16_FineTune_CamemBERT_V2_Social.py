#!/usr/bin/env python3
"""
16 — Fine-tuning CamemBERT V2 avec donnees FR sociales
=======================================================

Contexte :
    Le CamemBERT V1 (notebook 13) a ete fine-tune uniquement sur les articles
    Kaggle FR et obtient un score de 3/6 sur les phrases de test social media.
    Cette V2 integre le dataset synthetique de 10K posts FR sociaux pour
    ameliorer la detection sur les textes courts / registre informel.

Preconisation P2 : Re-fine-tune CamemBERT avec le dataset FR social synthetique.

Architecture :
    CamemBERT-base (couches 9-11 fine-tunees) + Head(768->256->2)
    Surpoids x2 sur les textes courts (< 30 mots)

Dataset : textes FR extraits du dataset bilingue V5
    - Kaggle FR articles + augmentation courte
    - Credibility Corpus FR tweets
    - 10K posts FR sociaux synthetiques (fr_social_media_synthetic.csv)

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
#  1. CHARGEMENT DATASET FR (avec donnees sociales)
# ============================================================
print("=" * 70)
print("FINE-TUNING CamemBERT V2 — Avec donnees FR sociales synthetiques")
print("=" * 70)

t0 = time.time()

print("\n[1/6] Chargement du dataset bilingue (extraction FR + social)...")

df_all = DatasetCleaner.prepare_bilingual_dataset(
    fake_path=os.path.join(DATA, 'Fake.csv'),
    true_path=os.path.join(DATA, 'True.csv'),
    kaggle_fr_dir=os.path.join(DATA, 'kaggle_fr'),
    fakenewsnet_dir=os.path.join(DATA, 'fakenewsnet'),
    constraint_dir=os.path.join(DATA, 'constraint'),
    credibility_dir=os.path.join(DATA, 'credibility_corpus'),
    french_oversample=1,
    social_oversample=1,
    fr_short_augment=True,
    fr_short_oversample=1,
    fr_social_path=os.path.join(DATA, 'fr_social_media_synthetic.csv'),
)

# Filtrer uniquement les textes FR
df_fr = df_all[df_all['language'] == 'fr'].copy()
df_fr['word_count'] = df_fr['text_original'].str.split().str.len()

print(f"  Textes FR total : {len(df_fr)}")
print(f"  Labels : {df_fr.label.value_counts().to_dict()}")
print(f"  < 15 mots : {(df_fr.word_count < 15).sum()}")
print(f"  15-30 mots : {((df_fr.word_count >= 15) & (df_fr.word_count < 30)).sum()}")
print(f"  30-100 mots : {((df_fr.word_count >= 30) & (df_fr.word_count < 100)).sum()}")
print(f"  > 100 mots : {(df_fr.word_count >= 100).sum()}")

# Identifier la part du dataset social synthetique
if 'source' in df_fr.columns:
    n_social = (df_fr['source'] == 'synthetic_fr_social').sum()
    print(f"  Dont posts sociaux synthetiques : {n_social}")

# ============================================================
#  2. SPLIT TRAIN/TEST 80/20 STRATIFIE
# ============================================================
print("\n[2/6] Split train/test 80/20 stratifie...")

df_train, df_test = train_test_split(
    df_fr, test_size=0.2, stratify=df_fr['label'], random_state=42
)
print(f"  Train : {len(df_train)} | Test : {len(df_test)}")
print(f"  Train labels : {df_train.label.value_counts().to_dict()}")
print(f"  Test labels  : {df_test.label.value_counts().to_dict()}")

# ============================================================
#  3. FINE-TUNING CamemBERT V2
# ============================================================
print("\n[3/6] Fine-tuning CamemBERT V2...")
print("  Epochs: 3 | Batch: 32 | LR: 2e-5 | Short text weight: 2.0")
print("  Couches gelees: 0-8 | Couches fine-tunees: 9-11 + head")
print("  Dataset enrichi avec 10K posts FR sociaux synthetiques")

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
print(f"  Epochs entraines : {metrics['epochs']}")
print(f"  Train size : {metrics['n_train']} | Val size : {metrics['n_val']}")
print(f"  Device : {metrics['device']}")
if 'co2_emissions_kg' in metrics:
    print(f"  Emissions CO2 : {metrics['co2_emissions_kg']:.6f} kg")

# ============================================================
#  4. EVALUATION DETAILLEE SUR HOLDOUT
# ============================================================
print("\n[4/6] Evaluation detaillee sur holdout test...")

test_texts = df_test['text_original'].astype(str).tolist()
test_labels = df_test['label'].astype(int).tolist()
results = classifier.predict(test_texts)

y_pred = results['predictions']
y_true = np.array(test_labels)
y_proba = results['probabilities']

# Metriques globales
acc_global = accuracy_score(y_true, y_pred)
f1_global = f1_score(y_true, y_pred)
prec_global = precision_score(y_true, y_pred)
rec_global = recall_score(y_true, y_pred)

print(f"\n  Metriques globales:")
print(f"    Accuracy  : {acc_global:.4f}")
print(f"    F1        : {f1_global:.4f}")
print(f"    Precision : {prec_global:.4f}")
print(f"    Recall    : {rec_global:.4f}")

# Par categorie de longueur
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

print(f"\n  Metriques par categorie de longueur:")
print(f"  {'Longueur':<20} {'N':>6} {'Accuracy':>9} {'F1':>7} {'Precision':>10} {'Recall':>7}")
print(f"  {'-'*60}")

for cat in ['ultra_court (<15)', 'court (15-30)', 'moyen (30-100)', 'long (>100)']:
    mask = df_test_eval['length_cat'] == cat
    sub = df_test_eval[mask]
    if len(sub) < 5:
        print(f"  {cat:<20} {len(sub):>6}   (trop peu d'echantillons)")
        continue
    yt = sub['label'].values
    yp = sub['y_pred'].values
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, zero_division=0)
    prec = precision_score(yt, yp, zero_division=0)
    rec = recall_score(yt, yp, zero_division=0)
    print(f"  {cat:<20} {len(sub):>6} {acc:>9.4f} {f1:>7.4f} {prec:>10.4f} {rec:>7.4f}")

# ============================================================
#  5. TESTS COMPARATIFS (V1 vs V2)
# ============================================================
print("\n[5/6] Tests comparatifs...")

# --- 5a. MEMES 6 phrases que notebook 13 (V1 = 3/6) ---
print("\n  --- Test V1 : memes 6 phrases que notebook 13 ---")
print("  (CamemBERT V1 obtient 3/6 sur ces phrases)")

test_samples_v1 = [
    ("SCANDALE ! On nous cache la verite sur les vaccins !", "suspect"),
    ("Le CNRS publie une etude sur le climat.", "fiable"),
    ("URGENT: puces 5G dans les vaccins !!", "suspect"),
    ("La mairie organise une fete ce weekend.", "fiable"),
    ("Partagez avant censure !! Revelations choc", "suspect"),
    ("Les resultats du bac sont disponibles.", "fiable"),
]

sample_results_v1 = classifier.predict([t[0] for t in test_samples_v1])
correct_v1 = 0
for i, (text, expected) in enumerate(test_samples_v1):
    label = "suspect" if sample_results_v1['predictions'][i] == 1 else "fiable"
    ok = "OK" if label == expected else "FAIL"
    if ok == "OK":
        correct_v1 += 1
    score = sample_results_v1['probabilities'][i]
    print(f"  [{ok}] {text:<55} score={score:.3f} -> {label} (attendu={expected})")

print(f"\n  Score V1 phrases : {correct_v1}/{len(test_samples_v1)}  (V1 avait 3/6)")

# --- 5b. Phrases supplementaires social media ---
print("\n  --- Test V2 : phrases supplementaires social media ---")

test_samples_v2 = [
    ("REVEILLEZ VOUS !! Le graphene dans les masques !!", "suspect"),
    ("Les cours reprennent lundi prochain.", "fiable"),
    ("ON NOUS MENT SUR TOUT !! Faites vos propres recherches", "suspect"),
    ("La bibliotheque municipale ouvre ses portes samedi.", "fiable"),
]

sample_results_v2 = classifier.predict([t[0] for t in test_samples_v2])
correct_v2 = 0
for i, (text, expected) in enumerate(test_samples_v2):
    label = "suspect" if sample_results_v2['predictions'][i] == 1 else "fiable"
    ok = "OK" if label == expected else "FAIL"
    if ok == "OK":
        correct_v2 += 1
    score = sample_results_v2['probabilities'][i]
    print(f"  [{ok}] {text:<55} score={score:.3f} -> {label} (attendu={expected})")

print(f"\n  Score V2 phrases : {correct_v2}/{len(test_samples_v2)}")

# --- Bilan total ---
total_correct = correct_v1 + correct_v2
total_phrases = len(test_samples_v1) + len(test_samples_v2)
print(f"\n  === BILAN TOTAL ===")
print(f"  Phrases V1 (notebook 13) : {correct_v1}/{len(test_samples_v1)}  (V1 avait 3/6)")
print(f"  Phrases V2 (social)      : {correct_v2}/{len(test_samples_v2)}")
print(f"  Total                    : {total_correct}/{total_phrases}")

# ============================================================
#  6. SAUVEGARDE MODELE V2
# ============================================================
print("\n[6/6] Sauvegarde CamemBERT V2...")

classifier.save('camembert_fr_v2')
print(f"  Sauvegarde dans {MODEL_DIR}/camembert_fr_v2.pt")

elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"Fine-tuning CamemBERT V2 termine en {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'=' * 70}")
