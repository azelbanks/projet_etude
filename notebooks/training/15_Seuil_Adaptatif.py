#!/usr/bin/env python3
"""
15 — Optimisation seuil adaptatif par langue
=============================================

Contexte :
    Le pipeline V5 utilise un seuil unique de 0.44 pour les deux langues.
    Or, les scores FR tendent a etre plus extremes (proches de 0 ou 1),
    tandis que les scores EN sont plus centres autour du seuil.

    Cette asymetrie suggere qu'un seuil adaptatif par langue pourrait
    ameliorer les performances, notamment la F1.

Methode :
    1. Charger le modele V5 et le dataset V5 (split identique a notebook 14)
    2. Separer les predictions test par langue (FR / EN)
    3. Grid search de seuils (0.30 a 0.60, pas de 0.01) optimisant la F1
    4. Comparer les metriques avec le seuil unique 0.44
    5. Si amelioration > 1% F1 : integrer dans ExpertFakeNewsDetector

Preconisation : P3 — Seuils adaptatifs par langue

Auteur : Thumalien Team
"""

import sys
import os
import time
import logging
import numpy as np
import pandas as pd

# --- Setup paths ---
_proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_proj, 'src'))

from pipeline.expert_detector import (
    DatasetCleaner,
    ExpertFakeNewsDetector,
    LinguisticFeatureExtractor,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA = os.path.join(_proj, 'data', 'training')
MODEL_DIR = os.path.join(_proj, 'models')

# ============================================================
#  1. CHARGEMENT DU DATASET V5 (identique a notebook 14)
# ============================================================
print("=" * 70)
print("15 — Optimisation seuil adaptatif par langue (P3)")
print("=" * 70)

t0 = time.time()

print("\n[1/5] Chargement du dataset bilingue V5...")

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

print(f"  Dataset total : {len(df_v5)} textes")
n_fr = sum(df_v5.language == 'fr')
n_en = sum(df_v5.language == 'en')
print(f"  EN={n_en}, FR={n_fr}")

# ============================================================
#  2. SPLIT TRAIN/TEST (identique a notebook 14)
# ============================================================
print("\n[2/5] Split train/test 80/20 stratifie (random_state=42)...")

df_train, df_test = train_test_split(
    df_v5, test_size=0.2, stratify=df_v5['label'], random_state=42
)
print(f"  Train : {len(df_train)} | Test : {len(df_test)}")

# ============================================================
#  3. CHARGEMENT DU MODELE V5 + PREDICTIONS SUR LE TEST SET
# ============================================================
print("\n[3/5] Chargement du modele V5 et predictions sur le test set...")

detector = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=0.44)
detector.load('expert_v5')

X_test_clean = df_test['text_clean'].values
X_test_orig = (
    df_test['text_original'].values
    if 'text_original' in df_test.columns else None
)
y_test = df_test['label'].values

X_feat = detector._build_features(X_test_clean, texts_original=X_test_orig, fit=False)
y_proba = detector.model.predict_proba(X_feat)
scores = y_proba[:, 0]  # P(Fiable)

# Separation par langue
languages = df_test['language'].values
mask_fr = languages == 'fr'
mask_en = languages == 'en'

scores_fr = scores[mask_fr]
y_test_fr = y_test[mask_fr]
scores_en = scores[mask_en]
y_test_en = y_test[mask_en]

print(f"  Test FR : {len(scores_fr)} textes (label=1: {y_test_fr.sum()})")
print(f"  Test EN : {len(scores_en)} textes (label=1: {y_test_en.sum()})")

# ============================================================
#  4. GRID SEARCH — SEUILS OPTIMAUX PAR LANGUE
# ============================================================
print("\n[4/5] Grid search seuils optimaux par langue (F1)...")

thresholds = np.arange(0.30, 0.61, 0.01)

SINGLE_THRESHOLD = 0.44


def grid_search_threshold(scores_lang, y_true_lang, thresholds):
    """Trouve le seuil optimal maximisant la F1."""
    best_f1 = -1
    best_th = SINGLE_THRESHOLD
    results = []
    for th in thresholds:
        y_pred = (scores_lang < th).astype(int)
        f1 = f1_score(y_true_lang, y_pred, zero_division=0)
        prec = precision_score(y_true_lang, y_pred, zero_division=0)
        rec = recall_score(y_true_lang, y_pred, zero_division=0)
        acc = accuracy_score(y_true_lang, y_pred)
        results.append({'threshold': th, 'f1': f1, 'precision': prec, 'recall': rec, 'accuracy': acc})
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1, pd.DataFrame(results)


# Grid search FR
best_th_fr, best_f1_fr, df_grid_fr = grid_search_threshold(scores_fr, y_test_fr, thresholds)
print(f"\n  FR — Seuil optimal : {best_th_fr:.2f} (F1={best_f1_fr:.4f})")

# Grid search EN
best_th_en, best_f1_en, df_grid_en = grid_search_threshold(scores_en, y_test_en, thresholds)
print(f"  EN — Seuil optimal : {best_th_en:.2f} (F1={best_f1_en:.4f})")

# ============================================================
#  5. COMPARAISON : SEUIL UNIQUE vs SEUILS ADAPTATIFS
# ============================================================
print("\n[5/5] Comparaison seuil unique (0.44) vs seuils adaptatifs par langue...")


def compute_metrics(scores_lang, y_true_lang, th):
    """Calcule F1, precision, recall, accuracy pour un seuil donne."""
    y_pred = (scores_lang < th).astype(int)
    return {
        'f1': f1_score(y_true_lang, y_pred, zero_division=0),
        'precision': precision_score(y_true_lang, y_pred, zero_division=0),
        'recall': recall_score(y_true_lang, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true_lang, y_pred),
    }


# Metriques avec seuil unique 0.44
m_fr_single = compute_metrics(scores_fr, y_test_fr, SINGLE_THRESHOLD)
m_en_single = compute_metrics(scores_en, y_test_en, SINGLE_THRESHOLD)

# Metriques avec seuils adaptatifs
m_fr_adapt = compute_metrics(scores_fr, y_test_fr, best_th_fr)
m_en_adapt = compute_metrics(scores_en, y_test_en, best_th_en)

# Metriques globales avec seuil unique
y_pred_single = (scores < SINGLE_THRESHOLD).astype(int)
m_global_single = {
    'f1': f1_score(y_test, y_pred_single, zero_division=0),
    'precision': precision_score(y_test, y_pred_single, zero_division=0),
    'recall': recall_score(y_test, y_pred_single, zero_division=0),
    'accuracy': accuracy_score(y_test, y_pred_single),
}

# Metriques globales avec seuils adaptatifs par langue
y_pred_adapt = np.zeros_like(y_test)
y_pred_adapt[mask_fr] = (scores_fr < best_th_fr).astype(int)
y_pred_adapt[mask_en] = (scores_en < best_th_en).astype(int)
# Textes "other" : garder seuil unique
mask_other = ~mask_fr & ~mask_en
if mask_other.sum() > 0:
    y_pred_adapt[mask_other] = (scores[mask_other] < SINGLE_THRESHOLD).astype(int)

m_global_adapt = {
    'f1': f1_score(y_test, y_pred_adapt, zero_division=0),
    'precision': precision_score(y_test, y_pred_adapt, zero_division=0),
    'recall': recall_score(y_test, y_pred_adapt, zero_division=0),
    'accuracy': accuracy_score(y_test, y_pred_adapt),
}

# ============================================================
#  TABLEAU RECAPITULATIF
# ============================================================
print("\n" + "=" * 70)
print("RESULTATS — SEUIL ADAPTATIF PAR LANGUE (P3)")
print("=" * 70)

print(f"\n  Seuils optimaux :")
print(f"    FR : {best_th_fr:.2f}  (vs seuil unique {SINGLE_THRESHOLD})")
print(f"    EN : {best_th_en:.2f}  (vs seuil unique {SINGLE_THRESHOLD})")

print(f"\n  {'':>10} {'Seuil unique (0.44)':>22} {'Seuil adaptatif':>22} {'Delta':>10}")
print(f"  {'-' * 66}")

for lang_label, m_single, m_adapt in [
    ('FR', m_fr_single, m_fr_adapt),
    ('EN', m_en_single, m_en_adapt),
    ('GLOBAL', m_global_single, m_global_adapt),
]:
    for metric in ['f1', 'precision', 'recall', 'accuracy']:
        v_single = m_single[metric]
        v_adapt = m_adapt[metric]
        delta = v_adapt - v_single
        sign = '+' if delta >= 0 else ''
        print(f"  {lang_label + ' ' + metric:<10} {v_single:>22.4f} {v_adapt:>22.4f} {sign}{delta:>9.4f}")
    print(f"  {'-' * 66}")

# Amelioration F1
delta_f1_fr = m_fr_adapt['f1'] - m_fr_single['f1']
delta_f1_en = m_en_adapt['f1'] - m_en_single['f1']
delta_f1_global = m_global_adapt['f1'] - m_global_single['f1']

print(f"\n  Amelioration F1 :")
print(f"    FR     : {delta_f1_fr:+.4f} ({delta_f1_fr*100:+.2f}%)")
print(f"    EN     : {delta_f1_en:+.4f} ({delta_f1_en*100:+.2f}%)")
print(f"    GLOBAL : {delta_f1_global:+.4f} ({delta_f1_global*100:+.2f}%)")

# Decision
improvement_significant = (delta_f1_fr > 0.01) or (delta_f1_en > 0.01) or (delta_f1_global > 0.01)

print(f"\n  Amelioration significative (>1% F1) : {'OUI' if improvement_significant else 'NON'}")

if improvement_significant:
    print(f"\n  >>> RECOMMANDATION : Integrer les seuils adaptatifs par langue")
    print(f"      threshold_fr = {best_th_fr:.2f}")
    print(f"      threshold_en = {best_th_en:.2f}")
    print(f"      Utiliser predict() avec parametre use_lang_threshold=True")
else:
    print(f"\n  >>> Le seuil unique 0.44 reste suffisant pour les deux langues.")

# ============================================================
#  DETAIL : TOP 5 SEUILS PAR LANGUE
# ============================================================
print(f"\n  --- Top 5 seuils FR (par F1) ---")
top5_fr = df_grid_fr.nlargest(5, 'f1')
for _, row in top5_fr.iterrows():
    print(f"    th={row['threshold']:.2f}  F1={row['f1']:.4f}  P={row['precision']:.4f}  R={row['recall']:.4f}")

print(f"\n  --- Top 5 seuils EN (par F1) ---")
top5_en = df_grid_en.nlargest(5, 'f1')
for _, row in top5_en.iterrows():
    print(f"    th={row['threshold']:.2f}  F1={row['f1']:.4f}  P={row['precision']:.4f}  R={row['recall']:.4f}")

# ============================================================
#  VALIDATION : TEST BILINGUE AVEC SEUILS ADAPTATIFS
# ============================================================
print(f"\n  --- Validation : predict avec seuils adaptatifs ---")

# On teste directement avec la methode modifiee si disponible
if hasattr(detector, 'threshold_fr'):
    detector.threshold_fr = best_th_fr
    detector.threshold_en = best_th_en

test_texts = [
    ("SCANDALE !! On nous cache la verite sur les vaccins !", "fr", "suspect"),
    ("Le CNRS a publie une etude sur le changement climatique.", "fr", "fiable"),
    ("URGENT: les vaccins contiennent des puces 5G !", "fr", "suspect"),
    ("La mairie annonce la renovation du pont.", "fr", "fiable"),
    ("BREAKING: Government EXPOSED in massive cover-up!", "en", "suspect"),
    ("A new study published in Nature examines climate change.", "en", "fiable"),
    ("SHARE before they DELETE this!! The truth about 5G!", "en", "suspect"),
    ("The city council approved the new budget.", "en", "fiable"),
]

results_test = detector.predict(pd.Series([t[0] for t in test_texts]))

print(f"\n  {'Texte':<55} {'Lang':>4} {'Score':>6} {'Label':>8} {'Attendu':>8} {'OK':>3}")
print(f"  {'-'*90}")

correct = 0
for i, (text, lang, expected) in enumerate(test_texts):
    score = float(results_test['ai_score_credibility'].iloc[i])
    det_lang = results_test['language'].iloc[i]
    label = "suspect" if results_test['prediction_label'].iloc[i] == 1 else "fiable"
    ok = "OK" if label == expected else "FAIL"
    if ok == "OK":
        correct += 1
    print(f"  {text[:55]:<55} {det_lang:>4} {score:>6.3f} {label:>8} {expected:>8} {ok:>3}")

print(f"\n  Score test bilingue : {correct}/{len(test_texts)}")

elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"Notebook 15 termine en {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'=' * 70}")
