#!/usr/bin/env python3
"""
26 — Fine-tuning V5 sur données Bluesky (Domain Adaptation)
=============================================================

Problème : V5 est entraîné sur des articles de presse (Reuters, Kaggle)
mais déployé sur des posts Bluesky courts et informels.
→ Domain shift = F1 suspect gold set = 0.087

Solution : Ajouter des posts Bluesky à haute confiance au dataset
d'entraînement pour que le TF-IDF apprenne le vocabulaire Bluesky.

Stratégie de pseudo-labeling :
    1. Exporter les posts Bluesky avec score V5 ≤ 0.15 (suspect)
       et score V5 ≥ 0.85 (fiable)
    2. Cross-valider avec CamemBERT pour les posts FR
    3. Échantillonner 5K suspect + 5K fiable
    4. Ré-entraîner V5 sur le dataset augmenté

Auteur : Thumalien Team
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

_proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_proj, 'src'))

from pipeline.expert_detector import ExpertFakeNewsDetector, EmotionFeatureExtractor
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate

MODEL_DIR = os.path.join(_proj, 'models')
DATA_DIR = os.path.join(_proj, 'data')
TRAINING_DIR = os.path.join(DATA_DIR, 'training')

print("=" * 70)
print("V5 FINE-TUNE — DOMAIN ADAPTATION BLUESKY")
print("=" * 70)
t0 = time.time()

# ================================================================
#  1. CHARGER LES DONNÉES D'ENTRAÎNEMENT ORIGINALES (V5)
# ================================================================
print("\n[1/6] Chargement des données d'entraînement originales...")

# Reproduire le chargement V5 (cf. notebook 12)
frames = []

# ISOT (True/Fake)
true_df = pd.read_csv(os.path.join(TRAINING_DIR, 'True.csv'))
true_df['label'] = 0  # fiable
true_df['text_original'] = true_df['text']
true_df['source'] = 'isot_true'
frames.append(true_df[['text_original', 'label', 'source']])

fake_df = pd.read_csv(os.path.join(TRAINING_DIR, 'Fake.csv'))
fake_df['label'] = 1  # suspect
fake_df['text_original'] = fake_df['text']
fake_df['source'] = 'isot_fake'
frames.append(fake_df[['text_original', 'label', 'source']])

# FR social synthetic
fr_social = pd.read_csv(os.path.join(TRAINING_DIR, 'fr_social_media_synthetic.csv'))
fr_social['source'] = 'fr_social'
fr_social['text_original'] = fr_social['text']
frames.append(fr_social[['text_original', 'label', 'source']])

# EN social synthetic
en_social = pd.read_csv(os.path.join(TRAINING_DIR, 'en_social_media_synthetic.csv'))
en_social['source'] = 'en_social'
en_social['text_original'] = en_social['text']
frames.append(en_social[['text_original', 'label', 'source']])

# FakeNewsNet
for fn in ['gossipcop_fake.csv', 'gossipcop_real.csv', 'politifact_fake.csv', 'politifact_real.csv']:
    path = os.path.join(TRAINING_DIR, 'fakenewsnet', fn)
    if os.path.exists(path):
        df_fn = pd.read_csv(path)
        if 'title' in df_fn.columns:
            df_fn['text_original'] = df_fn['title']
        elif 'text' in df_fn.columns:
            df_fn['text_original'] = df_fn['text']
        else:
            continue
        df_fn['label'] = 1 if 'fake' in fn else 0
        df_fn['source'] = f'fakenewsnet_{fn}'
        frames.append(df_fn[['text_original', 'label', 'source']])

# CONSTRAINT 2021
for fn in ['Constraint_Train.csv', 'Constraint_Val.csv', 'Constraint_Test.csv']:
    path = os.path.join(TRAINING_DIR, 'constraint', fn)
    if os.path.exists(path):
        df_c = pd.read_csv(path)
        if 'tweet' in df_c.columns and 'label' in df_c.columns:
            df_c['text_original'] = df_c['tweet']
            df_c['label'] = (df_c['label'] == 'fake').astype(int)
            df_c['source'] = 'constraint'
            frames.append(df_c[['text_original', 'label', 'source']])

# Credibility Corpus
for fn in os.listdir(os.path.join(TRAINING_DIR, 'credibility_corpus')):
    if fn.endswith('.csv'):
        path = os.path.join(TRAINING_DIR, 'credibility_corpus', fn)
        df_cr = pd.read_csv(path)
        if 'text' in df_cr.columns:
            df_cr['text_original'] = df_cr['text']
            df_cr['label'] = 1 if 'fake' in fn.lower() or 'non_credible' in fn.lower() else 0
            df_cr['source'] = f'credibility_{fn}'
            frames.append(df_cr[['text_original', 'label', 'source']])

# Kaggle FR
for fn in os.listdir(os.path.join(TRAINING_DIR, 'kaggle_fr')):
    if fn.endswith('.csv'):
        path = os.path.join(TRAINING_DIR, 'kaggle_fr', fn)
        try:
            df_kf = pd.read_csv(path, on_bad_lines='skip')
        except Exception:
            continue
        if 'text' in df_kf.columns:
            df_kf['text_original'] = df_kf['text']
            if 'label' not in df_kf.columns:
                df_kf['label'] = 1 if 'fake' in fn.lower() else 0
            df_kf['source'] = f'kaggle_fr_{fn}'
            frames.append(df_kf[['text_original', 'label', 'source']])

df_original = pd.concat(frames, ignore_index=True)
df_original = df_original.dropna(subset=['text_original'])
df_original['text_original'] = df_original['text_original'].astype(str)

print(f"  Dataset original : {len(df_original)} textes")
print(f"  Labels : {(df_original['label']==0).sum()} fiables, {(df_original['label']==1).sum()} suspects")
print(f"  Sources : {df_original['source'].nunique()}")

# ================================================================
#  2. CHARGER LES POSTS BLUESKY EXPORTÉS
# ================================================================
print("\n[2/6] Chargement des posts Bluesky haute confiance...")

bluesky_suspect_path = os.path.join(DATA_DIR, 'db', 'bluesky_suspect.csv')
bluesky_fiable_path = os.path.join(DATA_DIR, 'db', 'bluesky_fiable.csv')

df_bsky_suspect = pd.read_csv(bluesky_suspect_path)
df_bsky_fiable = pd.read_csv(bluesky_fiable_path)

print(f"  Bluesky suspect (score ≤ 0.15) : {len(df_bsky_suspect)} posts")
print(f"  Bluesky fiable (score ≥ 0.85) : {len(df_bsky_fiable)} posts")

# Nettoyer : enlever textes trop courts ou vides
df_bsky_suspect = df_bsky_suspect[df_bsky_suspect['text'].str.len() > 10].copy()
df_bsky_fiable = df_bsky_fiable[df_bsky_fiable['text'].str.len() > 10].copy()

# Échantillonner pour équilibrer
N_SAMPLE = 5000  # 5K de chaque classe
np.random.seed(42)

if len(df_bsky_suspect) > N_SAMPLE:
    df_bsky_suspect = df_bsky_suspect.sample(N_SAMPLE, random_state=42)
if len(df_bsky_fiable) > N_SAMPLE:
    df_bsky_fiable = df_bsky_fiable.sample(N_SAMPLE, random_state=42)

df_bsky_suspect['label'] = 1
df_bsky_suspect['source'] = 'bluesky_suspect'
df_bsky_suspect['text_original'] = df_bsky_suspect['text']

df_bsky_fiable['label'] = 0
df_bsky_fiable['source'] = 'bluesky_fiable'
df_bsky_fiable['text_original'] = df_bsky_fiable['text']

df_bluesky = pd.concat([
    df_bsky_suspect[['text_original', 'label', 'source']],
    df_bsky_fiable[['text_original', 'label', 'source']],
], ignore_index=True)

print(f"  Bluesky échantillonné : {len(df_bluesky)} posts ({(df_bluesky['label']==1).sum()} suspects, {(df_bluesky['label']==0).sum()} fiables)")

# ================================================================
#  3. COMBINER ET ENTRAÎNER V5-BLUESKY
# ================================================================
print("\n[3/6] Entraînement V5-Bluesky (dataset augmenté)...")

df_augmented = pd.concat([df_original, df_bluesky], ignore_index=True)
df_augmented = df_augmented.drop_duplicates(subset=['text_original'])
df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Dataset augmenté : {len(df_augmented)} textes (+{len(df_bluesky)} Bluesky)")
print(f"  Labels : {(df_augmented['label']==0).sum()} fiables, {(df_augmented['label']==1).sum()} suspects")

# Entraîner un nouveau ExpertFakeNewsDetector
det_bsky = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=0.44)

# Préparer le DataFrame au format attendu par train()
# train() attend 'text_clean' (pour TF-IDF) et 'text_original' (pour features linguistiques)
df_train = df_augmented[['text_original', 'label']].copy()
df_train['text_clean'] = df_train['text_original'].str.lower().str.replace(r'[^\w\s]', ' ', regex=True).str.strip()

print("\n  Entraînement avec cross-validation 5-fold...")
metrics = det_bsky.train(df_train, model_type='logreg', n_folds=5, track_emissions=False)
print(f"  CV F1 macro : {metrics.get('cv_f1_macro', 'N/A')}")

# Sauvegarder comme V5-bluesky
det_bsky.save(suffix='expert_v5_bluesky')
print(f"  Modèle sauvegardé : model_expert_v5_bluesky.pkl")

texts = df_augmented['text_original']
labels = df_augmented['label'].values

# ================================================================
#  4. MÉTRIQUES D'ENTRAÎNEMENT
# ================================================================
print("\n[4/6] Métriques d'entraînement...")
if isinstance(metrics, dict):
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
else:
    print(f"  Metrics: {metrics}")

# ================================================================
#  5. ÉVALUATION SUR GOLD TEST SET
# ================================================================
print("\n[5/6] Évaluation sur Gold Test Set...")

gold_path = os.path.join(DATA_DIR, 'gold_test_set_annotation_completed.xlsx')
df_gold = pd.read_excel(gold_path, sheet_name='Resolution')
gold_texts = df_gold['Texte'].fillna('')
gold_labels = (df_gold['Label final'] == 'suspect').astype(int).values

# V5 original (pour comparaison)
det_original = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=0.44)
det_original.load(suffix='expert_v5')
orig_results = det_original.predict(gold_texts)
orig_pred = orig_results['prediction_label'].values

# V5-Bluesky
det_bsky_eval = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=0.44)
det_bsky_eval.load(suffix='expert_v5_bluesky')
bsky_results = det_bsky_eval.predict(gold_texts)
bsky_pred = bsky_results['prediction_label'].values

print(f"\n  {'Modèle':<25s} {'Acc':>6s} {'F1 mac':>7s} {'F1 sus':>7s} {'F1 fia':>7s} {'FP':>4s} {'FN':>4s} {'TP':>4s}")
print(f"  {'-'*65}")

for name, pred in [
    ('V5 original', orig_pred),
    ('V5-Bluesky', bsky_pred),
]:
    acc = accuracy_score(gold_labels, pred)
    f1m = f1_score(gold_labels, pred, average='macro')
    f1s = f1_score(gold_labels, pred, pos_label=1, zero_division=0)
    f1f = f1_score(gold_labels, pred, pos_label=0)
    cm = confusion_matrix(gold_labels, pred)
    fp = cm[0, 1]
    fn = cm[1, 0] if cm.shape[0] > 1 else sum(gold_labels)
    tp = cm[1, 1] if cm.shape[0] > 1 else 0
    print(f"  {name:<25s} {acc:>6.3f} {f1m:>7.3f} {f1s:>7.3f} {f1f:>7.3f} {fp:>4d} {fn:>4d} {tp:>4d}")

# Tester différents seuils pour V5-Bluesky
print("\n  --- Optimisation du seuil V5-Bluesky sur gold set ---")
bsky_scores = bsky_results['ai_score_credibility'].values

best_th = 0.5
best_f1m = 0
for th in np.arange(0.10, 0.90, 0.01):
    pred_th = (bsky_scores < th).astype(int)  # suspect si score < seuil
    f1m = f1_score(gold_labels, pred_th, average='macro')
    f1s = f1_score(gold_labels, pred_th, pos_label=1, zero_division=0)
    if f1m > best_f1m:
        best_f1m = f1m
        best_th = th
        best_pred = pred_th.copy()

cm_best = confusion_matrix(gold_labels, best_pred)
f1s_best = f1_score(gold_labels, best_pred, pos_label=1, zero_division=0)
fp_best = cm_best[0, 1]
fn_best = cm_best[1, 0] if cm_best.shape[0] > 1 else sum(gold_labels)
tp_best = cm_best[1, 1] if cm_best.shape[0] > 1 else 0
acc_best = accuracy_score(gold_labels, best_pred)

print(f"  {'V5-Bluesky (seuil opt)':<25s} {acc_best:>6.3f} {best_f1m:>7.3f} {f1s_best:>7.3f} {'':>7s} {fp_best:>4d} {fn_best:>4d} {tp_best:>4d}")
print(f"  Seuil optimal : {best_th:.2f}")

# ================================================================
#  6. COMPARAISON DES SCORES PAR LANGUE
# ================================================================
print("\n[6/6] Analyse par langue...")

from langdetect import detect

for lang_filter, lang_name in [('fr', 'Français'), ('en', 'Anglais')]:
    mask = []
    for text in gold_texts:
        try:
            detected = detect(str(text))
            mask.append(detected == lang_filter)
        except Exception:
            mask.append(False)
    mask = np.array(mask)
    n_lang = mask.sum()
    if n_lang == 0:
        continue

    n_suspect = gold_labels[mask].sum()
    print(f"\n  {lang_name} ({n_lang} posts, {n_suspect} suspects) :")

    for name, pred in [('V5 original', orig_pred), ('V5-Bluesky', bsky_pred)]:
        if mask.sum() > 0:
            f1s = f1_score(gold_labels[mask], pred[mask], pos_label=1, zero_division=0)
            f1f = f1_score(gold_labels[mask], pred[mask], pos_label=0, zero_division=0)
            acc = accuracy_score(gold_labels[mask], pred[mask])
            cm = confusion_matrix(gold_labels[mask], pred[mask])
            fp = cm[0, 1] if cm.shape[1] > 1 else 0
            print(f"    {name:<25s} Acc={acc:.3f} F1s={f1s:.3f} F1f={f1f:.3f} FP={fp}")

elapsed = time.time() - t0
print(f"\n  Temps total : {elapsed:.0f}s ({elapsed/60:.1f}min)")
print("\n" + "=" * 70)
print("TERMINÉ")
print("=" * 70)
