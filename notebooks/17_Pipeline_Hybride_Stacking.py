#!/usr/bin/env python3
"""
17 — Pipeline Hybride Stacking : V5 TF-IDF + CamemBERT V2
==========================================================

Contexte :
    Preconisation P1 — Les deux modeles (V5 TF-IDF bilingue + CamemBERT V2 FR)
    ont des forces complementaires :
    - V5 TF-IDF : patterns explicites (MAJUSCULES, mots sensationnalistes, TF-IDF)
    - CamemBERT V2 : comprehension semantique (ironie, sous-entendus, contexte)

    Le stacking combine les scores de confiance des deux modeles via un
    meta-learner (LogisticRegression) pour obtenir une prediction finale
    superieure a chaque modele individuel.

Architecture :
    Pour les textes FR :
        score_tfidf  = V5_ExpertDetector.predict(text)  → [0, 1]
        score_camem  = CamemBERT_V2.predict(text)       → [0, 1]
        X_meta = [score_tfidf, score_camem, language_features...]
        y_meta = MetaLearner.predict(X_meta)

    Pour les textes EN :
        Le CamemBERT est FR-only, donc on utilise V5 seul pour EN.
        Un flag is_fr est ajoute pour que le meta-learner apprenne
        a ponderer differemment selon la langue.

Auteur : Thumalien Team
"""

import sys
import os
import time
import logging
import numpy as np
import pandas as pd
import joblib

# --- Setup paths ---
_proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_proj, 'src'))

from pipeline.expert_detector import (
    DatasetCleaner,
    ExpertFakeNewsDetector,
)
from pipeline.camembert_classifier import CamemBERTClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
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
#  1. CHARGEMENT DU DATASET V5 (meme split que notebooks 14/15)
# ============================================================
print("=" * 70)
print("PIPELINE HYBRIDE P1 — Stacking V5 TF-IDF + CamemBERT V2")
print("=" * 70)

t0 = time.time()

print("\n[1/8] Chargement du dataset V5...")

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

# Meme split que les autres notebooks
df_train, df_test = train_test_split(
    df_v5, test_size=0.2, stratify=df_v5['label'], random_state=42
)
print(f"  Train : {len(df_train)} | Test : {len(df_test)}")

# ============================================================
#  2. CHARGEMENT DES MODELES
# ============================================================
print("\n[2/8] Chargement des modeles V5 et CamemBERT V2...")

# V5 TF-IDF
detector_v5 = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=0.44)
detector_v5.load('expert_v5')
print("  V5 TF-IDF charge")

# CamemBERT V2
camembert = CamemBERTClassifier(model_dir=MODEL_DIR)
camembert_loaded = camembert.load(suffix='camembert_fr_v2')
if camembert_loaded:
    print("  CamemBERT V2 charge")
else:
    print("  ERREUR: CamemBERT V2 non disponible")
    sys.exit(1)

# ============================================================
#  3. GENERATION DES SCORES BASE LEARNERS (TRAIN)
# ============================================================
print("\n[3/8] Generation des scores base learners sur le train set...")
print("  Cela peut prendre quelques minutes (CamemBERT inference)...")

# --- V5 TF-IDF scores ---
X_train_clean = df_train['text_clean'].values
X_train_orig = df_train['text_original'].values
y_train = df_train['label'].values

X_feat_train = detector_v5._build_features(X_train_clean, texts_original=X_train_orig, fit=False)
if hasattr(detector_v5.model, 'predict_proba'):
    v5_train_scores = detector_v5.model.predict_proba(X_feat_train)[:, 1]
else:
    v5_train_scores = detector_v5.model.decision_function(X_feat_train)
# Note: predict_proba[:, 1] = P(label=1=FAKE), on veut P(FIABLE) = 1 - P(FAKE)
# En fait, dans expert_detector, ai_score_credibility = proba[:,0] (class 0 = VRAI)
# Mais le modele sklearn sort proba[:,1] pour la classe 1 (FAKE)
# Pour le stacking on utilise les scores bruts, la convention n'importe pas
# tant qu'on est coherent. Utilisons P(FIABLE) = proba[:,0]
v5_train_scores = detector_v5.model.predict_proba(X_feat_train)[:, 0]

print(f"  V5 scores train : min={v5_train_scores.min():.4f}, max={v5_train_scores.max():.4f}, mean={v5_train_scores.mean():.4f}")

# --- CamemBERT V2 scores (FR only) ---
# On predit sur tous les textes train, mais CamemBERT est FR-only
# Pour les textes EN, on met un score neutre (0.5) qui sera pondere
# par le meta-learner via le flag is_fr
train_langs = df_train['language'].values
fr_mask_train = train_langs == 'fr'
fr_indices_train = np.where(fr_mask_train)[0]

camembert_train_scores = np.full(len(df_train), 0.5)  # default pour EN

if len(fr_indices_train) > 0:
    fr_texts_train = X_train_clean[fr_indices_train].tolist()
    # Batch prediction CamemBERT
    batch_size = 500
    fr_scores = []
    for i in range(0, len(fr_texts_train), batch_size):
        batch = fr_texts_train[i:i+batch_size]
        batch_scores = camembert.predict_credibility_scores(batch)
        fr_scores.extend(batch_scores)
        if (i // batch_size) % 10 == 0:
            print(f"    CamemBERT batch {i//batch_size + 1}/{len(fr_texts_train)//batch_size + 1}")
    camembert_train_scores[fr_indices_train] = np.array(fr_scores)

print(f"  CamemBERT scores train (FR) : min={camembert_train_scores[fr_mask_train].min():.4f}, "
      f"max={camembert_train_scores[fr_mask_train].max():.4f}, "
      f"mean={camembert_train_scores[fr_mask_train].mean():.4f}")

# ============================================================
#  4. CONSTRUCTION DES FEATURES META
# ============================================================
print("\n[4/8] Construction des features meta...")

# Features pour le meta-learner :
# 1. V5 TF-IDF score (P(FIABLE))
# 2. CamemBERT score (P(FIABLE)) — 0.5 pour EN
# 3. is_fr (1/0) — permet au meta-learner de ponderer CamemBERT uniquement pour FR
# 4. score_diff = V5 - CamemBERT — signal de desaccord entre modeles
# 5. score_min = min(V5, CamemBERT) — cas ou les deux sont suspects
# 6. score_max = max(V5, CamemBERT) — cas ou au moins un est confiant

is_fr_train = fr_mask_train.astype(float)
score_diff_train = v5_train_scores - camembert_train_scores
score_min_train = np.minimum(v5_train_scores, camembert_train_scores)
score_max_train = np.maximum(v5_train_scores, camembert_train_scores)

X_meta_train = np.column_stack([
    v5_train_scores,
    camembert_train_scores,
    is_fr_train,
    score_diff_train,
    score_min_train,
    score_max_train,
])

feature_names = ['v5_score', 'camembert_score', 'is_fr', 'score_diff', 'score_min', 'score_max']

# Labels : 0 = VRAI, 1 = FAKE (inversion pour le meta-learner)
# On garde la meme convention que y_train
y_meta_train = y_train

print(f"  X_meta shape : {X_meta_train.shape}")
print(f"  Features : {feature_names}")
print(f"  Labels : 0={sum(y_meta_train == 0)}, 1={sum(y_meta_train == 1)}")

# ============================================================
#  5. ENTRAINEMENT META-LEARNER
# ============================================================
print("\n[5/8] Entrainement du meta-learner (LogReg, 5-fold CV)...")

meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

# Cross-validation sur le train
cv_f1 = cross_val_score(meta_model, X_meta_train, y_meta_train, cv=5, scoring='f1')
cv_acc = cross_val_score(meta_model, X_meta_train, y_meta_train, cv=5, scoring='accuracy')
cv_prec = cross_val_score(meta_model, X_meta_train, y_meta_train, cv=5, scoring='precision')
cv_rec = cross_val_score(meta_model, X_meta_train, y_meta_train, cv=5, scoring='recall')

print(f"  CV F1       : {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
print(f"  CV Accuracy : {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
print(f"  CV Precision: {cv_prec.mean():.4f}")
print(f"  CV Recall   : {cv_rec.mean():.4f}")

# Fit final sur tout le train
meta_model.fit(X_meta_train, y_meta_train)

# Coefficients du meta-learner
print(f"\n  Coefficients du meta-learner :")
for name, coef in zip(feature_names, meta_model.coef_[0]):
    print(f"    {name:>20} : {coef:+.4f}")
print(f"    {'intercept':>20} : {meta_model.intercept_[0]:+.4f}")

# ============================================================
#  6. GENERATION DES SCORES BASE LEARNERS (TEST)
# ============================================================
print("\n[6/8] Generation des scores base learners sur le test set...")

X_test_clean = df_test['text_clean'].values
X_test_orig = df_test['text_original'].values
y_test = df_test['label'].values
test_langs = df_test['language'].values

# V5 scores test
X_feat_test = detector_v5._build_features(X_test_clean, texts_original=X_test_orig, fit=False)
v5_test_scores = detector_v5.model.predict_proba(X_feat_test)[:, 0]

# CamemBERT scores test (FR only)
fr_mask_test = test_langs == 'fr'
fr_indices_test = np.where(fr_mask_test)[0]
camembert_test_scores = np.full(len(df_test), 0.5)

if len(fr_indices_test) > 0:
    fr_texts_test = X_test_clean[fr_indices_test].tolist()
    fr_scores_test = []
    for i in range(0, len(fr_texts_test), batch_size):
        batch = fr_texts_test[i:i+batch_size]
        batch_scores = camembert.predict_credibility_scores(batch)
        fr_scores_test.extend(batch_scores)
        if (i // batch_size) % 10 == 0:
            print(f"    CamemBERT test batch {i//batch_size + 1}/{len(fr_texts_test)//batch_size + 1}")
    camembert_test_scores[fr_indices_test] = np.array(fr_scores_test)

# Features meta test
is_fr_test = fr_mask_test.astype(float)
score_diff_test = v5_test_scores - camembert_test_scores
score_min_test = np.minimum(v5_test_scores, camembert_test_scores)
score_max_test = np.maximum(v5_test_scores, camembert_test_scores)

X_meta_test = np.column_stack([
    v5_test_scores,
    camembert_test_scores,
    is_fr_test,
    score_diff_test,
    score_min_test,
    score_max_test,
])

# ============================================================
#  7. EVALUATION HOLDOUT
# ============================================================
print("\n[7/8] Evaluation sur le holdout test...")

y_pred_meta = meta_model.predict(X_meta_test)
y_proba_meta = meta_model.predict_proba(X_meta_test)[:, 1]

# --- Metriques globales ---
acc = accuracy_score(y_test, y_pred_meta)
f1 = f1_score(y_test, y_pred_meta, zero_division=0)
prec = precision_score(y_test, y_pred_meta, zero_division=0)
rec = recall_score(y_test, y_pred_meta, zero_division=0)

print(f"\n  Metriques globales HYBRIDE :")
print(f"    Accuracy  : {acc:.4f}")
print(f"    F1        : {f1:.4f}")
print(f"    Precision : {prec:.4f}")
print(f"    Recall    : {rec:.4f}")

# --- V5-only baseline pour comparaison ---
v5_pred_test = detector_v5.model.predict(X_feat_test)
v5_f1 = f1_score(y_test, v5_pred_test, zero_division=0)
v5_acc = accuracy_score(y_test, v5_pred_test)

print(f"\n  Comparaison V5-only vs HYBRIDE :")
print(f"    {'Metrique':<20} {'V5 seul':>10} {'Hybride':>10} {'Delta':>10}")
print(f"    {'-'*50}")
print(f"    {'F1':<20} {v5_f1:>10.4f} {f1:>10.4f} {f1-v5_f1:>+10.4f}")
print(f"    {'Accuracy':<20} {v5_acc:>10.4f} {acc:>10.4f} {acc-v5_acc:>+10.4f}")

# --- Par langue ---
df_test_eval = df_test.copy()
df_test_eval['y_pred_hybrid'] = y_pred_meta
df_test_eval['y_pred_v5'] = v5_pred_test
df_test_eval['word_count'] = df_test_eval['text_clean'].str.split().str.len()

def length_cat(wc):
    if wc < 15:
        return 'ultra_court (<15)'
    elif wc < 30:
        return 'court (15-30)'
    elif wc < 100:
        return 'moyen (30-100)'
    elif wc < 300:
        return 'long (100-300)'
    else:
        return 'tres_long (>300)'

df_test_eval['length_cat'] = df_test_eval['word_count'].apply(length_cat)

print(f"\n  Comparaison V5 vs HYBRIDE par langue et longueur :")
print(f"  {'Langue':<5} {'Longueur':<20} {'N':>6} {'F1 V5':>8} {'F1 Hyb':>8} {'Delta':>8}")
print(f"  {'-'*60}")

for lang in ['fr', 'en']:
    for cat in ['ultra_court (<15)', 'court (15-30)', 'moyen (30-100)', 'long (100-300)', 'tres_long (>300)']:
        mask = (df_test_eval['language'] == lang) & (df_test_eval['length_cat'] == cat)
        sub = df_test_eval[mask]
        if len(sub) < 10:
            continue
        yt = sub['label'].values
        f1_v5_seg = f1_score(yt, sub['y_pred_v5'].values, zero_division=0)
        f1_hyb_seg = f1_score(yt, sub['y_pred_hybrid'].values, zero_division=0)
        delta = f1_hyb_seg - f1_v5_seg
        sign = '+' if delta >= 0 else ''
        print(f"  {lang.upper():<5} {cat:<20} {len(sub):>6} {f1_v5_seg:>8.4f} {f1_hyb_seg:>8.4f} {sign}{delta:>7.4f}")

# Resume par langue
print(f"\n  {'Langue':<5} {'N':>6} {'F1 V5':>8} {'F1 Hyb':>8} {'Delta':>8}")
print(f"  {'-'*40}")
for lang in ['fr', 'en']:
    mask = df_test_eval['language'] == lang
    sub = df_test_eval[mask]
    yt = sub['label'].values
    f1_v5_lang = f1_score(yt, sub['y_pred_v5'].values, zero_division=0)
    f1_hyb_lang = f1_score(yt, sub['y_pred_hybrid'].values, zero_division=0)
    delta = f1_hyb_lang - f1_v5_lang
    sign = '+' if delta >= 0 else ''
    print(f"  {lang.upper():<5} {len(sub):>6} {f1_v5_lang:>8.4f} {f1_hyb_lang:>8.4f} {sign}{delta:>7.4f}")

# ============================================================
#  8. SAUVEGARDE + TEST FINAL
# ============================================================
print("\n[8/8] Sauvegarde et test final bilingue...")

# Sauvegarder le meta-learner
meta_path = os.path.join(MODEL_DIR, 'hybrid_meta_learner.joblib')
joblib.dump({
    'meta_model': meta_model,
    'feature_names': feature_names,
    'cv_f1_mean': cv_f1.mean(),
    'cv_f1_std': cv_f1.std(),
}, meta_path)
print(f"  Meta-learner sauvegarde : {meta_path}")

# --- Test final bilingue ---
print("\n  --- Test final bilingue HYBRIDE ---")
test_texts = [
    # FR social media
    ("SCANDALE !! On nous cache la verite sur les vaccins !", "fr", "suspect"),
    ("Le CNRS a publie une etude sur le changement climatique.", "fr", "fiable"),
    ("URGENT: les vaccins contiennent des puces 5G !", "fr", "suspect"),
    ("La mairie annonce la renovation du pont.", "fr", "fiable"),
    ("Partagez massivement avant censure !! Info cachee", "fr", "suspect"),
    ("Les resultats du match : Lyon 2 - Marseille 1", "fr", "fiable"),
    ("REVEILLEZ VOUS !! Le graphene dans les masques !!", "fr", "suspect"),
    ("La meteo prevoit du soleil ce weekend.", "fr", "fiable"),
    # EN
    ("BREAKING: Government EXPOSED in massive cover-up!", "en", "suspect"),
    ("A new study published in Nature examines climate change.", "en", "fiable"),
    ("SHARE before they DELETE this!! The truth about 5G!", "en", "suspect"),
    ("The city council approved the new budget.", "en", "fiable"),
]

texts_raw = [t[0] for t in test_texts]
texts_langs = [t[1] for t in test_texts]

# V5 scores
v5_results = detector_v5.predict(pd.Series(texts_raw))
v5_scores_final = v5_results['ai_score_credibility'].values

# CamemBERT scores (FR only)
camembert_scores_final = np.full(len(texts_raw), 0.5)
fr_idx = [i for i, lang in enumerate(texts_langs) if lang == 'fr']
if fr_idx:
    fr_texts = [texts_raw[i] for i in fr_idx]
    fr_cam_scores = camembert.predict_credibility_scores(fr_texts)
    for j, idx in enumerate(fr_idx):
        camembert_scores_final[idx] = fr_cam_scores[j]

# Features meta
is_fr_final = np.array([1.0 if lang == 'fr' else 0.0 for lang in texts_langs])
diff_final = v5_scores_final - camembert_scores_final
min_final = np.minimum(v5_scores_final, camembert_scores_final)
max_final = np.maximum(v5_scores_final, camembert_scores_final)

X_meta_final = np.column_stack([
    v5_scores_final,
    camembert_scores_final,
    is_fr_final,
    diff_final,
    min_final,
    max_final,
])

# Prediction hybride
y_proba_final = meta_model.predict_proba(X_meta_final)[:, 0]  # P(VRAI) = P(class 0)

print(f"\n  {'Texte':<55} {'Lang':>4} {'V5':>6} {'Cam':>6} {'Hyb':>6} {'Label':>8} {'Att':>8} {'OK':>3}")
print(f"  {'-'*100}")

correct = 0
for i, (text, lang, expected) in enumerate(test_texts):
    v5_s = float(v5_scores_final[i])
    cam_s = float(camembert_scores_final[i])
    hyb_s = float(y_proba_final[i])
    label = "fiable" if hyb_s >= 0.5 else "suspect"
    ok = "OK" if label == expected else "FAIL"
    if ok == "OK":
        correct += 1
    print(f"  {text[:55]:<55} {lang:>4} {v5_s:>6.3f} {cam_s:>6.3f} {hyb_s:>6.3f} {label:>8} {expected:>8} {ok:>3}")

print(f"\n  Score test bilingue HYBRIDE : {correct}/{len(test_texts)}")

# --- Comparaison finale ---
print(f"\n  === BILAN COMPARATIF ===")
print(f"  V5 TF-IDF seul      : F1={v5_f1:.4f}, test bilingue=12/12")
print(f"  CamemBERT V2 seul   : F1=0.9661 (FR only), test=9/10")
print(f"  HYBRIDE (stacking)  : F1={f1:.4f}, test bilingue={correct}/{len(test_texts)}")

elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"Pipeline Hybride P1 termine en {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'=' * 70}")
