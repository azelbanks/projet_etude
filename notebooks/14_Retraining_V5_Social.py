#!/usr/bin/env python3
"""
14 — Retraining V5 : Integration donnees FR sociales
=====================================================

Contexte :
    V4 atteint F1=0.868 sur les textes FR ultra-courts (<15 mots), mais
    les donnees d'entrainement ne contiennent aucun post social FR natif.
    Le dataset synthetique de 10K posts FR (5K suspect + 5K fiable) genere
    par generate_fr_social_dataset.py comble ce manque.

Ameliorations V5 vs V4 :
    1. Integration de 10 000 posts FR sociaux synthetiques (fr_social_path)
    2. Dataset total : ~198 000 textes (FR=~86K / 43%, EN=~112K / 57%)
    3. Meilleure couverture des formulations social media FR

Datasets :
    - ISOT (True.csv + Fake.csv)
    - Kaggle FR (kaggle_fr/) x5 oversample
    - Kaggle FR short augmentation x3
    - FakeNewsNet (fakenewsnet/)
    - CONSTRAINT (constraint/)
    - Credibility Corpus (credibility_corpus/)
    - FR Social Media Synthetique (fr_social_media_synthetic.csv) — NOUVEAU V5

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
#  1. CHARGEMENT — DATASET V5 AVEC FR SOCIAL
# ============================================================
print("=" * 70)
print("RETRAINING V5 — Integration donnees FR sociales")
print("=" * 70)

t0 = time.time()

print("\n[1/7] Chargement du dataset bilingue V5 (+ 10K FR social)...")

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
    fr_social_path=os.path.join(DATA, 'fr_social_media_synthetic.csv'),  # V5
)

print(f"  Dataset total : {len(df_v5)} textes")
n_fr = sum(df_v5.language == 'fr')
n_en = sum(df_v5.language == 'en')
print(f"  EN={n_en}, FR={n_fr}")
print(f"  FR ratio : {n_fr / len(df_v5) * 100:.1f}%")
print(f"  Labels : {df_v5.label.value_counts().to_dict()}")
print(f"  Longueur moyenne : {df_v5.text_clean.str.split().str.len().mean():.1f} mots")

# Stats detaillees par langue et longueur
df_v5['word_count'] = df_v5['text_clean'].str.split().str.len()
for lang in ['fr', 'en']:
    subset = df_v5[df_v5.language == lang]
    short = subset[subset.word_count < 15]
    medium = subset[(subset.word_count >= 15) & (subset.word_count < 30)]
    long_ = subset[subset.word_count >= 30]
    print(f"\n  {lang.upper()} : {len(subset)} total | "
          f"<15 mots={len(short)} | 15-30={len(medium)} | >30={len(long_)}")
    print(f"    Labels suspect <15 mots : {short.label.sum()}/{len(short)}")

assert 'text_original' in df_v5.columns, "Colonne text_original manquante !"
print(f"\n  text_original present : oui")

# ============================================================
#  2. SPLIT TRAIN/TEST
# ============================================================
print("\n[2/7] Split train/test 80/20 stratifie...")

df_train, df_test = train_test_split(
    df_v5, test_size=0.2, stratify=df_v5['label'], random_state=42
)
print(f"  Train : {len(df_train)} | Test : {len(df_test)}")

# ============================================================
#  3. ENTRAINEMENT V5
# ============================================================
print("\n[3/7] Entrainement V5 (LogReg, 5-fold CV bilingue)...")
print("  Parametres : max_features=30000, ngram_range=(1,3), C=1.0, max_iter=10000")
print("  Features : TF-IDF + 15 linguistiques")
print("  Cela peut prendre plusieurs minutes...")

detector_v5 = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=0.44)
metrics_v5 = detector_v5.train(
    df_train,
    model_type='logreg',
    n_folds=5,
    track_emissions=True,
    emissions_dir=_proj,
)

print("\n  Metriques CV V5 :")
for k, v in sorted(metrics_v5.items()):
    if isinstance(v, float):
        print(f"    {k}: {v:.4f}")

# ============================================================
#  4. EVALUATION HOLDOUT GLOBALE
# ============================================================
print("\n[4/7] Evaluation sur le holdout test (20%)...")

eval_results = detector_v5.evaluate_holdout(df_test)

print(f"\n  Holdout Results V5:")
print(f"    Accuracy  : {eval_results['accuracy']:.4f}")
print(f"    F1        : {eval_results['f1']:.4f}")
print(f"    Precision : {eval_results['precision']:.4f}")
print(f"    Recall    : {eval_results['recall']:.4f}")
if 'roc_auc' in eval_results:
    print(f"    ROC AUC   : {eval_results['roc_auc']:.4f}")

print(f"\n  Classification Report V5 (holdout):")
print(eval_results['report_str'])

# ============================================================
#  5. EVALUATION DETAILLEE PAR LANGUE ET LONGUEUR
# ============================================================
print("\n[5/7] Evaluation detaillee par langue et longueur...")

X_test_clean = df_test['text_clean'].values
X_test_orig = df_test['text_original'].values if 'text_original' in df_test.columns else None
y_test = df_test['label'].values

X_feat = detector_v5._build_features(X_test_clean, texts_original=X_test_orig, fit=False)
y_pred = detector_v5.model.predict(X_feat)
if hasattr(detector_v5.model, 'predict_proba'):
    y_proba = detector_v5.model.predict_proba(X_feat)[:, 1]
else:
    y_proba = np.zeros(len(y_test))

df_test_eval = df_test.copy()
df_test_eval['y_pred'] = y_pred
df_test_eval['y_proba'] = y_proba
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

print(f"\n  {'Langue':<5} {'Longueur':<20} {'N':>6} {'Accuracy':>9} {'F1':>7} {'Precision':>10} {'Recall':>7}")
print(f"  {'-'*65}")

for lang in ['fr', 'en']:
    for cat in ['ultra_court (<15)', 'court (15-30)', 'moyen (30-100)', 'long (100-300)', 'tres_long (>300)']:
        mask = (df_test_eval['language'] == lang) & (df_test_eval['length_cat'] == cat)
        sub = df_test_eval[mask]
        if len(sub) < 10:
            continue
        yt = sub['label'].values
        yp = sub['y_pred'].values
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, zero_division=0)
        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        print(f"  {lang.upper():<5} {cat:<20} {len(sub):>6} {acc:>9.4f} {f1:>7.4f} {prec:>10.4f} {rec:>7.4f}")

# Resume par langue
print(f"\n  {'Langue':<5} {'N':>6} {'Accuracy':>9} {'F1':>7} {'Precision':>10} {'Recall':>7}")
print(f"  {'-'*50}")
for lang in ['fr', 'en']:
    mask = df_test_eval['language'] == lang
    sub = df_test_eval[mask]
    if len(sub) < 10:
        continue
    yt = sub['label'].values
    yp = sub['y_pred'].values
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, zero_division=0)
    prec = precision_score(yt, yp, zero_division=0)
    rec = recall_score(yt, yp, zero_division=0)
    print(f"  {lang.upper():<5} {len(sub):>6} {acc:>9.4f} {f1:>7.4f} {prec:>10.4f} {rec:>7.4f}")

# ============================================================
#  6. SAUVEGARDE V5
# ============================================================
print("\n[6/7] Sauvegarde du modele V5...")

detector_v5.save('expert_v5')
print(f"  Modele sauvegarde dans {MODEL_DIR} (suffix=expert_v5)")

# ============================================================
#  7. HEALTH CHECK + COMPARAISON V4 vs V5
# ============================================================
print("\n[7/7] Health check V5 et comparaison avec V4...")

hc_v5 = detector_v5.health_check()
print(f"\n  Health check V5 : {'PASS' if hc_v5['healthy'] else 'FAIL'}")
for detail in hc_v5['details']:
    status = "OK" if detail['passed'] else "FAIL"
    print(f"    [{status}] '{detail['text'][:60]}...' "
          f"label={detail['predicted_label']} (attendu={detail['expected_label']}) "
          f"score={detail['score']:.4f} (range={detail['expected_range']})")

# Comparaison V4 vs V5
print("\n  --- Comparaison V4 vs V5 ---")
try:
    detector_v4 = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=0.44)
    detector_v4.load('expert_v4')

    # V4 predictions on same test set
    X_v4_feat = detector_v4._build_features(X_test_clean, texts_original=X_test_orig, fit=False)
    y_v4_pred = detector_v4.model.predict(X_v4_feat)
    df_test_eval['y_pred_v4'] = y_v4_pred

    # Global comparison
    print(f"\n  {'Metrique':<20} {'V4':>10} {'V5':>10} {'Delta':>10}")
    print(f"  {'-'*50}")

    for lang_label, lang_mask in [('GLOBAL', pd.Series(True, index=df_test_eval.index)),
                                   ('FR', df_test_eval['language'] == 'fr'),
                                   ('EN', df_test_eval['language'] == 'en')]:
        sub = df_test_eval[lang_mask]
        yt = sub['label'].values
        f1_v4 = f1_score(yt, sub['y_pred_v4'].values, zero_division=0)
        f1_v5 = f1_score(yt, sub['y_pred'].values, zero_division=0)
        delta = f1_v5 - f1_v4
        sign = '+' if delta >= 0 else ''
        print(f"  F1 {lang_label:<16} {f1_v4:>10.4f} {f1_v5:>10.4f} {sign}{delta:>9.4f}")

    # Focus FR court V4 vs V5
    print("\n  --- Focus FR court : V4 vs V5 ---")
    for cat in ['ultra_court (<15)', 'court (15-30)']:
        mask_fr = (df_test_eval['language'] == 'fr') & (df_test_eval['length_cat'] == cat)
        sub = df_test_eval[mask_fr]
        if len(sub) < 10:
            continue
        yt = sub['label'].values
        f1_v4 = f1_score(yt, sub['y_pred_v4'].values, zero_division=0)
        f1_v5 = f1_score(yt, sub['y_pred'].values, zero_division=0)
        delta = f1_v5 - f1_v4
        sign = '+' if delta >= 0 else ''
        print(f"  FR {cat}: F1 V4={f1_v4:.4f} -> V5={f1_v5:.4f} ({sign}{delta:.4f})")

    # Focus EN court V4 vs V5
    print("\n  --- Focus EN court : V4 vs V5 ---")
    for cat in ['ultra_court (<15)', 'court (15-30)']:
        mask_en = (df_test_eval['language'] == 'en') & (df_test_eval['length_cat'] == cat)
        sub = df_test_eval[mask_en]
        if len(sub) < 10:
            continue
        yt = sub['label'].values
        f1_v4 = f1_score(yt, sub['y_pred_v4'].values, zero_division=0)
        f1_v5 = f1_score(yt, sub['y_pred'].values, zero_division=0)
        delta = f1_v5 - f1_v4
        sign = '+' if delta >= 0 else ''
        print(f"  EN {cat}: F1 V4={f1_v4:.4f} -> V5={f1_v5:.4f} ({sign}{delta:.4f})")

except Exception as e:
    print(f"  V4 non disponible pour comparaison : {e}")

# Test final bilingue
print("\n  --- Test final bilingue V5 ---")
test_texts = [
    # FR social media (cible V5)
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

results_v5 = detector_v5.predict(pd.Series([t[0] for t in test_texts]))
print(f"\n  {'Texte':<55} {'Lang':>4} {'Score':>6} {'Label':>8} {'Attendu':>8} {'OK':>3}")
print(f"  {'-'*90}")

correct = 0
for i, (text, lang, expected) in enumerate(test_texts):
    score = float(results_v5['ai_score_credibility'].iloc[i])
    label = "suspect" if score < 0.5 else "fiable"
    ok = "OK" if label == expected else "FAIL"
    if ok == "OK":
        correct += 1
    print(f"  {text[:55]:<55} {lang:>4} {score:>6.3f} {label:>8} {expected:>8} {ok:>3}")

print(f"\n  Score test bilingue V5 : {correct}/{len(test_texts)}")

elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"Retraining V5 termine en {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'=' * 70}")
