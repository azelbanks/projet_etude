#!/usr/bin/env python3
"""
12 — Retraining V4 : Amelioration performance FR court
=======================================================

Contexte :
    V3 corrigeait le bug des features linguistiques mais le modele
    reste faible sur les textes courts en francais (F1=0.65 pour <15 mots).
    C'est le cas d'usage principal de Bluesky.

Ameliorations V4 :
    1. Augmentation FR courte : extraction de premieres phrases et titres
       synthetiques depuis les articles KaggleFR
    2. french_oversample augmente de 3 a 5
    3. fr_short_augment active avec oversample x3
    4. 3 nouvelles features linguistiques :
       - all_caps_words_ratio (mots en MAJUSCULES)
       - interpellation_score (patterns manipulation sociale FR+EN)
       - is_short_text (indicateur texte < 20 mots)
    5. Vocabulaire sensationnaliste FR enrichi (+16 termes social media)
    6. Vocabulaire sensationnaliste EN enrichi (+3 termes)

Datasets :
    - ISOT (True.csv + Fake.csv)
    - Kaggle FR (kaggle_fr/) x5 oversample
    - Kaggle FR short augmentation x3
    - FakeNewsNet (fakenewsnet/)
    - CONSTRAINT (constraint/)
    - Credibility Corpus (credibility_corpus/)

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
#  1. CHARGEMENT — DATASET V4 AUGMENTE
# ============================================================
print("=" * 70)
print("RETRAINING V4 — Amelioration performance FR court")
print("=" * 70)

t0 = time.time()

print("\n[1/7] Chargement du dataset bilingue V4 (augmentation FR courte)...")

df_v4 = DatasetCleaner.prepare_bilingual_dataset(
    fake_path=os.path.join(DATA, 'Fake.csv'),
    true_path=os.path.join(DATA, 'True.csv'),
    kaggle_fr_dir=os.path.join(DATA, 'kaggle_fr'),
    fakenewsnet_dir=os.path.join(DATA, 'fakenewsnet'),
    constraint_dir=os.path.join(DATA, 'constraint'),
    credibility_dir=os.path.join(DATA, 'credibility_corpus'),
    french_oversample=5,       # V4 : augmente de 3 a 5
    social_oversample=2,
    fr_short_augment=True,     # V4 : activation augmentation courte
    fr_short_oversample=3,     # V4 : x3 sur les textes courts generes
)

print(f"  Dataset total : {len(df_v4)} textes")
print(f"  EN={sum(df_v4.language == 'en')}, FR={sum(df_v4.language == 'fr')}")
print(f"  Labels : {df_v4.label.value_counts().to_dict()}")
print(f"  Longueur moyenne : {df_v4.text_clean.str.split().str.len().mean():.1f} mots")

# Stats detaillees par langue et longueur
df_v4['word_count'] = df_v4['text_clean'].str.split().str.len()
for lang in ['fr', 'en']:
    subset = df_v4[df_v4.language == lang]
    short = subset[subset.word_count < 15]
    medium = subset[(subset.word_count >= 15) & (subset.word_count < 30)]
    long_ = subset[subset.word_count >= 30]
    print(f"\n  {lang.upper()} : {len(subset)} total | "
          f"<15 mots={len(short)} | 15-30={len(medium)} | >30={len(long_)}")
    print(f"    Labels suspect <15 mots : {short.label.sum()}/{len(short)}")

assert 'text_original' in df_v4.columns, "Colonne text_original manquante !"
print(f"\n  text_original present : oui")

# Verifier les nouvelles features
print("\n  Verification des 15 features linguistiques (V4)...")
sample = pd.Series(["SCANDALE ! On nous MENT sur les vaccins !! Partagez SVP", "Le CNRS publie une etude."])
feats = LinguisticFeatureExtractor.extract(sample)
print(f"    Features shape : {feats.shape}")
print(f"    Feature names : {LinguisticFeatureExtractor.FEATURE_NAMES}")
print(f"    Texte suspect  : all_caps_ratio={feats[0, 12]:.3f}, interp={feats[0, 13]:.0f}, is_short={feats[0, 14]:.0f}")
print(f"    Texte fiable   : all_caps_ratio={feats[1, 12]:.3f}, interp={feats[1, 13]:.0f}, is_short={feats[1, 14]:.0f}")

# ============================================================
#  2. SPLIT TRAIN/TEST
# ============================================================
print("\n[2/7] Split train/test 80/20 stratifie...")

df_train, df_test = train_test_split(
    df_v4, test_size=0.2, stratify=df_v4['label'], random_state=42
)
print(f"  Train : {len(df_train)} | Test : {len(df_test)}")

# ============================================================
#  3. ENTRAINEMENT V4
# ============================================================
print("\n[3/7] Entrainement V4 (LogReg, 5-fold CV bilingue)...")
print("  Parametres : max_features=30000, ngram_range=(1,3), C=1.0, max_iter=5000")
print("  Features : TF-IDF + 15 linguistiques (12 originales + 3 V4)")
print("  Cela peut prendre plusieurs minutes...")

detector_v4 = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=0.44)
metrics_v4 = detector_v4.train(
    df_train,
    model_type='logreg',
    n_folds=5,
    track_emissions=True,
    emissions_dir=_proj,
)

print("\n  Metriques CV V4 :")
for k, v in sorted(metrics_v4.items()):
    if isinstance(v, float):
        print(f"    {k}: {v:.4f}")

# ============================================================
#  4. EVALUATION HOLDOUT GLOBALE
# ============================================================
print("\n[4/7] Evaluation sur le holdout test (20%)...")

eval_results = detector_v4.evaluate_holdout(df_test)

print(f"\n  Holdout Results V4:")
print(f"    Accuracy  : {eval_results['accuracy']:.4f}")
print(f"    F1        : {eval_results['f1']:.4f}")
print(f"    Precision : {eval_results['precision']:.4f}")
print(f"    Recall    : {eval_results['recall']:.4f}")
if 'roc_auc' in eval_results:
    print(f"    ROC AUC   : {eval_results['roc_auc']:.4f}")

print(f"\n  Classification Report V4 (holdout):")
print(eval_results['report_str'])

# ============================================================
#  5. EVALUATION DETAILLEE PAR LANGUE ET LONGUEUR
# ============================================================
print("\n[5/7] Evaluation detaillee par langue et longueur...")

# Predictions sur le test set
X_test_clean = df_test['text_clean'].values
X_test_orig = df_test['text_original'].values if 'text_original' in df_test.columns else None
y_test = df_test['label'].values

# Build features and predict
X_feat = detector_v4._build_features(X_test_clean, texts_original=X_test_orig, fit=False)
y_pred = detector_v4.model.predict(X_feat)
if hasattr(detector_v4.model, 'predict_proba'):
    y_proba = detector_v4.model.predict_proba(X_feat)[:, 1]
else:
    y_proba = np.zeros(len(y_test))

df_test_eval = df_test.copy()
df_test_eval['y_pred'] = y_pred
df_test_eval['y_proba'] = y_proba
df_test_eval['word_count'] = df_test_eval['text_clean'].str.split().str.len()

# Categories de longueur
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

results_detail = []
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
        results_detail.append({
            'lang': lang, 'length': cat, 'n': len(sub),
            'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec,
        })

# Resume par langue
print(f"\n  {'Langue':<5} {'N':>6} {'Accuracy':>9} {'F1':>7} {'Precision':>10} {'Recall':>7}")
print(f"  {'-'*50}")
for lang in ['fr', 'en', 'other']:
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
#  6. SAUVEGARDE V4
# ============================================================
print("\n[6/7] Sauvegarde du modele V4...")

detector_v4.save('expert_v4')
print(f"  Modele sauvegarde dans {MODEL_DIR} (suffix=expert_v4)")

# ============================================================
#  7. HEALTH CHECK + COMPARAISON V3 vs V4
# ============================================================
print("\n[7/7] Health check V4 et comparaison avec V3...")

hc_v4 = detector_v4.health_check()
print(f"\n  Health check V4 : {'PASS' if hc_v4['healthy'] else 'FAIL'}")
for detail in hc_v4['details']:
    status = "OK" if detail['passed'] else "FAIL"
    print(f"    [{status}] '{detail['text'][:60]}...' "
          f"label={detail['predicted_label']} (attendu={detail['expected_label']}) "
          f"score={detail['score']:.4f} (range={detail['expected_range']})")

# Comparaison V3 vs V4
print("\n  --- Comparaison V3 vs V4 ---")
try:
    detector_v3 = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=0.44)
    detector_v3.load('expert_v3')

    eval_v3 = detector_v3.evaluate_holdout(df_test)

    print(f"\n  {'Metrique':<20} {'V3':>10} {'V4':>10} {'Delta':>10}")
    print(f"  {'-'*50}")
    for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
        v3_val = eval_v3.get(metric, 0)
        v4_val = eval_results.get(metric, 0)
        delta = v4_val - v3_val
        sign = '+' if delta >= 0 else ''
        print(f"  {metric:<20} {v3_val:>10.4f} {v4_val:>10.4f} {sign}{delta:>9.4f}")

    # Comparaison FR court V3 vs V4
    print("\n  --- Focus FR court : V3 vs V4 ---")
    # V3 predictions on same test set
    X_v3_feat = detector_v3._build_features(X_test_clean, texts_original=X_test_orig, fit=False)
    y_v3_pred = detector_v3.model.predict(X_v3_feat)
    df_test_eval['y_pred_v3'] = y_v3_pred

    for cat in ['ultra_court (<15)', 'court (15-30)']:
        mask_fr = (df_test_eval['language'] == 'fr') & (df_test_eval['length_cat'] == cat)
        sub = df_test_eval[mask_fr]
        if len(sub) < 10:
            continue
        yt = sub['label'].values
        f1_v3 = f1_score(yt, sub['y_pred_v3'].values, zero_division=0)
        f1_v4 = f1_score(yt, sub['y_pred'].values, zero_division=0)
        delta = f1_v4 - f1_v3
        sign = '+' if delta >= 0 else ''
        print(f"  FR {cat}: F1 V3={f1_v3:.4f} -> V4={f1_v4:.4f} ({sign}{delta:.4f})")

except Exception as e:
    print(f"  V3 non disponible pour comparaison : {e}")

# Test final bilingue
print("\n  --- Test final bilingue V4 ---")
test_texts = [
    ("SCANDALE !! On nous cache la verite sur les vaccins !", "fr", "suspect"),
    ("Le CNRS a publie une etude sur le changement climatique.", "fr", "fiable"),
    ("URGENT: les vaccins contiennent des puces 5G !", "fr", "suspect"),
    ("La mairie annonce la renovation du pont.", "fr", "fiable"),
    ("BREAKING: Government EXPOSED in massive cover-up!", "en", "suspect"),
    ("A new study published in Nature examines climate change.", "en", "fiable"),
    ("SHARE before they DELETE this!! The truth about 5G!", "en", "suspect"),
    ("The city council approved the new budget.", "en", "fiable"),
    ("Partagez massivement avant censure !! Info cachee", "fr", "suspect"),
    ("Les resultats du match : Lyon 2 - Marseille 1", "fr", "fiable"),
]

results_v4 = detector_v4.predict(pd.Series([t[0] for t in test_texts]))
print(f"\n  {'Texte':<55} {'Lang':>4} {'Score':>6} {'Label':>8} {'Attendu':>8} {'OK':>3}")
print(f"  {'-'*90}")

correct = 0
for i, (text, lang, expected) in enumerate(test_texts):
    score = float(results_v4['ai_score_credibility'].iloc[i])
    label = "suspect" if score < 0.5 else "fiable"
    ok = "OK" if label == expected else "FAIL"
    if ok == "OK":
        correct += 1
    print(f"  {text[:55]:<55} {lang:>4} {score:>6.3f} {label:>8} {expected:>8} {ok:>3}")

print(f"\n  Score test bilingue V4 : {correct}/{len(test_texts)}")

elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"Retraining V4 termine en {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'=' * 70}")

# Print scores for health check update
print("\n  --- Scores V4 pour mise a jour HEALTH_CHECK_CASES ---")
hc_texts = pd.Series([t for t, _, _, _ in ExpertFakeNewsDetector.HEALTH_CHECK_CASES])
hc_results = detector_v4.predict(hc_texts)
for i, (text, expected_label, _, _) in enumerate(ExpertFakeNewsDetector.HEALTH_CHECK_CASES):
    pred = int(hc_results['prediction_label'].iloc[i])
    score = float(hc_results['ai_score_credibility'].iloc[i])
    print(f"    text='{text[:60]}...' label={pred} score={score:.4f}")
