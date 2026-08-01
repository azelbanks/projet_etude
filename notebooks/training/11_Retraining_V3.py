#!/usr/bin/env python3
"""
11 — Retraining V3 : Exploitation des features linguistiques corrigees
======================================================================

Contexte :
    Le bug de preprocessing a ete corrige : les features linguistiques
    (caps_ratio, exclamation_count, question_count, punct_density,
    sentence_count) utilisent maintenant le texte original au lieu du
    texte nettoye. Le modele V2 a ete entraine avec le bug, donc ses
    predictions sont desormais inconsistantes. On reentrainne en V3.

Parametres identiques au V2 :
    - TF-IDF max_features=30000, ngram_range=(1,3), min_df=3, max_df=0.95
    - LogisticRegression C=1.0, max_iter=5000, class_weight='balanced'
    - Mode bilingue (FR/EN)
    - social_oversample=2
    - 5-fold stratified CV

Datasets :
    - ISOT (True.csv + Fake.csv)
    - Kaggle FR (kaggle_fr/)
    - FakeNewsNet (fakenewsnet/)
    - CONSTRAINT (constraint/)
    - Credibility Corpus (credibility_corpus/)

Auteur : Thumalien Team
"""

import sys
import os
import time
import logging

# --- Setup paths ---
_proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_proj, 'src'))

from pipeline.expert_detector import (
    DatasetCleaner,
    ExpertFakeNewsDetector,
    LinguisticFeatureExtractor,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA = os.path.join(_proj, 'data', 'training')
MODEL_DIR = os.path.join(_proj, 'models')

# ============================================================
#  1. CHARGEMENT ET PREPARATION DU DATASET (identique a V2)
# ============================================================
print("=" * 70)
print("RETRAINING V3 — Features linguistiques corrigees")
print("=" * 70)

t0 = time.time()

print("\n[1/6] Chargement du dataset bilingue complet...")

df_v3 = DatasetCleaner.prepare_bilingual_dataset(
    fake_path=os.path.join(DATA, 'Fake.csv'),
    true_path=os.path.join(DATA, 'True.csv'),
    kaggle_fr_dir=os.path.join(DATA, 'kaggle_fr'),
    fakenewsnet_dir=os.path.join(DATA, 'fakenewsnet'),
    constraint_dir=os.path.join(DATA, 'constraint'),
    credibility_dir=os.path.join(DATA, 'credibility_corpus'),
    social_oversample=2,
)

print(f"  Dataset total : {len(df_v3)} textes")
print(f"  EN={sum(df_v3.language == 'en')}, FR={sum(df_v3.language == 'fr')}")
print(f"  Labels : {df_v3.label.value_counts().to_dict()}")
print(f"  Longueur moyenne : {df_v3.text_clean.str.split().str.len().mean():.1f} mots")

# Verification que text_original est bien present (necessaire pour features corrigees)
assert 'text_original' in df_v3.columns, "Colonne text_original manquante !"
print(f"  text_original present : oui (features linguistiques utiliseront le texte original)")

# ============================================================
#  2. SPLIT TRAIN/TEST
# ============================================================
print("\n[2/6] Split train/test 80/20 stratifie...")

df_train, df_test = train_test_split(
    df_v3, test_size=0.2, stratify=df_v3['label'], random_state=42
)
print(f"  Train : {len(df_train)} | Test : {len(df_test)}")

# ============================================================
#  3. ENTRAINEMENT V3
# ============================================================
print("\n[3/6] Entrainement V3 (LogReg, 5-fold CV bilingue)...")
print("  Parametres : max_features=30000, ngram_range=(1,3), C=1.0, max_iter=5000")
print("  Cela peut prendre plusieurs minutes...")

detector_v3 = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=0.44)
metrics_v3 = detector_v3.train(
    df_train,
    model_type='logreg',
    n_folds=5,
    track_emissions=True,
    emissions_dir=_proj,
)

print("\n  Metriques CV V3 :")
for k, v in sorted(metrics_v3.items()):
    if isinstance(v, float):
        print(f"    {k}: {v:.4f}")

# ============================================================
#  4. EVALUATION HOLDOUT
# ============================================================
print("\n[4/6] Evaluation sur le holdout test (20%)...")

eval_results = detector_v3.evaluate_holdout(df_test)

print(f"\n  Holdout Results V3:")
print(f"    Accuracy  : {eval_results['accuracy']:.4f}")
print(f"    F1        : {eval_results['f1']:.4f}")
print(f"    Precision : {eval_results['precision']:.4f}")
print(f"    Recall    : {eval_results['recall']:.4f}")
if 'roc_auc' in eval_results:
    print(f"    ROC AUC   : {eval_results['roc_auc']:.4f}")

print(f"\n  Classification Report V3 (holdout):")
print(eval_results['report_str'])

# ============================================================
#  5. SAUVEGARDE V3
# ============================================================
print("\n[5/6] Sauvegarde du modele V3...")

detector_v3.save('expert_v3')
print(f"  Modele sauvegarde dans {MODEL_DIR} (suffix=expert_v3)")

# ============================================================
#  6. HEALTH CHECK + COMPARAISON V2 vs V3
# ============================================================
print("\n[6/6] Health check V3 et comparaison avec V2...")

hc_v3 = detector_v3.health_check()
print(f"\n  Health check V3 : {'PASS' if hc_v3['healthy'] else 'FAIL'}")
for detail in hc_v3['details']:
    status = "OK" if detail['passed'] else "FAIL"
    print(f"    [{status}] '{detail['text'][:60]}...' "
          f"label={detail['predicted_label']} (attendu={detail['expected_label']}) "
          f"score={detail['score']:.4f} (range={detail['expected_range']})")

# Charger V2 pour comparaison
print("\n  --- Comparaison V2 vs V3 ---")
try:
    detector_v2 = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=0.44)
    detector_v2.load('expert_v2')

    eval_v2 = detector_v2.evaluate_holdout(df_test)

    print(f"\n  {'Metrique':<20} {'V2':>10} {'V3':>10} {'Delta':>10}")
    print(f"  {'-'*50}")
    for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
        v2_val = eval_v2.get(metric, 0)
        v3_val = eval_results.get(metric, 0)
        delta = v3_val - v2_val
        sign = '+' if delta >= 0 else ''
        print(f"  {metric:<20} {v2_val:>10.4f} {v3_val:>10.4f} {sign}{delta:>9.4f}")

    # Health check V2
    hc_v2 = detector_v2.health_check()
    print(f"\n  Health check V2 : {'PASS' if hc_v2['healthy'] else 'FAIL'}")
    print(f"  Health check V3 : {'PASS' if hc_v3['healthy'] else 'FAIL'}")

except Exception as e:
    print(f"  V2 non disponible pour comparaison : {e}")

elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"Retraining V3 termine en {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'=' * 70}")

# Print the actual scores from health check so we can update HEALTH_CHECK_CASES
print("\n  --- Scores V3 pour mise a jour HEALTH_CHECK_CASES ---")
import pandas as pd
hc_texts = pd.Series([t for t, _, _, _ in ExpertFakeNewsDetector.HEALTH_CHECK_CASES])
hc_results = detector_v3.predict(hc_texts)
for i, (text, expected_label, _, _) in enumerate(ExpertFakeNewsDetector.HEALTH_CHECK_CASES):
    pred = int(hc_results['prediction_label'].iloc[i])
    score = float(hc_results['ai_score_credibility'].iloc[i])
    print(f"    text='{text[:60]}...' label={pred} score={score:.4f}")
