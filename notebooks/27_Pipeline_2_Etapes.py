#!/usr/bin/env python3
"""
27 — Pipeline 2 Etapes : Filtre Fait/Opinion + Detection Desinfo
=================================================================

Probleme : V5 produit 201 FP sur 500 posts Bluesky annotes (40% des fiables
sont flagges suspect) car il confond opinions virulentes et desinformation.

Solution : Pipeline en cascade :
  Etape 1 — Classifieur fait/opinion (TF-IDF + LogReg)
     → Les posts "opinion pure" sont directement classes "fiable"
  Etape 2 — V5 standard sur les posts factuels uniquement

Resultats (eval split 30%, non biaise, seuil calibre sur 70%) :
  - V5 seul :     Acc=0.553, F1_suspect=0.152, FP=66,  kappa=0.073, AC1=0.270
  - Cascade :     Acc=0.847, F1_suspect=0.148, FP=18,  kappa=0.085, AC1=0.817
  - Reduction FP : -73% (66 → 18)

Resultats (full 500 posts, biaise — seuil optimise sur meme jeu) :
  - V5 seul :     Acc=0.588, F1_suspect=0.156, FP=201, kappa=0.076, AC1=0.347
  - Cascade :     Acc=0.880, F1_suspect=0.250, FP=46,  kappa=0.196, AC1=0.859
  - Reduction FP : -77% (201 → 46)

Methodologie : split calibration/evaluation 70/30 stratifie (random_state=42)
  - Le seuil cascade (0.45) est optimise par grid search sur les 70% (calibration)
  - Les metriques finales sont rapportees sur les 30% (evaluation, non biaise)
  - Les resultats sur le jeu complet sont aussi fournis pour comparaison

Note : Le kappa est artificiellement supprime par le paradoxe de prevalence
(seulement 4.8% de suspects). Le Gwet's AC1, insensible a ce biais, montre
que la cascade atteint un accord de 0.802 (bon) vs 0.347 (faible) pour V5.

Auteur : Thumalien Team
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                             cohen_kappa_score, classification_report)
from sklearn.pipeline import Pipeline
from scipy.stats import fisher_exact


def gwet_ac1(y_true, y_pred):
    """Gwet's AC1 — robust alternative to Cohen's kappa for imbalanced data.
    Unlike kappa, AC1 is not suppressed by low prevalence (prevalence paradox)."""
    n = len(y_true)
    po = np.mean(np.array(y_true) == np.array(y_pred))
    pi_0 = (np.sum(np.array(y_true) == 0) + np.sum(np.array(y_pred) == 0)) / (2 * n)
    pi_1 = 1 - pi_0
    pe = 2 * pi_0 * pi_1
    return (po - pe) / (1 - pe) if pe < 1 else 0
import joblib

_proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(_proj, 'models')
DATA_DIR = os.path.join(_proj, 'data')

print("=" * 70)
print("PIPELINE 2 ETAPES — FILTRE FAIT/OPINION + V5")
print("=" * 70)
t0 = time.time()

# ================================================================
#  1. CHARGER LES DONNEES ANNOTEES
# ================================================================
print("\n[1/5] Chargement des 500 posts annotes...")

df = pd.read_excel(os.path.join(DATA_DIR, 'bluesky_500_annotation_completed.xlsx'),
                   sheet_name='Annotation')
texts = df['Texte'].fillna('').astype(str).values
human_labels = (df['Label_annotateur'] == 'suspect').astype(int).values
v5_labels = (df['Prediction_V5'] == 'suspect').astype(int).values

print(f"  {len(df)} posts ({(df['Langue']=='FR').sum()} FR, {(df['Langue']=='EN').sum()} EN)")
print(f"  Labels humains : {(human_labels==0).sum()} fiables, {(human_labels==1).sum()} suspects")
print(f"  Labels V5 :      {(v5_labels==0).sum()} fiables, {(v5_labels==1).sum()} suspects")

# ================================================================
#  2. CLASSIFICATION FAIT/OPINION (heuristique supervisee)
# ================================================================
print("\n[2/5] Classification fait/opinion par heuristiques...")

def classify_factuality(row):
    """Classify post as factuel/opinion/mixte based on text + comment signals."""
    text = str(row['Texte']).lower()
    comment = str(row['Commentaire']).lower() if pd.notna(row['Commentaire']) else ''

    fact_signals = [
        'alerte info', 'urgent :', 'breaking', 'confirmed',
        'selon ', 'according to', 'd\'après ', 'a annoncé', 'announced',
        'a confirmé', 'a déclaré', 'has confirmed', 'report',
        'étude montre', 'study shows', 'research found',
        'est mort', 'est décédé', 'has died', 'was killed',
        'a été capturé', 'was captured', 'a été arrêté',
        'quitte ', 'leaves ', 'resigns',
    ]
    comment_fact = any(w in comment for w in [
        'assertion', 'factuel', 'affirmation', 'non sourcé', 'non vérifié',
        'à vérifier', 'annonce', 'décès', 'capture', 'clickbait',
        'claim', 'fake news', 'complotist', 'antivax', 'conspirat',
        'diffamatoire', 'rumeur', 'inventé', 'fausse alerte',
        'insinuation', 'sans source', 'sans preuve', 'accusations',
    ])

    opinion_signals = [
        'je pense', 'i think', 'i believe', 'à mon avis', 'in my opinion',
        'je trouve', 'je crois que', 'i feel like', 'imo ', 'imho',
        'personnellement', 'personally', 'pour moi,',
    ]
    comment_opinion = any(w in comment for w in [
        'opinion', 'humour', 'ironie', 'personnel', 'subjectif',
        'satirique', 'artistique', 'commentaire personnel', 'réaction',
        'jugement de valeur', 'expression', 'sentiment',
    ])

    has_fact = any(m in text for m in fact_signals) or comment_fact
    has_opinion = any(m in text for m in opinion_signals) or comment_opinion

    if has_fact and has_opinion:
        return 'mixte'
    elif has_fact:
        return 'factuel'
    elif has_opinion:
        return 'opinion'
    else:
        return 'indetermine'

df['type_post'] = df.apply(classify_factuality, axis=1)

# Binary: factuel+mixte → needs checking, opinion+indetermine → safe
df['type_binary'] = df['type_post'].map({
    'factuel': 'factuel',
    'mixte': 'factuel',
    'opinion': 'opinion',
    'indetermine': 'opinion',
})
type_int = (df['type_binary'] == 'factuel').astype(int).values

print(f"  Distribution types : {df['type_post'].value_counts().to_dict()}")
print(f"  Binaire : {(type_int==1).sum()} factuels, {(type_int==0).sum()} opinions")

# ================================================================
#  3. VALIDATION STATISTIQUE
# ================================================================
print("\n[3/5] Test statistique fait/opinion → suspect...")

ct = pd.crosstab(df['type_binary'], df['Label_annotateur'])
table = [[ct.iloc[0,0], ct.iloc[0,1]], [ct.iloc[1,0], ct.iloc[1,1]]]
odds_ratio, p_value = fisher_exact(table)

rate_fact = human_labels[type_int == 1].mean() * 100
rate_opinion = human_labels[type_int == 0].mean() * 100

print(f"  Taux suspect factuels :  {rate_fact:.1f}%")
print(f"  Taux suspect opinions :  {rate_opinion:.1f}%")
print(f"  Odds ratio : {odds_ratio:.2f}")
print(f"  Fisher p-value : {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

# ================================================================
#  4. ENTRAINEMENT ETAPE 1 (classifieur fait/opinion)
# ================================================================
print("\n[4/5] Entrainement classifieur fait/opinion (CV 5-fold)...")

pipe_stage1 = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents='unicode',
    )),
    ('clf', LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stage1_pred = cross_val_predict(pipe_stage1, texts, type_int, cv=cv)
stage1_proba = cross_val_predict(pipe_stage1, texts, type_int, cv=cv, method='predict_proba')

print(f"  CV Accuracy :  {accuracy_score(type_int, stage1_pred):.3f}")
print(f"  CV F1 factuel: {f1_score(type_int, stage1_pred, pos_label=1):.3f}")
print(f"  CV F1 opinion: {f1_score(type_int, stage1_pred, pos_label=0):.3f}")
print(f"  CV F1 macro :  {f1_score(type_int, stage1_pred, average='macro'):.3f}")

# Train final model
pipe_stage1.fit(texts, type_int)

# ================================================================
#  5. SPLIT CALIBRATION / EVALUATION (70/30 stratifie)
# ================================================================
print("\n[5/7] Split calibration/evaluation 70/30 stratifie...")

indices = np.arange(len(df))
idx_calib, idx_eval = train_test_split(
    indices, test_size=0.30, stratify=human_labels, random_state=42
)

print(f"  Calibration : {len(idx_calib)} posts ({human_labels[idx_calib].sum()} suspects)")
print(f"  Evaluation  : {len(idx_eval)} posts ({human_labels[idx_eval].sum()} suspects)")

# ================================================================
#  6. OPTIMISATION SEUIL SUR CALIBRATION UNIQUEMENT
# ================================================================
print("\n[6/7] Optimisation seuil cascade sur calibration...")


def evaluate_cascade(threshold, idx, stage1_proba, v5_labels, human_labels):
    """Evaluate cascade at a given threshold on a subset of indices."""
    cascade = np.zeros(len(idx), dtype=int)
    for j, i in enumerate(idx):
        if stage1_proba[i, 1] < threshold:
            cascade[j] = 0  # opinion -> fiable
        else:
            cascade[j] = v5_labels[i]  # factuel -> V5
    y_true = human_labels[idx]
    acc = accuracy_score(y_true, cascade)
    f1m = f1_score(y_true, cascade, average='macro')
    f1s = f1_score(y_true, cascade, pos_label=1, zero_division=0)
    kappa = cohen_kappa_score(y_true, cascade)
    ac1 = gwet_ac1(y_true, cascade)
    cm = confusion_matrix(y_true, cascade)
    fp = cm[0, 1] if cm.shape[1] > 1 else 0
    fn = cm[1, 0] if cm.shape[0] > 1 else y_true.sum()
    return {'th': threshold, 'acc': acc, 'f1m': f1m, 'f1s': f1s,
            'fp': fp, 'fn': fn, 'kappa': kappa, 'ac1': ac1}


# Grid search on calibration set only
best_f1_calib = 0
best_th_calib = 0.5
results_calib = []

for th in np.arange(0.10, 0.90, 0.05):
    row = evaluate_cascade(th, idx_calib, stage1_proba, v5_labels, human_labels)
    results_calib.append(row)
    if row['f1m'] > best_f1_calib:
        best_f1_calib = row['f1m']
        best_th_calib = row['th']

print(f"  Seuil optimal (calibration) : {best_th_calib:.2f}")

# ================================================================
#  7. EVALUATION FINALE
# ================================================================
print("\n[7/7] Evaluation pipeline cascade...")


def print_comparison(label, idx_set, best_threshold):
    """Print comparison table for a given subset."""
    header = f"\n  --- {label} ({len(idx_set)} posts) ---"
    print(header)
    print(f"  {'Methode':<30s} {'Acc':>6s} {'F1mac':>6s} {'F1sus':>6s} {'FP':>5s} {'FN':>5s} {'kappa':>6s} {'AC1':>6s}")
    print(f"  {'-'*73}")

    # V5 baseline
    y_true = human_labels[idx_set]
    y_v5 = v5_labels[idx_set]
    acc = accuracy_score(y_true, y_v5)
    f1m = f1_score(y_true, y_v5, average='macro')
    f1s = f1_score(y_true, y_v5, pos_label=1, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_v5)
    ac1 = gwet_ac1(y_true, y_v5)
    cm = confusion_matrix(y_true, y_v5)
    fp = cm[0, 1] if cm.shape[1] > 1 else 0
    fn = cm[1, 0] if cm.shape[0] > 1 else y_true.sum()
    print(f"  {'V5 seul (baseline)':<30s} {acc:>6.3f} {f1m:>6.3f} {f1s:>6.3f} {fp:>5d} {fn:>5d} {kappa:>6.3f} {ac1:>6.3f}")
    baseline_fp = fp

    # Cascade at best threshold
    row = evaluate_cascade(best_threshold, idx_set, stage1_proba, v5_labels, human_labels)
    name = f"Cascade (seuil={best_threshold:.2f})"
    print(f"  {name:<30s} {row['acc']:>6.3f} {row['f1m']:>6.3f} {row['f1s']:>6.3f} {row['fp']:>5.0f} {row['fn']:>5.0f} {row['kappa']:>6.3f} {row['ac1']:>6.3f}")

    cascade_fp = int(row['fp'])
    if baseline_fp > 0:
        print(f"  Reduction FP : {baseline_fp} -> {cascade_fp} (-{(1 - cascade_fp/baseline_fp)*100:.0f}%)")
    return row


# A) Results on evaluation set (unbiased — threshold chosen on calibration)
eval_row = print_comparison("Results (eval split, unbiased)", idx_eval, best_th_calib)

# B) Results on full dataset for comparison (biased — same data for threshold + eval)
# Also run grid search on full dataset to get its own best threshold
best_f1_full = 0
best_th_full = 0.5
results_full = []

for th in np.arange(0.10, 0.90, 0.05):
    row = evaluate_cascade(th, indices, stage1_proba, v5_labels, human_labels)
    results_full.append(row)
    if row['f1m'] > best_f1_full:
        best_f1_full = row['f1m']
        best_th_full = row['th']

full_row = print_comparison("Results (full, biased)", indices, best_th_full)

print(f"\n  Seuil optimal (calibration 70%) : {best_th_calib:.2f}")
print(f"  Seuil optimal (full, biased)    : {best_th_full:.2f}")

# Save model (use threshold from proper calibration)
model_data = {
    'pipeline': pipe_stage1,
    'threshold': best_th_calib,
    'classes': ['opinion', 'factuel'],
    'metrics': {
        'cv_f1_macro_stage1': float(f1_score(type_int, stage1_pred, average='macro')),
        'cascade_f1_macro_eval': float(eval_row['f1m']),
        'cascade_f1_macro_full': float(full_row['f1m']),
        'cascade_fp_eval': int(eval_row['fp']),
        'cascade_fp_full': int(full_row['fp']),
        'baseline_fp': int(((v5_labels == 1) & (human_labels == 0)).sum()),
        'fisher_p_value': float(p_value),
        'odds_ratio': float(odds_ratio),
    }
}
model_path = os.path.join(MODEL_DIR, 'stage1_fact_opinion.joblib')
joblib.dump(model_data, model_path)
print(f"\n  Modele sauvegarde : {model_path}")

elapsed = time.time() - t0
print(f"\n  Temps total : {elapsed:.0f}s")
print("\n" + "=" * 70)
print("TERMINE")
print("=" * 70)
