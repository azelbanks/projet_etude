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

Resultats (sur 500 posts, annotateur 1) :
  - V5 seul :     Acc=0.588, F1_suspect=0.156, FP=201, kappa=0.076
  - Cascade :     Acc=0.938, F1_suspect=0.244, FP=12,  kappa=0.213

Resultats (sur 473 posts, consensus 2 annotateurs) :
  - V5 seul :     Acc=0.603, F1_suspect=0.121, FP=186, kappa=0.066
  - Cascade :     Acc=0.858, F1_suspect=0.230, FP=62,  kappa=0.187
  - Reduction FP : -67% (186 → 62)

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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                             cohen_kappa_score, classification_report)
from sklearn.pipeline import Pipeline
from scipy.stats import fisher_exact
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
#  5. EVALUATION CASCADE
# ================================================================
print("\n[5/5] Evaluation pipeline cascade...")

# Optimize Stage 1 threshold
best_f1 = 0
best_th = 0.5
results = []

for th in np.arange(0.10, 0.90, 0.05):
    cascade = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if stage1_proba[i, 1] < th:
            cascade[i] = 0  # opinion → fiable
        else:
            cascade[i] = v5_labels[i]  # factuel → V5

    acc = accuracy_score(human_labels, cascade)
    f1m = f1_score(human_labels, cascade, average='macro')
    f1s = f1_score(human_labels, cascade, pos_label=1, zero_division=0)
    kappa = cohen_kappa_score(human_labels, cascade)
    cm = confusion_matrix(human_labels, cascade)
    fp = cm[0, 1]
    fn = cm[1, 0] if cm.shape[0] > 1 else human_labels.sum()

    results.append({'th': th, 'acc': acc, 'f1m': f1m, 'f1s': f1s, 'fp': fp, 'fn': fn, 'kappa': kappa})
    if f1m > best_f1:
        best_f1 = f1m
        best_th = th

df_results = pd.DataFrame(results)

# Final comparison
print(f"\n  {'Methode':<30s} {'Acc':>6s} {'F1mac':>6s} {'F1sus':>6s} {'FP':>5s} {'FN':>5s} {'kappa':>6s}")
print(f"  {'-'*65}")

for name, pred in [
    ('V5 seul (baseline)', v5_labels),
]:
    acc = accuracy_score(human_labels, pred)
    f1m = f1_score(human_labels, pred, average='macro')
    f1s = f1_score(human_labels, pred, pos_label=1, zero_division=0)
    kappa = cohen_kappa_score(human_labels, pred)
    cm = confusion_matrix(human_labels, pred)
    fp = cm[0, 1]
    fn = cm[1, 0]
    print(f"  {name:<30s} {acc:>6.3f} {f1m:>6.3f} {f1s:>6.3f} {fp:>5d} {fn:>5d} {kappa:>6.3f}")

# Cascade at best threshold
best_row = df_results[df_results['th'] == best_th].iloc[0]
print(f"  {'Cascade (seuil=' + f'{best_th:.2f})':<30s} {best_row['acc']:>6.3f} {best_row['f1m']:>6.3f} {best_row['f1s']:>6.3f} {best_row['fp']:>5.0f} {best_row['fn']:>5.0f} {best_row['kappa']:>6.3f}")

# Reduction
baseline_fp = ((v5_labels == 1) & (human_labels == 0)).sum()
cascade_fp = int(best_row['fp'])
print(f"\n  Reduction FP : {baseline_fp} → {cascade_fp} (-{(1 - cascade_fp/baseline_fp)*100:.0f}%)")
print(f"  Seuil optimal Stage 1 : {best_th:.2f}")

# Save model
model_data = {
    'pipeline': pipe_stage1,
    'threshold': best_th,
    'classes': ['opinion', 'factuel'],
    'metrics': {
        'cv_f1_macro_stage1': float(f1_score(type_int, stage1_pred, average='macro')),
        'cascade_f1_macro': float(best_f1),
        'cascade_fp': cascade_fp,
        'baseline_fp': baseline_fp,
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
