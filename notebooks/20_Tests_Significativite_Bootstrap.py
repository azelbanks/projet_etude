#!/usr/bin/env python3
"""
20 — Tests de significativite statistique par bootstrap paire
=============================================================

Contexte :
    Les notebooks precedents comparent V4 vs V5 et analysent les performances
    par langue et par longueur, mais sans test statistique formel. Ce notebook
    applique un test bootstrap paire (1000 iterations) pour determiner si les
    differences observees sont statistiquement significatives.

Methode :
    - Echantillonnage avec remplacement N fois sur le holdout
    - Calcul du F1 sur chaque echantillon pour les deux conditions
    - Intervalle de confiance a 95% pour la difference
    - p-value = proportion d'echantillons ou la difference <= 0

Comparaisons testees :
    1. V4 vs V5 (F1 global)
    2. V5 FR vs V5 EN (F1 par langue)
    3. Comparaisons par segment de longueur (ultra-court, court, moyen, long)

Auteur : Thumalien Team
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# --- Setup paths ---
_proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_proj, 'src'))

from pipeline.expert_detector import (
    DatasetCleaner,
    ExpertFakeNewsDetector,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

DATA = os.path.join(_proj, 'data', 'training')
MODEL_DIR = os.path.join(_proj, 'models')

N_BOOTSTRAP = 1000

# ============================================================
#  UTILITAIRE : TEST BOOTSTRAP PAIRE
# ============================================================

def bootstrap_paired_test(y_true, y_pred_a, y_pred_b, n_bootstrap=1000, metric_fn=f1_score):
    """Bootstrap paired test. Returns CI and p-value."""
    np.random.seed(42)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        score_a = metric_fn(y_true[idx], y_pred_a[idx], zero_division=0)
        score_b = metric_fn(y_true[idx], y_pred_b[idx], zero_division=0)
        diffs.append(score_b - score_a)
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    p_value = np.mean(diffs <= 0)
    return {
        'mean_diff': np.mean(diffs),
        'ci_95': (ci_lower, ci_upper),
        'p_value': p_value,
        'significant': ci_lower > 0 or ci_upper < 0,
    }


def bootstrap_single_ci(y_true, y_pred, n_bootstrap=1000, metric_fn=f1_score):
    """Bootstrap CI for a single model's metric."""
    np.random.seed(42)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        s = metric_fn(y_true[idx], y_pred[idx], zero_division=0)
        scores.append(s)
    scores = np.array(scores)
    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'ci_lower': np.percentile(scores, 2.5),
        'ci_upper': np.percentile(scores, 97.5),
        'ci_width': np.percentile(scores, 97.5) - np.percentile(scores, 2.5),
    }


def length_category(wc):
    """Categorise un texte par nombre de mots."""
    if wc < 15:
        return 'ultra_court (<15)'
    elif wc < 30:
        return 'court (15-30)'
    elif wc < 100:
        return 'moyen (30-100)'
    else:
        return 'long (>=100)'


# ============================================================
#  1. CHARGEMENT DU DATASET V5
# ============================================================
print("=" * 70)
print("TESTS DE SIGNIFICATIVITE STATISTIQUE — BOOTSTRAP PAIRE")
print(f"N_BOOTSTRAP = {N_BOOTSTRAP}")
print("=" * 70)

t0 = time.time()

print("\n[1/6] Chargement du dataset bilingue V5...")

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
print(f"  EN={sum(df_v5.language == 'en')}, FR={sum(df_v5.language == 'fr')}")

# ============================================================
#  2. SPLIT IDENTIQUE AU TRAINING (random_state=42)
# ============================================================
print("\n[2/6] Split train/test 80/20 stratifie (random_state=42)...")

df_train, df_test = train_test_split(
    df_v5, test_size=0.2, stratify=df_v5['label'], random_state=42
)
print(f"  Train : {len(df_train)} | Test (holdout) : {len(df_test)}")

# ============================================================
#  3. CHARGEMENT DES MODELES V4 ET V5
# ============================================================
print("\n[3/6] Chargement des modeles V4 et V5...")

detector_v5 = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=0.44)
detector_v5.load('expert_v5')
print("  V5 charge.")

detector_v4 = None
v4_available = False
try:
    detector_v4 = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=0.44)
    detector_v4.load('expert_v4')
    v4_available = True
    print("  V4 charge.")
except Exception as e:
    print(f"  V4 non disponible : {e}")

# ============================================================
#  4. PREDICTIONS SUR LE HOLDOUT
# ============================================================
print("\n[4/6] Predictions sur le holdout...")

X_test_clean = df_test['text_clean'].values
X_test_orig = df_test['text_original'].values if 'text_original' in df_test.columns else None
y_test = df_test['label'].values

# --- V5 predictions ---
X_feat_v5 = detector_v5._build_features(X_test_clean, texts_original=X_test_orig, fit=False)
y_pred_v5 = detector_v5.model.predict(X_feat_v5)
f1_v5_global = f1_score(y_test, y_pred_v5, zero_division=0)
print(f"  V5 holdout F1 = {f1_v5_global:.4f}")

# --- V4 predictions ---
y_pred_v4 = None
if v4_available:
    X_feat_v4 = detector_v4._build_features(X_test_clean, texts_original=X_test_orig, fit=False)
    y_pred_v4 = detector_v4.model.predict(X_feat_v4)
    f1_v4_global = f1_score(y_test, y_pred_v4, zero_division=0)
    print(f"  V4 holdout F1 = {f1_v4_global:.4f}")

# Preparer les metadata du holdout
df_test_eval = df_test.copy()
df_test_eval['y_pred_v5'] = y_pred_v5
if y_pred_v4 is not None:
    df_test_eval['y_pred_v4'] = y_pred_v4
df_test_eval['word_count'] = df_test_eval['text_clean'].str.split().str.len()
df_test_eval['length_cat'] = df_test_eval['word_count'].apply(length_category)

# ============================================================
#  5. TESTS DE SIGNIFICATIVITE
# ============================================================
print("\n[5/6] Tests bootstrap de significativite...")
print(f"  {N_BOOTSTRAP} iterations par test\n")

# Collecter tous les resultats pour le tableau recapitulatif
summary_rows = []

# ----------------------------------------------------------
# 5a. V4 vs V5 — Global
# ----------------------------------------------------------
print("-" * 70)
print("TEST 1 : V4 vs V5 — F1 Global")
print("-" * 70)

if v4_available:
    result_v4v5 = bootstrap_paired_test(
        y_test, y_pred_v4, y_pred_v5,
        n_bootstrap=N_BOOTSTRAP, metric_fn=f1_score
    )
    print(f"  V4 F1 = {f1_v4_global:.4f}")
    print(f"  V5 F1 = {f1_v5_global:.4f}")
    print(f"  Difference moyenne (V5 - V4) = {result_v4v5['mean_diff']:+.4f}")
    print(f"  IC 95% = [{result_v4v5['ci_95'][0]:+.4f}, {result_v4v5['ci_95'][1]:+.4f}]")
    print(f"  p-value = {result_v4v5['p_value']:.4f}")
    print(f"  Significatif (alpha=0.05) : {'OUI' if result_v4v5['significant'] else 'NON'}")

    summary_rows.append({
        'Comparaison': 'V4 vs V5 (Global)',
        'Metrique': 'F1',
        'Score A': f"{f1_v4_global:.4f}",
        'Score B': f"{f1_v5_global:.4f}",
        'Diff moyenne': f"{result_v4v5['mean_diff']:+.4f}",
        'IC 95%': f"[{result_v4v5['ci_95'][0]:+.4f}, {result_v4v5['ci_95'][1]:+.4f}]",
        'p-value': f"{result_v4v5['p_value']:.4f}",
        'Significatif': 'OUI' if result_v4v5['significant'] else 'NON',
    })

    # V4 vs V5 par langue
    for lang in ['fr', 'en']:
        mask = df_test_eval['language'].values == lang
        if mask.sum() < 20:
            continue
        yt = y_test[mask]
        yp4 = y_pred_v4[mask]
        yp5 = y_pred_v5[mask]
        f1_4 = f1_score(yt, yp4, zero_division=0)
        f1_5 = f1_score(yt, yp5, zero_division=0)
        res = bootstrap_paired_test(yt, yp4, yp5, n_bootstrap=N_BOOTSTRAP)

        print(f"\n  V4 vs V5 — {lang.upper()} (n={mask.sum()}) :")
        print(f"    V4 F1={f1_4:.4f}, V5 F1={f1_5:.4f}")
        print(f"    Diff={res['mean_diff']:+.4f}, IC=[{res['ci_95'][0]:+.4f}, {res['ci_95'][1]:+.4f}]")
        print(f"    p={res['p_value']:.4f}, significatif={'OUI' if res['significant'] else 'NON'}")

        summary_rows.append({
            'Comparaison': f'V4 vs V5 ({lang.upper()})',
            'Metrique': 'F1',
            'Score A': f"{f1_4:.4f}",
            'Score B': f"{f1_5:.4f}",
            'Diff moyenne': f"{res['mean_diff']:+.4f}",
            'IC 95%': f"[{res['ci_95'][0]:+.4f}, {res['ci_95'][1]:+.4f}]",
            'p-value': f"{res['p_value']:.4f}",
            'Significatif': 'OUI' if res['significant'] else 'NON',
        })
else:
    print("  SKIP : modele V4 non disponible.")
    summary_rows.append({
        'Comparaison': 'V4 vs V5 (Global)',
        'Metrique': 'F1',
        'Score A': 'N/A', 'Score B': f"{f1_v5_global:.4f}",
        'Diff moyenne': 'N/A', 'IC 95%': 'N/A',
        'p-value': 'N/A', 'Significatif': 'N/A (V4 absent)',
    })

# ----------------------------------------------------------
# 5b. V5 FR vs V5 EN
# ----------------------------------------------------------
print("\n" + "-" * 70)
print("TEST 2 : V5 FR vs V5 EN — F1")
print("-" * 70)

mask_fr = df_test_eval['language'].values == 'fr'
mask_en = df_test_eval['language'].values == 'en'

f1_fr = f1_score(y_test[mask_fr], y_pred_v5[mask_fr], zero_division=0)
f1_en = f1_score(y_test[mask_en], y_pred_v5[mask_en], zero_division=0)

# Pour comparer FR vs EN, on ne peut pas faire un test paire direct
# car ce sont des sous-ensembles differents. On calcule les IC
# individuels et on verifie s'ils se chevauchent.
ci_fr = bootstrap_single_ci(y_test[mask_fr], y_pred_v5[mask_fr], n_bootstrap=N_BOOTSTRAP)
ci_en = bootstrap_single_ci(y_test[mask_en], y_pred_v5[mask_en], n_bootstrap=N_BOOTSTRAP)

# Chevauchement des IC
overlap = ci_fr['ci_upper'] >= ci_en['ci_lower'] and ci_en['ci_upper'] >= ci_fr['ci_lower']
significant_lang_diff = not overlap

print(f"  V5 FR : F1 = {f1_fr:.4f}  IC 95% = [{ci_fr['ci_lower']:.4f}, {ci_fr['ci_upper']:.4f}]  (n={mask_fr.sum()})")
print(f"  V5 EN : F1 = {f1_en:.4f}  IC 95% = [{ci_en['ci_lower']:.4f}, {ci_en['ci_upper']:.4f}]  (n={mask_en.sum()})")
print(f"  Difference FR - EN = {f1_fr - f1_en:+.4f}")
print(f"  Chevauchement des IC : {'OUI' if overlap else 'NON'}")
print(f"  Difference significative : {'OUI' if significant_lang_diff else 'NON'}")

summary_rows.append({
    'Comparaison': 'V5 FR vs V5 EN',
    'Metrique': 'F1',
    'Score A': f"{f1_fr:.4f} (FR)",
    'Score B': f"{f1_en:.4f} (EN)",
    'Diff moyenne': f"{f1_fr - f1_en:+.4f}",
    'IC 95%': f"FR:[{ci_fr['ci_lower']:.4f},{ci_fr['ci_upper']:.4f}] EN:[{ci_en['ci_lower']:.4f},{ci_en['ci_upper']:.4f}]",
    'p-value': 'IC non-overlap' if significant_lang_diff else 'IC overlap',
    'Significatif': 'OUI' if significant_lang_diff else 'NON',
})

# ----------------------------------------------------------
# 5c. Comparaisons par segment de longueur (V5)
# ----------------------------------------------------------
print("\n" + "-" * 70)
print("TEST 3 : V5 Intervalles de confiance par segment de longueur")
print("-" * 70)

length_cats_order = ['ultra_court (<15)', 'court (15-30)', 'moyen (30-100)', 'long (>=100)']

print(f"\n  {'Langue':<5} {'Segment':<20} {'N':>6} {'F1':>7} {'IC 95%':>22} {'Largeur IC':>11}")
print(f"  {'-' * 75}")

segment_ci_results = {}
for lang in ['fr', 'en']:
    for cat in length_cats_order:
        mask = (df_test_eval['language'].values == lang) & (df_test_eval['length_cat'].values == cat)
        n_seg = mask.sum()
        if n_seg < 20:
            print(f"  {lang.upper():<5} {cat:<20} {n_seg:>6}   (trop peu d'echantillons, skip)")
            continue
        yt_seg = y_test[mask]
        yp_seg = y_pred_v5[mask]
        f1_seg = f1_score(yt_seg, yp_seg, zero_division=0)
        ci_seg = bootstrap_single_ci(yt_seg, yp_seg, n_bootstrap=N_BOOTSTRAP)

        key = f"{lang.upper()} {cat}"
        segment_ci_results[key] = {
            'n': n_seg, 'f1': f1_seg,
            'ci_lower': ci_seg['ci_lower'], 'ci_upper': ci_seg['ci_upper'],
            'ci_width': ci_seg['ci_width'],
        }

        print(f"  {lang.upper():<5} {cat:<20} {n_seg:>6} {f1_seg:>7.4f} "
              f"[{ci_seg['ci_lower']:.4f}, {ci_seg['ci_upper']:.4f}] {ci_seg['ci_width']:>11.4f}")

# ----------------------------------------------------------
# 5d. V4 vs V5 par segment de longueur (si V4 dispo)
# ----------------------------------------------------------
if v4_available:
    print("\n" + "-" * 70)
    print("TEST 4 : V4 vs V5 par segment de longueur")
    print("-" * 70)

    for lang in ['fr', 'en']:
        for cat in length_cats_order:
            mask = (df_test_eval['language'].values == lang) & (df_test_eval['length_cat'].values == cat)
            n_seg = mask.sum()
            if n_seg < 20:
                continue
            yt_seg = y_test[mask]
            yp4_seg = y_pred_v4[mask]
            yp5_seg = y_pred_v5[mask]
            f1_4_seg = f1_score(yt_seg, yp4_seg, zero_division=0)
            f1_5_seg = f1_score(yt_seg, yp5_seg, zero_division=0)
            res_seg = bootstrap_paired_test(yt_seg, yp4_seg, yp5_seg, n_bootstrap=N_BOOTSTRAP)

            sig_marker = " ***" if res_seg['significant'] else ""
            print(f"  {lang.upper()} {cat:<20} (n={n_seg:>5}) : "
                  f"V4={f1_4_seg:.4f} -> V5={f1_5_seg:.4f} | "
                  f"diff={res_seg['mean_diff']:+.4f} | "
                  f"IC=[{res_seg['ci_95'][0]:+.4f}, {res_seg['ci_95'][1]:+.4f}] | "
                  f"p={res_seg['p_value']:.4f}{sig_marker}")

            summary_rows.append({
                'Comparaison': f'V4 vs V5 ({lang.upper()} {cat})',
                'Metrique': 'F1',
                'Score A': f"{f1_4_seg:.4f}",
                'Score B': f"{f1_5_seg:.4f}",
                'Diff moyenne': f"{res_seg['mean_diff']:+.4f}",
                'IC 95%': f"[{res_seg['ci_95'][0]:+.4f}, {res_seg['ci_95'][1]:+.4f}]",
                'p-value': f"{res_seg['p_value']:.4f}",
                'Significatif': 'OUI' if res_seg['significant'] else 'NON',
            })

# ============================================================
#  6. TABLEAU RECAPITULATIF
# ============================================================
print("\n\n" + "=" * 70)
print("[6/6] TABLEAU RECAPITULATIF — TESTS DE SIGNIFICATIVITE")
print("=" * 70)

df_summary = pd.DataFrame(summary_rows)

# Affichage formate
col_widths = {
    'Comparaison': 30,
    'Metrique': 8,
    'Score A': 12,
    'Score B': 12,
    'Diff moyenne': 12,
    'IC 95%': 32,
    'p-value': 12,
    'Significatif': 12,
}

header = ""
for col, w in col_widths.items():
    header += f"{col:<{w}} "
print(f"\n{header}")
print("-" * len(header))

for _, row in df_summary.iterrows():
    line = ""
    for col, w in col_widths.items():
        line += f"{str(row[col]):<{w}} "
    print(line)

# Resume
print("\n" + "=" * 70)
print("RESUME")
print("=" * 70)

n_sig = sum(1 for r in summary_rows if r['Significatif'] == 'OUI')
n_non_sig = sum(1 for r in summary_rows if r['Significatif'] == 'NON')
n_na = sum(1 for r in summary_rows if r['Significatif'] not in ('OUI', 'NON'))

print(f"\n  Nombre total de tests : {len(summary_rows)}")
print(f"  Significatifs (alpha=0.05) : {n_sig}")
print(f"  Non significatifs          : {n_non_sig}")
if n_na > 0:
    print(f"  Non applicables            : {n_na}")

# Intervalles de confiance serres
if segment_ci_results:
    print(f"\n  Largeur moyenne des IC (segments) : "
          f"{np.mean([v['ci_width'] for v in segment_ci_results.values()]):.4f}")
    tight = sum(1 for v in segment_ci_results.values() if v['ci_width'] < 0.05)
    total_seg = len(segment_ci_results)
    print(f"  Segments avec IC serre (<0.05) : {tight}/{total_seg}")

elapsed = time.time() - t0
print(f"\n  Duree totale : {elapsed:.0f}s ({elapsed / 60:.1f} min)")
print(f"\n{'=' * 70}")
print("Tests de significativite termines.")
print(f"{'=' * 70}")
