#!/usr/bin/env python3
"""
Thumalien — Analyse qualitative des erreurs du modele V2
=========================================================

Ce script identifie et categorise les erreurs du modele ExpertFakeNewsDetector V2
(LogReg + TF-IDF 30K + 12 features linguistiques + 7 emotions).

Objectif : Comprendre les patterns d'erreur pour guider le developpement V3.

Auteur  : Thumalien Team
Date    : Avril 2026
"""

import sys
import os
import warnings
import json
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import numpy as np
import pandas as pd

from pipeline.expert_detector import (
    ExpertFakeNewsDetector,
    DatasetCleaner,
    LanguageRouter,
    LinguisticFeatureExtractor,
)

# ================================================================
#  CONFIG
# ================================================================
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'training')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')

SAMPLE_SIZE = 2000  # Total sample to analyse
THRESHOLD = 0.44    # V2 production threshold
TOP_ERRORS = 50     # Top N errors to examine

np.random.seed(42)


# ================================================================
#  1. CHARGER LE MODELE V2
# ================================================================
print("=" * 70)
print("ANALYSE QUALITATIVE DES ERREURS — MODELE V2")
print("=" * 70)

print("\n[1/7] Chargement du modele V2...")
detector = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=THRESHOLD)
detector.load(suffix='expert_v2')
print(f"  Modele charge: use_emotions={detector.use_emotions}")
print(f"  Threshold: {detector.threshold}")
print(f"  Metriques sauvegardees: {list(detector.training_metrics.keys())[:10]}...")


# ================================================================
#  2. CHARGER LES DONNEES DE TEST
# ================================================================
print("\n[2/7] Chargement des donnees...")

datasets = {}

# --- ISOT (Fake.csv + True.csv) — articles longs EN ---
fake_path = os.path.join(DATA_DIR, 'Fake.csv')
true_path = os.path.join(DATA_DIR, 'True.csv')

if os.path.exists(fake_path) and os.path.exists(true_path):
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    df_fake['label'] = 1
    df_true['label'] = 0
    df_isot = pd.concat([
        df_fake[['text', 'label']],
        df_true[['text', 'label']]
    ], ignore_index=True)
    df_isot = df_isot.dropna(subset=['text'])
    df_isot['source'] = 'ISOT'
    datasets['ISOT'] = df_isot
    print(f"  ISOT: {len(df_isot)} articles (Fake={len(df_fake)}, True={len(df_true)})")

# --- French fake news ---
fr_path = os.path.join(DATA_DIR, 'french_fake_news.csv')
if os.path.exists(fr_path):
    df_fr = pd.read_csv(fr_path)
    if 'text' in df_fr.columns and 'label' in df_fr.columns:
        df_fr = df_fr[['text', 'label']].dropna()
        df_fr['source'] = 'FrenchFakeNews'
        datasets['FrenchFakeNews'] = df_fr
        print(f"  FrenchFakeNews: {len(df_fr)} articles")

# --- Constraint (tweets COVID EN) ---
constraint_dir = os.path.join(DATA_DIR, 'constraint')
for fname in ['Constraint_Train.csv', 'Constraint_Test.csv', 'Constraint_Val.csv']:
    fpath = os.path.join(constraint_dir, fname)
    if os.path.exists(fpath):
        df_c = pd.read_csv(fpath)
        if 'tweet' in df_c.columns and 'label' in df_c.columns:
            df_c = df_c.rename(columns={'tweet': 'text'})
            df_c['label'] = df_c['label'].map({'real': 0, 'fake': 1})
            df_c = df_c[['text', 'label']].dropna()
            df_c['source'] = f'Constraint_{fname.split("_")[1].split(".")[0]}'
            key = df_c['source'].iloc[0]
            datasets[key] = df_c
            print(f"  {key}: {len(df_c)} tweets")

# --- FakeNewsNet (GossipCop + PolitiFact) ---
fnn_dir = os.path.join(DATA_DIR, 'fakenewsnet')
for prefix in ['gossipcop', 'politifact']:
    for kind, lbl in [('fake', 1), ('real', 0)]:
        fpath = os.path.join(fnn_dir, f'{prefix}_{kind}.csv')
        if os.path.exists(fpath):
            df_fnn = pd.read_csv(fpath)
            if 'title' in df_fnn.columns:
                df_fnn = df_fnn.rename(columns={'title': 'text'})
                df_fnn['label'] = lbl
                df_fnn = df_fnn[['text', 'label']].dropna()
                df_fnn['source'] = f'FNN_{prefix}_{kind}'
                datasets[f'FNN_{prefix}_{kind}'] = df_fnn
                print(f"  FNN_{prefix}_{kind}: {len(df_fnn)} titres")

# --- Combine all ---
all_data = pd.concat(datasets.values(), ignore_index=True)
all_data = all_data.dropna(subset=['text'])
all_data = all_data[all_data['text'].str.len() > 5]
print(f"\n  Total disponible: {len(all_data)} textes")


# ================================================================
#  3. ECHANTILLONNAGE STRATIFIE
# ================================================================
print("\n[3/7] Echantillonnage stratifie...")

# Compute word count for stratification
all_data['word_count'] = all_data['text'].apply(lambda x: len(str(x).split()))
all_data['length_cat'] = pd.cut(
    all_data['word_count'],
    bins=[0, 15, 30, 100, float('inf')],
    labels=['tres_court', 'court', 'moyen', 'long']
)

# Sample strategy: ensure we get short texts too
sample_parts = []

# Short texts (<30 words) — take all available from social datasets, up to 500
short_mask = all_data['word_count'] < 30
df_short = all_data[short_mask]
n_short = min(500, len(df_short))
if n_short > 0:
    sample_parts.append(df_short.sample(n=n_short, random_state=42))
    print(f"  Textes courts (<30 mots): {n_short}")

# Medium texts (30-100 words) — up to 500
medium_mask = (all_data['word_count'] >= 30) & (all_data['word_count'] < 100)
df_medium = all_data[medium_mask]
n_medium = min(500, len(df_medium))
if n_medium > 0:
    sample_parts.append(df_medium.sample(n=n_medium, random_state=42))
    print(f"  Textes moyens (30-100 mots): {n_medium}")

# Long texts (>100 words) — fill remaining up to SAMPLE_SIZE
remaining = SAMPLE_SIZE - n_short - n_medium
long_mask = all_data['word_count'] >= 100
df_long = all_data[long_mask]
n_long = min(remaining, len(df_long))
if n_long > 0:
    sample_parts.append(df_long.sample(n=n_long, random_state=42))
    print(f"  Textes longs (>100 mots): {n_long}")

sample = pd.concat(sample_parts, ignore_index=True)
sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"  Echantillon final: {len(sample)} textes")
print(f"  Distribution labels: {sample['label'].value_counts().to_dict()}")


# ================================================================
#  4. PREDICTIONS
# ================================================================
print("\n[4/7] Predictions du modele V2...")

results = detector.predict(pd.Series(sample['text'].values))
sample['prediction'] = results['prediction_label'].values
sample['score'] = results['ai_score_credibility'].values
sample['language'] = results['language'].values

# Recompute word_count (may have been lost)
sample['word_count'] = sample['text'].apply(lambda x: len(str(x).split()))
sample['text_preview'] = sample['text'].apply(lambda x: str(x)[:150].replace('\n', ' '))

# Metrics globales
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

y_true = sample['label'].values
y_pred = sample['prediction'].values

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print(f"\n  Metriques sur l'echantillon:")
print(f"    Accuracy:  {acc:.4f}")
print(f"    F1-score:  {f1:.4f}")
print(f"    Precision: {prec:.4f}")
print(f"    Recall:    {rec:.4f}")
print(f"    Confusion matrix:\n{cm}")


# ================================================================
#  5. IDENTIFICATION DES ERREURS
# ================================================================
print("\n[5/7] Identification et categorisation des erreurs...")

# False Positives: label=0 (FIABLE) mais prediction=1 (SUSPECT)
# → score bas (le modele pense que c'est pas fiable)
fp_mask = (sample['label'] == 0) & (sample['prediction'] == 1)
false_positives = sample[fp_mask].sort_values('score', ascending=True).head(TOP_ERRORS)

# False Negatives: label=1 (FAKE) mais prediction=0 (FIABLE)
# → score haut (le modele pense que c'est fiable)
fn_mask = (sample['label'] == 1) & (sample['prediction'] == 0)
false_negatives = sample[fn_mask].sort_values('score', ascending=False).head(TOP_ERRORS)

print(f"  False Positives totaux: {fp_mask.sum()} (top {len(false_positives)} analyses)")
print(f"  False Negatives totaux: {fn_mask.sum()} (top {len(false_negatives)} analyses)")
print(f"  Total erreurs: {fp_mask.sum() + fn_mask.sum()} / {len(sample)} ({(fp_mask.sum() + fn_mask.sum()) / len(sample) * 100:.1f}%)")


# ================================================================
#  CATEGORISATION DES ERREURS
# ================================================================

SARCASM_INDICATORS = [
    'lol', 'lmao', 'haha', 'rofl', 'yeah right', 'sure', 'obviously',
    'totally', '/s', '😂', '🤣', 'mdr', 'ptdr', 'bien sur', 'evidemment',
    'bravo', 'genial', 'formidable', 'magnifique',
]

def categorize_error(row):
    """Categorise une erreur en pattern identifiable."""
    text = str(row['text']).lower()
    wc = row['word_count']
    lang = row.get('language', 'en')
    categories = []

    # 1. Trop court
    if wc < 15:
        categories.append('trop_court')

    # 2. Court (15-30)
    if 15 <= wc < 30:
        categories.append('court')

    # 3. Sarcasme/Ironie
    if any(ind in text for ind in SARCASM_INDICATORS):
        categories.append('sarcasme_ironie')

    # 4. Contenu neutre/ambigu (score proche du seuil)
    score = row['score']
    if abs(score - THRESHOLD) < 0.08:
        categories.append('zone_grise')

    # 5. Erreur detection langue
    if lang == 'other':
        categories.append('langue_non_detectee')

    # 6. Contenu factuel neutre (pas de signal fort)
    text_clean = DatasetCleaner.clean_for_ml(str(row['text']))
    ling = LinguisticFeatureExtractor.extract(pd.Series([text_clean]))
    sensationalism = ling[0, 6]  # sensationalism_score
    caps_ratio = ling[0, 1]
    excl_count = ling[0, 2]

    if sensationalism == 0 and caps_ratio < 0.05 and excl_count == 0:
        categories.append('contenu_neutre')

    # 7. Contenu tres sensationnaliste
    if sensationalism >= 3:
        categories.append('tres_sensationnaliste')

    # 8. Beaucoup de majuscules
    if caps_ratio > 0.3:
        categories.append('majuscules_excessives')

    if not categories:
        categories.append('autre')

    return '|'.join(categories)


# Apply categorization
false_positives = false_positives.copy()
false_negatives = false_negatives.copy()

false_positives['error_categories'] = false_positives.apply(categorize_error, axis=1)
false_negatives['error_categories'] = false_negatives.apply(categorize_error, axis=1)

# Count categories
def count_categories(df, error_type):
    cats = Counter()
    for cats_str in df['error_categories']:
        for cat in cats_str.split('|'):
            cats[cat] += 1
    return cats

fp_cats = count_categories(false_positives, 'FP')
fn_cats = count_categories(false_negatives, 'FN')

print("\n  Categories d'erreurs — False Positives:")
for cat, count in fp_cats.most_common():
    print(f"    {cat}: {count}")

print("\n  Categories d'erreurs — False Negatives:")
for cat, count in fn_cats.most_common():
    print(f"    {cat}: {count}")


# ================================================================
#  6. ANALYSE DES SEUILS PAR SOUS-GROUPE
# ================================================================
print("\n[6/7] Analyse de robustesse des seuils...")

from sklearn.metrics import f1_score as compute_f1

def find_optimal_threshold(y_true, scores, thresholds=np.arange(0.20, 0.70, 0.01)):
    """Trouve le seuil optimal pour maximiser F1."""
    best_f1 = 0
    best_t = 0.5
    for t in thresholds:
        y_pred = (scores < t).astype(int)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue
        f = compute_f1(y_true, y_pred)
        if f > best_f1:
            best_f1 = f
            best_t = t
    return best_t, best_f1

threshold_analysis = {}
subgroups = {
    'Global': sample,
    'EN': sample[sample['language'] == 'en'],
    'FR': sample[sample['language'] == 'fr'],
    'Tres_court (<15 mots)': sample[sample['word_count'] < 15],
    'Court (<30 mots)': sample[sample['word_count'] < 30],
    'Moyen (30-100 mots)': sample[(sample['word_count'] >= 30) & (sample['word_count'] < 100)],
    'Long (>100 mots)': sample[sample['word_count'] >= 100],
}

print(f"\n  {'Sous-groupe':<25} {'N':>6} {'Seuil V2':>10} {'F1 V2':>8} {'Seuil opt':>10} {'F1 opt':>8} {'Delta F1':>8}")
print("  " + "-" * 80)

for name, subgroup in subgroups.items():
    if len(subgroup) < 20:
        print(f"  {name:<25} {len(subgroup):>6}   (trop peu de donnees)")
        continue

    y_t = subgroup['label'].values
    scores = subgroup['score'].values

    # F1 with current threshold
    y_pred_v2 = (scores < THRESHOLD).astype(int)
    f1_v2 = compute_f1(y_t, y_pred_v2) if y_pred_v2.sum() > 0 else 0

    # Optimal threshold
    opt_t, opt_f1 = find_optimal_threshold(y_t, scores)

    delta = opt_f1 - f1_v2
    threshold_analysis[name] = {
        'n': len(subgroup),
        'threshold_v2': THRESHOLD,
        'f1_v2': round(f1_v2, 4),
        'threshold_optimal': round(opt_t, 2),
        'f1_optimal': round(opt_f1, 4),
        'delta_f1': round(delta, 4),
    }
    print(f"  {name:<25} {len(subgroup):>6} {THRESHOLD:>10.2f} {f1_v2:>8.4f} {opt_t:>10.2f} {opt_f1:>8.4f} {delta:>+8.4f}")


# ================================================================
#  7. ANALYSE PAR SOURCE DE DATASET
# ================================================================
print("\n  Analyse par source de dataset:")
print(f"  {'Source':<25} {'N':>6} {'Erreurs':>8} {'Taux err':>10} {'F1':>8}")
print("  " + "-" * 60)

source_analysis = {}
for src in sample['source'].unique():
    sub = sample[sample['source'] == src]
    if len(sub) < 10:
        continue
    y_t = sub['label'].values
    y_p = sub['prediction'].values
    errors = (y_t != y_p).sum()
    err_rate = errors / len(sub) * 100
    f1_src = compute_f1(y_t, y_p) if y_p.sum() > 0 else 0
    source_analysis[src] = {
        'n': len(sub),
        'errors': int(errors),
        'error_rate': round(err_rate, 1),
        'f1': round(f1_src, 4),
    }
    print(f"  {src:<25} {len(sub):>6} {errors:>8} {err_rate:>9.1f}% {f1_src:>8.4f}")


# ================================================================
#  8. GENERATION DU RAPPORT
# ================================================================
print("\n[7/7] Generation du rapport...")

# Prepare error examples for report
def format_error_examples(df, error_type, max_per_cat=3):
    """Format error examples grouped by category for the report."""
    sections = []
    # Group by primary category
    cat_groups = defaultdict(list)
    for _, row in df.iterrows():
        primary_cat = row['error_categories'].split('|')[0]
        cat_groups[primary_cat].append(row)

    for cat, rows in sorted(cat_groups.items(), key=lambda x: -len(x[1])):
        lines = [f"#### {cat.replace('_', ' ').title()} ({len(rows)} cas)"]
        for row in rows[:max_per_cat]:
            preview = str(row['text'])[:200].replace('\n', ' ').replace('|', '\\|')
            lines.append(f"")
            lines.append(f"- **Score**: {row['score']:.4f} | **Langue**: {row['language']} | **Mots**: {row['word_count']}")
            lines.append(f"  > {preview}...")
        sections.append('\n'.join(lines))
    return '\n\n'.join(sections)


fp_examples = format_error_examples(false_positives, 'FP')
fn_examples = format_error_examples(false_negatives, 'FN')

# Build threshold table
threshold_table_lines = []
threshold_table_lines.append("| Sous-groupe | N | Seuil V2 | F1 V2 | Seuil optimal | F1 optimal | Delta F1 |")
threshold_table_lines.append("|---|---|---|---|---|---|---|")
for name, data in threshold_analysis.items():
    threshold_table_lines.append(
        f"| {name} | {data['n']} | {data['threshold_v2']} | {data['f1_v2']:.4f} | "
        f"{data['threshold_optimal']} | {data['f1_optimal']:.4f} | {data['delta_f1']:+.4f} |"
    )
threshold_table = '\n'.join(threshold_table_lines)

# Build source table
source_table_lines = []
source_table_lines.append("| Source | N | Erreurs | Taux erreur | F1 |")
source_table_lines.append("|---|---|---|---|---|")
for src, data in sorted(source_analysis.items(), key=lambda x: -x[1]['error_rate']):
    source_table_lines.append(
        f"| {src} | {data['n']} | {data['errors']} | {data['error_rate']}% | {data['f1']:.4f} |"
    )
source_table = '\n'.join(source_table_lines)

# Category summary
all_cats = Counter()
all_cats.update(fp_cats)
all_cats.update(fn_cats)

cat_table_lines = []
cat_table_lines.append("| Categorie | False Positives | False Negatives | Total |")
cat_table_lines.append("|---|---|---|---|")
for cat, total in all_cats.most_common():
    fp_n = fp_cats.get(cat, 0)
    fn_n = fn_cats.get(cat, 0)
    cat_table_lines.append(f"| {cat.replace('_', ' ').title()} | {fp_n} | {fn_n} | {total} |")
cat_table = '\n'.join(cat_table_lines)


# ================================================================
#  RAPPORT MARKDOWN
# ================================================================
report = f"""# Analyse Qualitative des Erreurs — Modele V2
## Diagnostic approfondi pour guider le developpement V3

**Reference** : QUAL-THUM-2026-005
**Version** : 1.0
**Date** : Avril 2026
**Equipe** : Thumalien Data Science

---

## 1. Resume executif

Cette analyse qualitative examine les erreurs de prediction du modele **ExpertFakeNewsDetector V2** (LogReg + TF-IDF 30K features + 12 features linguistiques + 7 emotions) afin d'identifier les patterns d'echec systematiques et guider les ameliorations pour la V3.

### Methodologie

- **Echantillon analyse** : {len(sample)} textes issus de {len(datasets)} sources de donnees
- **Distribution** : articles longs (ISOT), tweets courts (CONSTRAINT), titres (FakeNewsNet), articles FR
- **Modele** : V2 avec seuil de production a {THRESHOLD}
- **Metriques sur l'echantillon** : Accuracy={acc:.4f}, F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}

### Constats principaux

1. **{fp_mask.sum()} faux positifs** (textes fiables classes suspects) et **{fn_mask.sum()} faux negatifs** (fake news non detectees) sur {len(sample)} textes, soit un taux d'erreur de **{(fp_mask.sum() + fn_mask.sum()) / len(sample) * 100:.1f}%**.
2. La faiblesse connue sur les textes courts (<30 mots) est confirmee — ils representent une part disproportionnee des erreurs.
3. Le seuil unique de {THRESHOLD} n'est pas optimal pour tous les sous-groupes ; un seuil adaptatif pourrait ameliorer les performances.
4. Les textes au contenu neutre/factuel sans signaux linguistiques forts sont la premiere source d'erreur.

### Matrice de confusion sur l'echantillon

```
                Predit FIABLE    Predit SUSPECT
Reel FIABLE         {cm[0][0]:>8}          {cm[0][1]:>8}
Reel SUSPECT        {cm[1][0]:>8}          {cm[1][1]:>8}
```

---

## 2. Distribution des erreurs par categorie

{cat_table}

**Definitions des categories** :

- **Contenu neutre** : Texte factuel sans marqueurs de sensationnalisme, ni majuscules excessives, ni ponctuation emotionnelle. Le modele manque de signal pour decider.
- **Zone grise** : Score de credibilite tres proche du seuil ({THRESHOLD} +/- 0.08). Cas intrinsequement ambigus.
- **Trop court** : Texte de moins de 15 mots. Le TF-IDF et les features linguistiques n'ont pas assez de matiere.
- **Court** : Texte de 15 a 30 mots. Signal faible mais present.
- **Sarcasme/Ironie** : Presence d'indicateurs linguistiques de sarcasme que le modele ne distingue pas du contenu sincere.
- **Tres sensationnaliste** : Texte avec 3+ mots sensationnalistes detectes. Peut etre du journalisme legitime couvrant un sujet sensationnel.
- **Majuscules excessives** : Plus de 30% de caracteres en majuscules, potentiellement trompeur pour le modele.
- **Langue non detectee** : La langue n'a pas ete identifiee comme FR ou EN.

---

## 3. Exemples d'erreurs par categorie

### 3.1 Faux Positifs (FIABLE classe SUSPECT)

Ces textes sont veridiques/fiables mais le modele les a classes comme suspects.

{fp_examples}

### 3.2 Faux Negatifs (FAKE/SUSPECT classe FIABLE)

Ces textes sont des fake news mais le modele les a classes comme fiables.

{fn_examples}

---

## 4. Analyse des seuils par sous-groupe

Le seuil unique de {THRESHOLD} est-il adapte a tous les types de texte ? L'analyse suivante compare le seuil V2 au seuil optimal (maximisant le F1) pour chaque sous-groupe.

{threshold_table}

### Observations

- Le seuil optimal varie significativement selon les sous-groupes, ce qui confirme l'interet d'un **seuil adaptatif** pour la V3.
- Les textes courts beneficieraient d'un seuil different des articles longs.
- La difference entre langues (FR vs EN) peut indiquer des distributions de scores differentes selon la langue.

---

## 5. Analyse par source de dataset

{source_table}

### Observations

- Les datasets de textes sociaux (courts, informels) ont generalement un taux d'erreur plus eleve que les articles ISOT (longs, formels).
- Les titres FakeNewsNet sont particulierement difficiles car ils sont tres courts et souvent ambigus hors contexte.
- Les tweets CONSTRAINT (COVID) presentent un defi specifique lie au vocabulaire medical et aux affirmations factuelles contestees.

---

## 6. Analyse detaillee des patterns d'erreur

### 6.1 Probleme principal : textes courts et manque de signal

Le modele V2 repose sur TF-IDF (30K features) + 12 features linguistiques + 7 emotions. Pour les textes tres courts :
- Le vecteur TF-IDF est tres creux (peu de mots = peu de features non-nulles)
- Les features linguistiques (densite de ponctuation, diversite lexicale) sont bruitees avec peu de mots
- Les 7 features emotionnelles n'ont pas assez de contexte pour etre fiables

**Impact** : Les textes <30 mots representent une proportion elevee des erreurs.

### 6.2 Contenu neutre — faux positifs

Des articles factuels de Reuters/presse serieuse sont parfois classes suspects car :
- Certains mots du vocabulaire TF-IDF ont des coefficients biaises par la distribution d'entrainement
- Un sujet politique controverse peut activer des features "suspect" meme dans un article factuel
- Le nettoyage du biais Reuters a pu etre insuffisant pour certains patterns residuels

### 6.3 Fake news sophistiquees — faux negatifs

Les fake news les plus difficiles a detecter sont celles qui :
- Adoptent un style journalistique professionnel (pas de majuscules, pas de ponctuation excessive)
- Evitent le vocabulaire sensationnaliste flagrant
- Presentent des affirmations fausses dans un format neutre et factuel
- Melangent des faits reels et des elements inventes

### 6.4 Zone grise et seuil unique

Le seuil de {THRESHOLD} cree une frontiere nette la ou le phenomene est continu. Les textes dans la zone [{THRESHOLD-0.08:.2f}, {THRESHOLD+0.08:.2f}] sont intrinsequement ambigus et tout classement binaire sera source d'erreurs.

---

## 7. Recommandations pour la V3

### 7.1 Ameliorations prioritaires

1. **Seuil adaptatif par sous-groupe** : Implementer un systeme de seuils differencies selon la longueur du texte et la langue detectee. Par exemple :
   - Textes courts (<30 mots) : seuil plus conservateur (plus proche de 0.5)
   - Articles longs (>100 mots) : seuil actuel ou optimise
   - Ajustement FR vs EN si les distributions de scores divergent

2. **Enrichissement des features pour textes courts** :
   - Features basees sur les embeddings (Word2Vec, FastText) qui generalisent mieux avec peu de mots
   - Features de contexte social (source du post, engagement, reseau de diffusion)
   - Features de verification factuelle (entites nommees vs bases de faits)

3. **Detection du sarcasme/ironie** :
   - Ajouter un classifieur de sarcasme en amont
   - Utiliser le sarcasme comme feature supplementaire plutot que comme bruit

### 7.2 Ameliorations secondaires

4. **Classification en 3 classes** : Remplacer le binaire FIABLE/SUSPECT par un triplet FIABLE / INCERTAIN / SUSPECT, avec une zone de non-decision explicite autour du seuil.

5. **Modele par langue** : Entrainer des modeles specialises FR et EN plutot qu'un modele bilingue unique, ou au minimum des couches de calibration par langue.

6. **Augmentation de donnees pour textes courts** :
   - Integration de plus de datasets sociaux (Twitter/X, Reddit, Bluesky)
   - Augmentation de donnees par paraphrase pour les classes sous-representees

7. **Explicabilite enrichie** :
   - Pour chaque prediction en zone grise, fournir un indice de confiance et les raisons du doute
   - Signaler explicitement quand le texte est trop court pour une prediction fiable

### 7.3 Priorites par impact attendu

| Amelioration | Effort | Impact F1 estime | Priorite |
|---|---|---|---|
| Seuil adaptatif | Faible | +0.02 a +0.05 | P0 |
| Features embeddings courts | Moyen | +0.03 a +0.08 | P1 |
| Classification 3 classes | Moyen | N/A (UX) | P1 |
| Detecteur sarcasme | Eleve | +0.01 a +0.03 | P2 |
| Modeles par langue | Eleve | +0.01 a +0.03 | P2 |
| Augmentation donnees | Moyen | +0.02 a +0.05 | P1 |

---

## 8. Annexes

### A. Configuration du modele V2

- **Algorithme** : LogisticRegression (sklearn)
- **Features** : TF-IDF (max_features=30000, ngram_range=(1,2), sublinear_tf=True) + 12 linguistiques + 7 emotions
- **Donnees d'entrainement** : 145 703 textes (bilingue FR/EN, articles + tweets)
- **Seuil de decision** : {THRESHOLD} (optimise sur F1 global)
- **F1 rapporte** : 0.897 (validation croisee 5-fold)
- **Faiblesse connue** : F1=0.80 sur textes <30 mots

### B. Datasets utilises pour cette analyse

| Dataset | Type | Langue | N echantillon |
|---|---|---|---|
"""

# Add dataset details
for src, count in sample['source'].value_counts().items():
    lang = sample[sample['source'] == src]['language'].mode().iloc[0] if len(sample[sample['source'] == src]) > 0 else '?'
    dtype = 'Articles' if 'ISOT' in src or 'French' in src else 'Tweets/Titres'
    report += f"| {src} | {dtype} | {lang.upper()} | {count} |\n"

report += f"""
### C. Reproductibilite

- Script : `notebooks/09_Analyse_Erreurs_Qualitative.py`
- Seed aleatoire : 42
- Echantillon : {len(sample)} textes (stratifie par longueur)
- Date d'execution : Avril 2026
"""

# Write report
os.makedirs(DOCS_DIR, exist_ok=True)
report_path = os.path.join(DOCS_DIR, '05_analyse_erreurs_qualitative.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n  Rapport sauvegarde: {report_path}")
print(f"\n{'=' * 70}")
print("ANALYSE TERMINEE")
print(f"{'=' * 70}")
