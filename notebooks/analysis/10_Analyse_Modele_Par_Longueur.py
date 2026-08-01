"""
Thumalien — Analyse du modèle ExpertFakeNewsDetector V2 par longueur de texte
==============================================================================

Produit :
- Matrice de confusion segmentée par longueur de texte
- Métriques par langue (FR vs EN)
- Analyse de calibration (ECE)
- Top coefficients TF-IDF (mots discriminants)
- Recommandations de mise à jour de la liste de mots sensationnalistes
- Rapport markdown sauvegardé dans docs/06_analyse_modele_par_longueur.md

Auteur : Thumalien Team
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Ajout du path projet
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix
)

from src.pipeline.expert_detector import (
    ExpertFakeNewsDetector,
    DatasetCleaner,
    LinguisticFeatureExtractor,
    LanguageRouter,
)

# ============================================================
# Configuration
# ============================================================
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'training')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')
THRESHOLD = 0.44

LENGTH_BINS = {
    'Ultra-court (<15 mots)': (0, 15),
    'Court (15-30 mots)': (15, 30),
    'Moyen (30-100 mots)': (30, 100),
    'Long (100-300 mots)': (100, 300),
    'Très long (>300 mots)': (300, float('inf')),
}

# ============================================================
# 1. Chargement du modèle V2
# ============================================================
print("=" * 70)
print("CHARGEMENT DU MODÈLE V2")
print("=" * 70)

detector = ExpertFakeNewsDetector(
    model_dir=MODEL_DIR,
    use_emotions=True,
    threshold=THRESHOLD,
)
detector.load(suffix='expert_v2')
print(f"Modèle chargé : {MODEL_DIR}/model_expert_v2.pkl")
print(f"Seuil : {THRESHOLD}")
print(f"Émotions : {detector.use_emotions}")
if detector.training_metrics:
    for k, v in detector.training_metrics.items():
        if k not in ('y_true', 'y_pred', 'y_proba'):
            print(f"  {k}: {v}")

# ============================================================
# 2. Chargement de TOUS les datasets
# ============================================================
print("\n" + "=" * 70)
print("CHARGEMENT DES DATASETS")
print("=" * 70)

datasets = {}

# 2a. ISOT (True.csv / Fake.csv) — articles longs EN
try:
    df_isot = DatasetCleaner.prepare_clean_dataset(
        fake_path=os.path.join(DATA_DIR, 'Fake.csv'),
        true_path=os.path.join(DATA_DIR, 'True.csv'),
        remove_short=5,  # On garde les courts pour l'analyse
    )
    df_isot['source'] = 'ISOT'
    df_isot['language_expected'] = 'en'
    datasets['ISOT'] = df_isot
    print(f"ISOT : {len(df_isot)} textes | labels: {df_isot['label'].value_counts().to_dict()}")
except Exception as e:
    print(f"ISOT : ERREUR — {e}")

# 2b. FakeNewsNet (titres courts EN)
try:
    df_fnn = DatasetCleaner.load_fakenewsnet(
        data_dir=os.path.join(DATA_DIR, 'fakenewsnet'),
        remove_short=3,
    )
    df_fnn['source'] = 'FakeNewsNet'
    df_fnn['language_expected'] = 'en'
    datasets['FakeNewsNet'] = df_fnn
    print(f"FakeNewsNet : {len(df_fnn)} textes | labels: {df_fnn['label'].value_counts().to_dict()}")
except Exception as e:
    print(f"FakeNewsNet : ERREUR — {e}")

# 2c. CONSTRAINT (tweets COVID EN)
try:
    df_cst = DatasetCleaner.load_constraint(
        data_dir=os.path.join(DATA_DIR, 'constraint'),
        remove_short=3,
    )
    df_cst['source'] = 'CONSTRAINT'
    df_cst['language_expected'] = 'en'
    datasets['CONSTRAINT'] = df_cst
    print(f"CONSTRAINT : {len(df_cst)} textes | labels: {df_cst['label'].value_counts().to_dict()}")
except Exception as e:
    print(f"CONSTRAINT : ERREUR — {e}")

# 2d. Kaggle FR (articles FR)
try:
    df_fr = DatasetCleaner.load_kaggle_french(
        kaggle_dir=os.path.join(DATA_DIR, 'kaggle_fr'),
        remove_short=5,
    )
    df_fr['source'] = 'KaggleFR'
    df_fr['language_expected'] = 'fr'
    datasets['KaggleFR'] = df_fr
    print(f"KaggleFR : {len(df_fr)} textes | labels: {df_fr['label'].value_counts().to_dict()}")
except Exception as e:
    print(f"KaggleFR : ERREUR — {e}")

# 2e. Credibility Corpus (tweets FR+EN)
try:
    df_cred = DatasetCleaner.load_credibility_corpus(
        data_dir=os.path.join(DATA_DIR, 'credibility_corpus'),
        remove_short=3,
    )
    df_cred['source'] = 'CredibilityCorpus'
    datasets['CredibilityCorpus'] = df_cred
    print(f"CredibilityCorpus : {len(df_cred)} textes | labels: {df_cred['label'].value_counts().to_dict()}")
except Exception as e:
    print(f"CredibilityCorpus : ERREUR — {e}")

# Combiner tous les datasets
all_dfs = list(datasets.values())
if not all_dfs:
    print("ERREUR CRITIQUE : aucun dataset chargé.")
    sys.exit(1)

df_all = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal combiné : {len(df_all)} textes")
print(f"Distribution labels : {df_all['label'].value_counts().to_dict()}")
print(f"Sources : {df_all['source'].value_counts().to_dict()}")

# ============================================================
# 3. Prédictions sur tous les textes
# ============================================================
print("\n" + "=" * 70)
print("PRÉDICTIONS (cela peut prendre quelques minutes...)")
print("=" * 70)

# On utilise text_original pour les prédictions (plus réaliste)
# Si text_original n'existe pas, fallback sur text_clean
texts_for_pred = df_all['text_original'] if 'text_original' in df_all.columns else df_all['text_clean']
texts_for_pred = texts_for_pred.fillna('').astype(str)

# Prédiction par batch pour éviter les problèmes mémoire
BATCH_SIZE = 5000
all_results = []
for i in range(0, len(texts_for_pred), BATCH_SIZE):
    batch = texts_for_pred.iloc[i:i+BATCH_SIZE].reset_index(drop=True)
    batch_results = detector.predict(pd.Series(batch))
    all_results.append(batch_results)
    print(f"  Batch {i//BATCH_SIZE + 1}: {len(batch)} textes traités")

results_df = pd.concat(all_results, ignore_index=True)

# Ajouter les résultats au dataframe principal
df_all['y_pred'] = results_df['prediction_label'].values
df_all['credibility_score'] = results_df['ai_score_credibility'].values
df_all['detected_language'] = results_df['language'].values

# Calculer le nombre de mots
df_all['word_count'] = df_all['text_clean'].fillna('').str.split().str.len()

print(f"Prédictions terminées : {len(df_all)} textes")
print(f"Distribution prédictions : {df_all['y_pred'].value_counts().to_dict()}")

# ============================================================
# 4. Analyse par segment de longueur
# ============================================================
print("\n" + "=" * 70)
print("ANALYSE PAR SEGMENT DE LONGUEUR")
print("=" * 70)

length_results = {}
for segment_name, (low, high) in LENGTH_BINS.items():
    mask = (df_all['word_count'] >= low) & (df_all['word_count'] < high)
    df_seg = df_all[mask]

    if len(df_seg) == 0:
        print(f"\n{segment_name}: AUCUN TEXTE")
        continue

    y_true = df_seg['label'].values
    y_pred = df_seg['y_pred'].values
    scores = df_seg['credibility_score'].values

    # Métriques
    acc = accuracy_score(y_true, y_pred)

    # Gestion des cas où une seule classe est présente
    try:
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
    except:
        f1 = prec = rec = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # Scores moyens
    true_pos_mask = (y_true == 1) & (y_pred == 1)
    true_neg_mask = (y_true == 0) & (y_pred == 0)
    avg_score_tp = scores[true_pos_mask].mean() if true_pos_mask.sum() > 0 else float('nan')
    avg_score_tn = scores[true_neg_mask].mean() if true_neg_mask.sum() > 0 else float('nan')

    result = {
        'count': len(df_seg),
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'avg_score_tp': avg_score_tp,
        'avg_score_tn': avg_score_tn,
        'sources': df_seg['source'].value_counts().to_dict(),
    }
    length_results[segment_name] = result

    print(f"\n{segment_name} (n={len(df_seg)}):")
    print(f"  Accuracy={acc:.4f} | F1={f1:.4f} | Precision={prec:.4f} | Recall={rec:.4f}")
    print(f"  TP={tp} | FP={fp} | TN={tn} | FN={fn}")
    print(f"  Score moyen TP (vrais suspects)={avg_score_tp:.4f} | TN (vrais fiables)={avg_score_tn:.4f}")
    print(f"  Sources: {result['sources']}")

# ============================================================
# 5. Analyse par langue
# ============================================================
print("\n" + "=" * 70)
print("ANALYSE PAR LANGUE")
print("=" * 70)

lang_results = {}
for lang in ['fr', 'en', 'other']:
    mask = df_all['detected_language'] == lang
    df_lang = df_all[mask]

    if len(df_lang) == 0:
        print(f"\n{lang.upper()}: AUCUN TEXTE")
        continue

    y_true = df_lang['label'].values
    y_pred = df_lang['y_pred'].values
    scores = df_lang['credibility_score'].values

    acc = accuracy_score(y_true, y_pred)
    try:
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
    except:
        f1 = prec = rec = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    true_pos_mask = (y_true == 1) & (y_pred == 1)
    true_neg_mask = (y_true == 0) & (y_pred == 0)
    avg_score_tp = scores[true_pos_mask].mean() if true_pos_mask.sum() > 0 else float('nan')
    avg_score_tn = scores[true_neg_mask].mean() if true_neg_mask.sum() > 0 else float('nan')

    result = {
        'count': len(df_lang),
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'avg_score_tp': avg_score_tp,
        'avg_score_tn': avg_score_tn,
        'sources': df_lang['source'].value_counts().to_dict(),
    }
    lang_results[lang] = result

    print(f"\n{lang.upper()} (n={len(df_lang)}):")
    print(f"  Accuracy={acc:.4f} | F1={f1:.4f} | Precision={prec:.4f} | Recall={rec:.4f}")
    print(f"  TP={tp} | FP={fp} | TN={tn} | FN={fn}")
    print(f"  Score moyen TP={avg_score_tp:.4f} | TN={avg_score_tn:.4f}")
    print(f"  Sources: {result['sources']}")

# ============================================================
# 5b. Croisement longueur x langue
# ============================================================
print("\n" + "=" * 70)
print("CROISEMENT LONGUEUR x LANGUE")
print("=" * 70)

cross_results = {}
for lang in ['fr', 'en']:
    for segment_name, (low, high) in LENGTH_BINS.items():
        mask = (df_all['word_count'] >= low) & (df_all['word_count'] < high) & (df_all['detected_language'] == lang)
        df_cross = df_all[mask]
        if len(df_cross) < 10:
            continue
        y_true = df_cross['label'].values
        y_pred = df_cross['y_pred'].values
        try:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
        except:
            f1 = acc = 0.0
        key = f"{lang.upper()} | {segment_name}"
        cross_results[key] = {'count': len(df_cross), 'f1': f1, 'accuracy': acc}
        print(f"  {key}: n={len(df_cross)} | F1={f1:.4f} | Acc={acc:.4f}")

# ============================================================
# 6. Analyse de calibration
# ============================================================
print("\n" + "=" * 70)
print("ANALYSE DE CALIBRATION")
print("=" * 70)

scores = df_all['credibility_score'].values
y_true = df_all['label'].values

# Le score est P(fiable), donc pour le calcul de calibration :
# - score élevé = prédit fiable (label 0)
# - score faible = prédit suspect (label 1)
# On inverse pour avoir P(suspect) = 1 - score
p_suspect = 1 - scores

n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
calibration_data = []
ece = 0.0

for i in range(n_bins):
    low_edge = bin_edges[i]
    high_edge = bin_edges[i + 1]
    mask = (p_suspect >= low_edge) & (p_suspect < high_edge)
    if i == n_bins - 1:  # dernier bin inclut 1.0
        mask = (p_suspect >= low_edge) & (p_suspect <= high_edge)

    n_in_bin = mask.sum()
    if n_in_bin == 0:
        calibration_data.append({
            'bin': f'{low_edge:.1f}-{high_edge:.1f}',
            'count': 0,
            'avg_predicted': float('nan'),
            'avg_actual': float('nan'),
            'gap': float('nan'),
        })
        continue

    avg_predicted = p_suspect[mask].mean()
    avg_actual = y_true[mask].mean()  # fraction réellement suspect (label=1)
    gap = abs(avg_actual - avg_predicted)
    ece += (n_in_bin / len(y_true)) * gap

    calibration_data.append({
        'bin': f'{low_edge:.1f}-{high_edge:.1f}',
        'count': int(n_in_bin),
        'avg_predicted': round(avg_predicted, 4),
        'avg_actual': round(avg_actual, 4),
        'gap': round(gap, 4),
    })

    print(f"  Bin [{low_edge:.1f}, {high_edge:.1f}): n={n_in_bin:>6} | "
          f"P(suspect) prédit={avg_predicted:.4f} | "
          f"Taux réel suspect={avg_actual:.4f} | "
          f"Écart={gap:.4f}")

print(f"\nECE (Expected Calibration Error) = {ece:.4f}")

# ============================================================
# 7. Analyse des coefficients TF-IDF
# ============================================================
print("\n" + "=" * 70)
print("ANALYSE DES COEFFICIENTS TF-IDF (MOTS DISCRIMINANTS)")
print("=" * 70)

# Extraire les coefficients du modèle
model = detector.model
vectorizer = detector.vectorizer

if hasattr(model, 'coef_'):
    coefs = model.coef_[0]  # Shape: (n_features,)
    feature_names = vectorizer.get_feature_names_out()
    n_tfidf = len(feature_names)

    # Les coefficients TF-IDF sont les premiers n_tfidf
    tfidf_coefs = coefs[:n_tfidf]

    # Coefficients linguistiques
    ling_names = LinguisticFeatureExtractor.FEATURE_NAMES
    ling_coefs = coefs[n_tfidf:n_tfidf + len(ling_names)]

    # Note: coef positif => pousse vers classe 1 (FAKE/SUSPECT)
    # coef négatif => pousse vers classe 0 (VRAI/FIABLE)

    # Top 30 vers SUSPECT (coefficients les plus positifs)
    top_suspect_idx = np.argsort(tfidf_coefs)[-30:][::-1]
    top_suspect_words = [(feature_names[i], tfidf_coefs[i]) for i in top_suspect_idx]

    # Top 30 vers FIABLE (coefficients les plus négatifs)
    top_fiable_idx = np.argsort(tfidf_coefs)[:30]
    top_fiable_words = [(feature_names[i], tfidf_coefs[i]) for i in top_fiable_idx]

    print("\n--- TOP 30 MOTS → SUSPECT (coef positif) ---")
    for word, coef in top_suspect_words:
        print(f"  {word:30s} coef={coef:+.4f}")

    print("\n--- TOP 30 MOTS → FIABLE (coef négatif) ---")
    for word, coef in top_fiable_words:
        print(f"  {word:30s} coef={coef:+.4f}")

    print("\n--- COEFFICIENTS DES FEATURES LINGUISTIQUES ---")
    for name, coef in zip(ling_names, ling_coefs):
        print(f"  {name:25s} coef={coef:+.4f}")

    # Features émotionnelles si présentes
    if detector.use_emotions:
        from src.pipeline.expert_detector import EmotionFeatureExtractor
        emo_names = EmotionFeatureExtractor.FEATURE_NAMES
        emo_start = n_tfidf + len(ling_names)
        emo_coefs = coefs[emo_start:emo_start + len(emo_names)]
        print("\n--- COEFFICIENTS DES FEATURES ÉMOTIONNELLES ---")
        for name, coef in zip(emo_names, emo_coefs):
            print(f"  {name:25s} coef={coef:+.4f}")

    # ============================================================
    # 8. Comparaison avec les listes de mots sensationnalistes
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSE DES MOTS SENSATIONNALISTES")
    print("=" * 70)

    current_en = LinguisticFeatureExtractor.SENSATIONALIST_EN
    current_fr = LinguisticFeatureExtractor.SENSATIONALIST_FR

    print(f"Mots sensationnalistes actuels EN : {len(current_en)}")
    print(f"Mots sensationnalistes actuels FR : {len(current_fr)}")

    # Identifier les mots TF-IDF fortement suspects qui ne sont pas dans les listes
    suspect_words_set = set(w for w, c in top_suspect_words)

    # Mots suspects dans le top TF-IDF mais absents des listes sensationnalistes
    missing_from_lists = []
    for word, coef in top_suspect_words:
        word_lower = word.lower()
        if word_lower not in current_en and word_lower not in current_fr:
            missing_from_lists.append((word, coef))

    print(f"\nMots discriminants SUSPECT absents des listes sensationnalistes ({len(missing_from_lists)}):")
    for word, coef in missing_from_lists:
        print(f"  {word:30s} coef={coef:+.4f}")

    # Vérifier si les mots des listes apparaissent dans le vocabulaire TF-IDF
    tfidf_vocab = set(feature_names)

    print("\nMots sensationnalistes EN dans le vocabulaire TF-IDF :")
    en_in_vocab = 0
    en_not_in_vocab = []
    for word in sorted(current_en):
        if word in tfidf_vocab:
            idx = list(feature_names).index(word)
            c = tfidf_coefs[idx]
            direction = "→ SUSPECT" if c > 0 else "→ FIABLE"
            print(f"  {word:30s} coef={c:+.4f} {direction}")
            en_in_vocab += 1
        else:
            en_not_in_vocab.append(word)
    print(f"  ({en_in_vocab}/{len(current_en)} dans le vocabulaire TF-IDF)")
    if en_not_in_vocab:
        print(f"  Absents du vocabulaire : {', '.join(en_not_in_vocab)}")

    print("\nMots sensationnalistes FR dans le vocabulaire TF-IDF :")
    fr_in_vocab = 0
    fr_not_in_vocab = []
    for word in sorted(current_fr):
        if word in tfidf_vocab:
            idx = list(feature_names).index(word)
            c = tfidf_coefs[idx]
            direction = "→ SUSPECT" if c > 0 else "→ FIABLE"
            print(f"  {word:30s} coef={c:+.4f} {direction}")
            fr_in_vocab += 1
        else:
            fr_not_in_vocab.append(word)
    print(f"  ({fr_in_vocab}/{len(current_fr)} dans le vocabulaire TF-IDF)")
    if fr_not_in_vocab:
        print(f"  Absents du vocabulaire : {', '.join(fr_not_in_vocab)}")

else:
    print("ATTENTION: Le modèle n'expose pas coef_ (pas un modèle linéaire)")
    top_suspect_words = []
    top_fiable_words = []
    missing_from_lists = []
    ling_coefs = []
    ling_names = []

# ============================================================
# 9. Génération du rapport Markdown
# ============================================================
print("\n" + "=" * 70)
print("GÉNÉRATION DU RAPPORT")
print("=" * 70)

report_lines = []
report_lines.append("# Analyse du modèle ExpertFakeNewsDetector V2 par longueur de texte")
report_lines.append("")
report_lines.append(f"**Date** : 2026-04-03  ")
report_lines.append(f"**Modèle** : ExpertFakeNewsDetector V2 (LogReg + TF-IDF + 12 linguistiques + 7 émotions)  ")
report_lines.append(f"**Seuil de décision** : {THRESHOLD}  ")
report_lines.append(f"**Données évaluées** : {len(df_all)} textes provenant de {len(datasets)} sources  ")
report_lines.append("")

# Sources
report_lines.append("## 1. Sources de données")
report_lines.append("")
report_lines.append("| Source | Nombre | Fiable (0) | Suspect (1) | Type |")
report_lines.append("|--------|--------|------------|-------------|------|")
for name, df_src in datasets.items():
    n = len(df_src)
    n0 = (df_src['label'] == 0).sum()
    n1 = (df_src['label'] == 1).sum()
    types = {
        'ISOT': 'Articles longs EN',
        'FakeNewsNet': 'Titres courts EN',
        'CONSTRAINT': 'Tweets COVID EN',
        'KaggleFR': 'Articles FR',
        'CredibilityCorpus': 'Tweets FR+EN',
    }
    report_lines.append(f"| {name} | {n:,} | {n0:,} | {n1:,} | {types.get(name, '?')} |")
report_lines.append(f"| **Total** | **{len(df_all):,}** | **{(df_all['label']==0).sum():,}** | **{(df_all['label']==1).sum():,}** | |")
report_lines.append("")

# Performance par longueur
report_lines.append("## 2. Performance par segment de longueur")
report_lines.append("")
report_lines.append("| Segment | N | Accuracy | F1 | Precision | Recall | TP | FP | TN | FN |")
report_lines.append("|---------|---|----------|-----|-----------|--------|----|----|----|----|")
for segment_name, r in length_results.items():
    report_lines.append(
        f"| {segment_name} | {r['count']:,} | {r['accuracy']:.4f} | {r['f1']:.4f} | "
        f"{r['precision']:.4f} | {r['recall']:.4f} | {r['tp']:,} | {r['fp']:,} | {r['tn']:,} | {r['fn']:,} |"
    )
report_lines.append("")

# Scores de crédibilité moyens
report_lines.append("### Scores de crédibilité moyens par segment")
report_lines.append("")
report_lines.append("| Segment | Score moyen TP (vrais suspects) | Score moyen TN (vrais fiables) | Écart |")
report_lines.append("|---------|--------------------------------|-------------------------------|-------|")
for segment_name, r in length_results.items():
    tp_s = f"{r['avg_score_tp']:.4f}" if not np.isnan(r['avg_score_tp']) else "N/A"
    tn_s = f"{r['avg_score_tn']:.4f}" if not np.isnan(r['avg_score_tn']) else "N/A"
    if not np.isnan(r['avg_score_tp']) and not np.isnan(r['avg_score_tn']):
        ecart = f"{r['avg_score_tn'] - r['avg_score_tp']:.4f}"
    else:
        ecart = "N/A"
    report_lines.append(f"| {segment_name} | {tp_s} | {tn_s} | {ecart} |")
report_lines.append("")

report_lines.append("> **Interprétation** : Le score de crédibilité est P(fiable). Un vrai suspect (TP) devrait avoir un score bas, un vrai fiable (TN) un score élevé. L'écart mesure la séparation entre les deux classes.")
report_lines.append("")

# Performance par langue
report_lines.append("## 3. Performance par langue")
report_lines.append("")
report_lines.append("| Langue | N | Accuracy | F1 | Precision | Recall | TP | FP | TN | FN |")
report_lines.append("|--------|---|----------|-----|-----------|--------|----|----|----|----|")
for lang, r in lang_results.items():
    report_lines.append(
        f"| {lang.upper()} | {r['count']:,} | {r['accuracy']:.4f} | {r['f1']:.4f} | "
        f"{r['precision']:.4f} | {r['recall']:.4f} | {r['tp']:,} | {r['fp']:,} | {r['tn']:,} | {r['fn']:,} |"
    )
report_lines.append("")

# Croisement longueur x langue
if cross_results:
    report_lines.append("### Croisement longueur x langue")
    report_lines.append("")
    report_lines.append("| Langue | Segment | N | F1 | Accuracy |")
    report_lines.append("|--------|---------|---|-----|----------|")
    for key, r in cross_results.items():
        parts = key.split(' | ', 1)
        lang_str = parts[0] if len(parts) > 1 else key
        seg_str = parts[1] if len(parts) > 1 else ""
        report_lines.append(f"| {lang_str} | {seg_str} | {r['count']:,} | {r['f1']:.4f} | {r['accuracy']:.4f} |")
    report_lines.append("")

# Calibration
report_lines.append("## 4. Analyse de calibration")
report_lines.append("")
report_lines.append(f"**ECE (Expected Calibration Error) = {ece:.4f}**")
report_lines.append("")
report_lines.append("> L'ECE mesure l'écart moyen entre la confiance du modèle et la précision réelle. ")
report_lines.append("> Un ECE < 0.05 indique un modèle bien calibré. Un ECE > 0.10 suggère une sur-confiance ou sous-confiance.")
report_lines.append("")
report_lines.append("| Bin P(suspect) | N | P(suspect) prédit | Taux réel suspect | Écart |")
report_lines.append("|----------------|---|-------------------|-------------------|-------|")
for row in calibration_data:
    n = row['count']
    pred = f"{row['avg_predicted']:.4f}" if not np.isnan(row.get('avg_predicted', float('nan'))) else "N/A"
    actual = f"{row['avg_actual']:.4f}" if not np.isnan(row.get('avg_actual', float('nan'))) else "N/A"
    gap = f"{row['gap']:.4f}" if not np.isnan(row.get('gap', float('nan'))) else "N/A"
    report_lines.append(f"| {row['bin']} | {n:,} | {pred} | {actual} | {gap} |")
report_lines.append("")

# Mots discriminants
report_lines.append("## 5. Mots les plus discriminants (coefficients TF-IDF)")
report_lines.append("")

if top_suspect_words:
    report_lines.append("### Top 30 mots poussant vers SUSPECT (coef positif)")
    report_lines.append("")
    report_lines.append("| Rang | Mot | Coefficient |")
    report_lines.append("|------|-----|-------------|")
    for i, (word, coef) in enumerate(top_suspect_words, 1):
        report_lines.append(f"| {i} | `{word}` | {coef:+.4f} |")
    report_lines.append("")

    report_lines.append("### Top 30 mots poussant vers FIABLE (coef négatif)")
    report_lines.append("")
    report_lines.append("| Rang | Mot | Coefficient |")
    report_lines.append("|------|-----|-------------|")
    for i, (word, coef) in enumerate(top_fiable_words, 1):
        report_lines.append(f"| {i} | `{word}` | {coef:+.4f} |")
    report_lines.append("")

# Features linguistiques
if ling_names and len(ling_coefs) > 0:
    report_lines.append("### Coefficients des features linguistiques")
    report_lines.append("")
    report_lines.append("| Feature | Coefficient | Direction |")
    report_lines.append("|---------|-------------|-----------|")
    for name, coef in sorted(zip(ling_names, ling_coefs), key=lambda x: abs(x[1]), reverse=True):
        direction = "SUSPECT" if coef > 0 else "FIABLE"
        report_lines.append(f"| `{name}` | {coef:+.4f} | {direction} |")
    report_lines.append("")

# Features émotionnelles
if detector.use_emotions and hasattr(model, 'coef_'):
    report_lines.append("### Coefficients des features émotionnelles")
    report_lines.append("")
    report_lines.append("| Émotion | Coefficient | Direction |")
    report_lines.append("|---------|-------------|-----------|")
    for name, coef in sorted(zip(emo_names, emo_coefs), key=lambda x: abs(x[1]), reverse=True):
        direction = "SUSPECT" if coef > 0 else "FIABLE"
        report_lines.append(f"| `{name}` | {coef:+.4f} | {direction} |")
    report_lines.append("")

# Recommandations mots sensationnalistes
report_lines.append("## 6. Mise à jour recommandée des listes de mots sensationnalistes")
report_lines.append("")

if missing_from_lists:
    report_lines.append("### Mots à fort coefficient SUSPECT absents des listes actuelles")
    report_lines.append("")
    report_lines.append("Ces mots apparaissent dans le top 30 des coefficients TF-IDF les plus positifs,")
    report_lines.append("mais ne figurent pas dans `SENSATIONALIST_EN` ni `SENSATIONALIST_FR` :")
    report_lines.append("")
    report_lines.append("| Mot | Coefficient | Recommandation |")
    report_lines.append("|-----|-------------|----------------|")
    for word, coef in missing_from_lists:
        # Heuristique simple pour la langue
        has_accent = any(c in word for c in 'àâäéèêëïîôùûüÿçœæ')
        lang_guess = "FR" if has_accent else "EN"
        report_lines.append(f"| `{word}` | {coef:+.4f} | Ajouter à `SENSATIONALIST_{lang_guess}` |")
    report_lines.append("")
else:
    report_lines.append("Tous les mots du top TF-IDF sont déjà couverts par les listes actuelles.")
    report_lines.append("")

report_lines.append(f"### État actuel des listes")
report_lines.append("")
report_lines.append(f"- `SENSATIONALIST_EN` : {len(current_en)} termes")
report_lines.append(f"- `SENSATIONALIST_FR` : {len(current_fr)} termes")
report_lines.append("")

# Recommandations V3
report_lines.append("## 7. Recommandations pour la V3")
report_lines.append("")
report_lines.append("### Constats principaux")
report_lines.append("")

# Générer les constats en fonction des résultats
if length_results:
    best_seg = max(length_results.items(), key=lambda x: x[1]['f1'])
    worst_seg = min(length_results.items(), key=lambda x: x[1]['f1'])
    report_lines.append(f"1. **Meilleur segment** : {best_seg[0]} (F1={best_seg[1]['f1']:.4f}, n={best_seg[1]['count']:,})")
    report_lines.append(f"2. **Pire segment** : {worst_seg[0]} (F1={worst_seg[1]['f1']:.4f}, n={worst_seg[1]['count']:,})")

if lang_results:
    for lang, r in lang_results.items():
        report_lines.append(f"3. **Langue {lang.upper()}** : F1={r['f1']:.4f} (n={r['count']:,})")

report_lines.append(f"4. **Calibration** : ECE={ece:.4f}")
report_lines.append("")

report_lines.append("### Pistes d'amélioration")
report_lines.append("")
report_lines.append("1. **Textes courts** : Augmenter les données d'entraînement pour les textes < 30 mots (titres, tweets). Envisager un modèle spécialisé court-texte ou un système d'ensemble avec routing par longueur.")
report_lines.append("2. **Calibration** : Si l'ECE > 0.05, appliquer une calibration post-hoc (Platt scaling ou isotonic regression) pour améliorer la fiabilité des scores de confiance.")
report_lines.append("3. **Équilibre FR/EN** : Vérifier que le modèle ne sur-apprend sur les patterns d'une langue au détriment de l'autre. Envisager des features cross-lingues (embeddings multilingues).")
report_lines.append("4. **Mots sensationnalistes** : Intégrer les mots discriminants identifiés dans l'analyse TF-IDF aux listes `SENSATIONALIST_EN`/`SENSATIONALIST_FR`.")
report_lines.append("5. **Features contextuelles** : Pour les textes courts, enrichir les features avec des métadonnées (source, heure de publication, compte auteur) lorsque disponibles.")
report_lines.append("6. **Seuil adaptatif** : Implémenter un seuil de décision dépendant de la longueur du texte pour compenser la perte de performance sur les textes courts.")
report_lines.append("")

# Écriture du rapport
os.makedirs(DOCS_DIR, exist_ok=True)
report_path = os.path.join(DOCS_DIR, '06_analyse_modele_par_longueur.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"\nRapport sauvegardé : {report_path}")
print("Terminé.")
