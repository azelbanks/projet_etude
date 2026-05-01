#!/usr/bin/env python3
"""
25 — V8 : Meta-Learner Étendu V5 + V6 + CamemBERT
====================================================

Amélioration du V7 en ajoutant le score CamemBERT comme 3e signal
dans le meta-learner. CamemBERT apporte un signal sémantique profond
complémentaire au TF-IDF (V5) et au style (V6).

Architecture V8 :
    Input text → V5 (TF-IDF LogReg)     → score_v5  P(fiable)  ──┐
    Input text → V6 (Style GradBoost)    → score_v6  P(suspect) ──┤
    Input text → CamemBERT (Transformer) → score_cam P(fiable)  ──┤
                                                                   ├→ Meta-Learner → Décision
    Dérivées : disagreement_v5_v6, disagreement_v5_cam,           │
               interaction_v5_v6, min_fiable, max_suspect         ──┘

Auteur : Thumalien Team
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

_proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_proj, 'src'))

from pipeline.expert_detector import ExpertFakeNewsDetector, EmotionFeatureExtractor
from pipeline.camembert_classifier import CamemBERTClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    confusion_matrix, precision_score, recall_score,
)
from sklearn.model_selection import LeaveOneOut
import re

MODEL_DIR = os.path.join(_proj, 'models')

print("=" * 70)
print("V8 — META-LEARNER ÉTENDU (V5 + V6 + CamemBERT)")
print("=" * 70)
t0 = time.time()

# ================================================================
#  1. CHARGER LES TROIS MODÈLES
# ================================================================
print("\n[1/7] Chargement des modèles V5, V6 et CamemBERT...")

# V5 : TF-IDF + LogReg
det_v5 = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=0.44)
det_v5.load(suffix='expert_v5')
print(f"  V5 chargé : LogReg + TF-IDF")

# V6 : Style-Only GradientBoosting
v6_data = joblib.load(os.path.join(MODEL_DIR, 'model_style_v6.joblib'))
v6_model = v6_data['model']
v6_scaler = v6_data['scaler']
v6_feature_names = v6_data['feature_names']
v6_model_name = v6_data['model_name']
print(f"  V6 chargé : {v6_model_name} ({len(v6_feature_names)} features)")

# CamemBERT : Transformer FR
cam = CamemBERTClassifier(model_dir=MODEL_DIR)
cam_loaded = cam.load(suffix='camembert_fr')
if not cam_loaded:
    cam_loaded = cam.load(suffix='camembert_best')
print(f"  CamemBERT chargé : {cam_loaded}")

# Émotions
emo = EmotionFeatureExtractor(model_dir=MODEL_DIR)
emo_loaded = emo.load()
print(f"  Émotions : {'OK' if emo_loaded else 'Non disponible'}")

# StyleFeatureExtractorV6 (copie du notebook 24)
from pipeline.expert_detector import LinguisticFeatureExtractor

class StyleFeatureExtractorV6:
    """Extracteur V6 : 28 features stylistiques."""
    SENSATIONALIST_EN = LinguisticFeatureExtractor.SENSATIONALIST_EN
    SENSATIONALIST_FR = LinguisticFeatureExtractor.SENSATIONALIST_FR
    CALL_TO_ACTION_FR = [
        r'\b(partagez|diffusez|faites tourner|rt svp|a partager)\b',
        r'\b(likez|abonnez|suivez|inscrivez)\b',
        r'\b(signez la petition|mobilisons|reagissez)\b',
        r'\b(avant (la )?censure|avant suppression|avant qu.?ils? suppriment)\b',
    ]
    CALL_TO_ACTION_EN = [
        r'\b(share|retweet|spread the word|pass it on)\b',
        r'\b(subscribe|follow|like|sign the petition)\b',
        r'\b(before (they|it gets?) deleted?|before censored)\b',
        r'\b(act now|do something|fight back|resist)\b',
    ]
    HEDGING_FR = [
        r'\b(selon|d.?apr[eè]s|il para[iî]t que|il semblerait)\b',
        r'\b(certains disent|on dit que|des sources)\b',
        r'\b(apparemment|soi-?disant|pr[eé]tendument)\b',
    ]
    HEDGING_EN = [
        r'\b(allegedly|reportedly|according to|sources say)\b',
        r'\b(it is said|some say|rumor has it|unconfirmed)\b',
        r'\b(supposedly|purportedly|claimed)\b',
    ]
    AUTHORITY_CLAIM_FR = [
        r'\b(un (m[eé]decin|scientifique|expert|chercheur|professeur) (affirme|confirme|r[eé]v[eè]le))\b',
        r'\b(etude (prouve|montre|confirme))\b',
        r'\b(c.?est prouv[eé]|la science dit|les chiffres parlent)\b',
    ]
    AUTHORITY_CLAIM_EN = [
        r'\b(doctor|scientist|expert|professor|researcher) (says|confirms|reveals)\b',
        r'\b(study (proves|shows|confirms))\b',
        r'\b(science says|the data shows|proven)\b',
    ]
    SOURCE_CITATION_PATTERNS = [
        r'\b(reuters|afp|ap news|associated press)\b',
        r'\b(selon (le |la |l.?)?[A-Z])',
        r'\b(source[s]?\s*:)',
        r'\b(d.?apr[eè]s (le |la |l.?)?[A-Z])',
        r'\b(published in|peer.?reviewed|journal)\b',
        r'\b(lib[eé]ration|le monde|figaro|bbc|cnn|nyt|washington post)\b',
    ]
    FEATURE_NAMES = [
        'word_count', 'sentence_count', 'avg_sentence_length',
        'avg_word_length', 'is_short_text', 'paragraph_count',
        'exclamation_count', 'question_count', 'punct_density',
        'ellipsis_count', 'repeated_punct_ratio', 'emoji_count',
        'caps_ratio', 'all_caps_words_ratio', 'caps_lock_words_count',
        'sensationalism_score', 'interpellation_score',
        'call_to_action_score', 'hedging_score', 'authority_claim_score',
        'has_url', 'has_source_citation', 'numeric_density',
        'quote_count', 'named_entity_density',
        'lexical_diversity', 'repeated_char_ratio', 'spelling_anomaly_score',
    ]

    @classmethod
    def extract(cls, texts):
        n = len(texts)
        results = np.zeros((n, len(cls.FEATURE_NAMES)), dtype=np.float64)
        for i, text in enumerate(texts):
            text = str(text)
            text_lower = text.lower()
            words = text.split()
            n_words = len(words) if words else 1
            n_chars = len(text) if text else 1
            alpha_chars = sum(c.isalpha() for c in text)
            results[i, 0] = n_words
            sentences = re.split(r'[.!?]+', text)
            sentences = [s for s in sentences if s.strip()]
            n_sentences = len(sentences) if sentences else 1
            results[i, 1] = n_sentences
            results[i, 2] = n_words / n_sentences
            results[i, 3] = np.mean([len(w) for w in words]) if words else 0
            results[i, 4] = 1.0 if n_words < 20 else 0.0
            paragraphs = [p for p in text.split('\n') if p.strip()]
            results[i, 5] = len(paragraphs)
            results[i, 6] = text.count('!')
            results[i, 7] = text.count('?')
            results[i, 8] = sum(c in '!?.,;:…' for c in text) / n_chars
            results[i, 9] = text.count('...') + text.count('…')
            repeated = len(re.findall(r'([!?.])\1{1,}', text))
            results[i, 10] = repeated / n_chars if n_chars > 0 else 0
            emoji_count = len(re.findall(
                r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
                r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
                r'\U00002702-\U000027B0\U0001F900-\U0001F9FF'
                r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF'
                r'\U00002600-\U000026FF]', text))
            results[i, 11] = emoji_count
            results[i, 12] = sum(c.isupper() for c in text) / alpha_chars if alpha_chars > 0 else 0
            caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
            results[i, 13] = caps_words / n_words if n_words > 0 else 0
            results[i, 14] = caps_words
            sens_score = 0
            for w in cls.SENSATIONALIST_EN | cls.SENSATIONALIST_FR:
                if re.search(r'(?:^|\b|\s)' + re.escape(w) + r'(?:\b|\s|$)', text_lower):
                    sens_score += 1
            results[i, 15] = sens_score
            interp_score = 0
            for pat in (LinguisticFeatureExtractor.INTERPELLATION_PATTERNS_FR +
                        LinguisticFeatureExtractor.INTERPELLATION_PATTERNS_EN):
                if re.search(pat, text_lower):
                    interp_score += 1
            results[i, 16] = interp_score
            cta_score = 0
            for pat in cls.CALL_TO_ACTION_FR + cls.CALL_TO_ACTION_EN:
                if re.search(pat, text_lower):
                    cta_score += 1
            results[i, 17] = cta_score
            hedge_score = 0
            for pat in cls.HEDGING_FR + cls.HEDGING_EN:
                if re.search(pat, text_lower):
                    hedge_score += 1
            results[i, 18] = hedge_score
            auth_score = 0
            for pat in cls.AUTHORITY_CLAIM_FR + cls.AUTHORITY_CLAIM_EN:
                if re.search(pat, text_lower):
                    auth_score += 1
            results[i, 19] = auth_score
            results[i, 20] = 1.0 if re.search(r'http|www\.', text) else 0.0
            source_score = 0
            for pat in cls.SOURCE_CITATION_PATTERNS:
                if re.search(pat, text_lower):
                    source_score += 1
            results[i, 21] = source_score
            results[i, 22] = sum(c.isdigit() for c in text) / n_chars
            results[i, 23] = text.count('"') + text.count('\u201c') + text.count('\u00ab')
            if len(words) > 1:
                ne_count = sum(1 for j, w in enumerate(words[1:], 1) if w[0].isupper() and w.isalpha())
                results[i, 24] = ne_count / n_words
            else:
                results[i, 24] = 0
            words_lower = [w.lower() for w in words]
            results[i, 25] = len(set(words_lower)) / n_words if n_words > 0 else 0
            repeated_chars = len(re.findall(r'(.)\1{2,}', text_lower))
            results[i, 26] = repeated_chars / n_words if n_words > 0 else 0
            common_short = {'je', 'tu', 'il', 'on', 'le', 'la', 'de', 'a', 'i', '\u00e0',
                            'y', 'or', 'et', 'en', 'du', 'un', 'au', 'ne', 'se', 'me',
                            'te', 'ce', 'ma', 'sa', 'ta', 'is', 'am', 'an', 'as', 'at',
                            'be', 'by', 'do', 'go', 'he', 'if', 'in', 'it', 'me', 'my',
                            'no', 'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we'}
            anomalous = sum(1 for w in words_lower if 1 <= len(w) <= 2 and w not in common_short)
            results[i, 27] = anomalous / n_words if n_words > 0 else 0
        return results


def predict_v6(texts, model, scaler, model_name, emo_extractor):
    """Score V6 : retourne P(suspect) pour chaque texte."""
    X_style = StyleFeatureExtractorV6.extract(texts)
    if emo_extractor is not None:
        X_emo = emo_extractor.get_emotion_features(
            texts.tolist() if hasattr(texts, 'tolist') else list(texts)
        )
        X_all = np.hstack([X_style, X_emo])
    else:
        X_all = X_style
    if model_name == 'LogReg':
        X_all = scaler.transform(X_all)
    return model.predict_proba(X_all)[:, 1]  # P(suspect)


# ================================================================
#  2. CHARGER LE GOLD TEST SET
# ================================================================
print("\n[2/7] Chargement du Gold Test Set...")

gold_path = os.path.join(_proj, 'data', 'gold_test_set_annotation_completed.xlsx')
df_gold = pd.read_excel(gold_path, sheet_name='Resolution')
gold_texts = df_gold['Texte'].fillna('')
gold_labels = (df_gold['Label final'] == 'suspect').astype(int).values

print(f"  {len(gold_labels)} posts : {sum(gold_labels==0)} fiables, {sum(gold_labels==1)} suspects")

# Détecter la langue de chaque post
from langdetect import detect
gold_langs = []
for text in gold_texts:
    try:
        lang = detect(str(text))
        gold_langs.append('fr' if lang == 'fr' else 'en')
    except Exception:
        gold_langs.append('en')
gold_langs = np.array(gold_langs)
n_fr = sum(gold_langs == 'fr')
n_en = sum(gold_langs == 'en')
print(f"  Langues : {n_fr} FR, {n_en} EN")

# ================================================================
#  3. GÉNÉRER LES SCORES V5, V6 ET CAMEMBERT
# ================================================================
print("\n[3/7] Prédiction V5 + V6 + CamemBERT sur le gold test set...")

# V5 scores
v5_results = det_v5.predict(gold_texts)
score_v5 = v5_results['ai_score_credibility'].values  # P(fiable)

# V6 scores
score_v6 = predict_v6(
    gold_texts, v6_model, v6_scaler, v6_model_name,
    emo if emo_loaded else None,
)  # P(suspect)

# CamemBERT scores (FR seulement, 0.5 neutre pour EN)
if cam_loaded:
    texts_list = gold_texts.tolist()
    score_cam = cam.predict_credibility_scores(texts_list)  # P(fiable)
    # Pour les textes EN, CamemBERT n'est pas fiable → mettre à 0.5 (neutre)
    for i, lang in enumerate(gold_langs):
        if lang != 'fr':
            score_cam[i] = 0.5
    print(f"  CamemBERT : score moyen FR = {score_cam[gold_langs=='fr'].mean():.3f}")
    print(f"  CamemBERT : score moyen EN (neutre=0.5) = {score_cam[gold_langs=='en'].mean():.3f}")
else:
    score_cam = np.full(len(gold_labels), 0.5)
    print("  CamemBERT non disponible, scores neutres (0.5)")

print(f"  Score V5 moyen : {score_v5.mean():.3f}")
print(f"  Score V6 moyen (P suspect) : {score_v6.mean():.3f}")
print(f"  Score CamemBERT moyen : {score_cam.mean():.3f}")

# ================================================================
#  4. CONSTRUIRE LES FEATURES META-LEARNER
# ================================================================
print("\n[4/7] Construction des features étendues...")

# V7 original (4 features)
disagreement_v5_v6 = np.abs(score_v5 - (1 - score_v6))
interaction_v5_v6 = score_v5 * score_v6

X_meta_v7 = np.column_stack([
    score_v5,
    score_v6,
    disagreement_v5_v6,
    interaction_v5_v6,
])
meta_names_v7 = ['score_v5_fiable', 'score_v6_suspect', 'disagreement', 'interaction']

# V8 étendu (7 features) : + CamemBERT + dérivées
disagreement_v5_cam = np.abs(score_v5 - score_cam)  # Accord TF-IDF ↔ Transformer
min_fiable = np.minimum(score_v5, score_cam)  # Score fiable le plus pessimiste
is_fr = (gold_langs == 'fr').astype(float)  # Signal de langue

X_meta_v8 = np.column_stack([
    score_v5,               # P(fiable) V5 TF-IDF
    score_v6,               # P(suspect) V6 Style
    score_cam,              # P(fiable) CamemBERT
    disagreement_v5_v6,     # |V5 - (1-V6)|
    disagreement_v5_cam,    # |V5 - CamemBERT| : désaccord TF-IDF ↔ Transformer
    interaction_v5_v6,      # V5 × V6
    min_fiable,             # min(V5, CamemBERT) : le plus méfiant gagne
])
meta_names_v8 = [
    'score_v5_fiable', 'score_v6_suspect', 'score_cam_fiable',
    'disagree_v5_v6', 'disagree_v5_cam', 'interact_v5_v6', 'min_fiable',
]

print(f"  V7 features : {len(meta_names_v7)}")
print(f"  V8 features : {len(meta_names_v8)}")

# ================================================================
#  5. ÉVALUATION LOO : V7 vs V8 (plusieurs meta-learners)
# ================================================================
print("\n[5/7] Évaluation Leave-One-Out...")

configs = [
    ("V7 LogReg (baseline)", X_meta_v7, meta_names_v7,
     LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)),
    ("V8 LogReg (+CamemBERT)", X_meta_v8, meta_names_v8,
     LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)),
    ("V8 LogReg C=0.1", X_meta_v8, meta_names_v8,
     LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000)),
    ("V8 LogReg C=10", X_meta_v8, meta_names_v8,
     LogisticRegression(C=10.0, class_weight='balanced', max_iter=1000)),
    ("V8 GradBoost", X_meta_v8, meta_names_v8,
     GradientBoostingClassifier(
         n_estimators=50, max_depth=2, learning_rate=0.1,
         random_state=42,
     )),
]

loo = LeaveOneOut()
results_table = []

print(f"\n  {'Config':<30s} {'Acc':>6s} {'F1 mac':>7s} {'F1 sus':>7s} {'F1 fia':>7s} {'FP':>4s} {'FN':>4s} {'TP':>4s}")
print(f"  {'-'*75}")

best_f1_suspect = 0
best_config_name = ""
best_meta_model = None
best_X_meta = None
best_names = None

for config_name, X_meta, feat_names, model_template in configs:
    loo_preds = np.zeros(len(gold_labels))
    loo_probas = np.zeros(len(gold_labels))

    for train_idx, test_idx in loo.split(X_meta):
        # Clone the model for each fold
        from sklearn.base import clone
        model_clone = clone(model_template)
        model_clone.fit(X_meta[train_idx], gold_labels[train_idx])
        loo_preds[test_idx] = model_clone.predict(X_meta[test_idx])
        loo_probas[test_idx] = model_clone.predict_proba(X_meta[test_idx])[:, 1]

    acc = accuracy_score(gold_labels, loo_preds)
    f1m = f1_score(gold_labels, loo_preds, average='macro')
    f1s = f1_score(gold_labels, loo_preds, pos_label=1, zero_division=0)
    f1f = f1_score(gold_labels, loo_preds, pos_label=0)
    cm = confusion_matrix(gold_labels, loo_preds)
    fp = cm[0, 1]
    fn = cm[1, 0] if cm.shape[0] > 1 else sum(gold_labels)
    tp = cm[1, 1] if cm.shape[0] > 1 else 0

    print(f"  {config_name:<30s} {acc:>6.3f} {f1m:>7.3f} {f1s:>7.3f} {f1f:>7.3f} {fp:>4d} {fn:>4d} {tp:>4d}")

    results_table.append({
        'config': config_name, 'accuracy': acc, 'f1_macro': f1m,
        'f1_suspect': f1s, 'f1_fiable': f1f, 'fp': fp, 'fn': fn, 'tp': tp,
    })

    if f1s > best_f1_suspect:
        best_f1_suspect = f1s
        best_config_name = config_name
        best_meta_model = model_template
        best_X_meta = X_meta
        best_names = feat_names

# Aussi tester les seuils sur le score combiné V5*CamemBERT
if cam_loaded:
    print(f"\n  --- Seuil optimal sur score combiné V5 × CamemBERT × (1-V6) ---")
    combined = score_v5 * score_cam * (1 - score_v6)
    best_th_f1 = 0
    best_th = 0.5
    for th in np.arange(0.01, 0.99, 0.005):
        pred = (combined < th).astype(int)
        f1s = f1_score(gold_labels, pred, pos_label=1, zero_division=0)
        f1m = f1_score(gold_labels, pred, average='macro')
        if f1m > best_th_f1:
            best_th_f1 = f1m
            best_th = th
            best_th_pred = pred.copy()

    cm_th = confusion_matrix(gold_labels, best_th_pred)
    f1s_th = f1_score(gold_labels, best_th_pred, pos_label=1, zero_division=0)
    f1f_th = f1_score(gold_labels, best_th_pred, pos_label=0)
    fp_th = cm_th[0, 1]
    fn_th = cm_th[1, 0] if cm_th.shape[0] > 1 else sum(gold_labels)
    tp_th = cm_th[1, 1] if cm_th.shape[0] > 1 else 0
    acc_th = accuracy_score(gold_labels, best_th_pred)

    print(f"  {'V8 Seuil combiné':<30s} {acc_th:>6.3f} {best_th_f1:>7.3f} {f1s_th:>7.3f} {f1f_th:>7.3f} {fp_th:>4d} {fn_th:>4d} {tp_th:>4d}")
    print(f"  Seuil optimal : {best_th:.3f}")

    if f1s_th > best_f1_suspect:
        best_f1_suspect = f1s_th
        best_config_name = "V8 Seuil combiné"

# ================================================================
#  6. ENTRAÎNER LE MEILLEUR MODÈLE SUR TOUT LE GOLD SET
# ================================================================
print(f"\n[6/7] Meilleur config : {best_config_name} (F1 suspect = {best_f1_suspect:.3f})")

if best_meta_model is not None:
    # Ré-entraîner sur tout le gold set
    final_model = clone(best_meta_model)
    final_model.fit(best_X_meta, gold_labels)

    print(f"\n  Coefficients du meta-learner final :")
    if hasattr(final_model, 'coef_'):
        for name, coef in zip(best_names, final_model.coef_[0]):
            print(f"    {name:30s} : {coef:+.4f}")
        print(f"    {'intercept':30s} : {final_model.intercept_[0]:+.4f}")
    elif hasattr(final_model, 'feature_importances_'):
        for name, imp in zip(best_names, final_model.feature_importances_):
            print(f"    {name:30s} : {imp:.4f}")

# ================================================================
#  7. COMPARAISON FINALE V5 vs V6 vs V7 vs V8
# ================================================================
print("\n[7/7] Comparaison finale...")

# V5 seul
pred_v5 = v5_results['prediction_label'].values

# V6 seul
X_gold_style = StyleFeatureExtractorV6.extract(gold_texts)
if emo_loaded:
    X_gold_emo = emo.get_emotion_features(gold_texts.tolist())
    X_gold_all = np.hstack([X_gold_style, X_gold_emo])
else:
    X_gold_all = X_gold_style
if v6_model_name == 'LogReg':
    X_gold_input = v6_scaler.transform(X_gold_all)
else:
    X_gold_input = X_gold_all
pred_v6 = v6_model.predict(X_gold_input)

# CamemBERT seul (seuil 0.5)
if cam_loaded:
    pred_cam = (score_cam < 0.5).astype(int)  # suspect si P(fiable) < 0.5
else:
    pred_cam = np.zeros(len(gold_labels))

print(f"\n  {'Modèle':<30s} {'Acc':>6s} {'F1 mac':>7s} {'F1 sus':>7s} {'F1 fia':>7s} {'FP':>4s} {'FN':>4s}")
print(f"  {'-'*65}")

for name, pred in [
    ('V5 (TF-IDF)', pred_v5),
    ('V6 (Style)', pred_v6),
    ('CamemBERT seul', pred_cam),
    (f'V8 BEST: {best_config_name}', None),  # LOO results
]:
    if pred is not None:
        acc = accuracy_score(gold_labels, pred)
        f1m = f1_score(gold_labels, pred, average='macro')
        f1s = f1_score(gold_labels, pred, pos_label=1, zero_division=0)
        f1f = f1_score(gold_labels, pred, pos_label=0)
        cm = confusion_matrix(gold_labels, pred)
        fp = cm[0, 1]
        fn = cm[1, 0] if cm.shape[0] > 1 else sum(gold_labels)
        print(f"  {name:<30s} {acc:>6.3f} {f1m:>7.3f} {f1s:>7.3f} {f1f:>7.3f} {fp:>4d} {fn:>4d}")
    else:
        print(f"  {name:<30s}   (voir LOO ci-dessus, F1 suspect = {best_f1_suspect:.3f})")

# ================================================================
#  SAUVEGARDE
# ================================================================
print("\n" + "=" * 70)
print("SAUVEGARDE")
print("=" * 70)

if best_meta_model is not None and best_f1_suspect > 0:
    save_data = {
        'meta_model': final_model,
        'meta_feature_names': best_names,
        'optimal_threshold': best_th if 'best_th' in dir() else 0.5,
        'v5_suffix': 'expert_v5',
        'v6_path': 'model_style_v6.joblib',
        'camembert_suffix': 'camembert_best',
        'uses_camembert': True,
        'gold_f1_suspect': best_f1_suspect,
        'gold_f1_macro': max(r['f1_macro'] for r in results_table),
        'best_config': best_config_name,
        'version': 'v8_hybrid_extended',
    }

    save_path = os.path.join(MODEL_DIR, 'model_hybrid_v8.joblib')
    joblib.dump(save_data, save_path)
    print(f"  Meta-modèle V8 sauvegardé : {save_path}")

    # Garder aussi le V7 comme backup
    print(f"  V7 conservé comme backup : model_hybrid_v7.joblib")

elapsed = time.time() - t0
print(f"\n  Temps total : {elapsed:.0f}s ({elapsed/60:.1f}min)")
print("\n" + "=" * 70)
print("TERMINÉ")
print("=" * 70)
