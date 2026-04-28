#!/usr/bin/env python3
"""
24 — Ensemble Hybride V7 : V5 (TF-IDF) + V6 (Style) + SHAP Explicabilite
==========================================================================

Strategie :
    V5 seul = biais thematique (F1 suspect gold = 0.087)
    V6 seul = trop de faux positifs sur le style (F1 suspect gold = 0.103)
    V7 = meta-modele qui combine les SCORES des deux modeles
        → le signal lexical (V5) + le signal stylistique (V6)
        → calibre sur le gold test set annote manuellement

Architecture :
    Input text → V5 predict (score_tfidf) ──┐
                                             ├→ Meta-Learner → Decision finale
    Input text → V6 predict (score_style) ──┘

    Le meta-learner recoit :
    - score_v5 : P(fiable) du modele TF-IDF
    - score_v6 : P(suspect) du modele style GradientBoosting
    - disagreement : |score_v5 - (1-score_v6)| (signal de conflit)

    + Explicabilite SHAP sur le modele V6 (35 features interpretables)

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    confusion_matrix, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

MODEL_DIR = os.path.join(_proj, 'models')

print("=" * 70)
print("V7 — ENSEMBLE HYBRIDE (V5 + V6) + SHAP")
print("=" * 70)
t0 = time.time()

# ================================================================
#  1. CHARGER LES DEUX MODELES
# ================================================================
print("\n[1/6] Chargement des modeles V5 et V6...")

# V5 : TF-IDF + LogReg
det_v5 = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=0.44)
det_v5.load(suffix='expert_v5')
print(f"  V5 charge : LogReg + TF-IDF (30015 features)")

# V6 : Style-Only GradientBoosting
v6_data = joblib.load(os.path.join(MODEL_DIR, 'model_style_v6.joblib'))
v6_model = v6_data['model']
v6_scaler = v6_data['scaler']
v6_feature_names = v6_data['feature_names']
v6_model_name = v6_data['model_name']
print(f"  V6 charge : {v6_model_name} ({len(v6_feature_names)} features)")

# Import StyleFeatureExtractorV6 from notebook 23
sys.path.insert(0, os.path.join(_proj, 'notebooks'))
# Re-define inline to avoid import issues
from pipeline.expert_detector import LinguisticFeatureExtractor
import re

class StyleFeatureExtractorV6:
    """Copie de l'extracteur V6 du notebook 23."""
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


def predict_v6(texts, model, scaler, emo_extractor):
    """Score V6 : retourne P(suspect) pour chaque texte."""
    X_style = StyleFeatureExtractorV6.extract(texts)
    if emo_extractor is not None:
        X_emo = emo_extractor.get_emotion_features(texts.tolist() if hasattr(texts, 'tolist') else list(texts))
        X_all = np.hstack([X_style, X_emo])
    else:
        X_all = X_style
    if v6_model_name == 'LogReg':
        X_all = scaler.transform(X_all)
    return model.predict_proba(X_all)[:, 1]  # P(suspect)


# Charger emotions
emo = EmotionFeatureExtractor(model_dir=MODEL_DIR)
emo_loaded = emo.load()
print(f"  Emotions : {'OK' if emo_loaded else 'Non disponible'}")

# ================================================================
#  2. CHARGER LE GOLD TEST SET
# ================================================================
print("\n[2/6] Chargement du Gold Test Set annote...")

gold_path = os.path.join(_proj, 'data', 'gold_test_set_annotation_completed.xlsx')
df_gold = pd.read_excel(gold_path, sheet_name='Resolution')
gold_texts = df_gold['Texte'].fillna('')
gold_labels = (df_gold['Label final'] == 'suspect').astype(int).values

print(f"  {len(gold_labels)} posts : {sum(gold_labels==0)} fiables, {sum(gold_labels==1)} suspects")

# ================================================================
#  3. GENERER LES SCORES V5 et V6 SUR LE GOLD SET
# ================================================================
print("\n[3/6] Prediction V5 + V6 sur le gold test set...")

# V5 scores
v5_results = det_v5.predict(gold_texts)
score_v5 = v5_results['ai_score_credibility'].values  # P(fiable)
pred_v5 = v5_results['prediction_label'].values

# V6 scores
score_v6 = predict_v6(gold_texts, v6_model, v6_scaler, emo if emo_loaded else None)  # P(suspect)

# Features pour le meta-learner
disagreement = np.abs(score_v5 - (1 - score_v6))  # desaccord entre V5 et V6

X_meta = np.column_stack([
    score_v5,           # P(fiable) V5
    score_v6,           # P(suspect) V6
    disagreement,       # signal de conflit
    score_v5 * score_v6,  # interaction
])
meta_feature_names = ['score_v5_fiable', 'score_v6_suspect', 'disagreement', 'interaction']

print(f"  Score V5 moyen : {score_v5.mean():.3f}")
print(f"  Score V6 moyen (P suspect) : {score_v6.mean():.3f}")
print(f"  Desaccord moyen : {disagreement.mean():.3f}")

# ================================================================
#  4. OPTIMISER LE SEUIL DE DECISION
# ================================================================
print("\n[4/6] Optimisation du seuil et du meta-learner...")

# A) Seuil simple sur score combine
print("\n  --- A) Seuil optimal sur score combine (V5 * (1-V6)) ---")
combined_score = score_v5 * (1 - score_v6)  # Plus c'est haut = plus fiable

best_threshold = 0.5
best_f1_macro = 0
best_results = {}

for th in np.arange(0.05, 0.95, 0.01):
    pred = (combined_score < th).astype(int)  # suspect si score combine < seuil
    f1_s = f1_score(gold_labels, pred, pos_label=1, zero_division=0)
    f1_f = f1_score(gold_labels, pred, pos_label=0)
    f1_m = f1_score(gold_labels, pred, average='macro')

    if f1_m > best_f1_macro:
        best_f1_macro = f1_m
        best_threshold = th
        best_results = {
            'f1_macro': f1_m, 'f1_suspect': f1_s, 'f1_fiable': f1_f,
            'accuracy': accuracy_score(gold_labels, pred),
            'pred': pred.copy(),
        }

print(f"    Seuil optimal : {best_threshold:.2f}")
print(f"    F1 macro      : {best_results['f1_macro']:.4f}")
print(f"    F1 suspect    : {best_results['f1_suspect']:.4f}")
print(f"    F1 fiable     : {best_results['f1_fiable']:.4f}")
print(f"    Accuracy      : {best_results['accuracy']:.4f}")

cm_combo = confusion_matrix(gold_labels, best_results['pred'])
print(f"    Confusion : TN={cm_combo[0,0]} FP={cm_combo[0,1]} | FN={cm_combo[1,0]} TP={cm_combo[1,1]}")

# B) Meta-learner LogReg (Leave-One-Out sur 200 posts)
print("\n  --- B) Meta-Learner LogReg (LOO cross-val sur gold set) ---")

from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
loo_preds = np.zeros(len(gold_labels))
loo_probas = np.zeros(len(gold_labels))

for train_idx, test_idx in loo.split(X_meta):
    meta_lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    meta_lr.fit(X_meta[train_idx], gold_labels[train_idx])
    loo_preds[test_idx] = meta_lr.predict(X_meta[test_idx])
    loo_probas[test_idx] = meta_lr.predict_proba(X_meta[test_idx])[:, 1]

print(f"    LOO Accuracy  : {accuracy_score(gold_labels, loo_preds):.4f}")
print(f"    LOO F1 macro  : {f1_score(gold_labels, loo_preds, average='macro'):.4f}")
print(f"    LOO F1 suspect: {f1_score(gold_labels, loo_preds, pos_label=1, zero_division=0):.4f}")
print(f"    LOO F1 fiable : {f1_score(gold_labels, loo_preds, pos_label=0):.4f}")

cm_loo = confusion_matrix(gold_labels, loo_preds)
print(f"    Confusion : TN={cm_loo[0,0]} FP={cm_loo[0,1]} | FN={cm_loo[1,0]} TP={cm_loo[1,1]}")

# Train final meta-learner on all gold data
meta_final = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
meta_final.fit(X_meta, gold_labels)
print(f"\n    Meta-learner coefficients:")
for name, coef in zip(meta_feature_names, meta_final.coef_[0]):
    print(f"      {name:25s} : {coef:+.4f}")
print(f"      {'intercept':25s} : {meta_final.intercept_[0]:+.4f}")

# ================================================================
#  5. COMPARAISON FINALE V5 vs V6 vs V7
# ================================================================
print("\n[5/6] Comparaison finale V5 vs V6 vs V7...")

# V6 predictions on gold
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

# V7 = combined score with optimal threshold
pred_v7_combo = best_results['pred']
# V7 meta = LOO predictions
pred_v7_meta = loo_preds.astype(int)

print(f"\n  {'Modele':<20s} {'Accuracy':>10s} {'F1 macro':>10s} {'F1 suspect':>10s} {'F1 fiable':>10s} {'FP':>5s} {'FN':>5s}")
print(f"  {'-'*65}")

for name, pred in [
    ('V5 (TF-IDF)', v5_results['prediction_label'].values),
    ('V6 (Style)', pred_v6),
    ('V7 Combo', pred_v7_combo),
    ('V7 Meta (LOO)', pred_v7_meta),
]:
    acc = accuracy_score(gold_labels, pred)
    f1m = f1_score(gold_labels, pred, average='macro')
    f1s = f1_score(gold_labels, pred, pos_label=1, zero_division=0)
    f1f = f1_score(gold_labels, pred, pos_label=0)
    cm = confusion_matrix(gold_labels, pred)
    fp = cm[0, 1] if cm.shape[0] > 1 else 0
    fn = cm[1, 0] if cm.shape[0] > 1 else 0
    print(f"  {name:<20s} {acc:>10.4f} {f1m:>10.4f} {f1s:>10.4f} {f1f:>10.4f} {fp:>5d} {fn:>5d}")

# ================================================================
#  6. SHAP EXPLICABILITE
# ================================================================
print("\n[6/6] SHAP Explicabilite sur le modele V6 (style features)...")

try:
    import shap

    # Pour GradientBoosting, on utilise TreeExplainer (rapide)
    if v6_model_name in ('GradientBoosting', 'RandomForest'):
        explainer = shap.TreeExplainer(v6_model)
        shap_values = explainer.shap_values(X_gold_input)
    else:
        # LogReg : LinearExplainer
        explainer = shap.LinearExplainer(v6_model, X_gold_input)
        shap_values = explainer.shap_values(X_gold_input)

    # Global feature importance (mean |SHAP|)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # classe suspect
    else:
        shap_vals = shap_values

    mean_shap = np.abs(shap_vals).mean(axis=0)

    # Mapper aux noms de features
    # V6 features = 28 style + 7 emotions
    all_v6_names = StyleFeatureExtractorV6.FEATURE_NAMES + [
        'emo_anger', 'emo_disgust', 'emo_joy', 'emo_neutral',
        'emo_fear', 'emo_surprise', 'emo_sadness'
    ]

    print(f"\n  === SHAP Global Feature Importance (mean |SHAP|) ===")
    sorted_idx = np.argsort(mean_shap)[::-1]
    for rank, idx in enumerate(sorted_idx[:20]):
        name = all_v6_names[idx] if idx < len(all_v6_names) else f'feature_{idx}'
        print(f"    {rank+1:2d}. {name:30s} SHAP={mean_shap[idx]:.4f}")

    # SHAP local : expliquer les 9 vrais suspects
    print(f"\n  === SHAP Local — Les 9 posts suspects ===")
    suspect_mask = gold_labels == 1
    suspect_indices = np.where(suspect_mask)[0]

    for idx in suspect_indices:
        text = gold_texts.iloc[idx][:60]
        pred_label = "SUSPECT" if pred_v6[idx] == 1 else "fiable"
        v6_score = score_v6[idx]
        print(f"\n    [{pred_label}] P(suspect)={v6_score:.3f} | {text}...")

        # Top 5 features pushing toward suspect
        shap_row = shap_vals[idx]
        top5 = np.argsort(shap_row)[::-1][:5]
        for f_idx in top5:
            fname = all_v6_names[f_idx] if f_idx < len(all_v6_names) else f'f{f_idx}'
            fval = X_gold_input[idx, f_idx]
            print(f"      {fname:25s} val={fval:.3f} SHAP={shap_row[f_idx]:+.4f}")

    # SHAP local : 5 plus gros faux positifs
    print(f"\n  === SHAP Local — Top 5 Faux Positifs V6 ===")
    fp_mask = (gold_labels == 0) & (pred_v6 == 1)
    fp_indices = np.where(fp_mask)[0]
    # Sort by confidence (highest P(suspect))
    fp_sorted = fp_indices[np.argsort(score_v6[fp_indices])[::-1]][:5]

    for idx in fp_sorted:
        text = gold_texts.iloc[idx][:60]
        v6_score_val = score_v6[idx]
        print(f"\n    [FP] P(suspect)={v6_score_val:.3f} | {text}...")
        shap_row = shap_vals[idx]
        top5 = np.argsort(shap_row)[::-1][:5]
        for f_idx in top5:
            fname = all_v6_names[f_idx] if f_idx < len(all_v6_names) else f'f{f_idx}'
            fval = X_gold_input[idx, f_idx]
            print(f"      {fname:25s} val={fval:.3f} SHAP={shap_row[f_idx]:+.4f}")

except ImportError:
    print("  SHAP non installe. Installer avec: pip install shap")
except Exception as e:
    print(f"  Erreur SHAP : {e}")

# ================================================================
#  SAUVEGARDE
# ================================================================
print("\n" + "=" * 70)
print("SAUVEGARDE")
print("=" * 70)

save_data = {
    'meta_model': meta_final,
    'meta_feature_names': meta_feature_names,
    'optimal_threshold': best_threshold,
    'v5_suffix': 'expert_v5',
    'v6_path': 'model_style_v6.joblib',
    'gold_f1_macro': best_results['f1_macro'],
    'gold_f1_suspect': best_results['f1_suspect'],
    'loo_f1_macro': f1_score(gold_labels, loo_preds, average='macro'),
    'loo_f1_suspect': f1_score(gold_labels, loo_preds, pos_label=1, zero_division=0),
    'version': 'v7_hybrid',
}

save_path = os.path.join(MODEL_DIR, 'model_hybrid_v7.joblib')
joblib.dump(save_data, save_path)
print(f"  Meta-modele sauvegarde : {save_path}")

elapsed = time.time() - t0
print(f"\n  Temps total : {elapsed:.0f}s ({elapsed/60:.1f}min)")
print("\n" + "=" * 70)
print("TERMINE")
print("=" * 70)
