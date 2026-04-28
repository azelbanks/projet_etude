#!/usr/bin/env python3
"""
23 — Modele Style-Only V6 : Detection de desinformation sans biais thematique
==============================================================================

Probleme identifie (gold test set, notebook 22) :
    Le modele V5 (TF-IDF 30K + 15 features linguistiques) apprend le SUJET
    ("coronavirus" → suspect, "reuters" → fiable) au lieu du STYLE de la
    desinformation. Resultat : F1 suspect = 0.087 sur le gold test set.

Solution V6 :
    Modele "style-only" : on SUPPRIME le TF-IDF et on enrichit les features
    linguistiques/stylistiques a ~30 features. Le modele ne peut apprendre
    QUE des patterns de style (ponctuation, majuscules, sensationnalisme,
    appels a l'action, structure) — il est topic-agnostic par construction.

Hypothese :
    Le F1 CV sera plus bas (~0.75 vs 0.90), mais les predictions sur des
    vrais posts Bluesky seront beaucoup plus coherentes car le modele ne
    peut pas tricher en reconnaissant les sujets.

Features V6 (30 features stylistiques) :
    Bloc 1 - Structure (6) : word_count, sentence_count, avg_sentence_length,
        avg_word_length, is_short_text, paragraph_count
    Bloc 2 - Ponctuation emotionnelle (6) : exclamation_count, question_count,
        punct_density, ellipsis_count, repeated_punct_ratio, emoji_count
    Bloc 3 - Majuscules (3) : caps_ratio, all_caps_words_ratio, caps_lock_words_count
    Bloc 4 - Lexique de manipulation (5) : sensationalism_score,
        interpellation_score, call_to_action_score, hedging_score,
        authority_claim_score
    Bloc 5 - Credibilite formelle (5) : has_url, has_source_citation,
        numeric_density, quote_count, named_entity_density
    Bloc 6 - Diversite/qualite (3) : lexical_diversity, repeated_char_ratio,
        spelling_anomaly_score
    Bloc 7 - Emotions MLP (7) : colere, degout, joie, neutre, peur,
        surprise, tristesse

Auteur : Thumalien Team
"""

import sys
import os
import time
import logging
import numpy as np
import pandas as pd
import re
import joblib
from collections import Counter

_proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_proj, 'src'))

from pipeline.expert_detector import (
    DatasetCleaner,
    LinguisticFeatureExtractor,
    EmotionFeatureExtractor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA = os.path.join(_proj, 'data', 'training')
MODEL_DIR = os.path.join(_proj, 'models')

# ================================================================
#  STYLE FEATURE EXTRACTOR V6
# ================================================================

class StyleFeatureExtractorV6:
    """
    Extracteur de features stylistiques enrichi — topic-agnostic.
    30 features de style + 7 emotions = 37 features totales.
    """

    # --- Lexiques de manipulation (FR + EN) ---
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
        r'\b(selon (le |la |l.?)?[A-Z])',  # "selon Le Monde"
        r'\b(source[s]?\s*:)',
        r'\b(d.?apr[eè]s (le |la |l.?)?[A-Z])',
        r'\b(published in|peer.?reviewed|journal)\b',
        r'\b(lib[eé]ration|le monde|figaro|bbc|cnn|nyt|washington post)\b',
    ]

    FEATURE_NAMES = [
        # Bloc 1 - Structure (6)
        'word_count', 'sentence_count', 'avg_sentence_length',
        'avg_word_length', 'is_short_text', 'paragraph_count',
        # Bloc 2 - Ponctuation emotionnelle (6)
        'exclamation_count', 'question_count', 'punct_density',
        'ellipsis_count', 'repeated_punct_ratio', 'emoji_count',
        # Bloc 3 - Majuscules (3)
        'caps_ratio', 'all_caps_words_ratio', 'caps_lock_words_count',
        # Bloc 4 - Lexique de manipulation (5)
        'sensationalism_score', 'interpellation_score',
        'call_to_action_score', 'hedging_score', 'authority_claim_score',
        # Bloc 5 - Credibilite formelle (5)
        'has_url', 'has_source_citation', 'numeric_density',
        'quote_count', 'named_entity_density',
        # Bloc 6 - Diversite/qualite (3)
        'lexical_diversity', 'repeated_char_ratio', 'spelling_anomaly_score',
    ]

    @classmethod
    def extract(cls, texts: pd.Series) -> np.ndarray:
        """Extraire 28 features stylistiques (sans emotions)."""
        n = len(texts)
        results = np.zeros((n, len(cls.FEATURE_NAMES)), dtype=np.float64)

        for i, text in enumerate(texts):
            text = str(text)
            text_lower = text.lower()
            words = text.split()
            n_words = len(words) if words else 1
            n_chars = len(text) if text else 1
            alpha_chars = sum(c.isalpha() for c in text)

            # --- Bloc 1 : Structure ---
            results[i, 0] = n_words

            sentences = re.split(r'[.!?]+', text)
            sentences = [s for s in sentences if s.strip()]
            n_sentences = len(sentences) if sentences else 1
            results[i, 1] = n_sentences
            results[i, 2] = n_words / n_sentences
            results[i, 3] = np.mean([len(w) for w in words]) if words else 0
            results[i, 4] = 1.0 if n_words < 20 else 0.0

            paragraphs = text.split('\n')
            paragraphs = [p for p in paragraphs if p.strip()]
            results[i, 5] = len(paragraphs)

            # --- Bloc 2 : Ponctuation emotionnelle ---
            results[i, 6] = text.count('!')
            results[i, 7] = text.count('?')
            results[i, 8] = sum(c in '!?.,;:…' for c in text) / n_chars

            ellipsis = text.count('...') + text.count('…')
            results[i, 9] = ellipsis

            # Ponctuation repetee (!!!, ???, etc.)
            repeated = len(re.findall(r'([!?.])\1{1,}', text))
            results[i, 10] = repeated / n_chars if n_chars > 0 else 0

            # Emojis (approximation unicode)
            emoji_count = len(re.findall(
                r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
                r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
                r'\U00002702-\U000027B0\U0001F900-\U0001F9FF'
                r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF'
                r'\U00002600-\U000026FF]', text))
            results[i, 11] = emoji_count

            # --- Bloc 3 : Majuscules ---
            results[i, 12] = (
                sum(c.isupper() for c in text) / alpha_chars
                if alpha_chars > 0 else 0
            )

            caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
            results[i, 13] = caps_words / n_words if n_words > 0 else 0
            results[i, 14] = caps_words

            # --- Bloc 4 : Lexique de manipulation ---
            # Sensationnalisme
            sens_score = 0
            for w in cls.SENSATIONALIST_EN | cls.SENSATIONALIST_FR:
                if re.search(r'(?:^|\b|\s)' + re.escape(w) + r'(?:\b|\s|$)', text_lower):
                    sens_score += 1
            results[i, 15] = sens_score

            # Interpellation
            interp_score = 0
            for pat in (LinguisticFeatureExtractor.INTERPELLATION_PATTERNS_FR +
                        LinguisticFeatureExtractor.INTERPELLATION_PATTERNS_EN):
                if re.search(pat, text_lower):
                    interp_score += 1
            results[i, 16] = interp_score

            # Call to action
            cta_score = 0
            for pat in cls.CALL_TO_ACTION_FR + cls.CALL_TO_ACTION_EN:
                if re.search(pat, text_lower):
                    cta_score += 1
            results[i, 17] = cta_score

            # Hedging (langage evasif)
            hedge_score = 0
            for pat in cls.HEDGING_FR + cls.HEDGING_EN:
                if re.search(pat, text_lower):
                    hedge_score += 1
            results[i, 18] = hedge_score

            # Authority claim sans source
            auth_score = 0
            for pat in cls.AUTHORITY_CLAIM_FR + cls.AUTHORITY_CLAIM_EN:
                if re.search(pat, text_lower):
                    auth_score += 1
            results[i, 19] = auth_score

            # --- Bloc 5 : Credibilite formelle ---
            results[i, 20] = 1.0 if re.search(r'http|www\.', text) else 0.0

            # Source citation
            source_score = 0
            for pat in cls.SOURCE_CITATION_PATTERNS:
                if re.search(pat, text_lower):
                    source_score += 1
            results[i, 21] = source_score

            results[i, 22] = sum(c.isdigit() for c in text) / n_chars  # numeric density
            results[i, 23] = text.count('"') + text.count('"') + text.count('«')  # quotes

            # Named entity density (approximation: mots commencant par majuscule hors debut de phrase)
            if len(words) > 1:
                ne_count = sum(1 for j, w in enumerate(words[1:], 1) if w[0].isupper() and w.isalpha())
                results[i, 24] = ne_count / n_words
            else:
                results[i, 24] = 0

            # --- Bloc 6 : Diversite/qualite ---
            words_lower = [w.lower() for w in words]
            results[i, 25] = len(set(words_lower)) / n_words if n_words > 0 else 0  # TTR

            # Repeated chars (e.g. "loooool", "noooon")
            repeated_chars = len(re.findall(r'(.)\1{2,}', text_lower))
            results[i, 26] = repeated_chars / n_words if n_words > 0 else 0

            # Spelling anomaly proxy: ratio of very short words (1-2 chars excl. common)
            common_short = {'je', 'tu', 'il', 'on', 'le', 'la', 'de', 'a', 'i', 'à',
                            'y', 'or', 'et', 'en', 'du', 'un', 'au', 'ne', 'se', 'me',
                            'te', 'ce', 'ma', 'sa', 'ta', 'is', 'am', 'an', 'as', 'at',
                            'be', 'by', 'do', 'go', 'he', 'if', 'in', 'it', 'me', 'my',
                            'no', 'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we'}
            anomalous = sum(1 for w in words_lower if 1 <= len(w) <= 2 and w not in common_short)
            results[i, 27] = anomalous / n_words if n_words > 0 else 0

        return results


# ================================================================
#  1. CHARGEMENT DES DONNEES
# ================================================================
print("=" * 70)
print("V6 — MODELE STYLE-ONLY (topic-agnostic)")
print("=" * 70)
t0 = time.time()

print("\n[1/8] Chargement du dataset bilingue V5...")

df_v6 = DatasetCleaner.prepare_bilingual_dataset(
    fake_path=os.path.join(DATA, 'Fake.csv'),
    true_path=os.path.join(DATA, 'True.csv'),
    kaggle_fr_dir=os.path.join(DATA, 'kaggle_fr'),
    fakenewsnet_dir=os.path.join(DATA, 'fakenewsnet'),
    constraint_dir=os.path.join(DATA, 'constraint'),
    credibility_dir=os.path.join(DATA, 'credibility_corpus'),
    french_oversample=5,
    social_oversample=2,
)

# Ajouter FR social synthetique
fr_social = os.path.join(DATA, 'fr_social_media_synthetic.csv')
if os.path.exists(fr_social):
    df_fr_social = pd.read_csv(fr_social)
    df_fr_social['language'] = 'fr'
    df_fr_social['text_clean'] = df_fr_social['text'].apply(
        lambda t: DatasetCleaner.clean_for_ml(str(t))
    )
    df_fr_social['text_original'] = df_fr_social['text']
    df_fr_social = df_fr_social[['text_clean', 'text_original', 'label', 'language']]
    df_v6 = pd.concat([df_v6, df_fr_social], ignore_index=True)
    print(f"  +{len(df_fr_social)} FR social synthetiques")

# Ajouter EN social si disponible
en_social = os.path.join(DATA, 'en_social_media_synthetic.csv')
if os.path.exists(en_social):
    df_en_social = pd.read_csv(en_social)
    if 'text' in df_en_social.columns and 'label' in df_en_social.columns:
        df_en_social['language'] = 'en'
        df_en_social['text_clean'] = df_en_social['text'].apply(
            lambda t: DatasetCleaner.clean_for_ml(str(t))
        )
        df_en_social['text_original'] = df_en_social['text']
        df_en_social = df_en_social[['text_clean', 'text_original', 'label', 'language']]
        df_v6 = pd.concat([df_v6, df_en_social], ignore_index=True)
        print(f"  +{len(df_en_social)} EN social synthetiques")

print(f"  Dataset total : {len(df_v6)} textes")
print(f"  EN={sum(df_v6.language=='en')}, FR={sum(df_v6.language=='fr')}")
print(f"  Labels : {df_v6.label.value_counts().to_dict()}")

# ================================================================
#  2. EXTRACTION DES FEATURES STYLE-ONLY
# ================================================================
print("\n[2/8] Extraction des features stylistiques (28 features)...")

# On utilise text_original pour les features (caps, ponctuation)
texts_for_features = df_v6['text_original'].fillna(df_v6['text_clean'])
X_style = StyleFeatureExtractorV6.extract(texts_for_features)
print(f"  Shape features style : {X_style.shape}")

# Ajouter les emotions (7 features)
print("\n[3/8] Extraction des features emotions (7 features)...")
emo = EmotionFeatureExtractor(model_dir=MODEL_DIR)
emo_loaded = emo.load()
if emo_loaded:
    X_emo = emo.get_emotion_features(texts_for_features.tolist())
    X_all = np.hstack([X_style, X_emo])
    all_feature_names = StyleFeatureExtractorV6.FEATURE_NAMES + [
        'emo_anger', 'emo_disgust', 'emo_joy', 'emo_neutral',
        'emo_fear', 'emo_surprise', 'emo_sadness'
    ]
    print(f"  Emotions chargees : {X_emo.shape[1]} features")
else:
    X_all = X_style
    all_feature_names = StyleFeatureExtractorV6.FEATURE_NAMES
    print("  Emotions non disponibles, style-only (28 features)")

y = df_v6['label'].values
print(f"  Shape finale : {X_all.shape}")
print(f"  Features : {len(all_feature_names)}")

# ================================================================
#  3. SPLIT TRAIN/TEST
# ================================================================
print("\n[4/8] Split train/test 80/20 stratifie...")
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, stratify=y, random_state=42
)
idx_train, idx_test = train_test_split(
    range(len(df_v6)), test_size=0.2, stratify=y, random_state=42
)
df_test = df_v6.iloc[idx_test].copy()
print(f"  Train : {len(X_train)} | Test : {len(X_test)}")

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================================================
#  4. ENTRAINEMENT AVEC 5-FOLD CV
# ================================================================
print("\n[5/8] Entrainement modele style-only (5-fold CV)...")

# Tester plusieurs modeles
models = {
    'LogReg': LogisticRegression(
        C=1.0, class_weight='balanced', max_iter=10000, solver='lbfgs'
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight='balanced',
        random_state=42, n_jobs=-1
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model_name = None
best_f1 = 0
best_model = None

for name, model in models.items():
    print(f"\n  --- {name} ---")
    X_input = X_train_scaled if name == 'LogReg' else X_train

    cv_results = cross_validate(
        model, X_input, y_train, cv=cv,
        scoring=['accuracy', 'f1', 'precision', 'recall'],
        return_train_score=True,
    )

    f1_mean = cv_results['test_f1'].mean()
    f1_std = cv_results['test_f1'].std()
    acc_mean = cv_results['test_accuracy'].mean()
    prec_mean = cv_results['test_precision'].mean()
    rec_mean = cv_results['test_recall'].mean()

    print(f"    CV F1     : {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"    CV Acc    : {acc_mean:.4f}")
    print(f"    CV Prec   : {prec_mean:.4f}")
    print(f"    CV Recall : {rec_mean:.4f}")
    print(f"    Train F1  : {cv_results['train_f1'].mean():.4f}")

    if f1_mean > best_f1:
        best_f1 = f1_mean
        best_model_name = name
        best_model = model

print(f"\n  Meilleur modele : {best_model_name} (F1={best_f1:.4f})")

# Entrainer le meilleur modele sur tout le train set
if best_model_name == 'LogReg':
    best_model.fit(X_train_scaled, y_train)
else:
    best_model.fit(X_train, y_train)

# ================================================================
#  5. EVALUATION HOLDOUT
# ================================================================
print("\n[6/8] Evaluation sur le holdout test (20%)...")

if best_model_name == 'LogReg':
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else np.zeros(len(y_test))

print(f"\n  Holdout Results V6 ({best_model_name}):")
print(f"    Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"    F1        : {f1_score(y_test, y_pred):.4f}")
print(f"    Precision : {precision_score(y_test, y_pred):.4f}")
print(f"    Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Fiable', 'Suspect']))
print(f"\n  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"    TN={cm[0,0]:6d}  FP={cm[0,1]:6d}")
print(f"    FN={cm[1,0]:6d}  TP={cm[1,1]:6d}")

# Feature importance
if best_model_name == 'LogReg':
    coefs = best_model.coef_[0]
    importance = np.abs(coefs)
elif best_model_name == 'GradientBoosting':
    importance = best_model.feature_importances_
elif best_model_name == 'RandomForest':
    importance = best_model.feature_importances_

print(f"\n  Feature Importance (top 15) :")
idx_sorted = np.argsort(importance)[::-1]
for rank, idx in enumerate(idx_sorted[:15]):
    name = all_feature_names[idx]
    imp = importance[idx]
    direction = ""
    if best_model_name == 'LogReg':
        direction = " → SUSPECT" if coefs[idx] > 0 else " → FIABLE"
    print(f"    {rank+1:2d}. {name:30s} {imp:.4f}{direction}")

# ================================================================
#  6. EVALUATION SUR GOLD TEST SET
# ================================================================
print("\n[7/8] Evaluation sur le Gold Test Set (200 posts annotes)...")

gold_path = os.path.join(_proj, 'data', 'gold_test_set_annotation_completed.xlsx')
if os.path.exists(gold_path):
    df_gold = pd.read_excel(gold_path, sheet_name='Resolution')

    gold_texts = df_gold['Texte'].fillna('')
    gold_labels = (df_gold['Label final'] == 'suspect').astype(int).values

    # Extraire features
    X_gold_style = StyleFeatureExtractorV6.extract(gold_texts)
    if emo_loaded:
        X_gold_emo = emo.get_emotion_features(gold_texts.tolist())
        X_gold = np.hstack([X_gold_style, X_gold_emo])
    else:
        X_gold = X_gold_style

    if best_model_name == 'LogReg':
        X_gold_input = scaler.transform(X_gold)
    else:
        X_gold_input = X_gold

    gold_pred = best_model.predict(X_gold_input)
    gold_proba = best_model.predict_proba(X_gold_input)[:, 1] if hasattr(best_model, 'predict_proba') else np.zeros(len(gold_labels))

    print(f"\n  Gold Test Set Results ({best_model_name}) :")
    print(f"    Accuracy  : {accuracy_score(gold_labels, gold_pred):.4f}")
    print(f"    F1 macro  : {f1_score(gold_labels, gold_pred, average='macro'):.4f}")
    print(f"    F1 suspect: {f1_score(gold_labels, gold_pred, pos_label=1, zero_division=0):.4f}")
    print(f"    F1 fiable : {f1_score(gold_labels, gold_pred, pos_label=0):.4f}")

    cm_gold = confusion_matrix(gold_labels, gold_pred)
    print(f"\n    Confusion Matrix (Gold) :")
    print(f"      TN={cm_gold[0,0]:4d}  FP={cm_gold[0,1]:4d}   (fiable)")
    print(f"      FN={cm_gold[1,0]:4d}  TP={cm_gold[1,1]:4d}   (suspect)")

    # Comparaison V5 vs V6
    print(f"\n  --- COMPARAISON V5 vs V6 sur Gold Test Set ---")
    print(f"  {'Metrique':<20s} {'V5':>10s} {'V6':>10s} {'Delta':>10s}")
    print(f"  {'-'*52}")

    # Charger V5 pour comparaison
    from pipeline.expert_detector import ExpertFakeNewsDetector
    det_v5 = ExpertFakeNewsDetector(model_dir=MODEL_DIR, threshold=0.44)
    det_v5.load(suffix='expert_v5')
    v5_results = det_v5.predict(gold_texts)
    v5_pred = v5_results['prediction_label'].values
    v5_scores = v5_results['ai_score_credibility'].values

    v5_f1_suspect = f1_score(gold_labels, v5_pred, pos_label=1, zero_division=0)
    v5_f1_fiable = f1_score(gold_labels, v5_pred, pos_label=0)
    v5_f1_macro = f1_score(gold_labels, v5_pred, average='macro')
    v5_acc = accuracy_score(gold_labels, v5_pred)

    v6_f1_suspect = f1_score(gold_labels, gold_pred, pos_label=1, zero_division=0)
    v6_f1_fiable = f1_score(gold_labels, gold_pred, pos_label=0)
    v6_f1_macro = f1_score(gold_labels, gold_pred, average='macro')
    v6_acc = accuracy_score(gold_labels, gold_pred)

    for metric, v5_val, v6_val in [
        ('Accuracy', v5_acc, v6_acc),
        ('F1 macro', v5_f1_macro, v6_f1_macro),
        ('F1 suspect', v5_f1_suspect, v6_f1_suspect),
        ('F1 fiable', v5_f1_fiable, v6_f1_fiable),
    ]:
        delta = v6_val - v5_val
        arrow = '↑' if delta > 0 else '↓' if delta < 0 else '='
        print(f"  {metric:<20s} {v5_val:>10.4f} {v6_val:>10.4f} {delta:>+10.4f} {arrow}")

    # Detail des erreurs V6 sur gold
    print(f"\n  --- Erreurs V6 sur Gold Test Set ---")
    for i in range(len(gold_labels)):
        if gold_pred[i] != gold_labels[i]:
            true_label = 'suspect' if gold_labels[i] == 1 else 'fiable'
            pred_label = 'suspect' if gold_pred[i] == 1 else 'fiable'
            text = gold_texts.iloc[i][:80]
            print(f"    [{true_label}→{pred_label}] proba={gold_proba[i]:.3f} | {text}")

    # Score distribution V6 sur gold
    print(f"\n  Score distribution (V6) :")
    for label_name, label_val in [('fiable', 0), ('suspect', 1)]:
        mask = gold_labels == label_val
        if mask.sum() > 0:
            scores = gold_proba[mask]
            print(f"    {label_name:8s}: n={mask.sum():3d}, "
                  f"mean={scores.mean():.3f}, med={np.median(scores):.3f}, "
                  f"min={scores.min():.3f}, max={scores.max():.3f}")
else:
    print("  ATTENTION : gold test set non trouve")

# ================================================================
#  7. TEST SUR POSTS BLUESKY REELS
# ================================================================
print("\n[8/8] Test sur exemples representatifs...")

test_texts = pd.Series([
    # Fiable - science/climat
    'Le réchauffement climatique est un enjeu majeur pour notre génération.',
    'New study in Nature shows Arctic ice is melting faster than predicted.',
    'WHO recommends updated COVID boosters for high-risk populations.',
    # Fiable - neutre
    'Il fait beau aujourd\'hui, je vais me promener.',
    'Just had a great coffee at the new place downtown.',
    # Suspect - style desinformation
    'ALERTE: On nous cache la VERITE!! Le vaccin contient des NANOPUCES!!!',
    'EXPOSED: Secret labs are using 5G to spread mind-control virus!!!',
    'SCANDALE: le gouvernement ment, le climat c\'est un HOAX pour nous taxer!!!',
    # Ambigu - opinions fortes mais pas fake
    'Ce gouvernement est une catastrophe absolue pour notre pays.',
    'Trump is the worst president in US history, period.',
])

X_test_examples = StyleFeatureExtractorV6.extract(test_texts)
if emo_loaded:
    X_test_emo = emo.get_emotion_features(test_texts.tolist())
    X_test_examples = np.hstack([X_test_examples, X_test_emo])

if best_model_name == 'LogReg':
    X_test_examples = scaler.transform(X_test_examples)

preds = best_model.predict(X_test_examples)
probas = best_model.predict_proba(X_test_examples)[:, 1] if hasattr(best_model, 'predict_proba') else np.zeros(len(test_texts))

expected = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
print(f"\n  {'Label':>8s} {'Pred':>6s} {'Score':>6s} {'OK':>4s} | Texte")
print(f"  {'-'*80}")
for i, text in enumerate(test_texts):
    label = 'suspect' if preds[i] == 1 else 'fiable'
    ok = '✓' if preds[i] == expected[i] else '✗'
    print(f"  {label:>8s} {preds[i]:>6d} {probas[i]:>6.3f} {ok:>4s} | {text[:60]}")

correct = sum(1 for i in range(len(expected)) if preds[i] == expected[i])
print(f"\n  Score exemples : {correct}/{len(expected)}")

# ================================================================
#  SAUVEGARDE
# ================================================================
print("\n" + "=" * 70)
print("SAUVEGARDE")
print("=" * 70)

save_data = {
    'model': best_model,
    'model_name': best_model_name,
    'scaler': scaler,
    'feature_names': all_feature_names,
    'n_style_features': len(StyleFeatureExtractorV6.FEATURE_NAMES),
    'n_emotion_features': 7 if emo_loaded else 0,
    'threshold': 0.5,
    'cv_f1': best_f1,
    'version': 'v6_style_only',
}

save_path = os.path.join(MODEL_DIR, 'model_style_v6.joblib')
joblib.dump(save_data, save_path)
print(f"  Modele sauvegarde : {save_path}")

elapsed = time.time() - t0
print(f"\n  Temps total : {elapsed:.0f}s ({elapsed/60:.1f}min)")
print("\n" + "=" * 70)
print("TERMINÉ")
print("=" * 70)
