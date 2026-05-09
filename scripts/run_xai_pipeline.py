#!/usr/bin/env python3
"""
Pipeline XAI complet — Thumalien
==================================

Reproduit toutes les analyses d'IA explicable et génère les figures du
rapport dans `docs/figures/xai/`. Conçu pour être lancé en une commande :

    python3 scripts/run_xai_pipeline.py [--skip transformer] [--skip ig]

Étapes :
    1. SHAP global sur V6 (beeswarm + dependence + bar)
    2. Faithfulness test sur V6 avec masquage top-k SHAP
    3. Décomposition méta-learner V8 sur 3 exemples (TP, FP, FN)
    4. Attention CamemBERT sur 3 exemples (TP, FP, FN)
    5. Layer Integrated Gradients via Captum sur 3 exemples
    6. Génère un index Markdown des figures

Le script calcule **toutes les features à la volée** depuis le gold set
annoté ; aucun cache préalable n'est requis (mais un cache est écrit pour
accélérer les runs suivants).

Tous les artefacts sont écrits dans :
    docs/figures/xai/                   <- PNG
    docs/figures/xai/results.json       <- métriques agrégées
    docs/figures/xai/INDEX.md           <- index pour le rapport
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

OUTPUT_DIR = PROJ_ROOT / "docs" / "figures" / "xai"
CACHE_DIR = PROJ_ROOT / "data" / "cache_xai"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_xai")


# =====================================================================
#  Helpers généraux
# =====================================================================

def load_gold_set() -> pd.DataFrame:
    """
    Charge le gold test set annoté. Colonnes normalisées :
    `text`, `label_int` (0=fiable, 1=suspect), `id`.
    """
    candidates = [
        PROJ_ROOT / "data" / "gold_test_set_annotation_completed.xlsx",
        PROJ_ROOT / "data" / "gold_test_set_annotation.xlsx",
        PROJ_ROOT / "data" / "gold_test_set.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        if p.suffix == ".xlsx":
            try:
                df = pd.read_excel(p, sheet_name="Resolution")
            except Exception:
                df = pd.read_excel(p)
            text_col = next((c for c in ["Texte", "text"] if c in df.columns), None)
            label_col = next(
                (c for c in ["Label final", "label", "y_true"] if c in df.columns), None
            )
            if text_col is None or label_col is None:
                continue
            df = df[[text_col, label_col]].rename(
                columns={text_col: "text", label_col: "label_raw"}
            )
            df["text"] = df["text"].fillna("").astype(str)
            df["label_int"] = (df["label_raw"].astype(str).str.lower() == "suspect").astype(int)
            df = df[df["text"].str.len() > 0].reset_index(drop=True)
            df["id"] = range(len(df))
            return df
        if p.suffix == ".csv":
            df = pd.read_csv(p)
            df = df.dropna(subset=["text"])
            if "label" in df.columns:
                df["label_int"] = (df["label"].astype(str).str.lower() == "suspect").astype(int)
            else:
                df["label_int"] = -1
            df["id"] = range(len(df))
            return df[["id", "text", "label_int"]]
    raise FileNotFoundError("Aucun gold test set trouvé sous data/")


def select_examples(df: pd.DataFrame, predictions: np.ndarray, k: int = 1) -> dict:
    """Sélectionne k exemples par classe d'erreur (TP, FP, FN, TN)."""
    out = {}
    for label, mask in [
        ("TP", (predictions == 1) & (df["label_int"] == 1)),
        ("FP", (predictions == 1) & (df["label_int"] == 0)),
        ("FN", (predictions == 0) & (df["label_int"] == 1)),
        ("TN", (predictions == 0) & (df["label_int"] == 0)),
    ]:
        idx = df.index[mask][:k].tolist()
        out[label] = df.loc[idx].to_dict("records") if idx else []
    return out


# =====================================================================
#  Feature engineering : V6 (style + emotions) au vol
# =====================================================================

def build_v6_features(texts: pd.Series) -> tuple[np.ndarray, list]:
    """
    Construit la matrice X (n, 35) — 28 style + 7 émotions — comme dans le
    notebook 24, et renvoie aussi la liste des 35 noms de features.
    """
    from pipeline.style_features import StyleFeatureExtractorV6
    from pipeline.expert_detector import EmotionFeatureExtractor

    X_style = StyleFeatureExtractorV6.extract(texts)
    feat_names = list(StyleFeatureExtractorV6.FEATURE_NAMES)

    emo = EmotionFeatureExtractor(model_dir=str(PROJ_ROOT / "models"))
    emo_loaded = emo.load()
    if emo_loaded:
        X_emo = emo.get_emotion_features(
            texts.tolist() if hasattr(texts, "tolist") else list(texts)
        )
        X = np.hstack([X_style, X_emo])
        # Noms d'émotions cohérents avec le dashboard
        emo_names = [
            "emo_anger", "emo_disgust", "emo_joy", "emo_neutral",
            "emo_fear", "emo_surprise", "emo_sadness",
        ]
        feat_names += emo_names[: X_emo.shape[1]]
    else:
        X = X_style
    return X, feat_names


def compute_v5_v6_scores(
    texts: pd.Series, X_v6_input: np.ndarray, v6_data: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule (score_v5_fiable, score_v6_suspect) sur les textes du gold.
    """
    from pipeline.expert_detector import ExpertFakeNewsDetector

    det = ExpertFakeNewsDetector(model_dir=str(PROJ_ROOT / "models"), threshold=0.44)
    det.load(suffix="expert_v5")
    v5_res = det.predict(texts)
    score_v5 = v5_res["ai_score_credibility"].values  # P(fiable)

    v6_model = v6_data["model"]
    v6_name = v6_data.get("model_name", "")
    v6_scaler = v6_data.get("scaler")
    if v6_name == "LogReg" and v6_scaler is not None:
        X_v6_input = v6_scaler.transform(X_v6_input)
    score_v6 = v6_model.predict_proba(X_v6_input)[:, 1]  # P(suspect)
    return score_v5, score_v6


# =====================================================================
#  Étape 1 : SHAP global sur V6
# =====================================================================

def step_shap_global(gold_df: pd.DataFrame) -> dict:
    log.info("[1/6] SHAP global sur V6")
    import joblib
    from explainability.shap_global import GlobalShapExplainer

    v6 = joblib.load(PROJ_ROOT / "models" / "model_style_v6.joblib")

    # X au vol — pas de dépendance à un cache
    X, feat_names = build_v6_features(gold_df["text"])
    np.save(CACHE_DIR / "X_gold_v6.npy", X)
    log.info("  X généré (%d, %d), cache=%s", *X.shape, CACHE_DIR / "X_gold_v6.npy")

    # Si V6 est LogReg, le scaler est requis pour SHAP
    X_in = X
    if v6.get("model_name") == "LogReg" and v6.get("scaler") is not None:
        X_in = v6["scaler"].transform(X)

    explainer = GlobalShapExplainer(
        model=v6["model"],
        feature_names=feat_names,
        output_dir=str(OUTPUT_DIR),
    )
    result = explainer.explain(X_in)
    log.info("  Beeswarm : %s", result.figures.get("beeswarm"))
    return json.loads(result.to_json())


# =====================================================================
#  Étape 2 : Faithfulness test
# =====================================================================

def step_faithfulness(gold_df: pd.DataFrame) -> dict:
    log.info("[2/6] Faithfulness test sur V6")
    import joblib
    import shap
    from explainability.faithfulness import FaithfulnessEvaluator

    v6 = joblib.load(PROJ_ROOT / "models" / "model_style_v6.joblib")

    # Réutilise le cache si possible, sinon régénère
    cache = CACHE_DIR / "X_gold_v6.npy"
    if cache.exists():
        X = np.load(cache)
    else:
        X, _ = build_v6_features(gold_df["text"])
        np.save(cache, X)

    X_in = X
    if v6.get("model_name") == "LogReg" and v6.get("scaler") is not None:
        X_in = v6["scaler"].transform(X)

    model = v6["model"]
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X_in)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]

    ev = FaithfulnessEvaluator(
        predict_proba_fn=model.predict_proba,
        class_index=1,
        output_dir=str(OUTPUT_DIR),
    )
    cmp = ev.compare_with_random(
        X_in, sv,
        n_random_seeds=5,
        max_k=min(20, X_in.shape[1]),
    )
    log.info(
        "  AOPC attribution=%.4f | random=%.4f (uplift=%+.4f)",
        cmp["aopc_attribution"], cmp["aopc_random_mean"], cmp["aopc_uplift"],
    )
    return {
        "aopc_attribution": cmp["aopc_attribution"],
        "aopc_random_mean": cmp["aopc_random_mean"],
        "aopc_uplift": cmp["aopc_uplift"],
        "comprehensiveness_at_k": cmp["result"].comprehensiveness_at_k,
        "sufficiency_at_k": cmp["result"].sufficiency_at_k,
        "figures": cmp["result"].figures,
    }


# =====================================================================
#  Étape 3 : Décomposition méta-learner V8
# =====================================================================

def step_meta_decomposition(gold_df: pd.DataFrame, cam_classifier=None) -> dict:
    """
    Calcule X_meta au vol pour le gold set, applique V8, décompose 3
    exemples (TP / FP / FN). Sauvegarde X_meta en cache.
    """
    log.info("[3/6] Décomposition méta-learner V8")
    import joblib
    from explainability.meta_decomposition import MetaLearnerDecomposer

    v8 = joblib.load(PROJ_ROOT / "models" / "model_hybrid_v8.joblib")
    decomposer = MetaLearnerDecomposer(v8)

    # Construire X_meta selon le format attendu par V8
    X_v6, _ = build_v6_features(gold_df["text"])
    v6 = joblib.load(PROJ_ROOT / "models" / "model_style_v6.joblib")
    score_v5, score_v6 = compute_v5_v6_scores(gold_df["text"], X_v6, v6)

    uses_cam = v8.get("uses_camembert", False)

    if uses_cam and cam_classifier is not None and cam_classifier._loaded:
        try:
            score_cam = np.array(
                cam_classifier.predict_credibility_scores(gold_df["text"].tolist())
            )
        except Exception as e:
            log.warning("  CamemBERT inference échec (%s) — score_cam=0.5", e)
            score_cam = np.full(len(gold_df), 0.5)
    else:
        score_cam = np.full(len(gold_df), 0.5)

    disagreement_v5_v6 = np.abs(score_v5 - (1 - score_v6))
    disagreement_v5_cam = np.abs(score_v5 - score_cam)
    interaction_v5_v6 = score_v5 * score_v6
    min_fiable = np.minimum(score_v5, score_cam)

    if uses_cam:
        X_meta = np.column_stack([
            score_v5, score_v6, score_cam,
            disagreement_v5_v6, disagreement_v5_cam,
            interaction_v5_v6, min_fiable,
        ])
    else:
        X_meta = np.column_stack([
            score_v5, score_v6, disagreement_v5_v6, interaction_v5_v6,
        ])

    np.save(CACHE_DIR / "X_meta_gold.npy", X_meta)

    # Predictions V8 et sélection TP/FP/FN
    p_suspect = decomposer.meta_model.predict_proba(X_meta)[:, 1]
    preds = (p_suspect >= decomposer.threshold).astype(int)
    selected = select_examples(gold_df, preds, k=1)

    out = {
        "feature_names": decomposer.feature_names,
        "coef": decomposer.coef.tolist(),
        "intercept": decomposer.intercept,
        "n_samples": int(len(gold_df)),
        "samples": [],
    }
    import matplotlib.pyplot as plt

    for err_type, examples in selected.items():
        for ex in examples:
            i = int(ex["id"])
            d = decomposer.decompose(X_meta[i])

            # Bar plot par exemple
            order = np.argsort(np.abs(d.contributions))[::-1]
            names_o = [d.feature_names[k] for k in order][::-1]
            vals_o = [d.contributions[k] for k in order][::-1]
            colors = ["#FF1744" if v > 0 else "#00E676" for v in vals_o]

            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.barh(range(len(names_o)), vals_o, color=colors)
            ax.set_yticks(range(len(names_o)))
            ax.set_yticklabels(names_o, fontsize=10)
            ax.axvline(0, color="k", linewidth=0.6)
            ax.set_xlabel("Contribution β·x au logit (+ = pousse vers SUSPECT)", fontsize=10)
            ax.set_title(
                f"Décomposition V8 — {err_type} (id={i})\n"
                f'"{ex["text"][:80]}{"..." if len(ex["text"])>80 else ""}"\n'
                f"logit z = {d.logit:+.3f} → P(suspect) = {d.proba_suspect:.3f} → {d.label}",
                fontsize=10,
            )
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            fig_path = OUTPUT_DIR / f"meta_decomposition_v8_{err_type.lower()}_{i}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            out["samples"].append({
                "error_type": err_type,
                "sample_idx": i,
                "text": ex["text"][:160],
                "logit": d.logit,
                "proba_suspect": d.proba_suspect,
                "label": d.label,
                "ground_truth": "SUSPECT" if ex["label_int"] == 1 else "FIABLE",
                "top_drivers": d.top_drivers(5),
                "figure": str(fig_path),
            })
            log.info(
                "  %s id=%d → P=%.3f (%s, vrai=%s)",
                err_type, i, d.proba_suspect, d.label,
                "SUSPECT" if ex["label_int"] == 1 else "FIABLE",
            )
    return out


# =====================================================================
#  Étape 4 : Attention CamemBERT
# =====================================================================

def _try_load_camembert():
    """Charge CamemBERT en essayant camembert_fr (full) puis camembert_best."""
    try:
        from pipeline.camembert_classifier import CamemBERTClassifier
    except ImportError as e:
        log.warning("  transformers/torch indisponible (%s)", e)
        return None
    clf = CamemBERTClassifier(model_dir=str(PROJ_ROOT / "models"))
    for suffix in ("camembert_fr", "camembert_best"):
        try:
            if clf.load(suffix=suffix):
                log.info("  CamemBERT chargé (%s.pt)", suffix)
                return clf
        except Exception as e:
            log.warning("  load(%s) a échoué : %s", suffix, e)
    return None


def step_attention(gold_df: pd.DataFrame, cam_classifier=None) -> dict:
    log.info("[4/6] Attention CamemBERT (TP, FP, FN)")
    if cam_classifier is None:
        cam_classifier = _try_load_camembert()
    if cam_classifier is None:
        return {"skipped": True, "reason": "camembert_not_loaded"}

    from explainability.attention_viz import CamembertAttentionExplainer
    explainer = CamembertAttentionExplainer(cam_classifier, output_dir=str(OUTPUT_DIR))

    # Filtre FR pour CamemBERT (heuristique simple)
    df_fr = gold_df.copy()
    # Non-capturing group (?:...) pour éviter le warning pandas sur les groupes
    df_fr["language"] = df_fr["text"].str.contains(
        r"\b(?:le|la|de|des|et|une|sont|avec|pour|dans|que|qui|pas|mais|c'est|n'est)\b",
        regex=True, case=False,
    ).map({True: "fr", False: "en"})
    df_fr = df_fr[df_fr["language"] == "fr"].reset_index(drop=True)
    if df_fr.empty:
        log.warning("  Aucun texte FR dans le gold — skip attention")
        return {"skipped": True, "reason": "no_french_in_gold"}

    preds = cam_classifier.predict(df_fr["text"].tolist())["predictions"]
    samples = select_examples(df_fr, np.asarray(preds), k=1)

    out = {}
    for err_type, examples in samples.items():
        for ex in examples:
            res = explainer.explain(
                ex["text"],
                ground_truth="SUSPECT" if ex["label_int"] == 1 else "FIABLE",
                tag=f"{err_type.lower()}_{ex['id']}",
            )
            out[err_type] = {
                "text": ex["text"][:160],
                "id": int(ex["id"]),
                "prediction": res.prediction_label,
                "p_suspect": res.prediction_proba_suspect,
                "figures": res.figures,
            }
            log.info(
                "  %s id=%d → %s (P=%.2f)",
                err_type, ex["id"], res.prediction_label, res.prediction_proba_suspect,
            )
    return out


# =====================================================================
#  Étape 5 : Layer Integrated Gradients
# =====================================================================

def step_integrated_gradients(gold_df: pd.DataFrame, cam_classifier=None) -> dict:
    """
    IG sur CamemBERT — sélectionne 1 TP, 1 FP et 1 FN pertinents
    (= où le modèle prédit avec une P_suspect informative, ni 0.0 ni 1.0).
    Filtre les textes FR uniquement (CamemBERT FR).
    """
    log.info("[5/6] Layer Integrated Gradients (Captum) sur CamemBERT")
    try:
        import captum  # noqa: F401
    except ImportError:
        log.warning("  captum non installé — pip install captum  (skip)")
        return {"skipped": True, "reason": "captum_not_installed"}

    if cam_classifier is None:
        cam_classifier = _try_load_camembert()
    if cam_classifier is None:
        return {"skipped": True, "reason": "camembert_not_loaded"}

    from explainability.integrated_gradients import IGExplainer
    explainer = IGExplainer(
        cam_classifier,
        output_dir=str(OUTPUT_DIR),
        n_steps=200,                    # transformers : 200 = bon compromis
        target_class=1,
        baseline_strategy="auto",
    )

    # Filtre FR (heuristique simple, idem step_attention)
    df = gold_df.copy()
    df["language"] = df["text"].str.contains(
        r"\b(?:le|la|de|des|et|une|sont|avec|pour|dans|que|qui|pas|mais|c'est|n'est)\b",
        regex=True, case=False,
    ).map({True: "fr", False: "en"})
    df_fr = df[(df["language"] == "fr") & (df["text"].str.len() > 30)].reset_index(drop=True)
    if df_fr.empty:
        return {"skipped": True, "reason": "no_french_in_gold"}

    # Predict CamemBERT pour identifier TP/FP/FN
    pred_dict = cam_classifier.predict(df_fr["text"].tolist())
    preds = np.asarray(pred_dict["predictions"])
    # `probabilities` = P(fiable) → P_suspect = 1 - p
    p_suspect = 1.0 - np.asarray(pred_dict["probabilities"])

    # Candidats : on veut des P_suspect ni saturés ni nuls
    # (sinon la vraie attribution IG est dégénérée)
    samples = []
    for err_type, mask in [
        ("TP", (preds == 1) & (df_fr["label_int"] == 1)),
        ("FP", (preds == 1) & (df_fr["label_int"] == 0)),
        ("FN", (preds == 0) & (df_fr["label_int"] == 1)),
    ]:
        candidates = df_fr.index[mask].tolist()
        if not candidates:
            continue
        # Trier par "informativité" : P_suspect proche de 0.5 = plus informatif
        candidates.sort(key=lambda i: abs(p_suspect[i] - 0.5))
        i = candidates[0]
        samples.append((err_type, i))

    out = {}
    for err_type, i in samples:
        row = df_fr.iloc[i]
        # Pour chaque exemple on explique la classe **prédite** par CamemBERT
        # — sinon les FN (modèle prédit FIABLE alors que c'est SUSPECT) donnent
        # des attributions IG dégénérées (gradients dans une zone saturée).
        # Cette pratique est recommandée par Captum (cf. tutoriel "BERT
        # Question Answering").
        target = int(preds[i])  # 0 = FIABLE, 1 = SUSPECT
        target_name = "SUSPECT" if target == 1 else "FIABLE"
        try:
            res = explainer.explain(
                row["text"],
                tag=f"{err_type.lower()}_{row['id']}",
                target_class=target,
            )
            ok = explainer.completeness_check(res)
            out[err_type] = {
                "text": row["text"][:160],
                "id": int(row["id"]),
                "p_suspect_camembert": float(p_suspect[i]),
                "p_suspect_ig": res.prediction_proba[1],
                "target_explained": target_name,
                "convergence_delta": res.convergence_delta,
                "completeness_ok": ok,
                "figures": res.figures,
            }
            log.info(
                "  %s id=%d P=%.3f target=%s Δ=%.2e completeness=%s",
                err_type, row["id"], p_suspect[i], target_name,
                res.convergence_delta, ok,
            )
        except Exception as e:
            log.warning("  IG échoué pour id=%d : %s", row["id"], e)
    return out


# =====================================================================
#  Étape 6 : Index Markdown
# =====================================================================

INDEX_TEMPLATE = """# Index des figures XAI — Thumalien V9

Généré automatiquement par `scripts/run_xai_pipeline.py` le {date}.
Gold set : {n_gold} posts annotés.

## 1. Explicabilité globale (V6 GradientBoosting + features de style)

| Figure | Fichier |
|---|---|
| Beeswarm summary | `shap_beeswarm_v6.png` |
| Bar global mean(\\|SHAP\\|) | `shap_global_bar_v6.png` |
| Dependence top-1 | `shap_dependence_*.png` |

## 2. Faithfulness / Fidélité

| Métrique | Valeur |
|---|---|
| AOPC (attribution) | {aopc_attr} |
| AOPC (random baseline) | {aopc_rand} |
| Uplift | {uplift} |
| Comprehensiveness@1 | {comp1} |
| Comprehensiveness@5 | {comp5} |

Figure : `faithfulness_aopc_curve.png`

## 3. Décomposition méta-learner V8

Coefficients V8 (LogReg sur 7 features V5+V6+CamemBERT) :
{meta_coefs}

Décomposition par exemple :
{meta_samples}

## 4. Attention CamemBERT (TP / FP / FN)

| Type | P(suspect) | Figure heatmap |
|---|---|---|
{attn_rows}

## 5. Layer Integrated Gradients (Captum)

Niveaux de Completeness (Sundararajan 2017, Kokhlikyan 2020) :
*axiomatique* (|Δ|<0.01), *pratique* (|Δ|<0.05), *indicatif* (|Δ|<0.15).
**Classe expliquée** = classe prédite par CamemBERT (recommandation Captum).

| Échantillon | P(suspect) | Classe expliquée | Δ_convergence | Niveau | Figure |
|---|---|---|---|---|---|
{ig_rows}

---

**Méthodes :**
- SHAP : Lundberg & Lee (NeurIPS 2017)
- Integrated Gradients : Sundararajan et al. (ICML 2017)
- ERASER faithfulness : DeYoung et al. (ACL 2020)

**Conformité AI Act** : ces analyses alimentent l'art. 13 (transparence)
et l'art. 14 (supervision humaine). Voir `docs/12_model_card.md` section 7.
"""


def build_index(results: dict, n_gold: int) -> Path:
    from datetime import date

    aopc = results.get("faithfulness", {})

    def fmt(x, fmt_spec=".4f"):
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return format(x, fmt_spec)
        return "N/A"

    aopc_attr = fmt(aopc.get("aopc_attribution"))
    aopc_rand = fmt(aopc.get("aopc_random_mean"))
    uplift = fmt(aopc.get("aopc_uplift"))
    comp = aopc.get("comprehensiveness_at_k", {})
    comp1 = fmt(comp.get(1))
    comp5 = fmt(comp.get(5))

    meta = results.get("meta_decomposition", {})
    meta_coefs = "\n".join(
        f"- `{n}` : β={c:+.3f}"
        for n, c in zip(meta.get("feature_names", []), meta.get("coef", []))
    ) or "_(non généré)_"
    meta_intercept = meta.get("intercept")
    if isinstance(meta_intercept, (int, float)):
        meta_coefs += f"\n- `intercept` : β₀={meta_intercept:+.3f}"

    meta_samples = "\n".join(
        f"- **{s['error_type']}** id={s['sample_idx']} → "
        f"P(suspect)={s['proba_suspect']:.3f} ({s['label']}, vrai={s['ground_truth']}) — "
        f"top contributeur : `{s['top_drivers'][0]['feature']}` "
        f"({s['top_drivers'][0]['contribution']:+.3f})"
        for s in meta.get("samples", [])
    ) or "_(non généré)_"

    attn = results.get("attention", {})
    attn_rows_list = []
    for k in ["TP", "FP", "FN", "TN"]:
        info = attn.get(k)
        if isinstance(info, dict) and "figures" in info:
            fig = os.path.basename(info["figures"].get("heatmap", ""))
            attn_rows_list.append(
                f"| {k} | {info.get('p_suspect', 0):.3f} | `{fig}` |"
            )
    attn_rows = "\n".join(attn_rows_list) or \
        "_(non généré — `transformers` ou modèle CamemBERT manquant)_"

    def _ig_level(d):
        a = abs(d)
        if a < 0.01: return "✓ axiomatique"
        if a < 0.05: return "✓ pratique"
        if a < 0.15: return "~ indicatif"
        return "✗ rejet"

    ig = results.get("integrated_gradients", {})
    ig_rows_list = []
    for sid, info in ig.items():
        if not isinstance(info, dict) or "figures" not in info:
            continue
        fig = os.path.basename(info["figures"].get("heatmap", ""))
        p = info.get("p_suspect_camembert") or info.get("p_suspect_ig") or 0
        delta = info.get("convergence_delta", 0)
        target = info.get("target_explained", "?")
        ig_rows_list.append(
            f"| {sid} | {p:.3f} | {target} | {delta:+.2e} | "
            f"{_ig_level(delta)} | `{fig}` |"
        )
    ig_rows = "\n".join(ig_rows_list) or \
        "_(non généré — `captum` ou modèle manquant)_"

    md = INDEX_TEMPLATE.format(
        date=date.today().isoformat(), n_gold=n_gold,
        aopc_attr=aopc_attr, aopc_rand=aopc_rand, uplift=uplift,
        comp1=comp1, comp5=comp5,
        meta_coefs=meta_coefs, meta_samples=meta_samples,
        attn_rows=attn_rows, ig_rows=ig_rows,
    )
    path = OUTPUT_DIR / "INDEX.md"
    path.write_text(md, encoding="utf-8")
    log.info("Index écrit : %s", path)
    return path


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip", action="append", default=[],
        choices=["shap", "faithfulness", "meta", "attention", "ig"],
        help="Étapes à sauter (peut être répété).",
    )
    args = parser.parse_args()

    t0 = time.time()
    gold = load_gold_set()
    log.info("Gold set chargé : %d posts (%d suspects, %d fiables)",
             len(gold), int((gold["label_int"] == 1).sum()),
             int((gold["label_int"] == 0).sum()))

    # CamemBERT chargé une seule fois (réutilisé par méta + attention + IG)
    cam_clf = None
    if not {"meta", "attention", "ig"}.issubset(args.skip):
        cam_clf = _try_load_camembert()

    results = {}
    if "shap" not in args.skip:
        try:
            results["shap_global"] = step_shap_global(gold)
        except Exception as e:
            log.error("step_shap_global a échoué : %s", e)
            results["shap_global"] = {"error": str(e)}
    if "faithfulness" not in args.skip:
        try:
            results["faithfulness"] = step_faithfulness(gold)
        except Exception as e:
            log.error("step_faithfulness a échoué : %s", e)
            results["faithfulness"] = {"error": str(e)}
    if "meta" not in args.skip:
        try:
            results["meta_decomposition"] = step_meta_decomposition(gold, cam_clf)
        except Exception as e:
            log.error("step_meta_decomposition a échoué : %s", e)
            results["meta_decomposition"] = {"error": str(e)}
    if "attention" not in args.skip:
        try:
            results["attention"] = step_attention(gold, cam_clf)
        except Exception as e:
            log.error("step_attention a échoué : %s", e)
            results["attention"] = {"error": str(e)}
    if "ig" not in args.skip:
        try:
            results["integrated_gradients"] = step_integrated_gradients(gold, cam_clf)
        except Exception as e:
            log.error("step_integrated_gradients a échoué : %s", e)
            results["integrated_gradients"] = {"error": str(e)}

    out_json = OUTPUT_DIR / "results.json"
    out_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log.info("Résultats sérialisés : %s", out_json)

    build_index(results, n_gold=len(gold))
    log.info("Pipeline XAI terminé en %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
