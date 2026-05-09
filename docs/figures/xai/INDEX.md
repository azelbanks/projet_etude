# Index des figures XAI — Thumalien V9

Généré automatiquement par `scripts/run_xai_pipeline.py` le 2026-05-09.
Gold set : 200 posts annotés.

## 1. Explicabilité globale (V6 GradientBoosting + features de style)

| Figure | Fichier |
|---|---|
| Beeswarm summary | `shap_beeswarm_v6.png` |
| Bar global mean(\|SHAP\|) | `shap_global_bar_v6.png` |
| Dependence top-1 | `shap_dependence_*.png` |

## 2. Faithfulness / Fidélité

| Métrique | Valeur |
|---|---|
| AOPC (attribution) | 0.2528 |
| AOPC (random baseline) | 0.0452 |
| Uplift | 0.2076 |
| Comprehensiveness@1 | 0.1561 |
| Comprehensiveness@5 | 0.2318 |

Figure : `faithfulness_aopc_curve.png`

## 3. Décomposition méta-learner V8

Coefficients V8 (LogReg sur 7 features V5+V6+CamemBERT) :
- `score_v5_fiable` : β=+0.393
- `score_v6_suspect` : β=+0.580
- `score_camembert_fiable` : β=-1.116
- `disagreement_v5_v6` : β=-2.939
- `disagreement_v5_cam` : β=+0.948
- `interaction_v5_v6` : β=+1.053
- `min_fiable` : β=-0.836
- `intercept` : β₀=+0.492

Décomposition par exemple :
- **TP** id=56 → P(suspect)=0.633 (SUSPECT, vrai=SUSPECT) — top contributeur : `score_camembert_fiable` (-1.113)
- **FP** id=0 → P(suspect)=0.663 (SUSPECT, vrai=FIABLE) — top contributeur : `score_camembert_fiable` (-1.115)
- **FN** id=9 → P(suspect)=0.449 (FIABLE, vrai=SUSPECT) — top contributeur : `score_camembert_fiable` (-1.004)
- **TN** id=2 → P(suspect)=0.151 (FIABLE, vrai=FIABLE) — top contributeur : `disagreement_v5_v6` (-1.725)

## 4. Attention CamemBERT (TP / FP / FN)

| Type | P(suspect) | Figure heatmap |
|---|---|---|
| TP | 0.986 | `camembert_attention_tp_88.png` |
| FP | 0.960 | `camembert_attention_fp_36.png` |
| FN | 0.101 | `camembert_attention_fn_9.png` |
| TN | 0.062 | `camembert_attention_tn_1.png` |

## 5. Layer Integrated Gradients (Captum)

Niveaux de Completeness (Sundararajan 2017, Kokhlikyan 2020) :
*axiomatique* (|Δ|<0.01), *pratique* (|Δ|<0.05), *indicatif* (|Δ|<0.15).
**Classe expliquée** = classe prédite par CamemBERT (recommandation Captum).

| Échantillon | P(suspect) | Classe expliquée | Δ_convergence | Niveau | Figure |
|---|---|---|---|---|---|
| TP | 0.797 | SUSPECT | +1.54e-01 | ✗ rejet | `ig_suspect_tp_180.png` |
| FP | 0.516 | SUSPECT | +6.27e-02 | ~ indicatif | `ig_suspect_fp_183.png` |
| FN | 0.101 | FIABLE | +4.00e-02 | ✓ pratique | `ig_fiable_fn_9.png` |

---

**Méthodes :**
- SHAP : Lundberg & Lee (NeurIPS 2017)
- Integrated Gradients : Sundararajan et al. (ICML 2017)
- ERASER faithfulness : DeYoung et al. (ACL 2020)

**Conformité AI Act** : ces analyses alimentent l'art. 13 (transparence)
et l'art. 14 (supervision humaine). Voir `docs/12_model_card.md` section 7.
