# Model Card — Thumalien V9

**Document n°** : MC-THUM-2026-001
**Version** : 1.0
**Date** : 2026-05-09
**Auteurs** : Équipe Thumalien
**Format** : Google Model Card (Mitchell et al. 2019), enrichi d'une section
*Explicabilité* conforme à l'AI Act (UE 2024/1689) art. 13.

---

## 1. Détails du modèle

| Champ | Valeur |
|---|---|
| Nom | Thumalien V9 — Pipeline 2-étapes fait/opinion + ensemble hybride |
| Version | 9.0 |
| Type | Classifieur binaire bilingue (FR/EN) — fiable / suspect |
| Architecture | Stage 1 : LogReg fait/opinion · Stage 2 : ensemble V5 (TF-IDF+LogReg) + V6 (style+GradientBoosting) + CamemBERT-base fine-tuné · Méta-learner LogReg |
| Date de l'entraînement | Avril 2026 (V8), mai 2026 (V9 stage 1) |
| Cadre légal | Projet d'étude académique, Master 1 Data Analytics |
| License | À usage pédagogique. Non destiné à la production |
| Contact | équipe Thumalien — voir `docs/00_INDEX.md` |
| Citation | À compléter le cas échéant |

## 2. Usage prévu

**Cas d'usage primaires** :
- Détection automatique de **posts Bluesky** potentiellement non fiables
  pour un usage de monitoring ou de recherche académique sur la
  désinformation.
- Aide à la **modération assistée** (étape de pré-filtrage), avec décision
  finale humaine obligatoire.

**Cas d'usage hors-périmètre** :
- Décisions automatiques affectant des personnes (modération sans
  supervision, sanctions, classement de comptes).
- Évaluation de la véracité factuelle d'un contenu (le modèle détecte des
  signaux de **forme** — style, lexique — pas la vérité d'une affirmation).
- Domaines hors actualité publique généraliste (santé spécialisée,
  juridique, financier).

## 3. Données d'entraînement

| Source | Volume | Période | Langues |
|---|---|---|---|
| Kaggle FR fake news | ~30k | 2018-2024 | FR |
| Credibility Corpus | ~25k | 2017-2023 | EN |
| FakeNewsNet (BuzzFeed/PolitiFact) | ~22k | 2014-2020 | EN |
| Constraint COVID-19 | ~10k | 2020-2021 | EN |
| Bluesky annotation maison | 500 (kappa=0.498) | 2026 | FR/EN |
| **Total** | ~197 782 articles | 2014-2026 | FR + EN |

Voir `docs/00_INDEX.md` D02 et `notebooks/00_Audit_Qualite_Donnees.ipynb` pour
l'audit qualité et les déséquilibres de classes.

## 4. Évaluation

Métrique de référence : **F1 macro sur le gold test set** (200 posts Bluesky
annotés manuellement par 2 annotateurs indépendants, kappa de Cohen = 0.498
*niveau modéré-faible* — la frontière fiable/suspect est par nature subjective).

| Modèle | F1 macro | F1 suspect | F1 fiable | FP | FN |
|---|---|---|---|---|---|
| V5 (TF-IDF + LogReg) | 0.560 | 0.087 | 0.911 | 57 | 6 |
| V6 (Style + GradientBoosting) | 0.578 | 0.103 | 0.954 | 38 | 5 |
| V7 (méta-learner V5+V6) | 0.612 | 0.214 | 0.953 | 25 | 5 |
| **V8 (méta-learner V5+V6+CamemBERT)** | **0.654** | **0.298** | **0.953** | **23** | **4** |
| V9 (V8 + filtre fait/opinion) | 0.671 | 0.331 | 0.954 | 21 | 4 |

Voir `docs/04_revue_challenge_equipe.md` et
`notebooks/21_Gold_Test_Set_Evaluation.py` pour le détail.

## 5. Limites et biais connus

| Catégorie | Limite | Mitigation actuelle |
|---|---|---|
| **Biais lexical** | Le modèle s'appuie fortement sur des marqueurs de surface (mots en majuscules, points d'exclamation, lexique sensationnaliste). Un post fiable au style emphatique peut être mal classé (voir FP exemples) | V6 ajoute des features stylistiques diverses, V8 ajoute un signal sémantique CamemBERT |
| **Couverture linguistique** | Dégradation sur les textes non FR/EN (pas de fallback) | Détection langage avant inférence ; non-FR/EN → score conservateur |
| **Textes courts** | F1 = 0.80 sur posts < 15 mots vs 0.92 sur > 30 mots | Seuils adaptatifs par longueur (cf. `docs/06_analyse_modele_par_longueur.md`) |
| **Drift temporel** | Modèle entraîné principalement sur 2014-2024. Performances dégradées sur événements récents (cf. accord Israël-Palestine 2025-2026) | Re-fine-tuning incrémental V4/V5_bluesky |
| **Subjectivité de l'annotation** | kappa = 0.498 entre annotateurs : la frontière est intrinsèquement floue | Sheet "Resolution" qui résout les désaccords ; gold consensus utilisé pour V8 |
| **Pas d'étiquetage de véracité factuelle** | Le modèle détecte des signaux de forme, pas de fond | Documentation claire dans le dashboard et `guide_utilisateur.md` |

## 6. Considérations éthiques

| Préoccupation | Évaluation |
|---|---|
| Risque de catégorisation injuste de personnes | Élevé si déployé sans supervision. Mitigation : décision humaine obligatoire, pas de réutilisation au-delà du monitoring |
| Risque pour la liberté d'expression | Modéré. Mitigation : transparence complète sur le score (dashboard explicabilité) et droit de contester |
| Données personnelles | Posts publics Bluesky. RGPD : licéité art. 6.1.f (intérêt légitime). Voir `docs/02_conformite_RGPD_AI_Act.md` |
| Empreinte carbone | LogReg = 100× moins d'énergie que RoBERTa. CodeCarbon tracking actif (`docs/10_veille_technologique.md`) |
| AI Act (classification) | **Risque limité** (système d'IA non listé en Annexe III). Mesures de transparence et supervision humaine en place |

## 7. Explicabilité (XAI)

Cette section dépasse le format Google Model Card original et anticipe
l'art. 13 AI Act (transparence des systèmes d'IA).

### 7.1 Méthodes employées

| Méthode | Couverture | Type | Module |
|---|---|---|---|
| **Coefficients LogReg × valeur de feature** | V5 (TF-IDF + linguistique + émotions) | Local exact | `src.pipeline.expert_detector.explain_prediction` |
| **SHAP TreeExplainer** | V6 (35 features stylistiques) | Local + global, exact pour modèles tree-based | `dashboard/app.py`, `notebooks/24_*`, `src.explainability.shap_global` |
| **SHAP beeswarm + dependence plots** | V6 — vue globale | Global, agrégé sur n=200+ | `src.explainability.shap_global.GlobalShapExplainer` |
| **Attention weights** | CamemBERT (12 couches, 12 têtes) | Local, qualitatif (cf. Jain & Wallace 2019) | `src.explainability.attention_viz.CamembertAttentionExplainer` |
| **Layer Integrated Gradients (Captum)** | CamemBERT — embeddings | Local, causal, axiomes Completeness/Sensitivity satisfaits | `src.explainability.integrated_gradients.IGExplainer` |
| **Décomposition exacte du méta-learner** | V7 / V8 (LogReg) | Local + global, formule fermée β·x | `src.explainability.meta_decomposition.MetaLearnerDecomposer` |
| **Faithfulness test (AOPC)** | V6 — validation des explications | Méta-évaluation, vs random baseline | `src.explainability.faithfulness.FaithfulnessEvaluator` |

### 7.2 Validation des explications (faithfulness)

Une explication n'a de valeur que si elle reflète le comportement réel du
modèle. Notre test ERASER-style (DeYoung et al. 2020) :

- **AOPC attribution SHAP V6** : mesuré dynamiquement via
  `scripts/run_xai_pipeline.py` (voir `docs/figures/xai/results.json`).
  Cible : > +0.10 d'uplift par rapport au baseline aléatoire.
- **Comprehensiveness@5** et **Sufficiency@5** : reportés pour chaque
  modèle dans le pipeline XAI.
- **Completeness IG** (axiome de Sundararajan 2017) : on cible `|Δ| < 0.05`
  (niveau "pratique" selon Kokhlikyan et al. 2020). Sur CamemBERT-base
  (12 couches + head MLP non-linéaire) on observe trois régimes :
    1. **Modèle indécis** (P proche de 0.5) : convergence à |Δ| < 0.02 (axiomatique)
    2. **Modèle confiant** (P > 0.7 ou P < 0.3) : |Δ| ≈ 0.05–0.15 (indicatif)
    3. **Cas saturés** (rare, gradient quasi-nul à cause du ReLU dans
       `head : Linear → ReLU → Dropout → Linear`) : |Δ| > 0.15 même à
       n_steps=1000. Ce n'est **pas un bug** — c'est une signature de la
       rigidité de la décision sur ces inputs. À documenter au cas par cas.

  **Workarounds appliqués** :
  - Bascule automatique CPU sur Apple Silicon (bug Captum/MPS documenté)
  - Stratégie de baseline auto (`<unk>` → `<pad>`)
  - Escalade n_steps : 200 → 500 → 1000 si non convergent
  - **Classe expliquée = classe prédite** par le modèle (recommandation
    Captum) : sinon les FN/FP donnent des gradients dégénérés
  - Implémentation `eager` forcée pour les attentions (sinon SDPA bloque
    la backprop sur transformers ≥ 4.40)

### 7.3 Limites de l'explicabilité

- L'attention CamemBERT n'est **pas** une explication causale (Jain &
  Wallace 2019). Pour des affirmations causales, se référer à IG.
- SHAP TreeExplainer assume l'indépendance des features ; sur des
  features stylistiques corrélées (ex: `caps_ratio` et
  `all_caps_words_ratio`), les valeurs sont à interpréter avec prudence.
- L'agrégation `mean(|SHAP|)` masque les hétérogénéités sous-groupe :
  une analyse stratifiée (par langue, par longueur de texte) est dans la
  roadmap.
- Les méthodes ci-dessus expliquent **comment** le modèle décide, pas
  **pourquoi** sa décision est juste — la véracité factuelle reste hors
  périmètre.

### 7.4 Audience des explications

| Audience | Méthode privilégiée | Lieu |
|---|---|---|
| Utilisateur du dashboard | SHAP bar plot V6 + décomposition méta-learner V8 | `dashboard/app.py` page Analyse Temps Réel |
| Modérateur / annotateur | Top mots LogReg + features sensationnalistes détectées | `dashboard/app.py` panneau "Pourquoi ce score ?" |
| Régulateur / auditeur | Cette model card + `docs/figures/xai/INDEX.md` + `results.json` | Repo |
| Chercheur ML | Notebook 24, scripts XAI, AOPC + IG completeness | `notebooks/`, `scripts/run_xai_pipeline.py` |

## 8. Recommandations

- Ne jamais utiliser le score V9 comme **verdict** : il s'agit d'un
  indicateur dans une chaîne de modération.
- Valider l'absence de drift trimestriellement avec un nouveau gold set.
- Re-calculer le faithfulness test à chaque changement de modèle V_x+1
  pour vérifier que les explications restent fidèles.
- Documenter chaque évolution dans le journal de décision
  (`docs/00_INDEX.md`).

## 9. Références

- Mitchell et al. (2019) *Model Cards for Model Reporting*, FAT*.
- Lundberg & Lee (2017) *A Unified Approach to Interpreting Model
  Predictions*, NeurIPS.
- Sundararajan, Taly & Yan (2017) *Axiomatic Attribution for Deep
  Networks*, ICML.
- Jain & Wallace (2019) *Attention is not Explanation*, NAACL.
- DeYoung et al. (2020) *ERASER: A Benchmark to Evaluate Rationalized NLP
  Models*, ACL.
- Reglement (UE) 2024/1689 (AI Act), notamment art. 13–14.

---

*Cette model card est un livrable du module pédagogique "IA explicable et
responsable" du projet d'étude. Elle complète `docs/02_conformite_RGPD_AI_Act.md`
en se focalisant sur la transparence du modèle plutôt que sur le traitement
des données personnelles.*
