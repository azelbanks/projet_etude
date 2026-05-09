# Politique de Veille Technologique
## Projet Thumalien - Suivi des évolutions technologiques et réglementaires

**Référence** : VEILLE-THUM-2026-001
**Version** : 1.1
**Date** : Avril 2026

---

## 1. Objectif

La veille technologique permet de maintenir le projet Thumalien à l'état de l'art en matière de NLP, détection de désinformation et conformité réglementaire. Elle alimente les décisions d'évolution du pipeline et anticipe les ruptures technologiques.

---

## 2. Domaines de veille

### 2.1 Veille technique (NLP / IA)

| Sujet | Sources | Frequence | Responsable |
|-------|---------|-----------|-------------|
| Modèles de langue (LLM, BERT, etc.) | Hugging Face blog, arXiv (cs.CL) | Hebdomadaire | Azélie |
| Détection de fake news (SOTA) | Papers With Code, ACL Anthology | Mensuelle | Azélie |
| Librairies Python (scikit-learn, PyTorch, transformers) | GitHub releases, changelogs | Mensuelle | Azélie |
| Techniques d'augmentation de données | arXiv, blogs ML | Mensuelle | Sébastien |
| Évaluation et métriques NLP | Conférences (ACL, EMNLP, NeurIPS) | Trimestrielle | Azélie |

### 2.2 Veille réglementaire

| Sujet | Sources | Fréquence | Responsable |
|-------|---------|-----------|-------------|
| RGPD (évolutions, jurisprudence) | CNIL, EUR-Lex | Trimestrielle | Azélie |
| AI Act (mise en application) | Commission Européenne, CNIL | Trimestrielle | Azélie |
| Éthique de l'IA | IEEE, ACM, rapports OCDE | Semestrielle | Sébastien |

### 2.3 Veille écosystème

| Sujet | Sources | Fréquence | Responsable |
|-------|---------|-----------|-------------|
| API AT Protocol (Bluesky) | GitHub bluesky-social, docs AT Proto | Hebdomadaire | Azélie |
| Concurrents (fact-checking, outils similaires) | ProductHunt, TechCrunch | Mensuelle | Sébastien |
| Frameworks dashboard (Streamlit, Gradio) | GitHub releases | Mensuelle | Azélie |
| Docker / MongoDB (mises à jour sécurité) | Docker Hub, MongoDB blog | Mensuelle | Azélie |

---

## 3. Technologies identifiées et évaluées

### 3.1 Technologies adoptées (intégrées au projet)

| Technologie | Date évaluation | Décision | Justification |
|------------|:--------------:|:--------:|---------------|
| CamemBERT (FR) | Avril 2026 | Adopté | F1 0.957 sur textes courts FR, supérieur au TF-IDF |
| RoBERTa (EN) | Avril 2026 | Adopté | F1 0.874 sur textes courts EN, complémentaire au pipeline |
| Pipeline hybride (stacking) | Avril 2026 | Adopté | Combine les forces TF-IDF (robustesse) et Transformer (précision courts) |
| CodeCarbon | Jan 2026 | Adopté | Monitoring CO2 transparent, zéro impact performance |
| fpdf2 | Avril 2026 | Adopté | Génération PDF depuis Python, pas de dépendance externe |
| Plotly | Mars 2026 | Adopté | Graphiques interactifs pour le dashboard, meilleur que matplotlib |

### 3.2 Technologies évaluées et non retenues

| Technologie | Date évaluation | Décision | Justification |
|------------|:--------------:|:--------:|---------------|
| TensorFlow (émotions) | Jan 2026 | Rejeté | Incompatible Apple Silicon M4, remplacé par PyTorch |
| GPT-4 pour détection | Fév 2026 | Non retenu | Coût prohibitif en inférence, latence trop élevée, dépendance API |
| Sentence-Transformers | Mars 2026 | Reporté | Prometteur mais V5 + Transformers couvrent le besoin actuel |
| FastAPI (serving) | Mars 2026 | Reporté | Pas encore nécessaire, inférence batch suffisante |
| Elasticsearch | Déc 2025 | Non retenu | MongoDB suffisant pour nos volumes, complexité inutile |

### 3.3 Technologies à surveiller (roadmap)

| Technologie | Intérêt pour Thumalien | Horizon | Priorité |
|------------|------------------------|:-------:|:--------:|
| RAG (Retrieval-Augmented Generation) | Cross-checker les claims avec une base factuelle (Wikipedia, Google Fact Check Tools API) | T3 2026 | **Haute** |
| LLM-as-Judge (GPT-4o, Claude) | Scoring zero-shot de la crédibilité, benchmark contre V7 | T3 2026 | **Haute** |
| Active Learning (modAL) | Cibler l'annotation sur les posts à fort disagreement V5/V6 | T2 2026 | **Haute** |
| Conformal Prediction | Intervalles de confiance calibrés au lieu de scores bruts | T3 2026 | Moyenne |
| ONNX Runtime | Export CamemBERT/RoBERTa en ONNX, -50% latence inférence | T3 2026 | Moyenne |
| MLflow / Weights & Biases | Tracking systématique des expériences ML | T2 2026 | Moyenne |
| Guardrails AI / NeMo Guardrails | Framework de safety pour systèmes IA de modération | T4 2026 | Moyenne |
| DPO/RLHF pour fake news | Fine-tuner un small LLM (Mistral 7B) avec paires vrai/faux | T4 2026 | Basse |
| Détection multimodale (CLIP) | Analyser les images associées aux posts pour détecter la manipulation visuelle | T4 2026 | Basse |
| Grafana + Prometheus | Monitoring avancé des performances système et drift detection | T2 2026 | Basse |

### 3.4 Tendances 2026 en détection de désinformation

| Tendance | Description | Impact pour Thumalien |
|----------|-------------|----------------------|
| LLM Fact-Checking | Les LLMs (GPT-4o, Claude, Gemini) sont utilisés pour vérifier les claims en les comparant à des sources fiables | Possibilité d'ajouter une couche de vérification factuelle au-dessus de la détection stylistique |
| Multimodal Misinformation | La désinformation s'appuie de plus en plus sur des images générées par IA (deepfakes, infographies truquées) | Extension future vers l'analyse d'images avec CLIP ou BLIP-2 |
| Explicabilité réglementaire | L'AI Act impose la transparence des systèmes IA à risque — SHAP devient un standard industriel | Notre intégration SHAP anticipe cette obligation |
| Federated Learning | Entraîner des modèles sans centraliser les données (privacy-preserving) | Pertinent si Thumalien est déployé chez plusieurs clients |
| Synthetic Data Augmentation | Génération de données synthétiques pour équilibrer les datasets (déjà utilisé en V4/V5) | Approfondir avec des LLMs pour générer des faux plus réalistes |

---

## 4. Processus de veille

### 4.1 Workflow

```
1. COLLECTE
   Sources : arXiv, Hugging Face, GitHub, CNIL, blogs
   Outils : RSS, newsletters, alertes Google Scholar
   Frequence : hebdomadaire

2. ANALYSE
   Pertinence pour Thumalien ? (oui/non)
   Impact estime (performance, cout, conformite)
   Effort d'integration

3. DECISION
   Adopter : integration dans le sprint suivant
   Reporter : ajouter a la roadmap
   Rejeter : documenter la justification

4. INTEGRATION
   Branche Git dediee
   Tests de non-regression
   Documentation mise a jour
```

### 4.2 Exemples concrets de veille appliquée

| Date | Découverte | Action | Résultat |
|------|-----------|--------|----------|
| Jan 2026 | PyTorch supporte Apple Silicon nativement | Migration TensorFlow -> PyTorch | Modèle émotions fonctionnel |
| Fév 2026 | Publication sur le débiaisage des datasets de presse | Implémentation BODY_AGENCY_TERMS | Réduction du biais Reuters |
| Mars 2026 | CamemBERT fine-tuné sur données courtes (blog HF) | Fine-tuning sur nos données FR | F1 0.957 sur ultra-courts |
| Avril 2026 | RoBERTa performant sur tweets EN (papier ACL) | Fine-tuning sur données EN | F1 0.874 sur ultra-courts |
| Avril 2026 | Technique de stacking pour NLP hybride | Pipeline hybride V5+CamemBERT | F1 FR +0.52% |
| Avril 2026 | SHAP TreeExplainer pour GradientBoosting (papier Lundberg 2017) | Intégration dans dashboard V7 | Explicabilité locale et globale des prédictions |
| Avril 2026 | Meta-learner stacking pour combiner modèles hétérogènes (technique Kaggle) | Architecture V7 hybride V5+V6 | Réduction FP de 57 à 25 sur gold set |
| Mai 2026 | Captum 0.9 (Meta) : Layer Integrated Gradients pour transformers | Module `src/explainability/integrated_gradients.py` avec LIG sur CamemBERT | Attribution causale par token, axiome de Completeness vérifié |
| Mai 2026 | Bug Captum/MPS sur Apple Silicon (issue PyTorch #128043) | Workaround : bascule CPU automatique pour les attributions IG | Δ_convergence divisé par 3 (de +0.35 à +0.04) |
| Mai 2026 | ERASER benchmark (DeYoung et al., ACL 2020) | Implémentation FaithfulnessEvaluator (AOPC, Comprehensiveness@k, Sufficiency@k vs random) | Validation quantitative des explications : uplift +0.21 vs baseline aléatoire |
| Mai 2026 | Mudrakarta et al. (2018) sur les baselines IG | Stratégie de baseline auto (`<unk>` → `<pad>`) avec retry n_steps escalade | Baseline `<unk>` préserve [CLS]/[SEP] = meilleure convergence Captum |
| Mai 2026 | Recommandation Captum : expliquer la classe prédite, pas la classe théorique | Patch step_integrated_gradients pour passer `target_class=preds[i]` | FN id=9 passe de Δ=0.47 (rejet) à Δ=0.04 (✓ pratique) |
| Mai 2026 | Google Model Card framework (Mitchell et al. 2019, FAT*) | Création `docs/12_model_card.md` avec section dédiée XAI (audience par persona, validation, limites) | Document auditeur conforme aux attentes AI Act art. 13 |

---

## 5. Ressources de veille

### 5.1 Sources principales

- **arXiv** (cs.CL, cs.AI) : publications académiques pre-print
- **Hugging Face Blog** : nouveaux modèles, techniques, datasets
- **Papers With Code** : benchmarks et SOTA détection fake news
- **CNIL** : actualités RGPD, guides pratiques IA
- **GitHub Trending** : librairies émergentes Python/ML
- **Google Scholar Alerts** : "fake news detection", "misinformation NLP"

### 5.2 Conférences et événements suivis

| Conférence | Domaine | Période |
|-----------|---------|---------|
| ACL (Association for Computational Linguistics) | NLP | Juillet |
| EMNLP (Empirical Methods in NLP) | NLP | Octobre |
| NeurIPS | Machine Learning | Décembre |
| CNIL Open Data Day | Réglementation | Variable |
| PyData | Python Data Science | Variable |

---

*Document validé par l'équipe projet - Avril 2026*
*Référence : VEILLE-THUM-2026-001 - Version 1.1*
