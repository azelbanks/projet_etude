# Politique de Veille Technologique
## Projet Thumalien - Suivi des evolutions technologiques et reglementaires

**Reference** : VEILLE-THUM-2026-001
**Version** : 1.1
**Date** : Avril 2026

---

## 1. Objectif

La veille technologique permet de maintenir le projet Thumalien a l'etat de l'art en matiere de NLP, detection de desinformation et conformite reglementaire. Elle alimente les decisions d'evolution du pipeline et anticipe les ruptures technologiques.

---

## 2. Domaines de veille

### 2.1 Veille technique (NLP / IA)

| Sujet | Sources | Frequence | Responsable |
|-------|---------|-----------|-------------|
| Modeles de langue (LLM, BERT, etc.) | Hugging Face blog, arXiv (cs.CL) | Hebdomadaire | Azelie |
| Detection de fake news (SOTA) | Papers With Code, ACL Anthology | Mensuelle | Azelie |
| Librairies Python (scikit-learn, PyTorch, transformers) | GitHub releases, changelogs | Mensuelle | Azelie |
| Techniques d'augmentation de donnees | arXiv, blogs ML | Mensuelle | Sebastien |
| Evaluation et metriques NLP | Conferences (ACL, EMNLP, NeurIPS) | Trimestrielle | Azelie |

### 2.2 Veille reglementaire

| Sujet | Sources | Frequence | Responsable |
|-------|---------|-----------|-------------|
| RGPD (evolutions, jurisprudence) | CNIL, EUR-Lex | Trimestrielle | Azelie |
| AI Act (mise en application) | Commission Europeenne, CNIL | Trimestrielle | Azelie |
| Ethique de l'IA | IEEE, ACM, rapports OCDE | Semestrielle | Sebastien |

### 2.3 Veille ecosysteme

| Sujet | Sources | Frequence | Responsable |
|-------|---------|-----------|-------------|
| API AT Protocol (Bluesky) | GitHub bluesky-social, docs AT Proto | Hebdomadaire | Azelie |
| Concurrents (fact-checking, outils similaires) | ProductHunt, TechCrunch | Mensuelle | Sebastien |
| Frameworks dashboard (Streamlit, Gradio) | GitHub releases | Mensuelle | Azelie |
| Docker / MongoDB (mises a jour securite) | Docker Hub, MongoDB blog | Mensuelle | Azelie |

---

## 3. Technologies identifiees et evaluees

### 3.1 Technologies adoptees (integrees au projet)

| Technologie | Date evaluation | Decision | Justification |
|------------|:--------------:|:--------:|---------------|
| CamemBERT (FR) | Avril 2026 | Adopte | F1 0.957 sur textes courts FR, superieur au TF-IDF |
| RoBERTa (EN) | Avril 2026 | Adopte | F1 0.874 sur textes courts EN, complementaire au pipeline |
| Pipeline hybride (stacking) | Avril 2026 | Adopte | Combine les forces TF-IDF (robustesse) et Transformer (precision courts) |
| CodeCarbon | Jan 2026 | Adopte | Monitoring CO2 transparent, zero impact performance |
| fpdf2 | Avril 2026 | Adopte | Generation PDF depuis Python, pas de dependance externe |
| Plotly | Mars 2026 | Adopte | Graphiques interactifs pour le dashboard, meilleur que matplotlib |

### 3.2 Technologies evaluees et non retenues

| Technologie | Date evaluation | Decision | Justification |
|------------|:--------------:|:--------:|---------------|
| TensorFlow (emotions) | Jan 2026 | Rejete | Incompatible Apple Silicon M4, remplace par PyTorch |
| GPT-4 pour detection | Fev 2026 | Non retenu | Cout prohibitif en inference, latence trop elevee, dependance API |
| Sentence-Transformers | Mars 2026 | Reporte | Prometteur mais V5 + Transformers couvrent le besoin actuel |
| FastAPI (serving) | Mars 2026 | Reporte | Pas encore necessaire, inference batch suffisante |
| Elasticsearch | Dec 2025 | Non retenu | MongoDB suffisant pour nos volumes, complexite inutile |

### 3.3 Technologies a surveiller (roadmap)

| Technologie | Interet pour Thumalien | Horizon | Priorite |
|------------|------------------------|:-------:|:--------:|
| RAG (Retrieval-Augmented Generation) | Cross-checker les claims avec une base factuelle (Wikipedia, Google Fact Check Tools API) | T3 2026 | **Haute** |
| LLM-as-Judge (GPT-4o, Claude) | Scoring zero-shot de la credibilite, benchmark contre V7 | T3 2026 | **Haute** |
| Active Learning (modAL) | Cibler l'annotation sur les posts a fort disagreement V5/V6 | T2 2026 | **Haute** |
| Conformal Prediction | Intervalles de confiance calibres au lieu de scores bruts | T3 2026 | Moyenne |
| ONNX Runtime | Export CamemBERT/RoBERTa en ONNX, -50% latence inference | T3 2026 | Moyenne |
| MLflow / Weights & Biases | Tracking systematique des experiences ML | T2 2026 | Moyenne |
| Guardrails AI / NeMo Guardrails | Framework de safety pour systemes IA de moderation | T4 2026 | Moyenne |
| DPO/RLHF pour fake news | Fine-tuner un small LLM (Mistral 7B) avec paires vrai/faux | T4 2026 | Basse |
| Detection multimodale (CLIP) | Analyser les images associees aux posts pour detecter la manipulation visuelle | T4 2026 | Basse |
| Grafana + Prometheus | Monitoring avance des performances systeme et drift detection | T2 2026 | Basse |

### 3.4 Tendances 2026 en detection de desinformation

| Tendance | Description | Impact pour Thumalien |
|----------|-------------|----------------------|
| LLM Fact-Checking | Les LLMs (GPT-4o, Claude, Gemini) sont utilises pour verifier les claims en les comparant a des sources fiables | Possibilite d'ajouter une couche de verification factuelle au-dessus de la detection stylistique |
| Multimodal Misinformation | La desinformation s'appuie de plus en plus sur des images generees par IA (deepfakes, infographies truquees) | Extension future vers l'analyse d'images avec CLIP ou BLIP-2 |
| Explicabilite reglementaire | L'AI Act impose la transparence des systemes IA a risque — SHAP devient un standard industriel | Notre integration SHAP anticipe cette obligation |
| Federated Learning | Entrainer des modeles sans centraliser les donnees (privacy-preserving) | Pertinent si Thumalien est deploye chez plusieurs clients |
| Synthetic Data Augmentation | Generation de donnees synthetiques pour equilibrer les datasets (deja utilise en V4/V5) | Approfondir avec des LLMs pour generer des faux plus realistes |

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

### 4.2 Exemples concrets de veille appliquee

| Date | Decouverte | Action | Resultat |
|------|-----------|--------|----------|
| Jan 2026 | PyTorch supporte Apple Silicon nativement | Migration TensorFlow -> PyTorch | Modele emotions fonctionnel |
| Fev 2026 | Publication sur le debiaisage des datasets de presse | Implementation BODY_AGENCY_TERMS | Reduction du biais Reuters |
| Mars 2026 | CamemBERT fine-tune sur donnees courtes (blog HF) | Fine-tuning sur nos donnees FR | F1 0.957 sur ultra-courts |
| Avril 2026 | RoBERTa performant sur tweets EN (papier ACL) | Fine-tuning sur donnees EN | F1 0.874 sur ultra-courts |
| Avril 2026 | Technique de stacking pour NLP hybride | Pipeline hybride V5+CamemBERT | F1 FR +0.52% |
| Avril 2026 | SHAP TreeExplainer pour GradientBoosting (papier Lundberg 2017) | Integration dans dashboard V7 | Explicabilite locale et globale des predictions |
| Avril 2026 | Meta-learner stacking pour combiner modeles heterogenes (technique Kaggle) | Architecture V7 hybride V5+V6 | Reduction FP de 57 a 25 sur gold set |

---

## 5. Ressources de veille

### 5.1 Sources principales

- **arXiv** (cs.CL, cs.AI) : publications academiques pre-print
- **Hugging Face Blog** : nouveaux modeles, techniques, datasets
- **Papers With Code** : benchmarks et SOTA detection fake news
- **CNIL** : actualites RGPD, guides pratiques IA
- **GitHub Trending** : librairies emergentes Python/ML
- **Google Scholar Alerts** : "fake news detection", "misinformation NLP"

### 5.2 Conferences et evenements suivis

| Conference | Domaine | Periode |
|-----------|---------|---------|
| ACL (Association for Computational Linguistics) | NLP | Juillet |
| EMNLP (Empirical Methods in NLP) | NLP | Octobre |
| NeurIPS | Machine Learning | Decembre |
| CNIL Open Data Day | Reglementation | Variable |
| PyData | Python Data Science | Variable |

---

*Document valide par l'equipe projet - Avril 2026*
*Reference : VEILLE-THUM-2026-001 - Version 1.1*
