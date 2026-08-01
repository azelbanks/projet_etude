# ThumaCheck — Social Media Intelligence & AI Monitor

> **Built by [Niamato Consulting](https://github.com/azelbanks/thumacheck) (Azélie Bernard & Sébastien Lazcanotegui) for [Thumalien](https://thumalien.com), a fact-checking and media monitoring company.**

![CI](https://github.com/azelbanks/thumacheck/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/Coverage-80%25-brightgreen?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-537%20passing-brightgreen?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.13+-blue?style=for-the-badge&logo=python)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker)
![MongoDB](https://img.shields.io/badge/MongoDB-Database-47A248?style=for-the-badge&logo=mongodb)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)


> **English summary below** — Detailed documentation in French follows.

### TL;DR (English)

ThumaCheck is a **real-time misinformation detection system** for Bluesky, built for Thumalien (fact-checking firm). It combines a V9 cascade NLP pipeline (CamemBERT FR F1: 0.957, RoBERTa EN F1: 0.874) with 7-class emotion analysis. The system processes **728 texts/second** at 1.5ms latency, includes **8 explainability mechanisms** (SHAP, attention, Integrated Gradients), and is backed by **537 tests at 80% coverage**. Architecture: FastAPI + Streamlit + MongoDB + Docker Compose.

**Quick start:** `git clone → cp .env.example .env → docker-compose up -d`

**Key commands:** `make test` · `make lint` · `make dashboard` · `make api`

---

## Project Description

**ThumaCheck** est une solution complète de surveillance et d'analyse des réseaux sociaux (Bluesky) en temps réel, développée par **Niamato Consulting** pour le compte de **Thumalien**. Le projet intègre un pipeline Data Engineering complet et deux modèles d'Intelligence Artificielle pour qualifier l'information.

L'objectif est de détecter les potentiels signaux faibles, les **Fake News** et d'analyser l'**ambiance émotionnelle** des discussions en ligne.

### Key Features
* **Collecte en temps réel :** Ingestion continue des posts Bluesky via l'API AT Protocol.
* **Détection de Fake News (V9) :** Pipeline cascade 2 étapes : filtre fait/opinion puis analyse V8 (meta-learner V5+V6+CamemBERT) + RoBERTa EN cascade (60/40 blend textes courts). Bilingue FR/EN, 17 features linguistiques (dont emoji) + 28 features stylistiques.
* **Analyse Émotionnelle (Deep Learning) :** Réseau de neurones MLP (PyTorch) classifiant les textes selon 7 émotions (Colère, Dégoût, Joie, Neutre, Peur, Surprise, Tristesse).
* **Modèles avancés :** CamemBERT (FR, F1 0.957) et RoBERTa (EN, F1 0.874) fine-tunés pour les textes ultra-courts type réseaux sociaux.
* **Explicabilité IA (XAI) complète :** Pipeline 8 mécanismes dans `src/explainability/` couvrant les 4 niveaux de l'IA explicable :
    * `explain_prediction()` — coefficients LogReg × TF-IDF par mot (exposé via API `/explain`)
    * SHAP global (beeswarm + dependence) sur V6
    * SHAP par instance sur V6
    * SHAP sur émotions (KernelExplainer sur MLP 7 classes)
    * Attention CamemBERT (CLS dernière couche + heatmap par couche)
    * Layer Integrated Gradients (Captum) avec axiome de Completeness vérifié
    * Décomposition exacte du méta-learner V8 (β·x) intégrée au dashboard
    * Validation faithfulness (AOPC, Comprehensiveness@k, Sufficiency@k vs random) — **uplift +0.21** sur le gold set
    * Model Card formelle (`docs/12_model_card.md`) avec section dédiée XAI
    * Reproductible en 1 commande : `python scripts/run_xai_pipeline.py`
* **Dashboard Interactif :** 5 pages Streamlit (Dashboard, Analyse IA, Explorateur, Performance, À propos). Glossaire pédagogique intégré, zone d'incertitude visuelle, distribution complète des 7 émotions.
* **API REST :** FastAPI avec endpoints `/predict`, `/explain` (XAI mot-par-mot), `/health`, `/energy`. Rate limiting (60 req/min), monitoring énergétique continu (CodeCarbon).
* **Scalabilité :** Prototype Kafka consumer pour architecture événementielle (batch processing, métriques, topic de sortie).
* **Green IT :** Monitoring de l'empreinte carbone des calculs IA via CodeCarbon (entraînement + API en temps réel).
* **Tests :** 537 tests unitaires et d'intégration (pytest, 80% couverture), benchmark latence automatisé.

### Key Metrics (V9)
* **537 845 posts** collectés depuis décembre 2025 (collecte finalisée — Juil 2026)
* **197 782 textes** d'entraînement (7 datasets, FR+EN)
* **F1-score V5** : 0.913 (CV), seuil de décision : 0.44
* **V9 Cascade** : faux positifs réduits de -67% (Fisher p=0.0005)
* **CamemBERT FR** : F1 0.957 sur textes ultra-courts
* **RoBERTa EN V2** : F1 0.874 sur textes ultra-courts (+4.3% vs V1)
* **Latence** : 1.5 ms/texte (728 textes/sec)
* **68,9%** des posts Bluesky classés fiables
* **Empreinte CO2** : 8,86 g (total entraînement, CodeCarbon mesuré — réf. ADEME 2024 : 52 m en voiture essence)

---

## Technical Architecture


### Baselines & Comparisons

| Model | F1 (FR short) | F1 (EN short) | Note |
|-------|:---:|:---:|------|
| Majority class | 0.50 | 0.50 | Naive baseline |
| V1 TF-IDF (biased) | 0.996* | — | *Reuters attribution leak |
| V2 TF-IDF (debiased) | 0.89 | — | Bias removed |
| **V5 Expert** | **0.904** | **0.774** | Production pipeline |
| **CamemBERT V2** | **0.957** | — | Fine-tuned on short texts |
| **RoBERTa EN V2** | — | **0.874** | Fine-tuned on short texts |
| **V9 Cascade** | — | — | FP reduced by 67% (p=0.0005) |

Le projet repose sur une architecture micro-services conteneurisée avec Docker.

```mermaid
graph LR
    A[Bluesky Network] -->|AT Protocol| B(Container: Collector)
    B -->|JSON| C[(Container: MongoDB)]
    D[Container: Jupyter/AI] -->|Training & Inference| C
    E[Container: Streamlit App] -->|Read & Visualize| C
    style C fill:#47A248,stroke:#333,stroke-width:2px
    style E fill:#FF4B4B,stroke:#333,stroke-width:2px
```

---

## Project Structure

```
thumacheck/
├── dashboard/              # Application Streamlit (Dashboard V5, 5 pages)
├── data/training/          # Datasets d'entraînement (FR+EN, 6 sources)
├── docs/                   # Documentation complète du projet
│   ├── pdf/                # Documents PDF exportés
│   └── references/         # Cadre pédagogique et cahier des charges institutionnel
├── models/                 # Modèles entraînés (.joblib, .pt)
├── notebooks/
│   ├── exploration/        # Data audits, Bluesky exploration, dataset integration
│   ├── training/           # Model training V1-V9, CamemBERT, RoBERTa fine-tuning
│   └── analysis/           # Error analysis, benchmarks, statistical tests, carbon audit
├── src/
│   ├── api/                # API FastAPI (predict, explain, energy, health)
│   ├── app/                # Point d'entrée application
│   ├── collection/         # Collecteur Bluesky + qualité des données
│   ├── explainability/     # Pipeline XAI : SHAP global, attention, IG, decomposition meta-learner, faithfulness
│   ├── monitoring/         # Monitoring hebdomadaire (drift detection, fairness audit)
│   ├── pipeline/           # Pipeline NLP expert + CamemBERT + RoBERTa + agrégations
│   └── scalability/        # Prototype Kafka consumer pour architecture événementielle
├── scripts/
│   └── run_xai_pipeline.py # Pipeline XAI complet en 1 commande (figures + INDEX.md + results.json)
├── docker-compose.yml
└── requirements.txt
```

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 00 | `00_Audit_Qualite_Donnees.ipynb` | Audit qualité des données collectées |
| 01 | `01_Exploration_Bluesky.ipynb` | Exploration initiale du réseau Bluesky |
| 02 | `02_Analyse_Emotions_MLP.ipynb` | Modèle MLP PyTorch — 7 émotions (early stopping + class weights) |
| 03 | `03_Mise_a_jour_Quotidienne.ipynb` | Pipeline de mise à jour quotidienne (incrémental MongoDB) |
| 04 | `04_Modele_Avance_RoBERTa.ipynb` | Prototype RoBERTa pour détection fake news |
| 05 | `05_Detection_Expert_Bilingue.ipynb` | Pipeline expert bilingue FR/EN (LogReg + TF-IDF) |
| 06 | `06_Documentation_Technique.ipynb` | Documentation technique du pipeline |
| 07 | `07_Analyse_Modele_GridSearch.ipynb` | GridSearch hyperparamètres (C, min_df, ngram) |
| 08 | `08_Integration_Datasets_V2.ipynb` | Intégration des 6 datasets V2 |
| 09 | `09_Analyse_Erreurs_Qualitative.py` | Analyse qualitative des erreurs sur 2000 textes |
| 10 | `10_Analyse_Modele_Par_Longueur.py` | Performance du modèle par longueur de texte |
| 11 | `11_Retraining_V3.py` | Réentraînement V3 (correction preprocessing) |
| 12 | `12_Retraining_V4.py` | Réentraînement V4 + CamemBERT FR |
| 13 | `13_FineTune_CamemBERT_FR.py` | Fine-tuning CamemBERT sur données FR |
| 14 | `14_Retraining_V5_Social.py` | V5 avec 10K textes sociaux FR synthétiques |
| 15 | `15_Seuil_Adaptatif.py` | Seuil adaptatif par longueur (non significatif) |
| 16 | `16_FineTune_CamemBERT_V2_Social.py` | CamemBERT V2 (F1 0.957 ultra-court) |
| 17 | `17_Pipeline_Hybride_Stacking.py` | Pipeline hybride stacking V5 + CamemBERT V2 |
| 18 | `18_FineTune_RoBERTa_EN.py` | RoBERTa EN V1 (F1 0.838) |
| 19 | `19_FineTune_RoBERTa_EN_V2.py` | RoBERTa EN V2 +10K synthétique (F1 0.874) |
| 20 | `20_Tests_Significativite_Bootstrap.py` | Tests de significativité bootstrap |
| 21 | `21_Gold_Test_Set_Evaluation.py` | Évaluation sur gold test set (ancien) |
| 22 | `22_Gold_Test_Set_Evaluation.py` | Évaluation pipeline V5 sur 200 posts annotés (F1 suspect=0.087) |
| 23 | `23_Style_Only_V6.py` | Modèle style-only V6 (GradientBoosting, 35 features, F1 suspect=0.103) |
| 24 | `24_Hybrid_Ensemble_V7_SHAP.py` | Ensemble hybride V5+V6 + SHAP explicabilité (F1 suspect=0.127) |
| 25 | `25_V8_Hybrid_Extended_CamemBERT.py` | V8 meta-learner V5+V6+CamemBERT (F1 suspect +28%) |
| 26 | `26_V5_Finetune_Bluesky.py` | Self-training sur Bluesky (échec documenté) |
| 27 | `27_Pipeline_2_Etapes.py` | V9 cascade fait/opinion (FP -67%, Fisher p=0.0005) |

---

## PDF Documentation

Tous les documents sont disponibles dans [`docs/pdf/`](docs/pdf/) :

| Document | Description |
|----------|-------------|
| [**Executive Summary**](docs/pdf/00_executive_summary.pdf) | **Synthèse 1 page : problème, solution, KPI, livrables, impact** |
| [Cahier des charges techniques](docs/pdf/01_cahier_des_charges_techniques.pdf) | Spécifications techniques détaillées du projet |
| [Conformité RGPD & AI Act](docs/pdf/02_conformite_RGPD_AI_Act.pdf) | Analyse de conformité réglementaire |
| [Méthodologie projet](docs/pdf/03_methodologie_projet.pdf) | Méthodologie et organisation du projet |
| [Revue & challenge équipe](docs/pdf/04_revue_challenge_equipe.pdf) | Revue critique et retours d'équipe |
| [Analyse erreurs qualitative](docs/pdf/05_analyse_erreurs_qualitative.pdf) | Analyse qualitative des erreurs du modèle |
| [Analyse par longueur de texte](docs/pdf/06_analyse_modele_par_longueur.pdf) | Performance du modèle selon la longueur |
| [Évolution des modèles V1→V5](docs/pdf/07_evolution_modeles_comparatif.pdf) | Comparatif de toutes les versions du modèle |
| [Planification & Gantt](docs/pdf/08_planification_gantt.pdf) | WBS, Gantt, dépendances, jalons et calendrier |
| [PRA/PCA](docs/pdf/09_PRA_PCA.pdf) | Plan de Reprise et Continuité d'Activité |
| [Veille technologique](docs/pdf/10_veille_technologique.pdf) | Politique de veille technique et réglementaire |
| [Accessibilité & handicap](docs/pdf/11_accessibilite_handicap.pdf) | Mesures d'accessibilité du système |
| [Rapport de projet](docs/pdf/rapport_projet_thumalien.pdf) | Rapport complet du projet ThumaCheck (pour Thumalien) |
| [Guide utilisateur](docs/pdf/guide_utilisateur.pdf) | Guide d'utilisation du système |
| [Rôles et compétences](docs/pdf/roles_et_competences_projet.pdf) | Distribution des rôles et compétences |
| [Rendu individuel Azelie](docs/pdf/rendu_individuel_azelie_bernard.pdf) | Bilan personnel et compétences |
| [Rendu individuel Sebastien](docs/pdf/rendu_individuel_sebastien_lazcanotegui.pdf) | Bilan personnel et compétences |

---

## Version History

| Version | Date | F1 global | F1 FR court | F1 EN court | Innovation clé |
|---------|------|-----------|-------------|-------------|----------------|
| V1.0 | Dec 2025 | 0.996 (biaisé) | N/A | N/A | Baseline TF-IDF EN (biais Reuters) |
| V1.5 | Jan 2026 | 0.986 | N/A | N/A | Bilingue + débiaisage Reuters + 12 features linguistiques |
| V2.0 | Fev 2026 | 0.897 | 0.650 | 0.763 | +3 datasets sociaux, seuil calibré 0.44, 73.4% Bluesky fiables |
| V3.0 | Mars 2026 | 0.900 | 0.650 | 0.763 | Bug fix features linguistiques (5/12 étaient nulles) |
| V4.0 | Avril 2026 | 0.905 | 0.860 | 0.752 | Augmentation FR court (+32% F1), +3 features, 187K textes |
| CamemBERT V1 | Avril 2026 | 0.950 (FR) | 0.901 | N/A | Transformer FR fine-tuné, test 3/6 |
| V5.0 | Avril 2026 | 0.913 | 0.904 | 0.774 | +10K FR social synthétique, test 12/12, 197K textes |
| CamemBERT V2 | Avril 2026 | 0.966 (FR) | 0.957 | N/A | +10K FR social, test 9/10 (+6.2% ultra-court) |
| Hybride P1 | Avril 2026 | 0.916 | 0.909 | 0.773 | Stacking V5 + CamemBERT V2, F1 FR +0.52% |
| RoBERTa EN V1 | Avril 2026 | 0.940 (EN) | N/A | 0.838 | Transformer EN fine-tuné, test 6/10 |
| RoBERTa EN V2 | Avril 2026 | 0.944 (EN) | N/A | 0.874 | +10K EN social, test 16/18 (+4.3% ultra-court) |
| V6 Style-Only | Avril 2026 | 0.830 | N/A | N/A | GradientBoosting 35 features style, topic-agnostic, F1 suspect gold +18% |
| V7 Hybride | Avril 2026 | N/A | N/A | N/A | Ensemble V5+V6, meta-learner LOO, F1 suspect gold 0.127 (+46% vs V5), SHAP |
| V8 Meta | Avril 2026 | N/A | N/A | N/A | Meta-learner V5+V6+CamemBERT, F1 suspect gold 0.163 (+28% vs V7) |
| **V9 Cascade** | **Mai 2026** | **N/A** | **N/A** | **N/A** | **Pipeline 2 étapes fait/opinion, FP -67%, Fisher p=0.0005** |

---

## Installation & Usage

```bash
# Cloner le projet
git clone https://github.com/azelbanks/thumacheck.git
cd thumacheck

# Lancer avec Docker Compose
docker-compose up -d

# Ou installation locale
pip install -r requirements.txt
```

### Prérequis
- Python 3.13+
- Docker & Docker Compose
- MongoDB
- GPU recommandé pour le fine-tuning des modèles Transformer

---

## Tests

```bash
# Lancer tous les tests
python3 -m pytest tests/ -v

# Avec rapport de couverture
python3 -m pytest tests/ --cov=src --cov=dashboard --cov-report=term-missing

# Benchmark de latence seul
python3 -m pytest tests/test_benchmark_latence.py -v -s
```

| Module testé | Tests | Couverture |
|-------------|:-----:|:----------:|
| Pipeline NLP (features, détecteur) | 53 | 81% |
| CamemBERT (architecture, dataset) | 13 | 97% |
| Dashboard (logique métier, helpers) | 42 | 73% |
| Collecteur Bluesky (validation, langue, indexes) | 28 | 92% |
| MongoDB (agrégations, requêtes) | 24 | 96% |
| Monitoring (scoring, rapports, main) | 17 | 100% |
| Explicabilité (SHAP, IG, attention, meta, faithfulness) | 29 | 79% |
| Qualité données | 10 | 92% |
| API FastAPI | 11 | 91% |
| Intégration pipeline | 11 | — |
| Benchmark latence | 3 | — |
| Sécurité / validation entrées | 7 | — |
| Fairness audit | 5 | — |
| Kafka consumer | 3 | — |
| **Total** | **537** | **80%** |

---

## Green IT

L'empreinte carbone de l'ensemble des entraînements est suivie via **CodeCarbon** :
- **Total CO2** : 8.86 g (6 sessions d'entraînement V1-V9 + CamemBERT + RoBERTa)
- Equivalent à moins d'une recherche Google (~7 g). Le choix de modèles frugaux (LogReg + fine-tuning court) limite l'empreinte

---


## Design Decisions

### Why TF-IDF + LogReg instead of transformers only?

CamemBERT and RoBERTa excel at contextual understanding but are expensive at inference. The TF-IDF+LogReg baseline runs in **0.02ms/text** versus **15ms/text** for a transformer. For high-throughput monitoring (728 texts/sec target), the ensemble approach (lightweight pipeline first, transformers for edge cases via cascade) maximizes throughput without sacrificing accuracy.

### Why a meta-learner (V8) instead of simple voting?

Simple majority voting treats all models equally. The V8 meta-learner learns **optimal weights** for each model based on their complementary strengths: V5 excels on formal text, V6 catches stylistic anomalies, CamemBERT handles short ambiguous posts. The stacking approach captures these synergies, yielding F1 0.94 vs 0.92 for simple voting.

### Why a 2-stage cascade (V9) instead of a single classifier?

The single-pass pipeline had a high false positive rate on opinion pieces. The V9 cascade adds a **fact/opinion gate** as a first stage: only factual claims proceed to fake news analysis. This reduced false positives by 67% (Fisher exact test, p=0.0005) without degrading recall.

### Why DuckDB for analytics instead of PostgreSQL?

This is a single-machine analytical workload (OLAP), not a multi-user transactional system (OLTP). DuckDB runs embedded with zero configuration, processes columnar data 10-100x faster than PostgreSQL for analytical queries, and requires no server deployment.

## Why MLP for Emotion Analysis?

The emotion classifier uses a lightweight MLP (Multi-Layer Perceptron) rather than a pre-trained transformer for three deliberate reasons:

1. **Latency constraint**: The MLP processes emotions in <0.5ms/text, keeping the total pipeline under 1.5ms. A transformer would add 10-50ms per inference, breaking real-time requirements.
2. **Task simplicity**: 7-class emotion classification on pre-extracted TF-IDF features doesn't benefit from contextual embeddings the way fake news detection does. The MLP achieves sufficient accuracy for the use case.
3. **Carbon footprint**: Fine-tuning a second transformer model would have multiplied the CO2 cost. The MLP trains in seconds vs. hours.

The fake news pipeline uses CamemBERT/RoBERTa where contextual understanding is critical — the architecture matches the right model to the right task.

---


---

## Visual Results

### SHAP Feature Importance (V6 Pipeline)

<p align="center">
  <img src="docs/figures/xai/shap_beeswarm_v6.png" width="700" alt="SHAP Beeswarm — Global feature importance" />
</p>

### CamemBERT Attention Heatmap

<p align="center">
  <img src="docs/figures/xai/camembert_attention_fp_demo.png" width="700" alt="CamemBERT attention on a false positive" />
</p>

### Integrated Gradients (Word-level Attribution)

<p align="center">
  <img src="docs/figures/xai/ig_suspect_demo.png" width="700" alt="Integrated Gradients attribution" />
</p>

### Meta-Learner V8 Decomposition

<p align="center">
  <img src="docs/figures/xai/meta_decomposition_v8_example.png" width="700" alt="V8 meta-learner decision decomposition" />
</p>

### Model Performance

<p align="center">
  <img src="docs/confusion_matrices_v2.png" width="600" alt="Confusion matrices" />
  <img src="docs/learning_curves.png" width="600" alt="Learning curves" />
</p>

## Deployment

### Local (development)
```bash
docker-compose up -d              # all services
docker-compose up -d mongodb collector dashboard  # minimal
```

### Production
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

Adds resource limits, restart policies, and log rotation. See `docker-compose.prod.yml`.

### Cloud deployment (architecture)

```
┌─────────────────────────────────────────────────┐
│                   Cloud Provider                │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Cloud Run│  │ Cloud Run│  │  Streamlit   │  │
│  │  (API)   │  │(Collector)│  │    Cloud     │  │
│  │ :8000    │  │          │  │   :8501      │  │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  │
│       │              │               │          │
│       └──────────┬───┘───────────────┘          │
│                  │                              │
│         ┌────────▼────────┐                     │
│         │  MongoDB Atlas  │                     │
│         │  (managed DB)   │                     │
│         └─────────────────┘                     │
└─────────────────────────────────────────────────┘
```

The architecture is decoupled — each service can be deployed independently:
- **API** → Cloud Run / Azure Container Apps / Railway
- **Collector** → Cloud Run Job (scheduled) or always-on container
- **Dashboard** → Streamlit Community Cloud (free) or container
- **Database** → MongoDB Atlas (free tier available)

No code changes needed — just swap environment variables.

---

## My Contributions (Azélie Bernard)

This project was developed as a two-person team. Here is my specific scope:

- **Full NLP pipeline architecture** (V1→V9): feature engineering (45+ linguistic/stylistic features), model selection, cascade design, meta-learner ensemble
- **CamemBERT fine-tuning** (FR): dataset curation, training loop, evaluation on ultra-short texts (F1: 0.957)
- **Explainability pipeline**: SHAP integration, attention heatmaps, Layer Integrated Gradients, faithfulness validation (AOPC/Comprehensiveness)
- **API design & implementation**: FastAPI endpoints, rate limiting, energy monitoring
- **Testing strategy**: 537 tests architecture, pytest fixtures, coverage configuration
- **Dashboard logic**: Streamlit pages (Dashboard, AI Analysis, Explorer, Performance), uncertainty visualization
- **Data engineering**: MongoDB aggregation pipelines, data quality controls, DuckDB integration
- **Documentation**: Technical specifications, GDPR/AI Act compliance analysis, model cards
- **Green IT**: CodeCarbon integration and carbon footprint tracking

Sébastien Lazcanotegui contributed to: RoBERTa EN fine-tuning, Bluesky data collection infrastructure, Kafka scalability prototype, and deployment configuration.
