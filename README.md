# Thumalien — Social Media Intelligence & AI Monitor

![Python](https://img.shields.io/badge/Python-3.13+-blue?style=for-the-badge&logo=python)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker)
![MongoDB](https://img.shields.io/badge/MongoDB-Database-47A248?style=for-the-badge&logo=mongodb)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)

## Description du Projet

**Thumalien** est une solution complète de surveillance et d'analyse des réseaux sociaux (Bluesky) en temps réel. Le projet intègre un pipeline Data Engineering complet et deux modèles d'Intelligence Artificielle pour qualifier l'information.

L'objectif est de détecter les potentiels signaux faibles, les **Fake News** et d'analyser l'**ambiance émotionnelle** des discussions en ligne.

### Fonctionnalités Clés
* **Collecte en temps réel :** Ingestion continue des posts Bluesky via l'API AT Protocol.
* **Détection de Fake News :** Pipeline NLP bilingue FR/EN (Régression Logistique + TF-IDF + 12 features linguistiques) pour évaluer la crédibilité (Score 0 à 1).
* **Analyse Émotionnelle (Deep Learning) :** Réseau de neurones MLP (PyTorch) classifiant les textes selon 7 émotions (Colère, Dégoût, Joie, Neutre, Peur, Surprise, Tristesse).
* **Modèles avancés :** CamemBERT (FR, F1 0.957) et RoBERTa (EN, F1 0.874) fine-tunés pour les textes ultra-courts type réseaux sociaux.
* **Pipeline hybride :** Stacking V5 + CamemBERT V2 pour une couverture bilingue optimale.
* **Dashboard Interactif :** Visualisation des données et KPI en temps réel via Streamlit.
* **Green IT :** Monitoring de l'empreinte carbone des calculs IA via CodeCarbon.

### Métriques clés (V2)
* **188 553 posts** collectés depuis décembre 2025
* **145 703 textes** d'entraînement (6 datasets, FR+EN)
* **F1-score** : 0.90 (holdout pipeline expert), seuil de décision : 0.44
* **CamemBERT FR** : F1 0.957 sur textes ultra-courts
* **RoBERTa EN V2** : F1 0.874 sur textes ultra-courts (+8.2% vs V1)
* **73.4%** des posts Bluesky classés fiables (vs 23% en V1.5)
* **Empreinte CO2** : 0.55 g (total entraînement)

---

## Architecture Technique

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

## Structure du Projet

```
projet_etude/
├── dashboard/              # Application Streamlit (Dashboard V3)
├── data/training/          # Datasets d'entraînement (FR+EN, 6 sources)
├── docs/                   # Documentation complète du projet
│   └── pdf/                # Documents PDF exportés
├── models/                 # Modèles entraînés (.joblib, .pt)
├── notebooks/              # Notebooks d'exploration, entraînement et analyse
├── src/
│   ├── app/                # Point d'entrée application
│   ├── collection/         # Collecteur Bluesky + qualité des données
│   ├── monitoring/         # Monitoring hebdomadaire (drift detection)
│   └── pipeline/           # Pipeline NLP expert + CamemBERT + agrégations
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

---

## Documentation PDF

Tous les documents sont disponibles dans [`docs/pdf/`](docs/pdf/) :

| Document | Description |
|----------|-------------|
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
| [Rapport de projet](docs/pdf/rapport_projet_thumalien.pdf) | Rapport complet du projet Thumalien |
| [Guide utilisateur](docs/pdf/guide_utilisateur.pdf) | Guide d'utilisation du système |
| [Rôles et compétences](docs/pdf/roles_et_competences_projet.pdf) | Distribution des rôles et compétences |
| [Rendu individuel Azelie](docs/pdf/rendu_individuel_azelie_bernard.pdf) | Bilan personnel et compétences |
| [Rendu individuel Sebastien](docs/pdf/rendu_individuel_sebastien_lazcanotegui.pdf) | Bilan personnel et compétences |

---

## Historique des versions

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

---

## Installation & Lancement

```bash
# Cloner le projet
git clone https://github.com/azelbanks/projet_etude.git
cd projet_etude

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

## Green IT

L'empreinte carbone de l'ensemble des entraînements est suivie via **CodeCarbon** :
- **Total CO2** : 0.55 g
- Le choix d'un modèle LogReg (vs Transformer) réduit drastiquement la consommation énergétique
