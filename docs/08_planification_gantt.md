# Planification detaillee du projet Thumalien
## Diagramme de Gantt, WBS, dependances et jalons

**Reference** : PLAN-THUM-2026-001
**Version** : 2.0
**Date** : Mai 2026
**Equipe** : Azelie Bernard (Lead technique), Sebastien Lazcanotegui (Consolidation ML)

---

## 1. Work Breakdown Structure (WBS)

### 1.1 Decomposition hierarchique

```
THUMALIEN - Projet Etude M1
|
|-- WP1 : Cadrage & Infrastructure (Dec 2025)
|   |-- T1.1 : Analyse des besoins et specification
|   |-- T1.2 : Choix technologiques (Python, MongoDB, Docker)
|   |-- T1.3 : Mise en place environnement Docker Compose
|   |-- T1.4 : Configuration MongoDB + volumes persistants
|   |-- T1.5 : Creation du depot Git et structure projet
|
|-- WP2 : Collecte de donnees (Dec 2025 - Continu)
|   |-- T2.1 : Developpement collecteur Bluesky (AT Protocol)
|   |-- T2.2 : Validation schema JSON + deduplication
|   |-- T2.3 : Audit qualite des donnees collectees
|   |-- T2.4 : Maintenance et monitoring continu
|
|-- WP3 : Modele d'emotions (Jan 2026)
|   |-- T3.1 : Exploration datasets emotions (GoEmotions, FR tweets)
|   |-- T3.2 : Developpement MLP PyTorch (7 classes)
|   |-- T3.3 : Early stopping + class weights
|   |-- T3.4 : Evaluation et validation
|
|-- WP4 : Pipeline NLP Expert (Jan - Avril 2026)
|   |-- T4.1 : Baseline V1.0 (TF-IDF + LogReg EN)
|   |-- T4.2 : Audit biais Reuters + debiaisage
|   |-- T4.3 : V1.5 bilingue + features linguistiques
|   |-- T4.4 : V2 integration datasets sociaux + seuil 0.44
|   |-- T4.5 : V3 correction preprocessing
|   |-- T4.6 : V4 augmentation FR court + 15 features
|   |-- T4.7 : V5 + 10K posts sociaux synthetiques
|   |-- T4.8 : GridSearch hyperparametres (C, min_df, ngram)
|
|-- WP5 : Modeles Transformer (Avril 2026)
|   |-- T5.1 : Fine-tuning CamemBERT V1 (FR)
|   |-- T5.2 : CamemBERT V2 + donnees sociales
|   |-- T5.3 : Fine-tuning RoBERTa EN V1
|   |-- T5.4 : RoBERTa EN V2 + 10K synthétique
|   |-- T5.5 : Pipeline hybride stacking V5 + CamemBERT V2
|
|-- WP6 : Dashboard & Visualisation (Mars 2026)
|   |-- T6.1 : Dashboard V1 (metriques basiques)
|   |-- T6.2 : Dashboard V2 (glassmorphism, 3 pages)
|   |-- T6.3 : Dashboard V3 (radar charts, live prediction)
|   |-- T6.4 : Integration weighted loss + securite
|
|-- WP7 : Documentation & Conformite (Fev - Avril 2026)
|   |-- T7.1 : Cahier des charges techniques
|   |-- T7.2 : Conformite RGPD & AI Act
|   |-- T7.3 : Rapport de projet
|   |-- T7.4 : Guide utilisateur
|   |-- T7.5 : Documentation technique (22 notebooks)
|   |-- T7.6 : Planification et gouvernance
|
|-- WP8 : Evaluation & Gold Test Set (Avril - Mai 2026)
|   |-- T8.1 : Gold set V1 (200 posts, kappa=0.808)
|   |-- T8.2 : Gold set V2 annotation (500 posts, 2 annotateurs, kappa=0.498)
|   |-- T8.3 : Evaluation systematique V5-V9 sur gold
|
|-- WP9 : Iterations avancees V6-V9 (Avril - Mai 2026)
|   |-- T9.1 : V6 Style-Only topic-agnostic (28 features, GBT)
|   |-- T9.2 : V7 Ensemble hybride V5+V6 + SHAP
|   |-- T9.3 : V8 Meta-learner V5+V6+CamemBERT
|   |-- T9.4 : Self-training Bluesky (echec documente)
|   |-- T9.5 : V9 Pipeline 2 etapes fait/opinion
|
|-- WP10 : Collecteur V3 & Reeequilibrage (Avril 2026)
|   |-- T10.1 : Reeequilibrage termes FR/EN (28 FR + 16 EN)
|   |-- T10.2 : Inference automatique emotions + V5
|   |-- T10.3 : Rate limiting & backoff progressif
|
|-- WP11 : Dashboard V4-V5 (Avril - Mai 2026)
|   |-- T11.1 : Dashboard V4 (V9 + SHAP + Explorateur)
|   |-- T11.2 : Dashboard V5 (5 pages, accents, Performance)
|
|-- WP12 : Tests & Qualite (Mai 2026)
|   |-- T12.1 : Tests unitaires (107 tests, 26% coverage)
|   |-- T12.2 : Benchmark latence (1.5ms/texte)
|   |-- T12.3 : Tests d'integration pipeline
|
|-- WP13 : Video MVP (Mai 2026)
|   |-- T13.1 : Script et storyboard
|   |-- T13.2 : Tournage demo live
|   |-- T13.3 : Montage et post-production
|   |-- T13.4 : Livraison finale
```

---

## 2. Diagramme de Gantt

### 2.1 Planning macro (semaines)

```
Tache                          | Dec 25  | Jan 26  | Fev 26  | Mars 26 | Avril 26| Mai 26  |
                               | S1 S2 S3| S4 S5 S6| S7 S8 S9|S10S11S12|S13S14S15|S16S17S18|
-------------------------------|---------|---------|---------|---------|---------|---------|
WP1 Cadrage & Infra            |[========]         |         |         |         |         |
  T1.1 Analyse besoins         |[===]    |         |         |         |         |         |
  T1.2 Choix techno            |[===]    |         |         |         |         |         |
  T1.3 Docker Compose          |   [====]|         |         |         |         |         |
  T1.4 MongoDB config          |   [====]|         |         |         |         |         |
  T1.5 Depot Git               |[==]     |         |         |         |         |         |
                               |         |         |         |         |         |         |
WP2 Collecte donnees           |   [=====|=========|=========|=========|=========|========]|
  T2.1 Collecteur Bluesky      |   [====]|         |         |         |         |         |
  T2.2 Validation schema       |   [====]|         |         |         |         |         |
  T2.3 Audit qualite           |         |   [====]|         |         |         |         |
  T2.4 Monitoring continu      |      [==|=========|=========|=========|=========|========]|
                               |         |         |         |         |         |         |
WP3 Modele emotions            |         |[========]         |         |         |         |
  T3.1 Exploration datasets    |         |[===]    |         |         |         |         |
  T3.2 MLP PyTorch             |         |[========]         |         |         |         |
  T3.3 Early stopping          |         |      [==]         |         |         |         |
  T3.4 Evaluation              |         |      [==]         |         |         |         |
                               |         |         |         |         |         |         |
WP4 Pipeline NLP (V1-V5)       |[===]    |[========|=========|=========|========]|         |
  T4.1 Baseline V1.0           |[===]    |         |         |         |         |         |
  T4.2 Audit biais Reuters     |         |[===]    |         |         |         |         |
  T4.3 V1.5 bilingue           |         |      [==|===]     |         |         |         |
  T4.4 V2 datasets sociaux     |         |         |[===]    |         |         |         |
  T4.5 V3 correction preproc   |         |         |         |[===]    |         |         |
  T4.6 V4 augmentation FR      |         |         |         |         |[===]    |         |
  T4.7 V5 posts synthetiques   |         |         |         |         |   [==]  |         |
  T4.8 GridSearch              |         |         |   [==]  |         |         |         |
                               |         |         |         |         |         |         |
WP5 Modeles Transformer        |         |         |         |         |[========]|         |
  T5.1 CamemBERT V1            |         |         |         |         |[===]    |         |
  T5.2 CamemBERT V2            |         |         |         |         |   [==]  |         |
  T5.3 RoBERTa EN V1           |         |         |         |         |   [===] |         |
  T5.4 RoBERTa EN V2           |         |         |         |         |      [=]|         |
  T5.5 Pipeline hybride        |         |         |         |         |      [=]|         |
                               |         |         |         |         |         |         |
WP6 Dashboard (V1-V3)          |         |         |         |[========|===]     |         |
  T6.1 Dashboard V1            |         |         |         |[===]    |         |         |
  T6.2 Dashboard V2            |         |         |         |   [====]|         |         |
  T6.3 Dashboard V3            |         |         |         |         |[===]    |         |
  T6.4 Securite + weighted     |         |         |         |         |   [==]  |         |
                               |         |         |         |         |         |         |
WP7 Documentation              |         |         |[==]     |         |[========|========]|
  T7.1 CDC techniques          |         |         |[==]     |         |         |         |
  T7.2 RGPD & AI Act           |         |         |[==]     |         |         |         |
  T7.3 Rapport projet          |         |         |         |         |[========|========]|
  T7.4 Guide utilisateur       |         |         |         |         |   [====]|         |
  T7.5 Doc technique           |         |         |         |         |[========|========]|
  T7.6 Planification           |         |         |         |         |      [==|==]      |
                               |         |         |         |         |         |         |
WP8 Gold Test Set              |         |         |         |         |[========|===]     |
  T8.1 Gold set V1 (200)       |         |         |         |         |[===]    |         |
  T8.2 Gold set V2 (500)       |         |         |         |         |   [====]|         |
  T8.3 Eval V5-V9 sur gold     |         |         |         |         |      [==|==]      |
                               |         |         |         |         |         |         |
WP9 Iterations V6-V9           |         |         |         |         |   [=====|====]    |
  T9.1 V6 Style-Only           |         |         |         |         |   [===] |         |
  T9.2 V7 Ensemble + SHAP      |         |         |         |         |      [==]         |
  T9.3 V8 Meta-learner         |         |         |         |         |      [==]         |
  T9.4 Self-training (echec)    |         |         |         |         |      [=]|         |
  T9.5 V9 Cascade fait/opinion |         |         |         |         |         |[===]    |
                               |         |         |         |         |         |         |
WP10 Collecteur V3             |         |         |         |         |   [====]|         |
  T10.1 Reequilibrage FR/EN    |         |         |         |         |   [==]  |         |
  T10.2 Inference auto         |         |         |         |         |   [===] |         |
  T10.3 Rate limiting          |         |         |         |         |      [=]|         |
                               |         |         |         |         |         |         |
WP11 Dashboard V4-V5           |         |         |         |         |      [==|====]    |
  T11.1 Dashboard V4 (V9+SHAP) |         |         |         |         |      [==]         |
  T11.2 Dashboard V5 (5 pages) |         |         |         |         |         |[===]    |
                               |         |         |         |         |         |         |
WP12 Tests & Qualite           |         |         |         |         |         |[======] |
  T12.1 Tests unitaires (94)   |         |         |         |         |         |[===]    |
  T12.2 Benchmark latence      |         |         |         |         |         |[==]     |
  T12.3 Tests integration      |         |         |         |         |         |   [==]  |
                               |         |         |         |         |         |         |
WP13 Video MVP                 |         |         |         |         |         |   [====]|
  T13.1 Script/storyboard      |         |         |         |         |         |   [==]  |
  T13.2 Tournage               |         |         |         |         |         |      [=]|
  T13.3 Montage                |         |         |         |         |         |      [=]|
  T13.4 Livraison              |         |         |         |         |         |      [=]|
```

### 2.2 Legende

- `[===]` : Duree de la tache
- S1-S18 : Semaines (Dec 2025 Semaine 1 a Mai 2026 Semaine 18)

---

## 3. Calendrier formel avec dates cles

| Date | Jalon | Livrable | Responsable |
|------|-------|----------|-------------|
| 01/12/2025 | J0 - Lancement projet | Repo Git, Docker, MongoDB | Azelie |
| 15/12/2025 | J1 - Collecteur operationnel | collect_bluesky.py fonctionnel | Azelie |
| 20/12/2025 | J2 - Baseline V1.0 | LogReg EN, F1=0.99 (biaise) | Azelie |
| 10/01/2026 | J3 - Audit qualite | Biais Reuters identifie | Azelie |
| 20/01/2026 | J4 - Modele emotions | MLP PyTorch 7 emotions | Azelie |
| 05/02/2026 | J5 - Pipeline V1.5 | Bilingue, 12 features, F1=0.986 | Azelie |
| 15/02/2026 | J6 - GridSearch | C=5.0, min_df=5, ngram=(1,2) | Sebastien |
| 25/02/2026 | J7 - Pipeline V2 | +3 datasets sociaux, seuil=0.44 | Azelie |
| 10/03/2026 | J8 - Dashboard V2 | Glassmorphism, 3 pages | Azelie |
| 20/03/2026 | J9 - Pipeline V3 | Correction features linguistiques | Azelie |
| 05/04/2026 | J10 - Pipeline V4 | Augmentation FR court, 15 features | Azelie |
| 10/04/2026 | J11 - Pipeline V5 | +10K FR synthetique, F1=0.913 | Azelie |
| 12/04/2026 | J12 - CamemBERT V1/V2 | F1 FR ultra-court = 0.957 | Azelie |
| 15/04/2026 | J13 - RoBERTa EN V1/V2 | F1 EN ultra-court = 0.874 | Azelie |
| 18/04/2026 | J14 - Pipeline hybride | Stacking V5 + CamemBERT V2 | Azelie |
| 20/04/2026 | J15 - Consolidation ML | Debiaisage, hyperparametres | Sebastien |
| 20/04/2026 | J16 - V6 Style-Only | GradientBoosting 28 features, topic-agnostic | Azelie |
| 22/04/2026 | J17 - V7 Ensemble + SHAP | Meta-learner V5+V6, explicabilite SHAP | Azelie |
| 24/04/2026 | J18 - V8 Meta-learner | V5+V6+CamemBERT, F1 suspect +28% | Azelie |
| 25/04/2026 | J19 - Gold Test Set V2 | 500 posts, 2 annotateurs, kappa=0.498 | Azelie |
| 28/04/2026 | J20 - Collecteur V3 | Reeequilibrage FR/EN + inference auto | Azelie |
| 02/05/2026 | J21 - V9 Cascade fait/opinion | FP -67%, Fisher p=0.0005 | Azelie |
| 05/05/2026 | J22 - Dashboard V5 | 5 pages, accents FR, page Performance | Azelie |
| 08/05/2026 | J23 - Tests & Benchmark | 107 tests, 1.5ms/texte, 728 textes/sec | Azelie |
| 12/05/2026 | J24 - Documentation finale | Rapport, planification, rendus individuels | Azelie |
| 15/05/2026 | J25 - Video MVP | Video 15-20 min face-cam + dossier | Azelie + Sebastien |
| **19/05/2026** | **DEADLINE** | **Livraison finale — dossier + video + code** | **Equipe** |

---

## 4. Dependances entre taches

### 4.1 Graphe de dependances

```
T1.3 (Docker) ──> T1.4 (MongoDB) ──> T2.1 (Collecteur)
                                          |
T1.5 (Git) ──────────────────────────────>|
                                          |
                                          v
                                     T2.3 (Audit qualite)
                                          |
                              +-----------+-----------+
                              |                       |
                              v                       v
                         T4.1 (V1.0)             T3.1 (Emotions)
                              |                       |
                              v                       v
                         T4.2 (Audit biais)      T3.2 (MLP)
                              |                       |
                              v                       |
                         T4.3 (V1.5) <────────────────+
                              |
                              v
                         T4.4 (V2) ──> T4.8 (GridSearch)
                              |
                              v
                         T4.5 (V3)
                              |
                              v
                         T4.6 (V4) ──> T5.1 (CamemBERT V1)
                              |              |
                              v              v
                         T4.7 (V5) ──> T5.2 (CamemBERT V2)
                              |              |
                              v              v
                         T5.3 (RoBERTa V1)  T5.5 (Hybride)
                              |
                              v
                         T5.4 (RoBERTa V2)

T4.4 (V2) ──> T6.1 (Dashboard V1) ──> T6.2 (V2) ──> T6.3 (V3)

T4.7 (V5) ──> T7.3 (Rapport) ──> T8.1 (Script video)
```

### 4.2 Matrice de dependances

| Tache | Depend de | Type | Critique |
|-------|-----------|------|----------|
| T1.4 MongoDB | T1.3 Docker | Fin-Debut | Oui |
| T2.1 Collecteur | T1.4 MongoDB | Fin-Debut | Oui |
| T4.1 Baseline | T2.1 Collecteur | Fin-Debut | Oui |
| T4.2 Audit biais | T4.1 Baseline | Fin-Debut | Oui |
| T4.3 V1.5 | T4.2 Audit + T3.4 Emotions | Fin-Debut | Oui |
| T4.4 V2 | T4.3 V1.5 | Fin-Debut | Oui |
| T4.5 V3 | T4.4 V2 | Fin-Debut | Oui |
| T4.6 V4 | T4.5 V3 | Fin-Debut | Oui |
| T4.7 V5 | T4.6 V4 | Fin-Debut | Oui |
| T5.1 CamemBERT | T4.6 V4 | Fin-Debut | Non |
| T5.3 RoBERTa | T4.7 V5 | Fin-Debut | Non |
| T5.5 Hybride | T5.2 CamemBERT V2 + T4.7 V5 | Fin-Debut | Non |
| T6.1 Dashboard | T4.4 V2 | Fin-Debut | Non |
| T7.3 Rapport | T4.7 V5 | Fin-Debut (contenu) | Non |
| T8.1 Gold set V1 | T4.7 V5 | Fin-Debut | Non |
| T8.3 Eval gold | T9.5 V9 + T8.2 Gold V2 | Fin-Debut | Non |
| T9.1 V6 Style | T4.7 V5 + T8.1 Gold V1 | Fin-Debut | Oui |
| T9.2 V7 Ensemble | T9.1 V6 | Fin-Debut | Oui |
| T9.3 V8 Meta | T9.2 V7 + T5.2 CamemBERT V2 | Fin-Debut | Oui |
| T9.5 V9 Cascade | T9.3 V8 + T8.2 Gold V2 | Fin-Debut | Oui |
| T11.1 Dashboard V4 | T9.3 V8 | Fin-Debut | Non |
| T11.2 Dashboard V5 | T11.1 Dashboard V4 + T9.5 V9 | Fin-Debut | Non |
| T12.1 Tests | T9.5 V9 + T11.2 Dashboard V5 | Fin-Debut | Non |
| T13.1 Video | T12.1 Tests + T7.3 Rapport | Fin-Debut | Oui |

### 4.3 Chemin critique

Le chemin critique du projet est :

```
T1.3 -> T1.4 -> T2.1 -> T4.1 -> T4.2 -> T4.3 -> T4.4 -> T4.5 -> T4.6 -> T4.7
  -> T9.1 (V6) -> T9.2 (V7) -> T9.3 (V8) -> T9.5 (V9) -> T7.3 (Rapport) -> T13.1 (Video)
```

Duree totale du chemin critique : **~24 semaines** (Dec 2025 - Mai 2026)
Deadline de livraison : **19 mai 2026**

Toute tache sur le chemin critique qui prend du retard retarde la livraison finale.

---

## 5. Repartition des responsabilites (RACI)

| Tache | Azelie Bernard | Sebastien Lazcanotegui |
|-------|:--------------:|:---------------------:|
| WP1 Infrastructure | R/A | I |
| WP2 Collecte | R/A | I |
| WP3 Emotions | R/A | I |
| WP4 Pipeline V1-V5 | R/A | C (GridSearch, debiaisage) |
| WP5 Transformers | R/A | I |
| WP6 Dashboard | R/A | I |
| WP7 Documentation | R/A | C (revue) |
| WP8 Gold Test Set | R/A | C (annotation) |
| WP9 Iterations V6-V9 | R/A | I |
| WP10 Collecteur V3 | R/A | I |
| WP11 Dashboard V4-V5 | R/A | I |
| WP12 Tests & Qualite | R/A | I |
| WP13 Video MVP | R | R |

**Legende RACI** : R = Responsable, A = Approbateur, C = Consulte, I = Informe

### 5.1 Charge de travail estimee

| Work Package | Azelie (heures) | Sebastien (heures) | Total |
|-------------|:----------------:|:-------------------:|:-----:|
| WP1 Cadrage & Infra | 20 | 0 | 20 |
| WP2 Collecte | 25 | 0 | 25 |
| WP3 Emotions | 30 | 0 | 30 |
| WP4 Pipeline NLP (V1-V5) | 80 | 10 | 90 |
| WP5 Transformers | 40 | 0 | 40 |
| WP6 Dashboard (V1-V3) | 20 | 0 | 20 |
| WP7 Documentation | 40 | 5 | 45 |
| WP8 Gold Test Set | 25 | 5 | 30 |
| WP9 Iterations V6-V9 | 50 | 0 | 50 |
| WP10 Collecteur V3 | 10 | 0 | 10 |
| WP11 Dashboard V4-V5 | 20 | 0 | 20 |
| WP12 Tests & Qualite | 15 | 0 | 15 |
| WP13 Video MVP | 10 | 10 | 20 |
| **Total** | **385** | **30** | **415** |

---

## 6. Outils de planification utilises

| Outil | Usage | Justification |
|-------|-------|---------------|
| **Git / GitHub** | Versioning du code et des documents | Tracabilite complete des modifications, historique des commits |
| **Docker Compose** | Orchestration des services | Reproductibilite de l'environnement, deploiement standardise |
| **Jupyter Notebooks** | Experimentation ML iterative | Documentation executable des experiences |
| **GitHub Issues** | Suivi des taches et bugs | Integration native avec Git, assignation, labels |
| **Markdown** | Documentation | Versionnable, lisible, convertible en PDF |
| **CodeCarbon** | Suivi empreinte carbone | Mesure automatique de la consommation energetique |

---

## 7. Indicateurs de suivi du planning

| KPI | Methode | Frequence | Seuil d'alerte |
|-----|---------|-----------|----------------|
| Avancement reel vs planifie | Comparaison jalons | Hebdomadaire | Retard > 1 semaine |
| Nombre de taches completees | Comptage Git commits | Hebdomadaire | < 3 commits/semaine |
| F1-score progression | Metriques ML | A chaque version | Regression > 2% |
| Volume donnees collectees | MongoDB count | Quotidien | < 500 posts/jour |
| Bugs ouverts | GitHub issues | Hebdomadaire | > 5 bugs critiques |

---

## 8. Gestion des risques lies au planning

| Risque | Probabilite | Impact | Mitigation |
|--------|:-----------:|:------:|------------|
| Retard sur le pipeline NLP (chemin critique) | Moyenne | Eleve | Parallelisation des taches non-critiques (dashboard, docs) |
| Indisponibilite API Bluesky | Faible | Moyen | Cache local, retry automatique, donnees existantes suffisantes |
| Probleme de convergence du modele | Moyenne | Moyen | Rollback vers version precedente validee |
| Surcharge d'un membre de l'equipe | Moyenne | Eleve | Repartition dynamique, priorisation des taches critiques |
| Changement de specifications | Faible | Moyen | Methodologie agile, iterations courtes |

---

*Document valide par l'equipe projet - Mai 2026*
*Reference : PLAN-THUM-2026-001 - Version 2.0*
*Deadline de livraison : 19 mai 2026*
