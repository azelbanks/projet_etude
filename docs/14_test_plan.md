# Plan de Tests — ThumaCheck

| Champ             | Valeur                          |
|-------------------|---------------------------------|
| **Reference**     | TEST-THUM-2026-001              |
| **Version**       | 1.0                             |
| **Date**          | Juillet 2026                    |
| **Projet**        | ThumaCheck — Detection de desinformation Bluesky |
| **Equipe**        | Azelie Bernard, Sebastien Lazcanotegui |
| **Client**        | Thumalien                       |

---

## 1. Objectifs de la strategie de test

La strategie de test de ThumaCheck vise a :

- **Garantir la fiabilite** du pipeline NLP de detection de desinformation (modeles V5, V6, meta-learner V8, CamemBERT).
- **Verifier la conformite** aux exigences du cahier des charges (latence < 100 ms/texte, F1 >= 0.85, couverture >= 75 %).
- **Detecter les regressions** via une integration continue automatisee (GitHub Actions).
- **Valider la securite** des entrees utilisateur (injection HTML, XSS, null bytes, unicode).
- **Mesurer la robustesse** du code face aux mutations (mutmut, kill rate >= 75 %).
- **Assurer l'equite** du systeme vis-a-vis des biais linguistiques et demographiques.

---

## 2. Perimetre

### 2.1 In-scope

| Composant                | Description                                          |
|--------------------------|------------------------------------------------------|
| Pipeline NLP (`src/pipeline/`) | ExpertFakeNewsDetector, inference, features linguistiques et stylistiques |
| Collection (`src/collection/`) | Collecte Bluesky, validation texte, nettoyage, qualite des donnees, monitoring |
| Explicabilite (`src/explainability/`) | MetaLearnerDecomposer, FaithfulnessEvaluator, SHAP global |
| API FastAPI (`src/api/`)  | Endpoints `/health`, `/predict`, `/explain`, validation Pydantic |
| Dashboard (`dashboard/`) | Logique metier, syntaxe Python, rendu Streamlit      |
| Modeles ML               | CamemBERT, emotions MLP, LogReg V5, meta-learner V8 |
| MongoDB                  | Aggregations, index, qualite des donnees             |

### 2.2 Out-of-scope

- Tests end-to-end navigateur (Selenium/Playwright) sur le dashboard Streamlit.
- Tests de charge distribues (Locust/k6) au-dela du benchmark local.
- Tests sur infrastructure de production (deploiement cloud).
- Penetration testing externe.

---

## 3. Types de tests

### 3.1 Tests unitaires

Les tests unitaires couvrent chaque module du systeme de maniere isolee, avec des mocks pour les dependances externes (MongoDB, modeles ML, torch).

#### Repartition par module

| Module               | Fichiers de test                                                        | Nb tests |
|----------------------|-------------------------------------------------------------------------|----------|
| **Pipeline / Detecteur** | `test_expert_detector.py`, `test_expert_detector_core.py`, `test_expert_detector_extended.py`, `test_detector_init.py` | 131 |
| **Collection Bluesky** | `test_collect_bluesky.py`, `test_collect_bluesky_extended.py`, `test_collection_cycle.py`, `test_collection_extended.py` | 79 |
| **Explicabilite (XAI)** | `test_explainability.py`, `test_explainability_extended.py`, `test_shap_global_extended.py` | 76 |
| **Dashboard**        | `test_dashboard_logic.py`, `test_dashboard_syntax.py`                   | 48 |
| **CamemBERT**        | `test_camembert_classifier.py`, `test_camembert_extended.py`            | 50 |
| **MongoDB**          | `test_mongo_aggregations.py`, `test_mongo_aggregations_extended.py`, `test_setup_indexes.py` | 25 |
| **Features**         | `test_linguistic_features.py`, `test_style_features.py`                 | 12 |
| **Monitoring**       | `test_pipeline_monitor.py`, `test_monitoring_extended.py`               | 25 |
| **Qualite donnees**  | `test_data_quality.py`, `test_dataset_cleaner.py`                       | 28 |
| **Emotions**         | `test_emotion_mlp.py`                                                   | 9  |
| **API FastAPI**      | `test_api.py`                                                           | 5  |
| **Score hebdomadaire** | `test_weekly_score_check.py`                                          | 11 |

**Total : 537 tests** repartis dans 31 fichiers de test.

### 3.2 Tests d'integration

| Fichier de test                | Description                                               | Nb tests |
|-------------------------------|-----------------------------------------------------------|----------|
| `test_pipeline_integration.py` | Pipeline complet : chargement modele, prediction, score   | 14       |
| `test_inference_cycle.py`      | Cycle d'inference end-to-end avec collecte et prediction  | 15       |
| `test_collection_cycle.py`     | Cycle de collecte Bluesky complet avec mocks MongoDB      | 9        |

Ces tests verifient l'interaction entre plusieurs modules (detecteur + features + meta-learner, collecte + stockage + inference).

### 3.3 Tests de performance (benchmark latence)

| Fichier                      | Scenario                    | Seuil CDC        |
|-----------------------------|-----------------------------|------------------|
| `test_benchmark_latence.py` | Texte unique (10 iterations) | < 200 ms (marge), CDC < 100 ms |
| `test_benchmark_latence.py` | Batch 10 textes (5 iterations) | < 1 000 ms total |
| `test_benchmark_latence.py` | Batch 100 textes             | < 10 000 ms total |

Les benchmarks mesurent le temps d'inference reel avec `time.perf_counter()` et affichent moyenne, P95 et debit (textes/sec). Ils requierent le modele `model_expert_v5.pkl`.

### 3.4 Tests de securite (validation des entrees)

Le fichier `test_input_validation.py` (7 tests) couvre :

- **Injection HTML/XSS** : verification que les balises `<script>` sont echappees avant rendu.
- **Texte tres long** : gestion de chaines de 100 000 caracteres.
- **Null bytes** : `\x00` dans le texte ne provoque pas de crash.
- **Unicode edge cases** : accents combinants, espaces zero-width, emojis 4-octets, texte RTL arabe.
- **Stabilite de prediction** : meme texte soumis deux fois retourne un score identique.
- **Entree minimale** : mot unique comme entree.

L'API FastAPI ajoute une couche de validation Pydantic (longueur min=1, max=10 000) testee dans `test_api.py`.

### 3.5 Tests de mutation (mutmut)

Configuration dans `pyproject.toml` :

```toml
[tool.mutmut]
paths_to_mutate = "src/explainability/meta_decomposition.py"
tests_dir = "tests/"
```

- **Cible** : module `meta_decomposition.py` (decomposition du meta-learner).
- **Objectif** : kill rate >= 75 %.
- **Resultat actuel** : 80.3 % de mutants tues.
- **Tests mutation-killing dedies** : classe `TestMetaDecompositionMutationKilling` dans `test_explainability.py` (26+ tests specifiques ciblant les mutants survivants).

### 3.6 Tests de fairness et biais

| Fichier                              | Tests                                                |
|--------------------------------------|------------------------------------------------------|
| `test_expert_detector_extended.py`   | `test_remove_agency_bias_reuters`, `test_remove_agency_bias_non_string` |
| `test_dataset_cleaner.py`            | Nettoyage des biais d'agence de presse dans les donnees d'entrainement |
| `test_collect_bluesky_extended.py`   | Validation equilibree FR/EN                          |
| `test_camembert_extended.py`         | Tests de robustesse multilingue                      |

Les biais sont controles a plusieurs niveaux :
- **Preprocessing** : suppression des mentions d'agences (Reuters, AP, AFP) qui biaisent le TF-IDF.
- **Evaluation** : comparaison des scores de credibilite entre textes FR et EN.
- **Dashboard** : section audit fairness avec metriques de parite demographique.

---

## 4. Matrice de tracabilite

| Exigence (Cahier des charges)                    | Tests associes                                          |
|-------------------------------------------------|--------------------------------------------------------|
| Detection de desinformation (F1 >= 0.85)        | `test_expert_detector_core.py`, `test_pipeline_integration.py`, `test_weekly_score_check.py` |
| Temps d'analyse < 100 ms/texte                  | `test_benchmark_latence.py`                             |
| Collecte Bluesky automatisee                    | `test_collect_bluesky.py`, `test_collection_cycle.py`, `test_collection_extended.py` |
| Explicabilite des predictions (XAI)             | `test_explainability.py`, `test_explainability_extended.py`, `test_shap_global_extended.py` |
| API REST fonctionnelle                          | `test_api.py`                                           |
| Dashboard de visualisation                      | `test_dashboard_logic.py`, `test_dashboard_syntax.py`   |
| Qualite des donnees                             | `test_data_quality.py`, `test_dataset_cleaner.py`       |
| Monitoring du pipeline                          | `test_pipeline_monitor.py`, `test_monitoring_extended.py` |
| Securite des entrees                            | `test_input_validation.py`, `test_api.py`               |
| Support multilingue (FR/EN)                     | `test_camembert_classifier.py`, `test_camembert_extended.py`, `test_expert_detector_extended.py` |
| Analyse des emotions                            | `test_emotion_mlp.py`                                   |
| Features linguistiques et stylistiques          | `test_linguistic_features.py`, `test_style_features.py` |
| Stockage MongoDB                                | `test_mongo_aggregations.py`, `test_mongo_aggregations_extended.py`, `test_setup_indexes.py` |
| Robustesse aux mutations de code                | `test_explainability.py` (classe `TestMetaDecompositionMutationKilling`) |

---

## 5. Couverture par module

| Module                          | Nb tests | Coverage cible | Fichiers de test principaux                     |
|---------------------------------|----------|----------------|-------------------------------------------------|
| `src/pipeline/`                 | 131      | >= 80 %        | `test_expert_detector*.py`, `test_detector_init.py` |
| `src/collection/`              | 79       | >= 75 %        | `test_collect_bluesky*.py`, `test_collection_*.py` |
| `src/explainability/`          | 76       | >= 80 %        | `test_explainability*.py`, `test_shap_global_extended.py` |
| `src/api/`                     | 5        | >= 75 %        | `test_api.py`                                   |
| `dashboard/`                   | 48       | >= 60 %        | `test_dashboard_logic.py`, `test_dashboard_syntax.py` |
| `src/collection/pipeline_monitor.py` | 25 | >= 75 %        | `test_pipeline_monitor.py`, `test_monitoring_extended.py` |

**Couverture globale cible : >= 75 %** (gate CI/CD).
**Couverture actuelle : ~80 %** (rapport `pytest-cov`).

---

## 6. Criteres d'acceptation

| Critere                        | Seuil minimum   | Outil de mesure    |
|-------------------------------|------------------|--------------------|
| Couverture de code            | >= 75 %          | `pytest-cov`       |
| Mutation kill rate            | >= 75 %          | `mutmut`           |
| F1-score detection            | >= 0.85          | `test_weekly_score_check.py` |
| Latence moyenne par texte     | < 200 ms         | `test_benchmark_latence.py` |
| 0 test en echec en CI         | 100 % pass       | GitHub Actions     |
| Validation Pydantic API       | 0 regression     | `test_api.py`      |

---

## 7. Outils utilises

| Outil           | Role                                      | Version    |
|-----------------|-------------------------------------------|------------|
| `pytest`        | Framework de test principal               | >= 8.0     |
| `pytest-cov`    | Mesure de couverture de code              | >= 5.0     |
| `mutmut`        | Tests de mutation                         | >= 2.4     |
| `unittest.mock` | Mocks et patches pour isolation           | stdlib     |
| `FastAPI TestClient` | Tests API HTTP                       | >= 0.100   |
| `Trivy`         | Scan de vulnerabilites des images Docker  | CI/CD      |
| `coverage`      | Rapport de couverture et gate CI          | >= 7.0     |

---

## 8. Environnement de test

### 8.1 CI/CD — GitHub Actions

Le workflow `.github/workflows/tests.yml` execute automatiquement les tests sur chaque push et pull request vers `main` :

- **Runner** : `ubuntu-latest`
- **Python** : 3.13
- **Etapes** :
  1. Checkout du code.
  2. Installation des dependances (`requirements.txt`, `pytest`, `pytest-cov`).
  3. Execution de la suite de tests avec couverture.
  4. Gate de couverture : echec si < 75 %.

**Tests exclus en CI** (necessitent des modeles ou des ressources lourdes) :
- `test_benchmark_latence.py` (necessite `model_expert_v5.pkl`)
- `test_pipeline_integration.py` (necessite `model_expert_v5.pkl`)
- `test_expert_detector.py` (necessite `model_expert_v5.pkl`)

### 8.2 Environnement local

- **OS** : macOS / Linux
- **Python** : 3.13
- **MongoDB** : via Docker Compose (`docker compose up -d mongo`)
- **Modeles** : `models/model_expert_v5.pkl` (requis pour les tests conditionels)

---

## 9. Procedure d'execution

### 9.1 Execution complete (local)

```bash
# Tous les tests
python3 -m pytest tests/ -v --cov=src --cov=dashboard --cov-report=term-missing

# Avec rapport HTML de couverture
python3 -m pytest tests/ -v --cov=src --cov=dashboard --cov-report=html:docs/coverage_html
```

### 9.2 Execution ciblee par module

```bash
# Pipeline uniquement
python3 -m pytest tests/test_expert_detector*.py tests/test_detector_init.py -v

# Explicabilite uniquement
python3 -m pytest tests/test_explainability*.py tests/test_shap_global_extended.py -v

# API uniquement
python3 -m pytest tests/test_api.py -v

# Securite uniquement
python3 -m pytest tests/test_input_validation.py -v

# Benchmark latence (necessite modele)
python3 -m pytest tests/test_benchmark_latence.py -v -s
```

### 9.3 Tests de mutation

```bash
# Lancer mutmut sur le module cible
mutmut run

# Voir les resultats
mutmut results

# Voir un mutant specifique
mutmut show <id>
```

### 9.4 Reproduction du pipeline CI

```bash
# Reproduire exactement le workflow GitHub Actions
python3 -m pytest tests/ -v \
  --ignore=tests/test_benchmark_latence.py \
  --ignore=tests/test_pipeline_integration.py \
  --ignore=tests/test_expert_detector.py \
  --cov=src --cov-report=term-missing

# Verifier le gate de couverture
coverage report --fail-under=75
```

---

## 10. Gestion des tests conditionnels

Certains tests necessitent des fichiers modeles volumineux non versiones dans Git. Le mecanisme de skip conditionnel est le suivant :

### 10.1 Skip par modele ML

```python
_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
_MODEL_EXISTS = os.path.exists(os.path.join(_MODEL_DIR, 'model_expert_v5.pkl'))

@pytest.mark.skipif(not _MODEL_EXISTS, reason="Model files not found")
class TestBenchmarkLatence:
    ...
```

**Fichiers concernes** : `test_benchmark_latence.py`, `test_pipeline_integration.py`, `test_input_validation.py` (classe `TestPipelineInputValidation`), `test_expert_detector_core.py`.

### 10.2 Skip par dependance PyTorch

```python
torch = pytest.importorskip("torch", reason="PyTorch required")
```

**Fichiers concernes** : `test_camembert_classifier.py`, `test_camembert_extended.py`.

### 10.3 Visibilite des tests ignores

Le fichier `tests/conftest.py` contient un hook `pytest_terminal_summary` qui affiche un avertissement visible en fin d'execution lorsque des tests ont ete ignores a cause de modeles absents :

```
==================== ATTENTION: tests ignores (modeles absents) ====================
  N tests ont ete ignores car les fichiers modeles ne sont pas presents dans models/.
  Pour les executer : assurez-vous que model_expert_v5.pkl est present.
```

---

## 11. Configuration de couverture

La couverture est configuree via `.coveragerc` :

- **Sources mesurees** : `src/`, `dashboard/`
- **Exclusions** : tests, migrations, notebooks, scripts
- **Lignes exclues** : `pragma: no cover`, `raise NotImplementedError`, `if __name__`, blocs d'affichage Streamlit (`st.markdown`, `st.plotly_chart`, etc.), fonctions de rendu de page dashboard
- **Precision** : 1 decimale
- **Missing lines** : affichees dans le rapport

---

## 12. Inventaire des fichiers de test

| # | Fichier                               | Nb tests | Module couvert                     |
|---|---------------------------------------|----------|------------------------------------|
| 1 | `tests/test_expert_detector_extended.py` | 66    | Pipeline / detecteur expert        |
| 2 | `tests/test_explainability.py`        | 50       | Explicabilite (XAI)                |
| 3 | `tests/test_dashboard_logic.py`       | 46       | Dashboard Streamlit                |
| 4 | `tests/test_expert_detector_core.py`  | 38       | Detecteur expert (core)            |
| 5 | `tests/test_camembert_extended.py`    | 37       | CamemBERT classifier               |
| 6 | `tests/test_collect_bluesky_extended.py` | 27    | Collecte Bluesky                   |
| 7 | `tests/test_collect_bluesky.py`       | 22       | Collecte Bluesky (base)            |
| 8 | `tests/test_detector_init.py`         | 21       | Initialisation detecteur           |
| 9 | `tests/test_collection_extended.py`   | 21       | Collection etendue                 |
| 10| `tests/test_explainability_extended.py` | 20     | Explicabilite etendue              |
| 11| `tests/test_dataset_cleaner.py`       | 19       | Nettoyage dataset                  |
| 12| `tests/test_monitoring_extended.py`   | 16       | Monitoring etendu                  |
| 13| `tests/test_inference_cycle.py`       | 15       | Cycle d'inference                  |
| 14| `tests/test_pipeline_integration.py`  | 14       | Integration pipeline               |
| 15| `tests/test_camembert_classifier.py`  | 13       | CamemBERT classifier (base)        |
| 16| `tests/test_weekly_score_check.py`    | 11       | Verification scores hebdomadaire   |
| 17| `tests/test_mongo_aggregations_extended.py` | 11 | Aggregations MongoDB etendues      |
| 18| `tests/test_mongo_aggregations.py`    | 11       | Aggregations MongoDB               |
| 19| `tests/test_pipeline_monitor.py`      | 9        | Monitoring pipeline                |
| 20| `tests/test_emotion_mlp.py`           | 9        | Modele emotions MLP                |
| 21| `tests/test_data_quality.py`          | 9        | Qualite des donnees                |
| 22| `tests/test_collection_cycle.py`      | 9        | Cycle de collecte                  |
| 23| `tests/test_style_features.py`        | 7        | Features stylistiques              |
| 24| `tests/test_input_validation.py`      | 7        | Validation / securite des entrees  |
| 25| `tests/test_shap_global_extended.py`  | 6        | SHAP global                        |
| 26| `tests/test_linguistic_features.py`   | 5        | Features linguistiques             |
| 27| `tests/test_expert_detector.py`       | 5        | Detecteur expert (integration)     |
| 28| `tests/test_api.py`                   | 5        | API FastAPI                        |
| 29| `tests/test_setup_indexes.py`         | 3        | Index MongoDB                      |
| 30| `tests/test_benchmark_latence.py`     | 3        | Benchmark latence                  |
| 31| `tests/test_dashboard_syntax.py`      | 2        | Syntaxe dashboard                  |
|   | **TOTAL**                             | **537**  |                                    |

---

*Document genere le 23 juillet 2026 — ThumaCheck v1.0*
