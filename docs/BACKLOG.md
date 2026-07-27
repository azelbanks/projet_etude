# Backlog Projet ThumaCheck

**Projet** : ThumaCheck — Detection de desinformation sur Bluesky
**Client** : Thumalien (fact-checking, veille media)
**Equipe** : Azelie Bernard (Lead technique), Sebastien Lazcanotegui (Consolidation ML)
**Periode** : Decembre 2025 — Juillet 2026
**Reference** : Derive du WBS `docs/08_planification_gantt.md`

---

## Legende

| Statut | Signification |
|--------|---------------|
| DONE | Livre et valide |
| DROPPED | Abandonne (justifie) |

---

## Sprint 1 — Cadrage & Infrastructure (Dec 2025)

| ID | Tache | Responsable | Statut | Livrable |
|----|-------|-------------|--------|----------|
| T1.1 | Analyse des besoins et specification | Azelie | DONE | `docs/01_cahier_des_charges_techniques.md` |
| T1.2 | Choix technologiques (Python, MongoDB, Docker) | Azelie | DONE | `docs/03_methodologie_projet.md` |
| T1.3 | Mise en place environnement Docker Compose | Azelie | DONE | `docker-compose.yml` |
| T1.4 | Configuration MongoDB + volumes persistants | Azelie | DONE | `docker-compose.yml` (service mongodb) |
| T1.5 | Creation du depot Git et structure projet | Azelie | DONE | `github.com/azelbanks/projet_etude` |

## Sprint 2 — Collecte & Emotions (Dec 2025 — Jan 2026)

| ID | Tache | Responsable | Statut | Livrable |
|----|-------|-------------|--------|----------|
| T2.1 | Developpement collecteur Bluesky (AT Protocol) | Azelie | DONE | `src/collection/collect_bluesky.py` |
| T2.2 | Validation schema JSON + deduplication | Azelie | DONE | Upsert MongoDB sur `uri` |
| T2.3 | Audit qualite des donnees collectees | Azelie | DONE | `notebooks/00_Audit_Qualite_Donnees.ipynb` |
| T2.4 | Maintenance et monitoring continu | Azelie | DONE | `src/collection/pipeline_monitor.py` |
| T3.1 | Exploration datasets emotions | Azelie | DONE | `notebooks/02_Analyse_Emotions_MLP.ipynb` |
| T3.2 | Developpement MLP PyTorch (7 classes) | Azelie | DONE | `src/pipeline/expert_detector.py` (EmotionFeatureExtractor) |
| T3.3 | Early stopping + class weights | Azelie | DONE | `models/emotion_bilingual.pt` |
| T3.4 | Evaluation et validation | Azelie | DONE | Accuracy 62% (7 classes) |

## Sprint 3 — Pipeline NLP V1-V2 (Jan — Mars 2026)

| ID | Tache | Responsable | Statut | Livrable |
|----|-------|-------------|--------|----------|
| T4.1 | Baseline V1.0 (TF-IDF + LogReg EN) | Azelie | DONE | F1 = 0.996 (biais Reuters) |
| T4.2 | Audit biais Reuters + debiaisage | Sebastien | DONE | `notebooks/07_Analyse_Modele_GridSearch.ipynb` |
| T4.3 | V1.5 bilingue + features linguistiques | Azelie | DONE | F1 = 0.986, 12 features |
| T4.8 | GridSearch hyperparametres (C, min_df, ngram) | Sebastien | DONE | C=5.0, min_df=5, ngram=(1,2) |
| T4.4 | V2 integration datasets sociaux + seuil 0.44 | Azelie | DONE | F1 = 0.897, 73.4% fiable Bluesky |

## Sprint 4 — Pipeline V3-V5 + Transformers (Mars — Avril 2026)

| ID | Tache | Responsable | Statut | Livrable |
|----|-------|-------------|--------|----------|
| T4.5 | V3 correction preprocessing | Azelie | DONE | `notebooks/11_Retraining_V3.py` |
| T4.6 | V4 augmentation FR court + 15 features | Azelie | DONE | `notebooks/12_Retraining_V4.py` |
| T4.7 | V5 + 10K posts sociaux synthetiques | Azelie | DONE | `notebooks/14_Retraining_V5_Social.py`, F1 = 0.913 |
| T5.1 | Fine-tuning CamemBERT V1 (FR) | Azelie | DONE | `notebooks/13_FineTune_CamemBERT_FR.py` |
| T5.2 | CamemBERT V2 + donnees sociales | Azelie | DONE | `notebooks/16_FineTune_CamemBERT_V2_Social.py` |
| T5.3 | Fine-tuning RoBERTa EN V1 | Azelie | DONE | `notebooks/18_FineTune_RoBERTa_EN.py` |
| T5.4 | RoBERTa EN V2 + 10K synthetique | Azelie | DONE | `notebooks/19_FineTune_RoBERTa_EN_V2.py` |
| T5.5 | Pipeline hybride stacking V5 + CamemBERT V2 | Azelie | DONE | `notebooks/17_Pipeline_Hybride_Stacking.py` |

## Sprint 5 — Dashboard & Documentation (Mars — Avril 2026)

| ID | Tache | Responsable | Statut | Livrable |
|----|-------|-------------|--------|----------|
| T6.1 | Dashboard V1 (metriques basiques) | Azelie | DONE | `dashboard/app.py` |
| T6.2 | Dashboard V2 (glassmorphism, 3 pages) | Azelie | DONE | Design system cyan/dark |
| T6.3 | Dashboard V3 (radar charts, live prediction) | Azelie | DONE | Analyse IA interactive |
| T6.4 | Integration weighted loss + securite | Azelie | DONE | Authentification bcrypt |
| T7.1 | Cahier des charges techniques | Azelie | DONE | `docs/01_cahier_des_charges_techniques.md` |
| T7.2 | Conformite RGPD & AI Act | Azelie | DONE | `docs/02_conformite_RGPD_AI_Act.md` |
| T7.3 | Rapport de projet | Azelie | DONE | `docs/rapport_projet_thumalien.md` |
| T7.4 | Guide utilisateur | Azelie | DONE | `docs/guide_utilisateur.md` |
| T7.5 | Documentation technique (28 notebooks) | Azelie | DONE | `notebooks/00-27` |
| T7.6 | Planification et gouvernance | Azelie | DONE | `docs/08_planification_gantt.md` |

## Sprint 6 — Gold Set & Iterations V6-V9 (Avril — Mai 2026)

| ID | Tache | Responsable | Statut | Livrable |
|----|-------|-------------|--------|----------|
| T8.1 | Gold set V1 (200 posts, kappa=0.808) | Sebastien | DONE | `scripts/extract_gold_test_set.py` |
| T8.2 | Gold set V2 (500 posts, 2 annotateurs, kappa=0.498) | Sebastien | DONE | `notebooks/22_Gold_Test_Set_Evaluation.py` |
| T8.3 | Evaluation systematique V5-V9 sur gold | Azelie | DONE | `docs/12_model_card.md` §4 |
| T9.1 | V6 Style-Only topic-agnostic (28 features, GBT) | Azelie | DONE | `notebooks/23_Style_Only_V6.py` |
| T9.2 | V7 Ensemble hybride V5+V6 + SHAP | Azelie | DONE | `notebooks/24_Hybrid_Ensemble_V7_SHAP.py` |
| T9.3 | V8 Meta-learner V5+V6+CamemBERT | Azelie | DONE | `notebooks/25_V8_Hybrid_Extended_CamemBERT.py` |
| T9.4 | Self-training Bluesky | Azelie | DROPPED | `docs/27_analyse_echec_self_training.md` (echec documente) |
| T9.5 | V9 Pipeline 2 etapes fait/opinion | Azelie | DONE | `notebooks/27_Pipeline_2_Etapes.py` |

## Sprint 7 — Collecte V3, Dashboard V5, Tests (Avril — Mai 2026)

| ID | Tache | Responsable | Statut | Livrable |
|----|-------|-------------|--------|----------|
| T10.1 | Reequilibrage termes FR/EN (210 keywords) | Azelie | DONE | `config/search_config.json` |
| T10.2 | Inference automatique emotions + V5 | Azelie | DONE | `scripts/batch_emotion_inference.py` |
| T10.3 | Rate limiting & backoff progressif | Azelie | DONE | Circuit breaker dans `collect_bluesky.py` |
| T11.1 | Dashboard V4 (V9 + SHAP + Explorateur) | Azelie | DONE | 4 pages Streamlit |
| T11.2 | Dashboard V5 (5 pages, Performance, A propos) | Azelie | DONE | `dashboard/app.py` (1 800+ LoC) |
| T12.1 | Tests unitaires (537 tests, 80% coverage) | Azelie | DONE | `tests/` (37 fichiers) |
| T12.2 | Benchmark latence (1.5ms/texte) | Azelie | DONE | `.github/workflows/ci.yml` |
| T12.3 | Tests d'integration pipeline | Azelie | DONE | `tests/test_pipeline_integration.py` |
| T12.4 | Tests explicabilite | Azelie | DONE | `tests/test_explainability.py` |

## Sprint 8 — XAI, CI/CD, Video (Mai 2026)

| ID | Tache | Responsable | Statut | Livrable |
|----|-------|-------------|--------|----------|
| T13.1 | Script et storyboard video | Azelie | DONE | `video_mvp/script_video_18min_v5.md` |
| T13.2 | Tournage demo live | Azelie + Sebastien | DONE | 28 rushes `.mov` |
| T13.3 | Montage et post-production | Azelie | DONE | `video_mvp/capcut/0723 (1).mp4` |
| T13.4 | Livraison finale | Azelie | DONE | MP4 18 min |
| T14.1 | SHAP global (beeswarm + dependence) | Azelie | DONE | `docs/figures/xai/` |
| T14.2 | Attention CamemBERT (CLS + heatmap) | Azelie | DONE | Dashboard heatmap interactive |
| T14.3 | Layer Integrated Gradients (Captum) | Azelie | DONE | `src/explainability/integrated_gradients.py` |
| T14.4 | Decomposition meta-learner V8 | Azelie | DONE | `docs/figures/xai/results.json` |
| T14.5 | Validation faithfulness (AOPC) | Azelie | DONE | `scripts/run_xai_pipeline.py` |
| T14.6 | Model Card formelle (Google standard) | Azelie | DONE | `docs/12_model_card.md` |
| T14.7 | Integration decomposition V8 dashboard | Azelie | DONE | Page "Analyse IA" |

## Sprint 9 — Refactoring & Packaging (Mai — Juillet 2026)

| ID | Tache | Responsable | Statut | Livrable |
|----|-------|-------------|--------|----------|
| T15.1 | Healthchecks MongoDB + depends_on | Azelie | DONE | `docker-compose.yml` |
| T15.2 | Utilisateur non-root + PYTHONPATH | Azelie | DONE | `Dockerfile` |
| T15.3 | API FastAPI (/predict, /health, /explain) | Azelie | DONE | `src/api/main.py` |
| T15.4 | Authentification dashboard | Azelie | DONE | `dashboard/auth_config.yaml` |
| T16.1 | Bootstrap IC 95% sur FP -67% | Azelie | DONE | `notebooks/20_Tests_Significativite_Bootstrap.py` |
| T16.2 | Justification CamemBERT non-prod | Azelie | DONE | `docs/rapport_projet_thumalien.md` §26 |
| T16.3 | Tableau risques enrichi (12 risques) | Azelie | DONE | `docs/13_risk_register.md` |
| T16.4 | Diagrammes C4 + sequence inference | Azelie | DONE | `docs/15_architecture_c4.md` |
| T16.5 | Etoffement rendu individuel Sebastien | Sebastien | DONE | `docs/rendu_individuel_sebastien_lazcanotegui.md` |
| T16.6 | Page de garde + footer pagine PDF | Azelie | DONE | `docs/pdf/` (17 PDF) |
| T16.7 | Script packaging nomenclature PE_2526 | Azelie | DONE | Nomenclature conforme |
| T16.8 | Regeneration 17 PDF depuis MD | Azelie | DONE | `docs/pdf/` |

---

## Metriques finales du backlog

| Indicateur | Valeur |
|------------|--------|
| Taches totales | 78 |
| Taches terminees (DONE) | 77 |
| Taches abandonnees (DROPPED) | 1 (self-training, echec documente) |
| Sprints | 9 (Dec 2025 — Juil 2026) |
| Charge totale | ~506 heures (Azelie 432h + Sebastien 74h) |
| Corpus final | 537 845 posts Bluesky |
| Modeles entraines | 9 versions (V1.0 — V9) |
| Tests | 537 pytest, 80% coverage, 80.3% mutation kill rate |
| Empreinte carbone | 8.86 g CO2 |
