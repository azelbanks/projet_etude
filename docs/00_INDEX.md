# Index Général de la Documentation Projet
## Thumalien -- Social Media Intelligence & AI Monitor

**Référence** : IDX-THUM-2026-001
**Version** : 2.0
**Statut** : En vigueur
**Date** : Avril 2026
**Responsable** : Chef de Projet Data

---

## 1. Documents de référence

| # | Document | Référence | Version | Statut | Chemin |
|---|----------|-----------|---------|--------|--------|
| D01 | Cahier des Charges Techniques | CDC-THUM-2026-001 | 2.0 | Validé | `docs/01_cahier_des_charges_techniques.md` |
| D02 | Conformité RGPD & AI Act | RGPD-THUM-2026-001 | 1.0 | En vigueur | `docs/02_conformite_RGPD_AI_Act.md` |
| D03 | Méthodologie et Gouvernance | METH-THUM-2026-001 | 1.0 | En vigueur | `docs/03_methodologie_projet.md` |
| D04 | Revue et Challenge Équipe | REV-THUM-2026-001 | 1.0 | En vigueur | `docs/04_revue_challenge_equipe.md` |
| D05 | Rôles et Compétences Projet | — | 1.0 | En vigueur | `docs/roles_et_competences_projet.md` |
| D06 | Rapport de Projet | — | 1.0 | En vigueur | `docs/rapport_projet_thumalien.md` |
| D07 | Guide Utilisateur | — | 1.0 | En vigueur | `docs/guide_utilisateur.md` |
| D08 | Index Général (ce document) | IDX-THUM-2026-001 | 2.0 | En vigueur | `docs/00_INDEX.md` |
| D12 | Model Card V9 (XAI) | MC-THUM-2026-001 | 1.0 | En vigueur | `docs/12_model_card.md` |
| D13 | Index figures XAI | — | 1.0 | Auto-généré | `docs/figures/xai/INDEX.md` |

---

## 2. Artefacts modèles

| Artefact | Chemin | Description | Version |
|----------|--------|-------------|---------|
| Modèle Expert V2 | `models/model_expert_v2.pkl` | LogisticRegression calibrée (TF-IDF 30K + 12 ling. + 7 émo.) | V2 |
| Vectorizer V2 | `models/tfidf_expert_v2.pkl` | TF-IDF 30 000 features, n-grams (1,3), sublinear TF | V2 |
| Métriques V2 | `models/metrics_expert_v2.pkl` | Scores CV, matrices de confusion, feature importance | V2 |
| Modèle Expert V1.5 | `models/model_expert.pkl` | LogReg bilingue (fallback) | V1.5 |
| Vectorizer V1.5 | `models/tfidf_expert.pkl` | TF-IDF 30K features | V1.5 |
| Métriques V1.5 | `models/metrics_expert.pkl` | Métriques d'entraînement V1.5 | V1.5 |
| Model Expert V3 | `models/model_expert_v3.pkl` | LogReg (bug fix preprocessing) | V3 |
| Vectorizer V3 | `models/tfidf_expert_v3.pkl` | TF-IDF V3 | V3 |
| Métriques V3 | `models/metrics_expert_v3.pkl` | Scores CV, matrices de confusion, feature importance | V3 |
| Model Expert V4 | `models/model_expert_v4.pkl` | LogReg (augmentation FR court) | V4 |
| Vectorizer V4 | `models/tfidf_expert_v4.pkl` | TF-IDF V4 | V4 |
| Métriques V4 | `models/metrics_expert_v4.pkl` | Scores CV, matrices de confusion, feature importance | V4 |
| Model Expert V5 | `models/model_expert_v5.pkl` | LogReg (197K texts, 15 ling + 7 émo) | V5 **Production** |
| Vectorizer V5 | `models/tfidf_expert_v5.pkl` | TF-IDF 30K features | V5 |
| Métriques V5 | `models/metrics_expert_v5.pkl` | Scores CV, matrices de confusion, feature importance | V5 |
| Model Style V6 | `models/model_style_v6.joblib` | GradientBoosting style-only (28 features + 7 émotions) | V6 |
| Model Hybrid V7 | `models/model_hybrid_v7.joblib` | Meta-learner LogReg (V5+V6 ensemble) | V7 |
| Hybrid Meta Learner | `models/hybrid_meta_learner.joblib` | Meta-learner for stacking | V7 |
| Model Hybrid V8 | `models/model_hybrid_v8.joblib` | Meta-learner V5+V6+CamemBERT (7 features) | V8 |
| Stage 1 Fait/Opinion | `models/stage1_fact_opinion.joblib` | Classifieur fait/opinion (pipeline 2 étapes) | V9 |
| MLP Émotions | `models/emotion_bilingual.pt` | MLP PyTorch 7 classes émotionnelles bilingue | 1.0 |
| Vocabulaire Émotions | `models/emotion_vocab_bilingual.pickle` | Mapping mot -> token (25 000 tokens) | 1.0 |
| Encodeur Labels | `models/emotion_label_encoder_bilingual.pickle` | 7 labels d'émotions | 1.0 |

---

## 3. Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 00 | `notebooks/00_Audit_Qualite_Donnees.ipynb` | Audit qualité des données d'entraînement, détection biais Reuters |
| 01 | `notebooks/01_Exploration_Bluesky.ipynb` | Exploration des posts Bluesky collectés |
| 02 | `notebooks/02_Analyse_Emotions_MLP.ipynb` | Entraînement MLP émotions bilingue (PyTorch) |
| 03 | `notebooks/03_Mise_a_jour_Quotidienne.ipynb` | Pipeline de mise à jour quotidienne |
| 04 | `notebooks/04_Modele_Avance_RoBERTa.ipynb` | Prototype RoBERTa (exploration, non déployé) |
| 05 | `notebooks/05_Detection_Expert_Bilingue.ipynb` | Pipeline expert bilingue + ablation study |
| 06 | `notebooks/06_Documentation_Technique.ipynb` | Documentation technique, limites, roadmap |
| 07 | `notebooks/07_Analyse_Modele_GridSearch.ipynb` | GridSearch hyperparamètres |
| 08 | `notebooks/08_Integration_Datasets_V2.ipynb` | Intégration 3 datasets sociaux |
| 09 | `notebooks/09_Analyse_Erreurs_Qualitative.py` | Analyse qualitative 2000 textes, 100 erreurs |
| 10 | `notebooks/10_Analyse_Modele_Par_Longueur.py` | Performance par longueur de texte |
| 11 | `notebooks/11_Retraining_V3.py` | Correction bug preprocessing |
| 12 | `notebooks/12_Retraining_V4.py` | Augmentation FR court (+32% F1) |
| 13 | `notebooks/13_FineTune_CamemBERT_FR.py` | Fine-tuning CamemBERT FR V1 |
| 14 | `notebooks/14_Retraining_V5_Social.py` | V5 +10K synthetic FR social |
| 15 | `notebooks/15_Seuil_Adaptatif.py` | Seuil adaptatif par longueur |
| 16 | `notebooks/16_FineTune_CamemBERT_V2_Social.py` | CamemBERT V2 (F1 0.957 ultra-court) |
| 17 | `notebooks/17_Pipeline_Hybride_Stacking.py` | Pipeline hybride stacking V5+CamemBERT |
| 18 | `notebooks/18_FineTune_RoBERTa_EN.py` | RoBERTa EN V1 (F1 0.838) |
| 19 | `notebooks/19_FineTune_RoBERTa_EN_V2.py` | RoBERTa EN V2 (F1 0.874) |
| 20 | `notebooks/20_Tests_Significativite_Bootstrap.py` | Tests significativité bootstrap |
| 21 | `notebooks/21_Gold_Test_Set_Evaluation.py` | Évaluation gold test set (200 posts) |
| 22 | `notebooks/22_Gold_Test_Set_Evaluation.py` | Évaluation gold test set V2 |
| 23 | `notebooks/23_Style_Only_V6.py` | Modèle style-only V6 (GradientBoosting) |
| 24 | `notebooks/24_Hybrid_Ensemble_V7_SHAP.py` | Ensemble hybride V7 + SHAP |
| 25 | `notebooks/25_V8_Hybrid_Extended_CamemBERT.py` | V8 méta-learner étendu (+CamemBERT) |
| 26 | `notebooks/26_V5_Finetune_Bluesky.py` | Self-training Bluesky (échec documenté) |
| 27 | `notebooks/27_Pipeline_2_Etapes.py` | Pipeline 2 étapes fait/opinion + V5 |

---

## 4. Monitoring et logs

| Fichier | Chemin | Description | Fréquence de mise à jour |
|---------|--------|-------------|--------------------------|
| Émissions carbone | `emissions.csv` | Bilan CodeCarbon (entraînement + inférence) | À chaque run CodeCarbon |
| Logs collecteur | Sortie standard du conteneur Docker `collector` | Volume, erreurs, heartbeat | Continu (toutes les 5 min) |
| Logs dashboard | Sortie standard du conteneur Docker `dashboard` | Accès, erreurs Streamlit | Continu |

---

## 5. Infrastructure

| Fichier | Chemin | Description |
|---------|--------|-------------|
| Docker Compose | `docker-compose.yml` | Orchestration des 4 services (MongoDB, Collector, Jupyter, Dashboard) |
| Dockerfile | `dockerfile` | Image Python 3.13-slim avec dépendances ML |
| Batch Inference | `scripts/batch_emotion_inference.py` | Inférence émotionnelle batch sur posts MongoDB |
| Requirements | `requirements.txt` | Dépendances Python du projet |
| Config Streamlit | `.streamlit/config.toml` | Thème dark + configuration serveur |
| Variables d'env | `.env` (non versionné) | Identifiants Bluesky, URI MongoDB |

---

## 6. Journal des décisions

| Date | Décision | Justification | Responsable | Référence |
|------|----------|---------------|-------------|-----------|
| Déc 2025 | Choix de Bluesky comme source de données | API ouverte (AT Protocol), collecte légale de posts publics | Chef de Projet | D06 sect. 1 |
| Déc 2025 | Choix de MongoDB pour le stockage | Base NoSQL adaptée aux documents JSON des posts | Data Engineer | D01 sect. 5.2 |
| Jan 2026 | Migration TensorFlow vers PyTorch | Incompatibilité TensorFlow avec Apple Silicon (M4 Pro) | ML Engineer | D06 sect. 5 |
| Jan 2026 | Nettoyage du biais Reuters | Le modèle apprenait le style Reuters (F1=0.99 artificiel) au lieu de détecter les fake news | Data Scientist | D06 sect. 4 |
| Fév 2026 | Choix de LogReg plutôt que RoBERTa | Performance comparable (F1 0.90 vs ~0.92), 100x moins d'énergie, interprétabilité totale | ML Engineer | D06 sect. 2 |
| Fév 2026 | Ajout de 12 features linguistiques | Captent des signaux structurels indépendants du contenu (caps, ponctuation, sensationnalisme) | ML Engineer | D06 sect. 6 |
| Fév 2026 | Intégration de 3 datasets sociaux (V2) | Domain shift : V1.5 classait 77% des posts Bluesky comme suspects | Data Scientist | D06 sect. 8 |
| Fév 2026 | Seuil de décision abaissé à 0.44 | Maximise simultanément le F1 holdout (0.9024) et la fiabilité Bluesky (73.4%) | Data Scientist + Chef de Projet | D06 sect. 9 |
| Mars 2026 | Dashboard glassmorphism 3 pages | Amélioration UX professionnelle avec explicabilité intégrée | Dashboard Dev | D01 sect. 3.5 |
| Avril 2026 | Mesure empreinte carbone inférence | Étendre CodeCarbon au-delà de l'entraînement pour un bilan complet | Expert Green IT | D04 sect. 9 |
| Avril 2026 | Création helper agrégation MongoDB | Déporter les calculs dans MongoDB pour améliorer la performance du dashboard | Data Engineer | D04 sect. 7 |
| Avril 2026 | V6 Style-Only model | 28 features style, topic-agnostic | ML Engineer | D06 sect. 16 |
| Avril 2026 | V7 Ensemble hybride V5+V6 + SHAP explicabilité | Combinaison V5+V6 avec SHAP pour explicabilité | ML Engineer | D06 sect. 17 |
| Avril 2026 | Gold test set 200 posts annotés (Cohen kappa 0.808) | Évaluation sur jeu de test annoté manuellement | Data Scientist | D06 sect. 14 |
| Avril 2026 | Tests significativité bootstrap sur toutes les versions | Validation statistique des améliorations entre versions | Data Scientist | D06 sect. 20 |
| Avril 2026 | V8 intégration CamemBERT dans méta-learner | 3e signal sémantique FR, F1 suspect +28% | ML Engineer | D06 sect. 18 |
| Avril 2026 | Self-training Bluesky : échec documenté | Pseudo-labeling circulaire, abandon motivé | Data Scientist | D06 sect. 19 |
| Mai 2026 | Annotation humaine 500 posts (2 annotateurs, kappa=0.498) | Gold standard fiable pour évaluation Bluesky | Data Scientist | D06 sect. 20 |
| Mai 2026 | V9 pipeline 2 étapes fait/opinion | Filtre opinions avant détection, FP -67% | ML Engineer | D06 sect. 21 |
| Mai 2026 | Audit corpus et rééquilibrage collecte V3 | Biais émotionnel (75% joie), déséquilibre FR/EN (87.5% EN), 28 termes FR + 16 termes EN | Data Scientist | D06 sect. 22 |
| Mai 2026 | Inférence batch émotions (214K posts) | Couverture émotionnelle 100%, inférence auto intégrée au collecteur | ML Engineer | D06 sect. 22 |
| Mai 2026 | Refactoring Docker Compose professionnel | Healthchecks MongoDB, démarrage ordonné, PYTHONPATH unifié | DevOps | D06 sect. 22 |

---

## 7. Roadmap des versions modèle

| Version | Date | Statut | F1 | Bluesky % fiable | Innovation principale |
|---------|------|--------|-----|-------------------|----------------------|
| V1.0 | Déc 2025 | Obsolète | 0.99 (biaisé) | ~0% | Baseline LogReg EN |
| V1.5 | Fév 2026 | Fallback | 0.986 | 23% | Bilingue + nettoyage Reuters + 12 features linguistiques |
| V2 | Fév 2026 | **Historique** | 0.897 | 73.4% | 3 datasets sociaux + seuil 0.44 + 7 émotions |
| V3 | Mar 2026 | Fait | 0.90 | — | Correction bug preprocessing |
| V4 | Mar 2026 | Fait | FR court 0.86 | — | Augmentation FR court +32% |
| V5 | Mar 2026 | **Production** | 0.913 | — | +10K synthetic FR social |
| V6 | Avr 2026 | Fait | CV 0.830 | — | Style-only GradientBoosting topic-agnostic |
| V7 | Avr 2026 | Fait | Gold F1 suspect=0.127 | — | Ensemble hybride V5+V6 + SHAP |
| V8 | Avr 2026 | Fait | Gold F1 suspect=0.163 | — | V7 + CamemBERT (3e signal sémantique) |
| V9 | Mai 2026 | **Dernier** | Consensus kappa=0.187 | — | Pipeline 2 étapes fait/opinion, FP -67% |

---

*Document mis à jour en Mai 2026*
*Référence : IDX-THUM-2026-001 -- Version 3.1*
