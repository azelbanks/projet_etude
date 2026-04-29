# Index General de la Documentation Projet
## Thumalien -- Social Media Intelligence & AI Monitor

**Reference** : IDX-THUM-2026-001
**Version** : 2.0
**Statut** : En vigueur
**Date** : Avril 2026
**Responsable** : Chef de Projet Data

---

## 1. Documents de reference

| # | Document | Reference | Version | Statut | Chemin |
|---|----------|-----------|---------|--------|--------|
| D01 | Cahier des Charges Techniques | CDC-THUM-2026-001 | 2.0 | Valide | `docs/01_cahier_des_charges_techniques.md` |
| D02 | Conformite RGPD & AI Act | RGPD-THUM-2026-001 | 1.0 | En vigueur | `docs/02_conformite_RGPD_AI_Act.md` |
| D03 | Methodologie et Gouvernance | METH-THUM-2026-001 | 1.0 | En vigueur | `docs/03_methodologie_projet.md` |
| D04 | Revue et Challenge Equipe | REV-THUM-2026-001 | 1.0 | En vigueur | `docs/04_revue_challenge_equipe.md` |
| D05 | Roles et Competences Projet | — | 1.0 | En vigueur | `docs/roles_et_competences_projet.md` |
| D06 | Rapport de Projet | — | 1.0 | En vigueur | `docs/rapport_projet_thumalien.md` |
| D07 | Guide Utilisateur | — | 1.0 | En vigueur | `docs/guide_utilisateur.md` |
| D08 | Index General (ce document) | IDX-THUM-2026-001 | 2.0 | En vigueur | `docs/00_INDEX.md` |

---

## 2. Artefacts modeles

| Artefact | Chemin | Description | Version |
|----------|--------|-------------|---------|
| Modele Expert V2 | `models/model_expert_v2.pkl` | LogisticRegression calibree (TF-IDF 30K + 12 ling. + 7 emo.) | V2 |
| Vectorizer V2 | `models/tfidf_expert_v2.pkl` | TF-IDF 30 000 features, n-grams (1,3), sublinear TF | V2 |
| Metriques V2 | `models/metrics_expert_v2.pkl` | Scores CV, matrices de confusion, feature importance | V2 |
| Modele Expert V1.5 | `models/model_expert.pkl` | LogReg bilingue (fallback) | V1.5 |
| Vectorizer V1.5 | `models/tfidf_expert.pkl` | TF-IDF 30K features | V1.5 |
| Metriques V1.5 | `models/metrics_expert.pkl` | Metriques d'entrainement V1.5 | V1.5 |
| Model Expert V3 | `models/model_expert_v3.pkl` | LogReg (bug fix preprocessing) | V3 |
| Vectorizer V3 | `models/tfidf_expert_v3.pkl` | TF-IDF V3 | V3 |
| Metriques V3 | `models/metrics_expert_v3.pkl` | Scores CV, matrices de confusion, feature importance | V3 |
| Model Expert V4 | `models/model_expert_v4.pkl` | LogReg (augmentation FR court) | V4 |
| Vectorizer V4 | `models/tfidf_expert_v4.pkl` | TF-IDF V4 | V4 |
| Metriques V4 | `models/metrics_expert_v4.pkl` | Scores CV, matrices de confusion, feature importance | V4 |
| Model Expert V5 | `models/model_expert_v5.pkl` | LogReg (197K texts, 15 ling + 7 emo) | V5 **Production** |
| Vectorizer V5 | `models/tfidf_expert_v5.pkl` | TF-IDF 30K features | V5 |
| Metriques V5 | `models/metrics_expert_v5.pkl` | Scores CV, matrices de confusion, feature importance | V5 |
| Model Style V6 | `models/model_style_v6.joblib` | GradientBoosting style-only (28 features + 7 emotions) | V6 |
| Model Hybrid V7 | `models/model_hybrid_v7.joblib` | Meta-learner LogReg (V5+V6 ensemble) | V7 |
| Hybrid Meta Learner | `models/hybrid_meta_learner.joblib` | Meta-learner for stacking | V7 |
| MLP Emotions | `models/emotion_bilingual.pt` | MLP PyTorch 7 classes emotionnelles bilingue | 1.0 |
| Vocabulaire Emotions | `models/emotion_vocab_bilingual.pickle` | Mapping mot -> token (25 000 tokens) | 1.0 |
| Encodeur Labels | `models/emotion_label_encoder_bilingual.pickle` | 7 labels d'emotions | 1.0 |

---

## 3. Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 00 | `notebooks/00_Audit_Qualite_Donnees.ipynb` | Audit qualite des donnees d'entrainement, detection biais Reuters |
| 01 | `notebooks/01_Exploration_Bluesky.ipynb` | Exploration des posts Bluesky collectes |
| 02 | `notebooks/02_Analyse_Emotions_MLP.ipynb` | Entrainement MLP emotions bilingue (PyTorch) |
| 03 | `notebooks/03_Mise_a_jour_Quotidienne.ipynb` | Pipeline de mise a jour quotidienne |
| 04 | `notebooks/04_Modele_Avance_RoBERTa.ipynb` | Prototype RoBERTa (exploration, non deploye) |
| 05 | `notebooks/05_Detection_Expert_Bilingue.ipynb` | Pipeline expert bilingue + ablation study |
| 06 | `notebooks/06_Documentation_Technique.ipynb` | Documentation technique, limites, roadmap |
| 07 | `notebooks/07_Analyse_Modele_GridSearch.ipynb` | GridSearch hyperparametres |
| 08 | `notebooks/08_Integration_Datasets_V2.ipynb` | Integration 3 datasets sociaux |
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
| 20 | `notebooks/20_Tests_Significativite_Bootstrap.py` | Tests significativite bootstrap |
| 21 | `notebooks/21_Gold_Test_Set_Evaluation.py` | Evaluation gold test set (200 posts) |
| 22 | `notebooks/22_Gold_Test_Set_Evaluation.py` | Evaluation gold test set V2 |
| 23 | `notebooks/23_Style_Only_V6.py` | Modele style-only V6 (GradientBoosting) |
| 24 | `notebooks/24_Hybrid_Ensemble_V7_SHAP.py` | Ensemble hybride V7 + SHAP |

---

## 4. Monitoring et logs

| Fichier | Chemin | Description | Frequence de mise a jour |
|---------|--------|-------------|--------------------------|
| Emissions carbone | `emissions.csv` | Bilan CodeCarbon (entrainement + inference) | A chaque run CodeCarbon |
| Logs collecteur | Sortie standard du conteneur Docker `collector` | Volume, erreurs, heartbeat | Continu (toutes les 5 min) |
| Logs dashboard | Sortie standard du conteneur Docker `dashboard` | Acces, erreurs Streamlit | Continu |

---

## 5. Infrastructure

| Fichier | Chemin | Description |
|---------|--------|-------------|
| Docker Compose | `docker-compose.yml` | Orchestration des 4 services (MongoDB, Collector, Jupyter, Dashboard) |
| Dockerfile | `dockerfile` | Image Python 3.9-slim avec dependances ML |
| Requirements | `requirements.txt` | Dependances Python du projet |
| Config Streamlit | `.streamlit/config.toml` | Theme dark + configuration serveur |
| Variables d'env | `.env` (non versionne) | Identifiants Bluesky, URI MongoDB |

---

## 6. Journal des decisions

| Date | Decision | Justification | Responsable | Reference |
|------|----------|---------------|-------------|-----------|
| Dec 2025 | Choix de Bluesky comme source de donnees | API ouverte (AT Protocol), collecte legale de posts publics | Chef de Projet | D06 sect. 1 |
| Dec 2025 | Choix de MongoDB pour le stockage | Base NoSQL adaptee aux documents JSON des posts | Data Engineer | D01 sect. 5.2 |
| Jan 2026 | Migration TensorFlow vers PyTorch | Incompatibilite TensorFlow avec Apple Silicon (M4 Pro) | ML Engineer | D06 sect. 5 |
| Jan 2026 | Nettoyage du biais Reuters | Le modele apprenait le style Reuters (F1=0.99 artificiel) au lieu de detecter les fake news | Data Scientist | D06 sect. 4 |
| Fev 2026 | Choix de LogReg plutot que RoBERTa | Performance comparable (F1 0.90 vs ~0.92), 100x moins d'energie, interpretabilite totale | ML Engineer | D06 sect. 2 |
| Fev 2026 | Ajout de 12 features linguistiques | Captent des signaux structurels independants du contenu (caps, ponctuation, sensationnalisme) | ML Engineer | D06 sect. 6 |
| Fev 2026 | Integration de 3 datasets sociaux (V2) | Domain shift : V1.5 classait 77% des posts Bluesky comme suspects | Data Scientist | D06 sect. 8 |
| Fev 2026 | Seuil de decision abaisse a 0.44 | Maximise simultanement le F1 holdout (0.9024) et la fiabilite Bluesky (73.4%) | Data Scientist + Chef de Projet | D06 sect. 9 |
| Mars 2026 | Dashboard glassmorphism 3 pages | Amelioration UX professionnelle avec explicabilite integree | Dashboard Dev | D01 sect. 3.5 |
| Avril 2026 | Mesure empreinte carbone inference | Etendre CodeCarbon au-dela de l'entrainement pour un bilan complet | Expert Green IT | D04 sect. 9 |
| Avril 2026 | Creation helper aggregation MongoDB | Deporter les calculs dans MongoDB pour ameliorer la performance du dashboard | Data Engineer | D04 sect. 7 |
| Avril 2026 | V6 Style-Only model | 28 features style, topic-agnostic | ML Engineer | D06 sect. 16 |
| Avril 2026 | V7 Ensemble hybride V5+V6 + SHAP explicabilite | Combinaison V5+V6 avec SHAP pour explicabilite | ML Engineer | D06 sect. 17 |
| Avril 2026 | Gold test set 200 posts annotes (Cohen kappa 0.808) | Evaluation sur jeu de test annote manuellement | Data Scientist | D06 sect. 14 |
| Avril 2026 | Tests significativite bootstrap sur toutes les versions | Validation statistique des ameliorations entre versions | Data Scientist | D06 sect. 20 |

---

## 7. Roadmap des versions modele

| Version | Date | Statut | F1 | Bluesky % fiable | Innovation principale |
|---------|------|--------|-----|-------------------|----------------------|
| V1.0 | Dec 2025 | Obsolete | 0.99 (biaise) | ~0% | Baseline LogReg EN |
| V1.5 | Fev 2026 | Fallback | 0.986 | 23% | Bilingue + nettoyage Reuters + 12 features linguistiques |
| V2 | Fev 2026 | **Production** | 0.897 | 73.4% | 3 datasets sociaux + seuil 0.44 + 7 emotions |
| V3 | Mar 2026 | Fait | 0.90 | — | Correction bug preprocessing |
| V4 | Mar 2026 | Fait | FR court 0.86 | — | Augmentation FR court +32% |
| V5 | Mar 2026 | **Production** | 0.913 | — | +10K synthetic FR social |
| V6 | Avr 2026 | Fait | CV 0.830 | — | Style-only GradientBoosting topic-agnostic |
| V7 | Avr 2026 | **Dernier** | Gold F1 suspect=0.127 | — | Ensemble hybride V5+V6 + SHAP |
| V8 | — | Planifie | — | — | Active Learning + RAG fact-checking |

---

*Document mis a jour en Avril 2026*
*Reference : IDX-THUM-2026-001 -- Version 2.0*
