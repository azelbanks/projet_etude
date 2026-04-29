# Rendu Individuel - Azélie Bernard
## Projet Thumalien - Master Big Data, Sup de Vinci

**Étudiant** : Azélie Bernard
**Rôle principal** : Lead technique (Data Engineer, ML Engineer, DevOps, Dashboard Dev)
**Date** : Avril 2026

---

## 1. Mon rôle dans le projet

J'ai assuré le rôle de **lead technique** sur le projet Thumalien, couvrant l'ensemble de la chaîne de valeur : de la collecte des données sur Bluesky jusqu'au déploiement du dashboard de visualisation, en passant par le développement des modèles de détection de fake news.

### Responsabilités principales
- Architecture technique et infrastructure Docker
- Développement du collecteur Bluesky (AT Protocol)
- Pipeline NLP complet (V1.0 à V7)
- Fine-tuning des modèles Transformer (CamemBERT, RoBERTa)
- Explicabilité SHAP
- Gold Test Set annotation
- Dashboard Streamlit interactif
- Documentation technique
- Conformité RGPD et AI Act

---

## 2. Contributions techniques détaillées

### 2.1 Infrastructure et collecte (Déc 2025)
- Mise en place de l'architecture Docker Compose (MongoDB, Jupyter, Streamlit, Collecteur)
- Développement du collecteur Bluesky avec gestion des erreurs et déduplication
- Configuration MongoDB avec index unique et volumes persistants
- **Résultat** : 228 000+ posts collectés depuis décembre 2025

### 2.2 Pipeline NLP (Jan - Avril 2026)
- Baseline V1.0 : TF-IDF + LogisticRegression (F1=0.99, identifié le biais Reuters)
- Audit qualité : découverte du data leakage Reuters (99.2% identifiable par style)
- V1.5 : pipeline bilingue + 12 features linguistiques + modèle émotions
- V2.0 : intégration 3 datasets sociaux, seuil calibré 0.44
- V3.0 : correction du bug preprocessing (5/12 features nulles)
- V4.0 : augmentation FR court (+32% F1)
- V5.0 : +10K posts synthétiques, F1 global = 0.913
- V6.0 : modèle style-only topic-agnostic (28 features stylistiques + 7 émotions, GradientBoosting, CV F1=0.830)
- V7.0 : ensemble hybride meta-learner V5+V6 avec explicabilité SHAP, FP réduits de 57 à 25 sur gold set
- **30+ commits** sur le dépôt Git

### 2.3 Modèles Transformer (Avril 2026)
- Fine-tuning CamemBERT V1/V2 pour le français (F1 ultra-court = 0.957)
- Fine-tuning RoBERTa EN V1/V2 pour l'anglais (F1 ultra-court = 0.874)
- Pipeline hybride stacking V5 + CamemBERT V2

### 2.4 Dashboard Streamlit (Mars 2026)
- Design glassmorphism dark theme
- 3 pages : vue globale, prédiction live, explicabilité
- Radar charts, métriques temps réel, connexion MongoDB

### 2.5 Modèle d'émotions (Jan 2026)
- MLP PyTorch bilingue, 7 classes d'émotions
- Early stopping + class weights pour gérer le déséquilibre
- Intégration comme features dans le pipeline NLP

### 2.6 Gold Test Set (Avril 2026)
- Création et annotation manuelle de 200 posts Bluesky
- Double annotation avec calcul du kappa de Cohen (0.808)
- Évaluation systématique de tous les modèles V5-V7
- Identification du biais thématique (mot "coronavirus" coefficient +9.72)

### 2.7 Modèle Style-Only V6 (Avril 2026)
- Conception de 28 features stylistiques en 6 blocs (structure, ponctuation, majuscules, lexique manipulation, crédibilité, diversité)
- GradientBoosting topic-agnostic
- Gold F1 suspect = 0.103 (+18% vs V5)

### 2.8 Ensemble Hybride V7 + SHAP (Avril 2026)
- Meta-learner LogReg combinant scores V5 et V6
- Intégration de SHAP TreeExplainer pour l'explicabilité
- V7 Combo accuracy = 0.840 (vs 0.685 V5), FP = 25 (vs 57 V5)
- Intégration complète dans le dashboard Streamlit

---

## 3. Compétences mobilisées et acquises

### 3.1 Compétences techniques

| Compétence | Niveau avant | Niveau après | Contexte d'application |
|-----------|:------------:|:------------:|----------------------|
| Python avancé | Intermédiaire | Avancé | Pipeline complet, 25 notebooks |
| NLP / Text Mining | Débutant | Avancé | TF-IDF, features linguistiques, tokenization |
| Deep Learning (PyTorch) | Débutant | Intermédiaire | MLP émotions, fine-tuning Transformers |
| Transformers (Hugging Face) | Débutant | Intermédiaire | CamemBERT, RoBERTa, stacking |
| Docker / Docker Compose | Intermédiaire | Avancé | Architecture micro-services |
| MongoDB | Débutant | Intermédiaire | Schéma, index, agrégations |
| Streamlit | Débutant | Intermédiaire | Dashboard interactif 3 pages |
| Git / GitHub | Intermédiaire | Avancé | Versioning, merge, collaboration |
| Explicabilité IA (SHAP) | Débutant | Intermédiaire | TreeExplainer, feature importance, intégration dashboard |
| Feature Engineering avancé | Intermédiaire | Avancé | 28 features stylistiques, meta-features, stacking |
| Ensemble Learning | Débutant | Intermédiaire | Meta-learner, stacking, GradientBoosting |

### 3.2 Compétences transversales

| Compétence | Mise en pratique |
|-----------|-----------------|
| Gestion de projet | Planification itérative, priorisation des tâches, respect des délais |
| Résolution de problèmes | Debugging du biais Reuters, correction preprocessing V3, migration TF->PyTorch |
| Communication technique | Rédaction de 10+ documents, 25 notebooks commentés |
| Esprit critique | Analyse des erreurs, tests de significativité, identification des biais |
| Autonomie | Apprentissage de PyTorch, CamemBERT, AT Protocol en autodidacte |

---

## 4. Défis rencontrés et solutions

| Défi | Contexte | Solution | Apprentissage |
|------|----------|----------|---------------|
| Biais Reuters (F1=0.99 biaisé) | Le modèle détectait le style Reuters, pas les fake news | Débiaisage (BODY_AGENCY_TERMS), nettoyage des artefacts | L'évaluation est aussi importante que l'entraînement |
| TensorFlow incompatible M4 | Apple Silicon non supporté par TF | Migration vers PyTorch | Toujours vérifier la compatibilité hardware |
| Features nulles (5/12) | Bug de preprocessing dans le pipeline V2 | Debug, correction, retraining V3 | Les tests unitaires sont essentiels |
| F1 FR court = 0.65 | Les textes courts de type réseau social mal classifiés | Augmentation + vocabulaire enrichi + CamemBERT | Les modèles généralistes ne suffisent pas pour les textes courts |
| Merge Git avec conflits | Dépôt recréé par un collègue | `--allow-unrelated-histories` + résolution manuelle | Toujours communiquer avant de restructurer un dépôt |
| Biais thématique (V5 détecte le sujet, pas la désinformation) | Gold test set révélait 57 FP sur posts "coronavirus" fiables | V6 style-only + V7 meta-learner réduisant FP de 57 à 25 | L'évaluation sur données réelles est indispensable |

---

## 5. Analyse critique et recul

### Ce qui a bien fonctionné
- L'approche itérative (V1 -> V5) a permis d'améliorer continuellement les performances
- La documentation au fil de l'eau (notebooks) facilite la traçabilité
- Le choix de Docker garantit la reproductibilité
- L'ajout de CamemBERT et RoBERTa a résolu le problème des textes courts
- Le gold test set a révélé des faiblesses invisibles en cross-validation
- L'approche V6 style-only a prouvé qu'on peut détecter des patterns de désinformation sans analyser le contenu
- SHAP apporte une transparence complète sur les raisons de chaque prédiction

### Ce que j'aurais fait différemment
- Commencer par des tests unitaires dès le début (le bug des features nulles aurait été détecté plus tôt)
- Utiliser MLflow pour le tracking des expériences au lieu de notebooks individuels
- Prévoir une stratégie d'annotation manuelle plus tôt dans le projet
- Mieux répartir la charge de travail dans l'équipe

### Limites identifiées
- Le modèle reste dépendant des datasets d'entraînement (pas de fact-checking réel)
- L'absence de données annotées spécifiquement pour Bluesky est une faiblesse
- Le monitoring en production est minimal (pas de Grafana/Prometheus)

---

## 6. Bilan personnel

Ce projet m'a permis de mener un projet Data/IA de bout en bout, de la collecte au déploiement. J'ai particulièrement progressé en NLP et Deep Learning, domaines dans lesquels j'étais débutante en début de projet. La complexité du problème (détection de fake news dans un contexte bilingue, multi-longueur) m'a appris que les solutions simples sont souvent les meilleures (TF-IDF + features > modèle complexe mal entraîné), et que l'itération constante est la clé de l'amélioration.

La dimension éthique (biais, conformité RGPD, AI Act) a été particulièrement formatrice et sera un atout pour ma carrière professionnelle.

---

*Rendu individuel - Azélie Bernard - Avril 2026*
