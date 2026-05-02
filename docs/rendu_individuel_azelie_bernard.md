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
- Pipeline NLP complet (V1.0 à V9)
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
- Rééquilibrage FR/EN de la collecte (V3 collecteur : 28 termes FR + 16 termes EN, suppression des biais émotionnels)
- Inférence automatique intégrée au collecteur (émotions + V5 à chaque cycle)
- **Résultat** : 239 000+ posts collectés depuis décembre 2025, 100% annotés émotionnellement

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
- V8.0 : intégration CamemBERT comme 3e signal sémantique, F1 suspect +28%
- Self-training Bluesky : tentative de domain adaptation par pseudo-labeling, échec documenté (circularité)
- Annotation humaine : 500 posts annotés par 2 annotateurs, kappa inter-annotateurs = 0.498
- V9.0 : pipeline 2 étapes fait/opinion, réduction FP de 186 à 62 (-67%), validation statistique (Fisher p=0.0005)
- **40+ commits** sur le dépôt Git

### 2.3 Modèles Transformer (Avril 2026)
- Fine-tuning CamemBERT V1/V2 pour le français (F1 ultra-court = 0.957)
- Fine-tuning RoBERTa EN V1/V2 pour l'anglais (F1 ultra-court = 0.874)
- Pipeline hybride stacking V5 + CamemBERT V2

### 2.4 Dashboard Streamlit (Mars - Mai 2026)
- Design glassmorphism dark theme, interface intégralement en français
- 5 pages : Dashboard (vue globale), Analyse IA (prédiction live V9), Explorateur (posts MongoDB), Performance (métriques temps réel), À propos
- Radar charts, distribution des scores, analyse SHAP intégrée, connexion MongoDB temps réel
- V4 : intégration pipeline V9 cascade fait/opinion + SHAP
- V5 : refactoring complet Plotly, accents français, page Performance avec benchmarks

### 2.5 Modèle d'émotions (Jan 2026)
- MLP PyTorch bilingue, 7 classes d'émotions
- Early stopping + class weights pour gérer le déséquilibre
- Intégration comme features dans le pipeline NLP

### 2.6 Gold Test Set et annotation humaine (Avril-Mai 2026)
- Création et annotation manuelle de 200 posts Bluesky (gold set V1, kappa = 0.808)
- Annotation de 500 posts Bluesky par 2 annotateurs indépendants (gold set V2, kappa = 0.498)
- Échantillonnage stratifié : 250 FR + 250 EN, 50 par tranche de score
- Évaluation systématique de tous les modèles V5-V9
- Identification du biais thématique (mot "coronavirus" coefficient +9.72)
- Découverte de la distinction fait/opinion comme facteur discriminant (odds ratio 4.67, p=0.0005)

### 2.7 Modèle Style-Only V6 (Avril 2026)
- Conception de 28 features stylistiques en 6 blocs (structure, ponctuation, majuscules, lexique manipulation, crédibilité, diversité)
- GradientBoosting topic-agnostic
- Gold F1 suspect = 0.103 (+18% vs V5)

### 2.8 Ensemble Hybride V7 + SHAP (Avril 2026)
- Meta-learner LogReg combinant scores V5 et V6
- Intégration de SHAP TreeExplainer pour l'explicabilité
- V7 Combo accuracy = 0.840 (vs 0.685 V5), FP = 25 (vs 57 V5)
- Intégration complète dans le dashboard Streamlit

### 2.9 Tests et qualité (Mai 2026)
- Suite de 94 tests unitaires et d'intégration (pytest) couvrant 26% du code source
- Tests des modules critiques : collecteur (validation texte, détection langue), pipeline NLP (features linguistiques, features stylistiques V6, détecteur expert V5), CamemBERT (architecture réseau, dataset PyTorch), MongoDB (agrégations, requêtes), monitoring (scoring, rapports)
- Benchmark de latence automatisé : 1.5 ms/texte (728 textes/sec), 66x sous l'exigence CDC (100 ms)
- Tests d'intégration bilingues, edge cases (textes vides, emojis, textes longs, caractères spéciaux)

---

## 3. Compétences mobilisées et acquises

### 3.1 Compétences techniques

| Compétence | Niveau avant | Niveau après | Contexte d'application |
|-----------|:------------:|:------------:|----------------------|
| Python avancé | Intermédiaire | Avancé | Pipeline complet, 28 notebooks (00-27) |
| NLP / Text Mining | Débutant | Avancé | TF-IDF, features linguistiques, tokenization |
| Deep Learning (PyTorch) | Débutant | Intermédiaire | MLP émotions, fine-tuning Transformers |
| Transformers (Hugging Face) | Débutant | Intermédiaire | CamemBERT, RoBERTa, stacking |
| Docker / Docker Compose | Intermédiaire | Avancé | Architecture micro-services |
| MongoDB | Débutant | Intermédiaire | Schéma, index, agrégations |
| Streamlit | Débutant | Avancé | Dashboard interactif 5 pages, Plotly avancé |
| Git / GitHub | Intermédiaire | Avancé | Versioning, merge, collaboration |
| Explicabilité IA (SHAP) | Débutant | Intermédiaire | TreeExplainer, feature importance, intégration dashboard |
| Feature Engineering avancé | Intermédiaire | Avancé | 28 features stylistiques, meta-features, stacking |
| Ensemble Learning | Débutant | Intermédiaire | Meta-learner, stacking, GradientBoosting |
| Tests / CI | Débutant | Intermédiaire | pytest, mocking, benchmarking, coverage |

### 3.2 Compétences transversales

| Compétence | Mise en pratique |
|-----------|-----------------|
| Gestion de projet | Planification itérative, priorisation des tâches, respect des délais |
| Résolution de problèmes | Debugging du biais Reuters, correction preprocessing V3, migration TF->PyTorch |
| Communication technique | Rédaction de 10+ documents, 28 notebooks commentés |
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
- L'approche itérative (V1 → V9) a permis d'améliorer continuellement les performances
- La documentation au fil de l'eau (notebooks) facilite la traçabilité
- Le choix de Docker garantit la reproductibilité
- L'ajout de CamemBERT et RoBERTa a résolu le problème des textes courts
- Le gold test set a révélé des faiblesses invisibles en cross-validation
- L'approche V6 style-only a prouvé qu'on peut détecter des patterns de désinformation sans analyser le contenu
- SHAP apporte une transparence complète sur les raisons de chaque prédiction

### Ce que j'aurais fait différemment
- **Tests unitaires dès le début** : le bug des 5 features nulles (V2→V3) aurait été détecté immédiatement avec des tests de non-régression. La suite de tests (94 tests, 26% coverage) a été ajoutée tardivement mais reste un acquis méthodologique important.
- **MLflow pour le tracking** : les 28 notebooks documentent chaque expérience, mais un outil dédié (MLflow, W&B) aurait permis un suivi plus systématique des hyperparamètres et des métriques.
- **Annotation humaine plus précoce** : le gold test set a révélé un écart majeur entre la performance en cross-validation (F1=0.90) et la performance réelle sur Bluesky. Cette prise de conscience, arrivée tardivement, a motivé les itérations V6-V9.
- **Répartition de la charge** : en tant que lead technique, j'ai centralisé trop de responsabilités. Déléguer certains modules (dashboard, documentation) plus tôt aurait accéléré le projet.

### Axes d'amélioration identifiés
- **Fact-checking réel** : le pipeline détecte des signaux stylistiques de désinformation, mais ne vérifie pas les faits. L'intégration d'une API de fact-checking (ClaimBuster, Google Fact Check) serait un gain majeur.
- **Gold set plus équilibré** : le consensus (473 posts, 15 suspects) est très déséquilibré. Un échantillonnage ciblé de posts suspects permettrait une évaluation plus robuste.
- **Labels fait/opinion dédiés** : la distinction fait/opinion (V9) repose sur des heuristiques lexicales. Des labels humains explicites amélioreraient la cascade (l'oracle montre un F1 suspect potentiel de 0.545).
- **Monitoring production** : le weekly score check (JSONL) est fonctionnel mais rudimentaire. Prometheus + Grafana permettraient des alertes temps réel et des dashboards de drift monitoring.
- **Tests end-to-end** : les tests unitaires couvrent les composants isolés, mais un test d'intégration complet (collecte → MongoDB → inference → dashboard) validerait la chaîne complète.

---

## 6. Bilan personnel

Ce projet représente 6 mois de travail intensif sur un problème complexe et ouvert : la détection de désinformation sur les réseaux sociaux. En tant que lead technique, j'ai mené un projet Data/IA de bout en bout — de la collecte de 239 000+ posts Bluesky au déploiement d'un dashboard interactif, en passant par 9 itérations du pipeline NLP.

Les apprentissages clés :
- **L'évaluation est plus importante que l'entraînement** : un F1=0.99 en cross-validation masquait un biais Reuters. Le gold test set a été un tournant dans ma compréhension de l'évaluation des modèles.
- **L'itération constante est la seule voie** : chaque version (V1→V9) a corrigé une faiblesse identifiée en production. Les "échecs" (self-training, seuil adaptatif) sont aussi riches d'enseignements que les succès.
- **Les solutions simples dominent** : TF-IDF + features linguistiques (V5, 1.5ms/texte) reste plus robuste en production qu'un Transformer seul. L'architecture hybride (V8/V9) combine le meilleur des deux approches.
- **L'éthique n'est pas optionnelle** : la conformité RGPD, l'AI Act, le bilan carbone (CodeCarbon) et l'explicabilité (SHAP) m'ont appris que la responsabilité d'un ingénieur IA dépasse la performance technique.

Ce projet m'a fait passer de débutante à un niveau intermédiaire-avancé en NLP/Deep Learning. La rigueur scientifique acquise (tests de significativité, kappa inter-annotateurs, validation sur données réelles) sera un atout déterminant pour ma carrière professionnelle en data science.

---

*Rendu individuel - Azélie Bernard - Mai 2026*
