# Roles et Competences Projet Thumalien
## Cartographie des professionnels necessaires a l'excellence du projet

**Projet** : Thumalien — Social Media Intelligence & AI Monitor
**Auteur** : Azelie Bernard
**Formation** : Master Big Data
**Date** : Avril 2026

---

## Preambule — Pourquoi cette reflexion ?

Un projet comme Thumalien ne se resume pas a du code. Il mobilise des competences qui traversent l'ingenierie des donnees, l'intelligence artificielle, l'infrastructure, la visualisation et la gouvernance. Chaque brique du systeme — de la collecte Bluesky en temps reel jusqu'a la prediction de credibilite affichee sur le dashboard — exige un savoir-faire specifique.

Ce document identifie les **roles professionnels indispensables**, non pas comme une liste theorique, mais comme une reflexion ancree dans la realite concrete du projet : ses choix techniques, ses difficultes rencontrees et ses ambitions futures. Chaque role est pousse au plus haut niveau de competence attendu dans l'industrie, car la detection de desinformation est un sujet ou l'approximation n'est pas acceptable.

L'objectif est double :
1. **Comprendre** quelles expertises sont mobilisees et pourquoi elles sont chacune critiques
2. **Projeter** le projet vers un niveau d'excellence professionnel en identifiant les standards de chaque discipline

---

## Table des matieres

1. [Vue d'ensemble de l'equipe](#1-vue-densemble-de-lequipe)
2. [Data Engineer — Architecte du pipeline de donnees](#2-data-engineer--architecte-du-pipeline-de-donnees)
3. [Machine Learning Engineer / NLP Specialist](#3-machine-learning-engineer--nlp-specialist)
4. [Data Scientist Senior — Methodologue et analyste](#4-data-scientist-senior--methodologue-et-analyste)
5. [MLOps Engineer — Industrialisation des modeles](#5-mlops-engineer--industrialisation-des-modeles)
6. [Developpeur Dashboard / Data Visualization Engineer](#6-developpeur-dashboard--data-visualization-engineer)
7. [DevOps / Cloud Architect](#7-devops--cloud-architect)
8. [Chef de Projet Data / Product Owner](#8-chef-de-projet-data--product-owner)
9. [Expert Green IT & IA Responsable](#9-expert-green-it--ia-responsable)
10. [Matrice de responsabilites RACI](#10-matrice-de-responsabilites-raci)
11. [Reflexion — La synergie comme condition de l'excellence](#11-reflexion--la-synergie-comme-condition-de-lexcellence)

---

## 1. Vue d'ensemble de l'equipe

### Architecture du projet et mapping des competences

Le projet Thumalien repose sur 4 briques techniques, chacune necessitant des competences specifiques :

```
COLLECTE          STOCKAGE           INTELLIGENCE          RESTITUTION
(Bluesky API)     (MongoDB)          (NLP + Deep Learning) (Dashboard)
    |                 |                     |                    |
Data Engineer    Data Engineer        ML Engineer           Dashboard Dev
                 DevOps              Data Scientist         UX/Data Viz
                                     MLOps Engineer
                                     Expert Green IT

                        --- Transversal ---
                    Chef de Projet / Product Owner
```

### Organisation reelle du projet : binome complementaire

Le projet Thumalien a ete realise en **binome** par Azelie Bernard et Sebastien Lazcanotegui, avec une repartition des responsabilites fondee sur la complementarite des competences :

| Membre | Role principal | Contributions cles |
|--------|---------------|-------------------|
| **Azelie Bernard** | Lead technique & developpement | Pipeline ML (V1→V9), collecteur Bluesky, dashboard Streamlit, infrastructure Docker, conformite RGPD/AI Act, documentation technique |
| **Sebastien Lazcanotegui** | Validation & qualite | Annotation humaine (gold test set, 2e annotateur pour le kappa inter-annotateurs), revue et validation de la documentation, tests fonctionnels, support sur le GridSearch et le debiaisage, co-production video MVP |

### Projection vers une equipe d'excellence (vision industrielle)

| Configuration | Effectif | Compromis |
|---------------|----------|-----------|
| Binome (etat actuel) | 2 personnes | Polyvalence maximale, complementarite dev/validation |
| Equipe minimale | 4 personnes | Data Engineer + ML Engineer + DevOps + Chef de projet |
| Equipe optimale | 6 personnes | Ajout Data Scientist dedie + Dashboard Dev |
| Equipe d'excellence | 8 personnes | Tous les roles specialises, expertise maximale par domaine |

**Reflexion** : le projet Thumalien a ete realise en binome, ce qui a impose une polyvalence forte a Azelie sur le plan technique et a confie a Sebastien un role structurant de validation et de qualite. Cette organisation a l'avantage de la rapidite de decision et de la communication directe. Cependant, pour atteindre un niveau de production industrielle — ou chaque composant est pousse a son maximum — la specialisation de chaque role devient indispensable. Un ingenieur ML qui doit aussi gerer l'infrastructure Docker perd du temps sur l'optimisation de ses modeles. Un Data Engineer qui doit aussi concevoir le dashboard ne peut pas pousser la resilience de son pipeline au maximum.

---

## 2. Data Engineer — Architecte du pipeline de donnees

### Identite du role

| Attribut | Detail |
|----------|--------|
| **Intitule** | Data Engineer Senior / Lead Data Engineer |
| **Seniorite requise** | 5 a 8 ans d'experience |
| **Rattachement** | Direction Technique / CTO |
| **Certification de reference** | Google Professional Data Engineer, AWS Data Analytics, Databricks Certified |

### Pourquoi ce role est obligatoire pour Thumalien

Le coeur de Thumalien est un flux de donnees continu : des posts Bluesky sont collectes en temps reel via WebSocket, transformes, stockes dans MongoDB, puis consommes par les modeles IA et le dashboard. Sans un Data Engineer de haut niveau, ce pipeline est fragile : perte de donnees en cas de deconnexion, absence de deduplication, pas de monitoring de la qualite des donnees entrantes.

Actuellement, le collecteur (`src/collection/collect_bluesky.py`) fonctionne avec une boucle de recherche par mots-cles toutes les 5 minutes avec 3 tentatives et backoff exponentiel. C'est une base solide, mais un Data Engineer senior pousse cette architecture beaucoup plus loin.

### Missions detaillees

#### Mission 1 — Conception et industrialisation du pipeline d'ingestion

**Objectif** : Transformer le collecteur actuel en un pipeline de production resilient, scalable et observable.

**Taches** :
- Remplacer la boucle de polling par une architecture event-driven basee sur le Firehose Bluesky (AT Protocol `com.atproto.sync.subscribeRepos`) pour une collecte veritablement temps reel
- Implementer un systeme de message queue (Apache Kafka ou Redis Streams) entre le collecteur et MongoDB pour decoupler ingestion et stockage
- Concevoir un schema de deduplication robuste base sur le `uri` des posts (index unique MongoDB + bloom filter en memoire pour les verifications rapides)
- Implementer un mecanisme de replay : si le pipeline tombe, il doit pouvoir reprendre exactement la ou il s'est arrete grace a un cursor persistant
- Gerer le backpressure : si MongoDB est lent, le collecteur doit ralentir sans perdre de messages

**Indicateurs de performance** :
- Zero perte de donnees mesurable sur 30 jours
- Latence ingestion < 5 secondes entre l'emission du post Bluesky et son stockage en base
- Taux de disponibilite du pipeline > 99.5%

#### Mission 2 — Architecture et optimisation de la base de donnees

**Objectif** : Faire de MongoDB un socle performant capable de servir simultanement l'ingestion temps reel, les requetes analytiques et le dashboard.

**Taches** :
- Concevoir le schema MongoDB optimal pour les posts : choisir entre un schema plat (actuel) et un schema avec des sous-documents pour les enrichissements (emotions, score de credibilite)
- Creer les index adaptes : index compose sur `(created_at, search_term)` pour les requetes temporelles, index texte pour la recherche full-text, index TTL pour l'archivage automatique
- Implementer une strategie de partitionnement temporel (collections mensuelles ou sharding sur `created_at`) pour maintenir les performances quand la base depasse le million de documents
- Mettre en place des aggregation pipelines optimises pour les KPI du dashboard (repartition emotions, evolution temporelle, top termes)
- Configurer les write concerns et read preferences pour equilibrer coherence et performance

**Indicateurs de performance** :
- Temps de reponse des requetes dashboard < 500ms pour 1M+ documents
- Taille des index < 20% de la taille des donnees
- Zero downtime lors des migrations de schema

#### Mission 3 — Qualite et observabilite des donnees

**Objectif** : Garantir que les donnees qui alimentent les modeles IA sont fiables, completes et conformes aux attentes.

**Taches** :
- Implementer des checks de qualite automatises a l'ingestion : validation du schema JSON, detection des champs manquants, verification de la langue
- Creer un pipeline de data profiling quotidien : distribution des langues, longueur moyenne des textes, volume par mot-cle, detection d'anomalies (chute ou pic soudain de volume)
- Mettre en place des alertes (Slack, email) en cas d'anomalie : arret de la collecte, changement dans l'API Bluesky, degradation de la qualite
- Documenter le data lineage : d'ou vient chaque donnee, quelles transformations elle a subi, quand elle a ete collectee
- Implementer un systeme de data versioning pour les datasets d'entrainement (DVC — Data Version Control)

**Indicateurs de performance** :
- 100% des posts stockes ont tous les champs obligatoires
- Anomalies detectees en < 15 minutes
- Lineage tracable de bout en bout pour chaque prediction

### Competences techniques requises au plus haut niveau

| Domaine | Technologies | Niveau attendu |
|---------|-------------|----------------|
| Streaming | Kafka, Redis Streams, AT Protocol Firehose | Expert — conception d'architectures event-driven |
| Bases NoSQL | MongoDB (sharding, aggregation framework, change streams) | Expert — optimisation pour > 10M documents |
| Orchestration | Airflow, Prefect, Dagster | Avance — conception de DAGs de production |
| Qualite | Great Expectations, dbt tests, DVC | Avance — mise en place de data contracts |
| Python | asyncio, aiohttp, pymongo (async) | Expert — programmation asynchrone performante |
| Monitoring | Prometheus, Grafana, ELK Stack | Avance — dashboards d'observabilite pipeline |

---

## 3. Machine Learning Engineer / NLP Specialist

### Identite du role

| Attribut | Detail |
|----------|--------|
| **Intitule** | Machine Learning Engineer specialise NLP |
| **Seniorite requise** | 5 a 10 ans d'experience dont 3+ en NLP |
| **Rattachement** | Equipe IA / Direction R&D |
| **Certification de reference** | Google ML Engineer, DeepLearning.AI NLP Specialization |

### Pourquoi ce role est obligatoire pour Thumalien

Thumalien repose sur deux modeles d'IA qui sont le coeur de sa proposition de valeur : la detection de fake news (LogReg + TF-IDF, pipeline expert V2) et l'analyse emotionnelle (MLP PyTorch). Le projet a deja rencontre des problemes majeurs que seul un ML Engineer senior aurait anticipes : le biais Reuters (le modele apprenait le style d'ecriture au lieu de la desinformation), le domain shift (F1 excellent sur articles longs mais 77% de faux suspects sur Bluesky), et le choix du seuil de decision (0.44 au lieu de 0.50).

Ces problemes sont classiques en NLP applique, mais ils sont subtils et destructeurs. Un ML Engineer de haut niveau les previent avant qu'ils n'impactent la production.

### Missions detaillees

#### Mission 1 — Conception et optimisation des modeles de detection

**Objectif** : Porter le pipeline de detection de fake news au niveau de l'etat de l'art tout en preservant l'interpretabilite et la frugalite qui font la force du projet.

**Taches** :
- Auditer le pipeline V2 actuel (TF-IDF 30K features + 12 features linguistiques + 7 emotions → LogReg) et identifier les axes d'amelioration quantifies
- Concevoir le pipeline V3 avec Sentence-Transformers (all-MiniLM-L6-v2 ou multilingual-e5-base) : encoder les textes en vecteurs denses de 384 dimensions qui capturent la semantique, pas juste la frequence des mots
- Implementer une architecture hybride : embeddings semantiques + features linguistiques artisanales. Les features linguistiques (caps_ratio, sensationalism_score, punct_density) capturent des signaux structurels que les embeddings seuls ratent
- Concevoir un systeme d'ensemble : combiner les predictions du pipeline TF-IDF V2 et du pipeline V3 embeddings par stacking ou blending pour maximiser la robustesse
- Mener des experimentations rigoureuses avec cross-validation stratifiee, analyse des erreurs par type de texte (court/long, FR/EN, politique/sante)
- Optimiser le seuil de decision par une approche systematique : courbe precision-recall, analyse du cout metier des faux positifs vs faux negatifs, calibration de Platt

**Indicateurs de performance** :
- F1 > 0.85 sur textes courts (< 30 mots), contre 0.80 actuellement
- Maintien du F1 > 0.98 sur articles longs (pas de regression)
- Temps d'inference < 50ms par texte pour le serving temps reel

#### Mission 2 — Modele d'analyse emotionnelle avance

**Objectif** : Faire evoluer le MLP PyTorch actuel vers un modele plus performant et mieux calibre.

**Taches** :
- Evaluer le MLP actuel (Embedding 25K x 64 → FC 64→48→24 → 7 classes) sur des benchmarks standards (GoEmotions, SemEval)
- Concevoir un modele base sur des embeddings pre-entraines multilingues (XLM-RoBERTa distille) avec fine-tuning sur les 7 emotions cibles
- Implementer une calibration des probabilites (temperature scaling) pour que les scores d'emotions soient des probabilites fiables, pas juste des scores relatifs
- Ajouter la detection de sarcasme et d'ironie, qui sont des vecteurs majeurs de desinformation sur les reseaux sociaux et qui perturbent l'analyse emotionnelle naive
- Creer un dataset d'evaluation bilingue specifique a Bluesky en annotant manuellement 1 000 posts (schema d'annotation inter-annotateurs, score kappa de Cohen)

**Indicateurs de performance** :
- Accuracy > 70% sur les 7 emotions (benchmark GoEmotions)
- Calibration ECE (Expected Calibration Error) < 0.05
- Support effectif des textes FR et EN sans degradation croisee

#### Mission 3 — Gestion du biais et de l'equite

**Objectif** : Garantir que les modeles ne discriminent pas systematiquement certains types de contenus, langues ou communautes.

**Taches** :
- Mener un audit de biais systematique : performance par langue (FR vs EN), par sujet (politique vs sante vs technologie), par longueur de texte
- Identifier et quantifier les biais residuels : le modele est-il plus severe sur les textes en francais ? Sur certains sujets ?
- Implementer des metriques d'equite (Equalized Odds, Demographic Parity) adaptees au contexte de la detection de desinformation
- Concevoir un pipeline de debiasing : reechantillonnage, regularisation adversariale, ou post-processing des scores
- Documenter les biais connus et les limites du modele dans une model card standardisee

**Indicateurs de performance** :
- Ecart de F1 entre FR et EN < 3 points
- Ecart de F1 entre sujets < 5 points
- Model card publiee et mise a jour a chaque version

### Competences techniques requises au plus haut niveau

| Domaine | Technologies | Niveau attendu |
|---------|-------------|----------------|
| NLP classique | TF-IDF, n-grams, feature engineering linguistique | Expert — maitrise fine des parametres et de leurs impacts |
| NLP moderne | Transformers, Sentence-BERT, XLM-RoBERTa | Expert — fine-tuning, distillation, quantization |
| Frameworks ML | scikit-learn, PyTorch, Hugging Face | Expert — contribution-level understanding |
| Evaluation | Cross-validation, ablation studies, error analysis | Expert — methodologie experimentale rigoureuse |
| Multilingue | langdetect, polyglot, FastText language ID | Avance — gestion des specificites linguistiques FR/EN |
| Explicabilite | SHAP, LIME, coefficients LogReg, attention weights | Expert — interpretation et communication des decisions du modele |

---

## 4. Data Scientist Senior — Methodologue et analyste

### Identite du role

| Attribut | Detail |
|----------|--------|
| **Intitule** | Data Scientist Senior / Lead Data Scientist |
| **Seniorite requise** | 5 a 8 ans d'experience |
| **Rattachement** | Equipe Data Science / Direction R&D |
| **Certification de reference** | PhD ou Master en statistiques/ML, publications |

### Pourquoi ce role est obligatoire pour Thumalien

Le Data Scientist est le gardien de la rigueur methodologique. Dans Thumalien, c'est lui qui pose les bonnes questions : le F1 de 0.90 est-il reellement fiable ou est-il gonfle par un desequilibre dans le jeu de test ? Le seuil de 0.44 est-il robuste ou va-t-il deriver dans le temps ? Les 73.4% de posts "fiables" sur Bluesky refletent-ils la realite ou un biais du modele ?

Le rapport du projet montre deja une excellente demarche scientifique (analyse du biais Reuters, GridSearch systematique, comparaison de strategies d'adaptation). Le Data Scientist senior pousse cette rigueur encore plus loin.

### Missions detaillees

#### Mission 1 — Cadrage methodologique et design experimental

**Taches** :
- Definir le protocole experimental pour chaque iteration du modele : hypothese, metriques, dataset de test, critere de succes
- Concevoir la strategie de split train/validation/test en tenant compte des biais temporels (les fake news evoluent dans le temps : un modele entraine sur 2020 ne detecte pas forcement les patterns de 2026)
- Implementer un framework de tests statistiques pour comparer les modeles : test de McNemar pour les classifieurs, bootstrap confidence intervals pour les metriques
- Definir les metriques metier en plus des metriques techniques : quel est le cout reel d'un faux positif (classer un texte fiable comme suspect) vs un faux negatif (laisser passer une fake news) ?
- Valider que le seuil de decision (actuellement 0.44) est optimal par rapport au cout metier, pas seulement au F1

#### Mission 2 — Analyse exploratoire et signal faibles

**Taches** :
- Concevoir des analyses exploratoires avancees sur les 245 000+ posts collectes (collecte continue) : clustering thematique (LDA, BERTopic), detection de communautes, evolution temporelle des sujets
- Identifier les signaux faibles : emergence de nouveaux themes, changement de tonalite emotionnelle sur un sujet, augmentation soudaine du score de suspicion sur un mot-cle
- Creer des rapports d'analyse automatises hebdomadaires : tendances, anomalies, sujets emergents
- Mener des analyses de correlation entre le score de credibilite et les features emotionnelles : les posts suspects sont-ils systematiquement plus emotionnels ? Quelles emotions sont les plus correlees a la desinformation ?

#### Mission 3 — Validation et audit des modeles

**Taches** :
- Realiser des audits de robustesse : comment se comporte le modele face a des attaques adversariales (textes volontairement deformes pour tromper le classifieur) ?
- Tester la calibration du modele : quand il predit 70% de chance d'etre fiable, est-ce vraiment fiable 70% du temps ? (reliability diagram)
- Mener des analyses d'erreurs qualitatives : lire les 100 plus grosses erreurs du modele, les categoriser, en deduire des pistes d'amelioration
- Mesurer la derive du modele dans le temps (concept drift) : les performances se degradent-elles semaine apres semaine a mesure que les sujets d'actualite changent ?
- Proposer une strategie de retraining : a quelle frequence faut-il reentrainer le modele ? Sur quels criteres ?

### Competences techniques requises au plus haut niveau

| Domaine | Technologies | Niveau attendu |
|---------|-------------|----------------|
| Statistiques | Tests d'hypotheses, intervalles de confiance, bootstrap | Expert — rigueur academique |
| Exploration | pandas, matplotlib, seaborn, Plotly | Expert — storytelling par la donnee |
| Topic modeling | LDA, BERTopic, UMAP, HDBSCAN | Avance — extraction de themes non supervises |
| Evaluation ML | Precision-recall, ROC-AUC, calibration, ablation | Expert — au-dela des metriques de surface |
| Communication | Rapports, presentations, vulgarisation | Expert — traduire la complexite en decisions |

---

## 5. MLOps Engineer — Industrialisation des modeles

### Identite du role

| Attribut | Detail |
|----------|--------|
| **Intitule** | MLOps Engineer / ML Platform Engineer |
| **Seniorite requise** | 3 a 6 ans d'experience |
| **Rattachement** | Equipe Infrastructure / Equipe IA |
| **Certification de reference** | Google MLOps, AWS ML Specialty |

### Pourquoi ce role est obligatoire pour Thumalien

Actuellement, le modele Thumalien est un fichier `.pkl` charge manuellement par le dashboard. Il n'y a pas de versioning des modeles, pas de pipeline de retraining automatise, pas de monitoring de la performance en production. Le jour ou le modele commence a deriver (parce que les sujets d'actualite changent, parce que les patterns de desinformation evoluent), personne ne le saura.

Le MLOps Engineer est celui qui transforme un prototype de notebook en un systeme de production fiable et auto-surveillant.

### Missions detaillees

#### Mission 1 — Pipeline de training automatise

**Taches** :
- Mettre en place un pipeline de training reproductible et automatise avec MLflow ou Weights & Biases : chaque run enregistre les hyperparametres, les metriques, les artefacts (modele, vectorizer, scaler)
- Implementer le versioning des modeles : chaque modele entraine recoit un identifiant unique, est teste automatiquement, et n'est promu en production que s'il passe les tests de non-regression
- Creer un pipeline de retraining declenche par des criteres automatiques : derive detectee, nouveau dataset disponible, planification mensuelle
- Automatiser la generation des model cards et des rapports de performance a chaque retraining
- Gerer les artefacts : stockage des modeles (model_expert_v2.pkl, vectorizer_expert_v2.pkl, emotion_model.pth) dans un registre centralisé avec historique complet

#### Mission 2 — Serving et inference en production

**Taches** :
- Concevoir l'architecture de serving : remplacer le chargement direct du `.pkl` dans Streamlit par une API de prediction (FastAPI + model server)
- Implementer le batching des predictions : quand le dashboard charge 2 000 posts, les predictions doivent etre parallelisees et cachees
- Optimiser le temps d'inference : profiler le pipeline (TF-IDF transform + LogReg predict + emotions predict), identifier les goulots d'etranglement
- Mettre en place un A/B testing framework : pouvoir servir 2 versions du modele simultanement et comparer leurs performances sur le trafic reel
- Gerer le fallback : si le modele V3 echoue, revenir automatiquement sur le V2

#### Mission 3 — Monitoring et alerting en production

**Taches** :
- Implementer le monitoring de la derive des donnees (data drift) : la distribution des textes Bluesky entrants change-t-elle par rapport aux donnees d'entrainement ?
- Implementer le monitoring de la derive du modele (concept drift) : la distribution des scores de credibilite evolue-t-elle dans le temps ?
- Creer des dashboards MLOps : taux de prediction par heure, distribution des scores, temps d'inference, erreurs
- Configurer des alertes : si le taux de posts suspects depasse 50% sur 24h (contre 26.6% attendu), declencher une investigation
- Mettre en place des health checks automatises : le modele repond-il ? Ses predictions sont-elles coherentes ?

### Competences techniques requises au plus haut niveau

| Domaine | Technologies | Niveau attendu |
|---------|-------------|----------------|
| ML Tracking | MLflow, W&B, Neptune | Expert — conception de plateformes ML |
| Serving | FastAPI, BentoML, TF Serving, Triton | Avance — optimisation de latence |
| CI/CD ML | GitHub Actions, Jenkins, Argo Workflows | Expert — pipelines de test automatises |
| Monitoring | Prometheus, Grafana, Evidently AI | Expert — detection de derive |
| Conteneurs | Docker, Kubernetes, Helm | Avance — deploiement de services ML |

---

## 6. Developpeur Dashboard / Data Visualization Engineer

### Identite du role

| Attribut | Detail |
|----------|--------|
| **Intitule** | Data Visualization Engineer / Developpeur Full-Stack Data |
| **Seniorite requise** | 3 a 5 ans d'experience |
| **Rattachement** | Equipe Produit / Equipe Data |
| **Certification de reference** | Certifications Tableau/Plotly, UX Design |

### Pourquoi ce role est obligatoire pour Thumalien

Le dashboard est l'interface entre l'intelligence du systeme et ses utilisateurs. Un modele IA aussi performant soit-il est inutile si sa restitution est confuse, lente ou peu lisible. Le dashboard Streamlit actuel offre deja 3 pages (Vue Globale, Analyse temps reel, Metriques & Transparence), mais un specialiste de la visualisation transforme cette interface en un veritable outil d'aide a la decision.

### Missions detaillees

#### Mission 1 — Conception de l'experience utilisateur analytique

**Taches** :
- Mener des interviews utilisateurs (analystes OSINT, journalistes, fact-checkers) pour comprendre leurs besoins reels : quelles questions posent-ils aux donnees ? Quels workflows suivent-ils ?
- Concevoir des wireframes et prototypes pour chaque vue du dashboard, en respectant les principes de data visualization d'Edward Tufte : maximiser le ratio donnees/encre, eliminer le chartjunk
- Implementer des interactions avancees : drill-down sur un post suspect pour voir les features qui ont contribue a son score, filtrage temporel fluide, comparaison de periodes
- Creer un systeme d'alertes visuelles : mettre en evidence les anomalies (pic de posts suspects, emergence d'un nouveau sujet) sans surcharger l'interface
- Optimiser l'accessibilite : contraste, taille de police, navigation clavier, support ecran large et mobile

#### Mission 2 — Visualisations avancees et storytelling

**Taches** :
- Concevoir des visualisations adaptees a chaque type de donnee :
  - **Timeline interactive** : evolution du ratio fiable/suspect au fil du temps avec annotations des evenements marquants
  - **Heatmap emotionnelle** : croisement emotions x sujets x temps pour identifier les combinaisons a risque
  - **Network graph** : relations entre termes de recherche, emotions dominantes et score de credibilite
  - **Radar multi-dimensionnel** : profil emotionnel d'un post ou d'un sujet sur les 7 emotions
- Implementer l'explicabilite visuelle : quand un utilisateur clique sur un post, afficher les top features (mots TF-IDF, features linguistiques) qui expliquent le score
- Creer des rapports exportables (PDF, CSV) pour les analyses qui doivent etre partagees en dehors du dashboard

#### Mission 3 — Performance et architecture frontend

**Taches** :
- Optimiser les requetes MongoDB depuis Streamlit : pagination, caching (`st.cache_data`, `st.cache_resource`), pre-calcul des aggregations
- Implementer le chargement progressif : afficher les metriques principales immediatement, charger les details en arriere-plan
- Gerer l'etat applicatif proprement via `session_state` : eviter les recalculs inutiles, maintenir les filtres utilisateur entre les pages
- Concevoir une architecture modulaire : chaque composant du dashboard (graphique, KPI, tableau) est un composant reutilisable et testable

### Competences techniques requises au plus haut niveau

| Domaine | Technologies | Niveau attendu |
|---------|-------------|----------------|
| Dashboard | Streamlit, Dash, Panel | Expert — composants customs, optimisation |
| Visualisation | Plotly, D3.js, Altair | Expert — visualisations sur mesure |
| UX/UI | Figma, principes Tufte, accessibilite | Avance — design centre utilisateur |
| Backend | MongoDB aggregation, caching, pagination | Avance — optimisation des performances |
| Python | pandas, numpy, async | Avance — manipulation de donnees efficace |

---

## 7. DevOps / Cloud Architect

### Identite du role

| Attribut | Detail |
|----------|--------|
| **Intitule** | DevOps Engineer / Cloud Infrastructure Architect |
| **Seniorite requise** | 5 a 8 ans d'experience |
| **Rattachement** | Direction Infrastructure / CTO |
| **Certification de reference** | AWS Solutions Architect, CKA (Kubernetes), Terraform Associate |

### Pourquoi ce role est obligatoire pour Thumalien

Thumalien repose sur une architecture Docker Compose avec 4 services (MongoDB, Collector, Jupyter, Streamlit). C'est adapte au developpement et au prototypage, mais insuffisant pour une production serieuse. Un DevOps senior transforme cette architecture en un systeme resilient, securise, scalable et observable.

### Missions detaillees

#### Mission 1 — Infrastructure as Code et conteneurisation avancee

**Taches** :
- Migrer de Docker Compose a Kubernetes (ou Docker Swarm pour une premiere etape) pour beneficier de l'auto-scaling, du self-healing et du rolling update
- Ecrire l'infrastructure en code (Terraform ou Pulumi) : chaque composant est reproductible, versionne et auditable
- Optimiser les images Docker : multi-stage builds pour reduire la taille (l'image Python avec TensorFlow + PyTorch + scikit-learn peut depasser 5 GB), separation des images par service
- Implementer des health checks et liveness probes pour chaque service : si MongoDB ne repond plus, Kubernetes le redemarrera automatiquement
- Gerer les secrets proprement : les identifiants Bluesky (`.env`) doivent etre stockes dans un vault (HashiCorp Vault, AWS Secrets Manager), pas dans des fichiers

#### Mission 2 — CI/CD et automatisation

**Taches** :
- Concevoir le pipeline CI/CD complet : a chaque push sur `main`, les tests unitaires tournent, les images Docker sont reconstruites, le deploiement est automatique si les tests passent
- Implementer des tests d'integration : le pipeline doit verifier que le collecteur se connecte bien a MongoDB, que le modele charge et predit correctement, que le dashboard s'affiche
- Automatiser le deploiement blue/green : la nouvelle version du dashboard est deployee en parallele de l'ancienne, le trafic est bascule progressivement
- Mettre en place des rollbacks automatiques : si la nouvelle version a un taux d'erreur superieur a 1%, retour automatique a la version precedente

#### Mission 3 — Securite et conformite

**Taches** :
- Auditer la securite de l'infrastructure : scans de vulnerabilites sur les images Docker (Trivy, Snyk), analyse des dependances Python (safety, pip-audit)
- Implementer le chiffrement : donnees au repos (MongoDB encryption at rest), donnees en transit (TLS entre les services), secrets en vault
- Gerer les acces : RBAC (Role-Based Access Control) pour les differents utilisateurs du dashboard et de l'API
- Assurer la conformite RGPD : les posts Bluesky collectes contiennent des donnees personnelles (handles, textes), un mecanisme de suppression sur demande doit etre implemente
- Mettre en place des logs d'audit : qui a accede a quoi, quand, pourquoi

### Competences techniques requises au plus haut niveau

| Domaine | Technologies | Niveau attendu |
|---------|-------------|----------------|
| Conteneurs | Docker, Kubernetes, Helm | Expert — architecture de production |
| IaC | Terraform, Pulumi, Ansible | Expert — infrastructure reproductible |
| CI/CD | GitHub Actions, ArgoCD, Jenkins | Expert — pipelines complexes |
| Cloud | AWS/GCP/Azure (au moins un) | Expert — services manages (EKS, Atlas, CloudRun) |
| Securite | Vault, TLS, RBAC, scanning | Expert — securite by design |
| Monitoring | Prometheus, Grafana, ELK, Jaeger | Expert — observabilite complete |

---

## 8. Chef de Projet Data / Product Owner

### Identite du role

| Attribut | Detail |
|----------|--------|
| **Intitule** | Chef de Projet Data / Product Owner Intelligence |
| **Seniorite requise** | 7 a 10 ans d'experience dont 3+ en data/IA |
| **Rattachement** | Direction Produit / Direction Generale |
| **Certification de reference** | PMP, Scrum Product Owner (PSPO), CDMP |

### Pourquoi ce role est obligatoire pour Thumalien

Thumalien n'est pas qu'un exercice technique. C'est un outil de detection de desinformation qui touche a des questions ethiques, legales et societales. Le Chef de Projet est celui qui maintient la vision : pourquoi construit-on cet outil ? Pour qui ? Avec quelles limites ? Il arbitre entre les ambitions techniques et les contraintes reelles.

### Missions detaillees

#### Mission 1 — Vision produit et roadmap

**Taches** :
- Definir la vision produit a 6, 12 et 24 mois : quelles fonctionnalites, pour quels utilisateurs, avec quels criteres de succes
- Prioriser le backlog en fonction de l'impact utilisateur et de la faisabilite technique : la V3 avec Sentence-Transformers est-elle plus prioritaire que l'API REST ? L'amelioration du dashboard est-elle plus urgente que le support de nouvelles langues ?
- Mener des etudes utilisateurs : interviews avec des analystes OSINT, des journalistes, des fact-checkers, des chercheurs en sciences de l'information
- Definir les KPI produit : nombre d'utilisateurs actifs, temps moyen passe sur le dashboard, taux de confiance dans les predictions, NPS
- Piloter les sprints et les ceremonies Agile : planning, review, retrospective

#### Mission 2 — Gouvernance des donnees et conformite

**Taches** :
- Assurer la conformite RGPD du projet : les posts Bluesky sont des donnees publiques, mais les handles sont des donnees personnelles. Faut-il les anonymiser ? Les conserver combien de temps ?
- Anticiper le cadre de l'AI Act europeen : Thumalien est-il un systeme d'IA a haut risque ? Quelles obligations de transparence, d'explicabilite et de supervision humaine s'appliquent ?
- Rediger la documentation de conformite : registre de traitement, analyse d'impact (AIPD), politique de retention des donnees
- Definir les limites ethiques du systeme : Thumalien ne doit pas devenir un outil de censure. Comment garantir que le label "suspect" est utilise pour informer, pas pour restreindre ?
- Coordonner avec le DPO (Delegue a la Protection des Donnees) si l'organisation en dispose

#### Mission 3 — Coordination et communication

**Taches** :
- Faciliter la communication entre les profils techniques (Data Engineer, ML Engineer, DevOps) et les parties prenantes non techniques (direction, utilisateurs, partenaires)
- Gerer les risques projet : identifier les dependances critiques (API Bluesky peut changer, datasets d'entrainement peuvent devenir indisponibles), les mitiger
- Produire des rapports d'avancement synthetiques pour la direction
- Organiser les revues de modeles : chaque nouvelle version du modele est presentee, ses performances discutees, ses limites documentees avant mise en production
- Gerer les partenariats externes : fournisseurs de datasets, communaute Bluesky, autres equipes de fact-checking

### Competences requises au plus haut niveau

| Domaine | Competence | Niveau attendu |
|---------|-----------|----------------|
| Gestion de projet | Agile/Scrum, Kanban, OKR | Expert — pilotage de projets data complexes |
| Reglementation | RGPD, AI Act, DSA | Expert — conformite reglementaire |
| Data Literacy | Comprehension des pipelines ML, metriques | Avance — dialoguer avec les equipes techniques |
| Communication | Vulgarisation, presentations, rapports | Expert — pont entre technique et metier |
| Ethique IA | Biais, equite, transparence | Avance — cadrage ethique des systemes IA |

---

## 9. Expert Green IT & IA Responsable

### Identite du role

| Attribut | Detail |
|----------|--------|
| **Intitule** | Expert Green IT / Responsable IA Responsable |
| **Seniorite requise** | 3 a 6 ans d'experience |
| **Rattachement** | Equipe Transverse / RSE |
| **Certification de reference** | INR (Institut du Numerique Responsable), ISO 14001, AI Ethics |

### Pourquoi ce role est obligatoire pour Thumalien

Thumalien integre deja CodeCarbon pour mesurer l'empreinte carbone des entrainements (0.55 g CO2 au total — exemplaire). Mais la demarche Green IT ne se limite pas a la mesure : elle doit guider les choix d'architecture, de modeles et d'infrastructure. Le choix de LogReg plutot que RoBERTa est un choix Green IT autant que technique (10-100x moins d'energie). Cet expert formalise et pousse cette logique.

### Missions detaillees

#### Mission 1 — Mesure et optimisation de l'empreinte environnementale

**Taches** :
- Etendre la mesure CodeCarbon au-dela de l'entrainement : mesurer l'inference (chaque prediction consomme de l'energie), le stockage (MongoDB tourne 24/7), la collecte (requetes API continues)
- Calculer l'empreinte totale du projet : entrainement + inference + infrastructure + stockage + reseau
- Definir un budget carbone pour le projet : combien de g CO2 par mois sont acceptables ? Comment le reduire ?
- Optimiser l'efficacite energetique : batch les predictions au lieu de les faire une par une, eteindre les services non utilises (Jupyter la nuit), compresser les donnees en base
- Comparer l'empreinte de Thumalien avec des solutions equivalentes : combien consommerait un pipeline base sur GPT-4 ou RoBERTa ? Quantifier l'avantage du choix frugal

#### Mission 2 — IA de confiance et explicabilite

**Taches** :
- Implementer SHAP (SHapley Additive exPlanations) pour expliquer chaque prediction : quels mots et features ont le plus contribue au score de credibilite
- Concevoir l'interface d'explicabilite dans le dashboard : quand un utilisateur voit un post classe "suspect", il doit pouvoir comprendre pourquoi en un clic
- Rediger la documentation d'explicabilite exigee par l'AI Act : comment le modele fonctionne, quelles sont ses limites, comment contester une prediction
- Implementer un mecanisme de feedback utilisateur : si un analyste juge qu'une prediction est erronee, cette information est enregistree et utilisee pour ameliorer le modele
- Auditer les biais en collaboration avec le ML Engineer : les populations francophones sont-elles traitees equitablement ? Les sujets sensibles (sante, religion) sont-ils geres sans discrimination ?

#### Mission 3 — Gouvernance et sensibilisation

**Taches** :
- Former l'equipe aux pratiques de numerique responsable : choix d'algorithmes frugaux, optimisation des requetes, dimensionnement adequat de l'infrastructure
- Produire un rapport Green IT trimestriel : evolution de l'empreinte, actions d'optimisation, comparaison avec les objectifs
- Participer aux choix d'architecture avec un regard environnemental : faut-il migrer vers le cloud (plus efficient car mutualise) ou rester on-premise (plus controle mais moins optimise) ?
- Veiller a la conformite avec les standards emergents : AI Act europeen, norme ISO 42001 (IA), Green Software Foundation

### Competences techniques requises au plus haut niveau

| Domaine | Technologies | Niveau attendu |
|---------|-------------|----------------|
| Mesure carbone | CodeCarbon, Green Algorithms, Carbontracker | Expert — mesure et optimisation |
| Explicabilite | SHAP, LIME, ELI5, Captum (PyTorch) | Avance — integration dans les pipelines |
| Ethique IA | Fairness metrics, model cards, datasheets | Avance — audit systematique |
| Reglementation | AI Act, RGPD, normes ISO | Avance — veille reglementaire |
| Eco-conception | Optimisation algorithmique, frugalite | Avance — choix d'architecture verts |

---

## 10. Matrice de responsabilites RACI — Repartition reelle du binome

La matrice RACI definit, pour chaque activite du projet, qui est **R**esponsable (fait le travail), **A**pprobateur (valide), **C**onsulte (donne un avis) et **I**nforme (tenu au courant).

| Activite | Azelie Bernard | Sebastien Lazcanotegui |
|----------|:--------------:|:---------------------:|
| Pipeline collecte Bluesky | **R/A** | I |
| Architecture MongoDB | **R/A** | I |
| Pipeline ML fake news (V1→V9) | **R** | C (GridSearch, debiaisage) |
| Modele emotions (MLP PyTorch) | **R/A** | I |
| GridSearch / optimisation | **R** | **R** (contribution directe) |
| Modeles Transformer (CamemBERT, RoBERTa) | **R/A** | I |
| Dashboard Streamlit (V1→V5) | **R/A** | I |
| Infrastructure Docker | **R/A** | I |
| Gold test set (annotation) | **R** | **R** (2e annotateur, kappa inter-annotateurs) |
| Tests fonctionnels | **R** | C (validation) |
| Conformite RGPD / AI Act | **R** | C (relecture) |
| Documentation technique | **R** | C (revue et relecture) |
| Bilan carbone (CodeCarbon) | **R/A** | I |
| Explicabilite (SHAP) | **R/A** | I |
| Video MVP | **R** | **R** (co-production) |

### 10.1 Projection RACI pour une equipe d'excellence (vision industrielle)

A titre de reflexion professionnelle, voici comment les memes activites se repartiraient dans une equipe complete et specialisee :

| Activite | Data Eng. | ML Eng. | Data Sci. | MLOps | Dashboard | DevOps | Chef Projet | Green IT |
|----------|:---------:|:-------:|:---------:|:-----:|:---------:|:------:|:-----------:|:--------:|
| Pipeline collecte | **R** | I | C | C | I | A | I | C |
| Architecture MongoDB | **R** | C | C | I | C | A | I | I |
| Modele fake news | C | **R** | A | C | I | I | I | C |
| Modele emotions | I | **R** | A | C | I | I | I | C |
| GridSearch / optim | I | **R** | A | I | I | I | I | I |
| MLOps / serving | C | C | I | **R** | I | A | I | I |
| Dashboard UI/UX | I | C | C | I | **R** | I | A | C |
| Visualisations | I | C | **R** | I | **R** | I | A | I |
| Infrastructure Docker | C | I | I | C | I | **R** | I | C |
| CI/CD | I | I | I | C | I | **R** | I | I |
| Securite / RGPD | I | I | I | I | I | **R** | A | C |
| Conformite AI Act | I | C | C | I | I | I | **R** | **R** |
| Bilan carbone | I | C | I | I | I | C | I | **R** |
| Explicabilite | I | **R** | C | I | C | I | A | **R** |
| Roadmap produit | C | C | C | C | C | C | **R** | C |

---

## 11. Reflexion — La synergie comme condition de l'excellence

### Ce que le projet Thumalien nous enseigne

Thumalien, dans sa forme actuelle, est un projet remarquable par sa completude : un pipeline de bout en bout, de la collecte temps reel a la visualisation, en passant par deux modeles d'IA, le tout conteneurise et mesure en empreinte carbone. Le fait qu'il ait ete realise par un binome dans le cadre d'un Master — Azelie assurant le developpement technique et Sebastien la validation qualite — en temoigne d'une organisation efficace et d'une polyvalence forte.

Mais c'est precisement cette polyvalence qui revele la necessite des roles specialises. Chaque defi rencontre dans le projet — le biais Reuters, le domain shift, le choix du seuil 0.44, la gestion du multilingue — est un probleme classique pour le specialiste concerne, mais un obstacle potentiellement bloquant pour un generaliste qui decouvre le probleme en contexte.

### La hierarchie invisible des competences

Dans un projet d'excellence, les roles ne sont pas simplement juxtaposes : ils s'emboitent selon une logique de dependances qui revele leur importance relative.

```
                    Chef de Projet
                   (Vision & cadrage)
                         |
            +------------+------------+
            |                         |
      Data Engineer              ML Engineer
    (Donnees fiables)         (Modeles performants)
            |                         |
            +-----+     +------+-----+
                  |     |      |
               Data Scientist  MLOps
              (Rigueur)      (Production)
                  |              |
            +-----+------+------+
            |            |
        Dashboard    Green IT
       (Restitution) (Responsabilite)
```

**Le Data Engineer et le ML Engineer sont les deux piliers** : sans donnees fiables, le meilleur modele est inutile ; sans modele performant, les meilleures donnees sont inexploitees. Le Data Scientist garantit que le lien entre les deux est rigoureux. Le MLOps transforme le prototype en produit. Le Dashboard rend le produit utilisable. Le Green IT rend le tout responsable. Et le Chef de Projet s'assure que l'ensemble converge vers un objectif clair.

### L'excellence n'est pas la perfection technique — c'est l'adequation

Un projet d'excellence n'est pas celui qui utilise les technologies les plus avancees. C'est celui qui fait les bons choix pour le bon contexte. Thumalien l'illustre parfaitement :

- **Choisir LogReg plutot que RoBERTa** n'est pas un compromis, c'est un choix d'ingenierie : F1 comparable, 100x moins d'energie, interpretabilite totale, deploiement trivial
- **Choisir un seuil de 0.44 plutot que 0.50** n'est pas un hack, c'est une calibration metier : optimiser pour le cas d'usage reel (textes courts Bluesky), pas pour une metrique abstraite
- **Ajouter 3 datasets sociaux** pour resoudre le domain shift n'est pas du bricolage, c'est de l'ingenierie des donnees d'entrainement au service du cas d'usage

Chaque professionnel decrit dans ce document doit porter cette philosophie : **la meilleure solution est celle qui resout le probleme de l'utilisateur, pas celle qui impressionne le plus sur un CV**.

### Le facteur humain

Au-dela des competences techniques, l'excellence d'une equipe data repose sur des qualites humaines souvent sous-estimees :

- **La curiosite intellectuelle** : comprendre pourquoi un modele echoue, pas juste comment le corriger
- **L'humilite methodologique** : remettre en question ses propres resultats, chercher les failles avant les autres
- **La communication** : un ML Engineer qui ne sait pas expliquer son modele a un non-technique produit un outil que personne n'adoptera
- **L'ethique** : nous construisons un outil qui etiquette des contenus comme "suspects". Cette responsabilite ne doit jamais etre prise a la legere

### Conclusion

Ce document n'est pas une fiche de poste. C'est une vision de ce que signifie porter un projet de Social Media Intelligence au plus haut niveau d'excellence. Chaque role decrit ici est une composante indispensable d'un systeme ou la qualite de l'ensemble depend de la qualite de chaque partie.

Le projet Thumalien a demontre qu'un binome complementaire — Azelie sur le developpement technique, Sebastien sur la validation et la qualite — peut construire un systeme fonctionnel et intelligent de bout en bout. L'etape suivante — transformer ce prototype en un produit d'excellence industrielle — necessiterait une equipe ou chaque membre apporte une expertise profonde dans son domaine, tout en comprenant suffisamment les autres domaines pour collaborer efficacement.

**L'excellence n'est pas un etat, c'est une discipline collective.**

---

*Document genere dans le cadre du projet Thumalien — Master Big Data — Avril 2026*
