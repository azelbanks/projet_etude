# Rendu Individuel - Azelie Bernard
## Projet Thumalien - Master Big Data, Sup de Vinci

**Etudiant** : Azelie Bernard
**Role principal** : Lead technique (Data Engineer, ML Engineer, DevOps, Dashboard Dev)
**Date** : Avril 2026

---

## 1. Mon role dans le projet

J'ai assure le role de **lead technique** sur le projet Thumalien, couvrant l'ensemble de la chaine de valeur : de la collecte des donnees sur Bluesky jusqu'au deploiement du dashboard de visualisation, en passant par le developpement des modeles de detection de fake news.

### Responsabilites principales
- Architecture technique et infrastructure Docker
- Developpement du collecteur Bluesky (AT Protocol)
- Pipeline NLP complet (V1.0 a V5.0)
- Fine-tuning des modeles Transformer (CamemBERT, RoBERTa)
- Dashboard Streamlit interactif
- Documentation technique
- Conformite RGPD et AI Act

---

## 2. Contributions techniques detaillees

### 2.1 Infrastructure et collecte (Dec 2025)
- Mise en place de l'architecture Docker Compose (MongoDB, Jupyter, Streamlit, Collecteur)
- Developpement du collecteur Bluesky avec gestion des erreurs et deduplication
- Configuration MongoDB avec index unique et volumes persistants
- **Resultat** : 188 553 posts collectes depuis decembre 2025

### 2.2 Pipeline NLP (Jan - Avril 2026)
- Baseline V1.0 : TF-IDF + LogisticRegression (F1=0.99, identifie le biais Reuters)
- Audit qualite : decouverte du data leakage Reuters (99.2% identifiable par style)
- V1.5 : pipeline bilingue + 12 features linguistiques + modele emotions
- V2.0 : integration 3 datasets sociaux, seuil calibre 0.44
- V3.0 : correction du bug preprocessing (5/12 features nulles)
- V4.0 : augmentation FR court (+32% F1)
- V5.0 : +10K posts synthetiques, F1 global = 0.913
- **22 commits** sur le depot Git

### 2.3 Modeles Transformer (Avril 2026)
- Fine-tuning CamemBERT V1/V2 pour le francais (F1 ultra-court = 0.957)
- Fine-tuning RoBERTa EN V1/V2 pour l'anglais (F1 ultra-court = 0.874)
- Pipeline hybride stacking V5 + CamemBERT V2

### 2.4 Dashboard Streamlit (Mars 2026)
- Design glassmorphism dark theme
- 3 pages : vue globale, prediction live, explicabilite
- Radar charts, metriques temps reel, connexion MongoDB

### 2.5 Modele d'emotions (Jan 2026)
- MLP PyTorch bilingue, 7 classes d'emotions
- Early stopping + class weights pour gerer le desequilibre
- Integration comme features dans le pipeline NLP

---

## 3. Competences mobilisees et acquises

### 3.1 Competences techniques

| Competence | Niveau avant | Niveau apres | Contexte d'application |
|-----------|:------------:|:------------:|----------------------|
| Python avance | Intermediaire | Avance | Pipeline complet, 22 notebooks |
| NLP / Text Mining | Debutant | Avance | TF-IDF, features linguistiques, tokenization |
| Deep Learning (PyTorch) | Debutant | Intermediaire | MLP emotions, fine-tuning Transformers |
| Transformers (Hugging Face) | Debutant | Intermediaire | CamemBERT, RoBERTa, stacking |
| Docker / Docker Compose | Intermediaire | Avance | Architecture micro-services |
| MongoDB | Debutant | Intermediaire | Schema, index, aggregations |
| Streamlit | Debutant | Intermediaire | Dashboard interactif 3 pages |
| Git / GitHub | Intermediaire | Avance | Versioning, merge, collaboration |

### 3.2 Competences transversales

| Competence | Mise en pratique |
|-----------|-----------------|
| Gestion de projet | Planification iterative, priorisation des taches, respect des delais |
| Resolution de problemes | Debugging du biais Reuters, correction preprocessing V3, migration TF->PyTorch |
| Communication technique | Redaction de 10+ documents, 22 notebooks commentes |
| Esprit critique | Analyse des erreurs, tests de significativite, identification des biais |
| Autonomie | Apprentissage de PyTorch, CamemBERT, AT Protocol en autodidacte |

---

## 4. Defis rencontres et solutions

| Defi | Contexte | Solution | Apprentissage |
|------|----------|----------|---------------|
| Biais Reuters (F1=0.99 biaise) | Le modele detectait le style Reuters, pas les fake news | Debiaisage (BODY_AGENCY_TERMS), nettoyage des artefacts | L'evaluation est aussi importante que l'entrainement |
| TensorFlow incompatible M4 | Apple Silicon non supporte par TF | Migration vers PyTorch | Toujours verifier la compatibilite hardware |
| Features nulles (5/12) | Bug de preprocessing dans le pipeline V2 | Debug, correction, retraining V3 | Les tests unitaires sont essentiels |
| F1 FR court = 0.65 | Les textes courts de type reseau social mal classifies | Augmentation + vocabulaire enrichi + CamemBERT | Les modeles generalistes ne suffisent pas pour les textes courts |
| Merge Git avec conflits | Depot recree par un collegue | `--allow-unrelated-histories` + resolution manuelle | Toujours communiquer avant de restructurer un depot |

---

## 5. Analyse critique et recul

### Ce qui a bien fonctionne
- L'approche iterative (V1 -> V5) a permis d'ameliorer continuellement les performances
- La documentation au fil de l'eau (notebooks) facilite la tracabilite
- Le choix de Docker garantit la reproductibilite
- L'ajout de CamemBERT et RoBERTa a resolu le probleme des textes courts

### Ce que j'aurais fait differemment
- Commencer par des tests unitaires des le debut (le bug des features nulles aurait ete detecte plus tot)
- Utiliser MLflow pour le tracking des experiences au lieu de notebooks individuels
- Prevoir une strategie d'annotation manuelle plus tot dans le projet
- Mieux repartir la charge de travail dans l'equipe

### Limites identifiees
- Le modele reste dependant des datasets d'entrainement (pas de fact-checking reel)
- L'absence de donnees annotees specifiquement pour Bluesky est une faiblesse
- Le monitoring en production est minimal (pas de Grafana/Prometheus)

---

## 6. Bilan personnel

Ce projet m'a permis de mener un projet Data/IA de bout en bout, de la collecte au deploiement. J'ai particulierement progresse en NLP et Deep Learning, domaines dans lesquels j'etais debutante en debut de projet. La complexite du probleme (detection de fake news dans un contexte bilingue, multi-longueur) m'a appris que les solutions simples sont souvent les meilleures (TF-IDF + features > modele complexe mal entraine), et que l'iteration constante est la cle de l'amelioration.

La dimension ethique (biais, conformite RGPD, AI Act) a ete particulierement formatrice et sera un atout pour ma carriere professionnelle.

---

*Rendu individuel - Azelie Bernard - Avril 2026*
