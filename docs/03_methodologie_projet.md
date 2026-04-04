# Methodologie et Gouvernance du Projet
## Cadrage methodologique, cycle de vie et processus qualite

**Reference** : METH-THUM-2026-001
**Version** : 1.0
**Date** : Avril 2026
**Chef de Projet** : Direction Projet Data

---

## 1. Cadre methodologique

### 1.1 Methodologie retenue : Agile adaptee Data/IA

Le projet Thumalien suit une methodologie **Agile adaptee aux projets Data/IA**, inspiree du framework CRISP-DM (Cross-Industry Standard Process for Data Mining) combine avec des pratiques Scrum pour la gestion iterative.

**Pourquoi cette approche** :
- Les projets IA sont par nature experimentaux : on ne sait pas a l'avance quel modele, quel dataset ou quel seuil donnera les meilleurs resultats
- L'approche iterative permet de livrer de la valeur incrementalement (V1 → V1.5 → V2 → V3)
- Chaque iteration est un cycle complet : donnees → modele → evaluation → decision

### 1.2 Phases CRISP-DM appliquees a Thumalien

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    1. COMPREHENSION        2. COMPREHENSION                 │
│       METIER                  DES DONNEES                   │
│    (Detecter les           (Explorer Bluesky,               │
│     fake news)              auditer ISOT)                   │
│         │                       │                           │
│         └──────────┬────────────┘                           │
│                    │                                        │
│              3. PREPARATION                                 │
│                 DES DONNEES                                 │
│              (Nettoyage Reuters,                            │
│               integration V2)                               │
│                    │                                        │
│              4. MODELISATION                                │
│              (TF-IDF + LogReg,                              │
│               MLP emotions,                                 │
│               GridSearch)                                   │
│                    │                                        │
│              5. EVALUATION                                  │
│              (CV 5-fold, ablation,                          │
│               test Bluesky)                                 │
│                    │                                        │
│              6. DEPLOIEMENT                                 │
│              (Docker, Dashboard,                            │
│               API future)                                   │
│                    │                                        │
│                    └──── Retour Phase 1 (iteration) ────→   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Cycles iteratifs du projet

| Iteration | Periode | Objectif | Livrable |
|-----------|---------|----------|----------|
| **V1.0** | Dec 2025 | Baseline : TF-IDF + LogReg anglais uniquement | F1 = 0.99 (biaise par Reuters) |
| **V1.5** | Jan-Fev 2026 | Correction biais, bilingue, emotions, features linguistiques | F1 = 0.986, pipeline expert complet |
| **V2** | Fev 2026 | Integration datasets sociaux, adaptation textes courts | F1 = 0.897, 73.4% fiable sur Bluesky |
| **V3** | Mars 2026 | Correction features linguistiques (bug preprocessing) | F1 = 0.900, Precision +19.3% |
| **V4** | Avril 2026 | Amelioration FR court : augmentation, 15 features, vocabulaire enrichi | F1 global = 0.905, FR court F1 = 0.86 (+32%) |
| **V5** (en cours) | Avril 2026 | Fine-tuning CamemBERT pour FR court | F1 FR court cible > 0.92 |

---

## 2. Gouvernance du projet

### 2.1 Organisation de l'equipe

```
                    ┌──────────────────┐
                    │  Chef de Projet  │
                    │  (Pilotage,      │
                    │   conformite,    │
                    │   arbitrages)    │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────┴──────┐  ┌─────┴──────┐  ┌──────┴───────┐
    │ Pole Data    │  │ Pole IA    │  │ Pole Infra   │
    │              │  │            │  │              │
    │ Data Engineer│  │ ML Engineer│  │ DevOps       │
    │              │  │ Data Sci.  │  │ Dashboard Dev│
    │              │  │ MLOps      │  │              │
    │              │  │ Green IT   │  │              │
    └──────────────┘  └────────────┘  └──────────────┘
```

### 2.2 Instances de gouvernance

| Instance | Frequence | Participants | Objectif |
|----------|-----------|-------------|----------|
| **Daily standup** | Quotidien (15 min) | Toute l'equipe | Synchronisation, blocages |
| **Sprint review** | Bi-hebdomadaire | Toute l'equipe + parties prenantes | Demo des livrables, feedback |
| **Sprint planning** | Bi-hebdomadaire | Toute l'equipe | Planification des taches |
| **Retrospective** | Bi-hebdomadaire | Equipe technique | Amelioration continue |
| **Comite de pilotage** | Mensuel | Chef de projet + direction | Avancement, risques, budget |
| **Revue de modele** | A chaque nouvelle version | ML Eng. + Data Sci. + Chef projet | Validation des performances |
| **Audit de conformite** | Trimestriel | Chef projet + DPO | Conformite RGPD/AI Act |

### 2.3 Processus de decision

| Type de decision | Decideur | Processus |
|-----------------|----------|-----------|
| Choix d'architecture technique | Data Engineer + DevOps | Proposition → revue → validation Chef de projet |
| Choix de modele/algorithme | ML Engineer + Data Scientist | Experimentation → metriques → revue de modele → validation |
| Seuil de decision (0.44, etc.) | Data Scientist + Chef de projet | Analyse quantitative → impact metier → validation |
| Ajout de dataset | ML Engineer + Data Scientist | Evaluation qualite → test integration → validation |
| Mise en production | MLOps + Chef de projet | Tests de non-regression → validation → deploiement |
| Conformite/ethique | Chef de projet | Analyse d'impact → avis DPO → decision |

---

## 3. Processus qualite

### 3.1 Qualite des donnees

La qualite des donnees est le fondement de tout projet IA. Un modele entraine sur des donnees biaisees produira des predictions biaisees (principe "Garbage In, Garbage Out").

#### Controles a l'ingestion

| Controle | Methode | Frequence | Responsable |
|----------|---------|-----------|-------------|
| Completude des champs | Validation du schema JSON (tous les champs requis presents) | Chaque insertion | Data Engineer |
| Deduplication | Index unique sur `uri` dans MongoDB | Chaque insertion | Data Engineer |
| Detection de langue | langdetect sur les 500 premiers caracteres | Chaque insertion | Pipeline |
| Longueur minimale | Rejet des textes < 3 mots | Chaque insertion | Pipeline |
| Volume d'ingestion | Monitoring du nombre de posts/heure | Continu | Data Engineer |

#### Controles sur les datasets d'entrainement

| Controle | Methode | Resultat Thumalien |
|----------|---------|-------------------|
| Biais d'agence (Reuters) | `audit_reuters_leakage()` | Detecte (99.2%) et corrige |
| Equilibre des classes | Comptage par label | True: 63.1%, Fake: 36.9% (V4, 187K textes) |
| Distribution linguistique | Comptage par langue | EN: 72%, FR: 28% (avant oversampling) |
| Distribution de longueur | Histogramme des word counts | Bimodale : articles (340 mots) + social (20 mots) |
| Corruption CSV | Detection des lignes malformees | 39 lignes corrigees dans Fake.csv |
| Coherence des labels | Verification manuelle d'un echantillon | 50 textes verifies, 96% de coherence |

### 3.2 Qualite des modeles

#### Protocole d'evaluation standard

Chaque version du modele est evaluee selon le protocole suivant :

```
1. CROSS-VALIDATION (5-fold stratifie)
   - Metriques : F1, Precision, Recall, Accuracy
   - Stratification par label ET par langue
   - Resultat : intervalle de confiance a 95%

2. HOLDOUT TEST (20% des donnees, jamais vus en entrainement)
   - Metriques : F1, Precision, Recall, Accuracy, matrice de confusion
   - Analyse par segment : langue (FR/EN), longueur (court/long), sujet

3. TEST OPERATIONNEL (posts Bluesky reels)
   - Echantillon : 2 000 posts reels
   - Metriques : distribution des scores, % fiable/suspect
   - Comparaison avec les attendus (70-80% fiable pour un flux non cible)

4. ANALYSE D'ERREURS
   - Les 100 plus grosses erreurs sont lues manuellement
   - Categorisation : faux positifs (fiable classe suspect) et faux negatifs
   - Identification de patterns communs → pistes d'amelioration

5. ABLATION STUDY
   - Impact de chaque groupe de features (TF-IDF, linguistiques, emotions)
   - Identification des contributions marginales
```

#### Criteres de promotion en production

Un modele ne peut etre deploye en production que s'il satisfait TOUS les criteres suivants :

| Critere | Seuil | V4 actuel |
|---------|-------|-----------|
| F1 CV global | >= 0.85 | 0.905 |
| F1 holdout | >= 0.85 | 0.90 |
| Accuracy holdout | >= 85% | 93% |
| F1 EN test | >= 0.85 | 0.988 |
| F1 FR test | >= 0.75 | 0.985 |
| Ecart F1 FR-EN | < 15 points | 0.3 point |
| % fiable sur Bluesky | 60-85% | 73.4% |
| Pas de regression vs version precedente | F1 articles longs stable | 0.988 (identique V1.5) |
| Empreinte carbone | < 1 g CO2 | 0.30 g |
| Temps d'inference | < 100ms/texte | ~50ms |

### 3.3 Qualite du code

| Pratique | Implementation | Responsable |
|----------|---------------|-------------|
| Structure modulaire | `src/collection/`, `src/pipeline/`, `dashboard/` | Toute l'equipe |
| Versioning Git | Commits reguliers, messages descriptifs | Toute l'equipe |
| Documentation | Docstrings, notebooks commentes, README | Toute l'equipe |
| Tests | Tests d'integration (pipeline charge et predit) | ML Engineer + DevOps |
| Code review | Revue par un pair avant merge | Toute l'equipe |
| Environnement reproductible | requirements.txt, Docker, random_state=42 | DevOps + ML Engineer |

---

## 4. Gestion des risques

### 4.1 Registre des risques

| ID | Risque | Probabilite | Impact | Criticite | Mitigation |
|----|--------|:-----------:|:------:|:---------:|-----------|
| R01 | Changement de l'API Bluesky (breaking change) | Moyenne | Eleve | Haute | Veille sur les releases AT Protocol. Version pinee de la librairie atproto |
| R02 | Derive du modele (concept drift) | Elevee | Moyen | Haute | Monitoring mensuel de la distribution des scores. Retraining planifie |
| R03 | Biais non detecte dans les predictions | Moyenne | Eleve | Haute | Audit de biais trimestriel. Diversification continue des datasets |
| R04 | Panne infrastructure (MongoDB, Docker) | Faible | Eleve | Moyenne | Volumes persistants, restart: always dans Docker Compose |
| R05 | Depassement de la capacite de stockage | Faible | Moyen | Faible | Monitoring de l'espace disque. Politique de retention 12 mois |
| R06 | Demande d'effacement RGPD massive | Tres faible | Moyen | Faible | Procedure d'effacement automatisee et documentee |
| R07 | Performance insuffisante sur nouvelle thematique | Moyenne | Moyen | Moyenne | Datasets complementaires thematiques. Retraining cible |
| R08 | Depart d'un membre cle de l'equipe | Faible | Eleve | Moyenne | Documentation exhaustive. Bus factor > 1 pour chaque composant |
| R09 | Non-conformite reglementaire (evolution RGPD/AI Act) | Moyenne | Eleve | Haute | Veille reglementaire trimestrielle. Audit de conformite |
| R10 | Mauvaise interpretation des scores par les utilisateurs | Moyenne | Eleve | Haute | Formation utilisateurs. Mentions explicites des limites dans le dashboard |

### 4.2 Plan de continuite d'activite (PCA)

| Scenario | Impact | Action | RTO | RPO |
|----------|--------|--------|-----|-----|
| Panne MongoDB | Arret collecte + dashboard | Redemarrage conteneur. Donnees sur volume persistant | 5 min | 0 (pas de perte) |
| Panne collecteur | Arret collecte (pas d'impact dashboard) | Redemarrage automatique (`restart: always`) | 1 min | 5 min de posts perdus |
| Corruption modele .pkl | Predictions erronees | Chargement du modele precedent (V1.5 en fallback) | 2 min | 0 |
| Perte de la machine | Perte totale | Restauration depuis Git (code) + backup MongoDB | 1 jour | Dernier backup |
| API Bluesky indisponible | Arret collecte | Attente avec retry. Dashboard fonctionne sur donnees existantes | N/A | Periode d'indisponibilite |

**RTO** = Recovery Time Objective (temps maximal pour reprendre le service)
**RPO** = Recovery Point Objective (perte de donnees maximale acceptable)

---

## 5. Planification et jalons

### 5.1 Historique des jalons (realises)

| Jalon | Date | Livrable | Statut |
|-------|------|----------|--------|
| J0 — Cadrage projet | Dec 2025 | Specification des besoins, choix technologiques | Realise |
| J1 — Collecteur operationnel | Dec 2025 | collect_bluesky.py + MongoDB + Docker | Realise |
| J2 — Baseline V1.0 | Dec 2025 | LogReg anglais, F1=0.99 (biaise) | Realise |
| J3 — Audit qualite | Jan 2026 | Identification biais Reuters, corruption CSV | Realise |
| J4 — Modele emotions | Jan 2026 | MLP PyTorch bilingue, 7 emotions | Realise |
| J5 — Pipeline V1.5 | Fev 2026 | Bilingue + features linguistiques + emotions, F1=0.986 | Realise |
| J6 — GridSearch | Fev 2026 | Optimisation hyperparametres, analyse de robustesse | Realise |
| J7 — Pipeline V2 | Fev 2026 | +3 datasets sociaux, seuil 0.44, 73.4% fiable | Realise |
| J8 — Dashboard V2 | Mars 2026 | 3 pages, glassmorphism, explicabilite | Realise |
| J9 — Pipeline V3 | Mars 2026 | Correction features linguistiques, retraining | Realise |
| J10 — Pipeline V4 | Avril 2026 | Augmentation FR court, 15 features, F1 FR=0.935 | Realise |
| J11 — CamemBERT FR | Avril 2026 | Fine-tuning CamemBERT pour textes courts FR | En cours |
| J9 — Documentation | Avril 2026 | Rapport, guide utilisateur, CDC, RGPD | En cours |

### 5.2 Jalons futurs (planifies)

| Jalon | Date cible | Livrable | Responsable |
|-------|-----------|----------|-------------|
| J10 — API REST | T2 2026 | FastAPI pour l'inference en temps reel | MLOps + DevOps |
| J11 — Monitoring derive | T2 2026 | Detection automatique du concept drift | MLOps + Data Scientist |
| J12 — Pipeline V3 | T3 2026 | Sentence-Transformers, embeddings 384D | ML Engineer |
| J13 — Annotation Bluesky | T3 2026 | 1 000 posts Bluesky annotes manuellement | Data Scientist |
| J14 — Pipeline V4 | T4 2026 | Fine-tuning CamemBERT/RoBERTa | ML Engineer |
| J15 — Production cloud | T4 2026 | Migration vers infrastructure cloud (GCP/AWS) | DevOps |

---

## 6. Indicateurs de suivi (KPI projet)

### 6.1 KPI techniques

| KPI | Methode de mesure | Frequence | Cible | Actuel |
|-----|-------------------|-----------|-------|--------|
| F1-score modele | Cross-validation 5-fold | A chaque retraining | >= 0.85 | 0.905 |
| % fiable sur Bluesky | Distribution des predictions sur 2 000 posts | Mensuel | 60-85% | 73.4% |
| Volume de posts collectes | `db.raw_posts.countDocuments()` | Quotidien | > 1 000/jour | ~2 000/jour |
| Uptime collecteur | Logs de collecte | Mensuel | > 95% | ~90% (estimé) |
| Temps d'inference | Benchmark sur 1 000 textes | A chaque version | < 100ms/texte | ~50ms |
| Empreinte carbone | CodeCarbon | A chaque entrainement | < 1 g CO2 | 0.30 g |
| Taille du dataset d'entrainement | Comptage | A chaque version | > 100 000 | 187 782 |

### 6.2 KPI projet

| KPI | Methode de mesure | Frequence | Cible |
|-----|-------------------|-----------|-------|
| Nombre de notebooks documentes | Comptage | Mensuel | 9 (00-08) |
| Couverture de la documentation | Checklist des livrables | Trimestriel | 100% |
| Nombre de risques ouverts critiques | Registre des risques | Mensuel | 0 |
| Conformite RGPD | Audit trimestriel | Trimestriel | 100% |
| Satisfaction utilisateurs | Enquete (quand applicable) | Semestriel | > 7/10 |

---

## 7. Communication et reporting

### 7.1 Matrice de communication

| Public | Contenu | Format | Frequence | Canal |
|--------|---------|--------|-----------|-------|
| Equipe technique | Avancement, blocages | Standup oral | Quotidien | Reunion |
| Direction | Synthese, risques, decisions | Comite de pilotage | Mensuel | Presentation |
| Utilisateurs | Nouvelles fonctionnalites, limites | Release notes | A chaque version | Email/doc |
| Regulateur | Conformite, mesures | Rapport AIPD | Annuel | Document formel |
| Communaute academique | Resultats, methodologie | Rapport technique | A chaque publication | Document/article |

### 7.2 Modele de rapport d'avancement

```
RAPPORT D'AVANCEMENT — [Mois/Annee]

1. RESUME EXECUTIF (3 lignes max)
2. JALONS ATTEINTS CETTE PERIODE
3. METRIQUES CLES (F1, volume, uptime)
4. RISQUES ET ACTIONS
5. PROCHAINES ETAPES
6. DECISIONS REQUISES
```

---

## 8. Gestion du changement

### 8.1 Processus de changement

Tout changement significatif (nouveau dataset, modification d'architecture, changement de seuil) suit le processus :

```
1. DEMANDE
   - Description du changement propose
   - Justification (donnees, metriques, besoin metier)
   - Impact estime (performance, securite, conformite)

2. ANALYSE
   - Le responsable technique evalue la faisabilite
   - Le Data Scientist evalue l'impact sur les metriques
   - Le Chef de projet evalue l'impact sur le planning

3. DECISION
   - Validation par le Chef de projet
   - Si impact conformite : avis DPO
   - Si impact performance : revue de modele

4. IMPLEMENTATION
   - Branche Git dediee
   - Tests de non-regression
   - Documentation mise a jour

5. VALIDATION
   - Revue par un pair
   - Tests d'acceptation
   - Merge et deploiement
```

### 8.2 Historique des changements majeurs

| Date | Changement | Justification | Impact |
|------|-----------|---------------|--------|
| Dec 2025 | Creation du projet | Besoin initial | N/A |
| Jan 2026 | Nettoyage biais Reuters | F1 biaise a 0.99 par data leakage | F1 reel mesurable |
| Jan 2026 | Passage TensorFlow → PyTorch | Incompatibilite Apple Silicon | Modele emotions fonctionnel |
| Fev 2026 | Ajout features linguistiques (12) | Enrichir le signal au-dela du TF-IDF | +2 points F1 FR |
| Fev 2026 | Integration emotions (7 features) | Signal emotionnel correle a la desinformation | +0.5 points F1 |
| Fev 2026 | 3 datasets sociaux (V2) | Domain shift : 77% suspect sur Bluesky | 73.4% fiable (vs 23%) |
| Mars 2026 | Correction features ling. (V3) | 5/12 features etaient nulles (bug preprocessing) | F1 +0.3%, Precision +19.3% |
| Avril 2026 | Augmentation FR court (V4) | FR court F1=0.65 insuffisant pour Bluesky | FR court F1=0.86 (+32%), FR global F1=0.935 |
| Fev 2026 | Seuil 0.44 (vs 0.50) | Calibration pour textes courts | +7 points de fiabilite Bluesky |
| Mars 2026 | Dashboard glassmorphism | Amelioration UX/UI | Dashboard professionnel |

---

*Document valide par la Direction Projet — Avril 2026*
*Reference : METH-THUM-2026-001 — Version 1.0*
