# Methodologie et Gouvernance du Projet
## Cadrage methodologique, cycle de vie et processus qualite

**Reference** : METH-THUM-2026-001
**Version** : 1.0
**Date** : Avril 2026
**Equipe** : Azelie Bernard (Lead technique), Sebastien Lazcanotegui (Validation & Qualite)

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
| **V5** | Avril 2026 | Integration 10K posts FR sociaux synthetiques, F1 FR court 0.65->0.90 | F1 global = 0.913, F1 FR = 0.944, FR court F1 = 0.904, F1 EN = 0.894 |
| **V6** | Avril 2026 | Modele style-only topic-agnostic (28 features stylistiques) | GradientBoosting CV F1 = 0.830 |
| **V7** | Avril 2026 | Ensemble hybride meta-learner V5+V6 + SHAP | FP reduits de 57 a 25 sur gold set |
| **V8** | Avril 2026 | Integration CamemBERT comme 3e signal semantique | F1 suspect +28% |
| **V9** | Mai 2026 | Pipeline 2 etapes fait/opinion | FP -67% (186 a 62), kappa 0.187 |
| **Collecteur V3** | Mai 2026 | Reequilibrage FR/EN (28 FR + 16 EN), suppression biais emotionnel, inference auto V9 | 245 000+ posts (collecte continue), 100% annotes |

---

## 2. Gouvernance du projet

### 2.1 Organisation de l'equipe

Le projet Thumalien est realise en **binome** dans le cadre d'un Master Big Data. La repartition des responsabilites reflette les competences complementaires des deux membres :

```
┌──────────────────────────────────────────────────────────┐
│                   BINOME PROJET                          │
│                                                          │
│  Azelie Bernard              Sebastien Lazcanotegui      │
│  ─────────────────           ──────────────────────      │
│  Lead technique              Validation & Qualite        │
│  - Pipeline ML (V1→V9)      - Annotation gold test set  │
│  - Collecteur Bluesky        - Revue documentation       │
│  - Dashboard Streamlit       - Tests fonctionnels        │
│  - Infrastructure Docker     - Support GridSearch         │
│  - Documentation technique   - Video MVP                 │
│  - Conformite RGPD/AI Act                                │
└──────────────────────────────────────────────────────────┘
```

Cette organisation en binome compact permet une prise de decision rapide et une communication directe, sans les frais de coordination d'une equipe plus large.

### 2.2 Instances de gouvernance

Adaptees a un binome, les instances de gouvernance privilegient la legerete et l'efficacite sur le formalisme :

| Instance | Frequence | Participants | Objectif |
|----------|-----------|-------------|----------|
| **Point de synchronisation** | Hebdomadaire (~30 min) | Azelie + Sebastien | Avancement, blocages, prochaines priorites |
| **Revue d'iteration** | Bi-mensuelle (~1h) | Azelie + Sebastien | Demo des livrables, feedback mutuel, ajustement du backlog |
| **Bilan mensuel** | Mensuel (~30 min) | Azelie + Sebastien | Synthese d'avancement, risques, arbitrages |
| **Revue de modele** | A chaque nouvelle version | Azelie (presentation) + Sebastien (relecture) | Validation des performances et des metriques |
| **Point conformite** | Ponctuel (a chaque livrable RGPD/AI Act) | Azelie + Sebastien | Verification de la conformite des livrables |

### 2.3 Processus de decision

| Type de decision | Responsable | Processus |
|-----------------|----------|-----------|
| Choix d'architecture technique | Azelie | Proposition → discussion en binome → implementation |
| Choix de modele/algorithme | Azelie | Experimentation → metriques → revue en binome → validation |
| Seuil de decision (0.44, etc.) | Azelie + Sebastien | Analyse quantitative → discussion impact → validation conjointe |
| Ajout de dataset | Azelie | Evaluation qualite → test integration → validation |
| Mise en production | Azelie | Tests de non-regression → validation → deploiement |
| Conformite/ethique | Azelie + Sebastien | Analyse d'impact → relecture croisee → decision |

---

## 3. Processus qualite

### 3.1 Qualite des donnees

La qualite des donnees est le fondement de tout projet IA. Un modele entraine sur des donnees biaisees produira des predictions biaisees (principe "Garbage In, Garbage Out").

#### Controles a l'ingestion

| Controle | Methode | Frequence | Responsable |
|----------|---------|-----------|-------------|
| Completude des champs | Validation du schema JSON (tous les champs requis presents) | Chaque insertion | Azelie |
| Deduplication | Index unique sur `uri` dans MongoDB | Chaque insertion | Azelie |
| Detection de langue | langdetect sur les 500 premiers caracteres | Chaque insertion | Pipeline (automatise) |
| Longueur minimale | Rejet des textes < 3 mots | Chaque insertion | Pipeline (automatise) |
| Volume d'ingestion | Monitoring du nombre de posts/heure | Continu | Azelie |

#### Controles sur les datasets d'entrainement

| Controle | Methode | Resultat Thumalien |
|----------|---------|-------------------|
| Biais d'agence (Reuters) | `audit_reuters_leakage()` | Detecte (99.2%) et corrige |
| Equilibre des classes | Comptage par label | True: 63.1%, Fake: 36.9% (V5, 197 782 textes) |
| Distribution linguistique | Comptage par langue | FR: 86K (43.5%), EN: 112K (56.5%) |
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

| Critere | Seuil | V5 actuel |
|---------|-------|-----------|
| F1 CV global | >= 0.85 | 0.913 |
| F1 holdout | >= 0.85 | 0.91 |
| Accuracy holdout | >= 85% | 93% |
| F1 EN test | >= 0.85 | 0.894 |
| F1 FR test | >= 0.75 | 0.944 |
| Ecart F1 FR-EN | < 15 points | 5.0 points |
| % fiable sur Bluesky | 60-85% | 73.4% |
| Pas de regression vs version precedente | F1 articles longs stable | 0.988 (identique V1.5) |
| Empreinte carbone | < 10 g CO2 (total projet) | 6.14 g |
| Temps d'inference | < 100ms/texte | ~50ms |

### 3.3 Qualite du code

| Pratique | Implementation | Responsable |
|----------|---------------|-------------|
| Structure modulaire | `src/collection/`, `src/pipeline/`, `dashboard/` | Azelie |
| Versioning Git | Commits reguliers, messages descriptifs | Azelie + Sebastien |
| Documentation | Docstrings, notebooks commentes, README | Azelie (redaction), Sebastien (relecture) |
| Tests | Tests d'integration (pipeline charge et predit) | Azelie (dev), Sebastien (validation) |
| Code review | Relecture croisee des livrables critiques | Azelie + Sebastien |
| Environnement reproductible | requirements.txt, Docker, random_state=42 | Azelie |

---

## 4. Gestion des risques

### 4.1 Registre des risques

| ID | Risque | Probabilite | Impact | Criticite | Mitigation | Statut |
|----|--------|:-----------:|:------:|:---------:|-----------|:------:|
| R01 | Changement de l'API Bluesky (breaking change) | Moyenne (30%) | Eleve | Haute | Veille sur les releases AT Protocol. Version pinee de la librairie atproto. Abstraction dans `collect_bluesky.py` pour isoler les appels API | Ouvert — surveille |
| R02 | Derive du modele (concept drift) | Elevee (60%) | Moyen | Haute | Monitoring hebdomadaire (`weekly_score_check.py`) : alerte si delta score moyen > 5%. Retraining planifie semestriel | Mitige — monitoring actif |
| R03 | Biais non detecte dans les predictions | Moyenne (40%) | Eleve | Haute | Audit de biais via gold set (kappa mesure). Analyse F1 par segment (FR/EN, court/long). Diversification continue des datasets | Mitige — gold set V2 |
| R04 | Panne infrastructure (MongoDB, Docker) | Faible (10%) | Eleve | Moyenne | Volumes persistants Docker, `restart: always`, healthchecks Mongo (`mongosh --eval`). PCA documente dans `09_PRA_PCA.md` | Mitige — healthchecks |
| R05 | Depassement de la capacite de stockage | Faible (15%) | Moyen | Faible | Monitoring de l'espace disque. Politique de retention 12 mois. 245K posts = ~200 MB | Accepte |
| R06 | Demande d'effacement RGPD massive | Tres faible (5%) | Moyen | Faible | Procedure d'effacement automatisee documentee dans `02_conformite_RGPD_AI_Act.md` §3.2. Index MongoDB sur `author_handle` | Mitige |
| R07 | Performance insuffisante sur nouvelle thematique (ex: elections, pandemie) | Moyenne (35%) | Moyen | Moyenne | V6 style-only est topic-agnostic par conception. Datasets complementaires thematiques pre-identifies. Active learning sur les FP | Mitige — V6 topic-agnostic |
| R08 | Indisponibilite d'un membre du binome | Faible (10%) | Eleve | Moyenne | Documentation exhaustive (12 documents). Code commente. Tous les modeles et scripts reproductibles | Mitige — documentation |
| R09 | Non-conformite reglementaire (evolution RGPD/AI Act) | Moyenne (30%) | Eleve | Haute | Veille reglementaire (section 10). AIPD complete. Model Card conforme Mitchell 2019. Classification AI Act explicite | Mitige — conformite |
| R10 | Mauvaise interpretation des scores par les utilisateurs | Moyenne (40%) | Eleve | Haute | Explicabilite XAI multi-niveaux (SHAP, decomposition, attention). Avertissement explicite dans le dashboard. Guide utilisateur | Mitige — XAI pipeline |
| R11 | Sur-ajustement du seuil Stage 1 sur le gold set | Moyenne (45%) | Moyen | Moyenne | Seuil 0.40 derive des memes 500 posts que l'evaluation. Un jeu de calibration independant eliminerait le biais. Documente en §26 | Ouvert — risque accepte |
| R12 | Circularite du self-training (domain adaptation) | Realisee (100%) | Moyen | N/A | Self-training V5 sur Bluesky = echec documente (§22). Alternative : annotation humaine (§23). Le risque s'est materialise et a ete traite | Ferme — echec documente |

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
| J11 — Pipeline V5 | Avril 2026 | Integration 10K posts FR sociaux synthetiques, F1 global=0.913, FR court=0.904 | Realise |
| J12 — CamemBERT FR | Avril 2026 | Fine-tuning CamemBERT V1/V2, F1 ultra-court 0.957 | Realise |
| J13 — RoBERTa EN | Avril 2026 | Fine-tuning RoBERTa V1/V2, F1 ultra-court 0.874 | Realise |
| J14 — Pipeline hybride V8 | Avril 2026 | Meta-learner V5+V6+CamemBERT, F1 suspect +28% | Realise |
| J15 — Pipeline V9 cascade | Mai 2026 | Pipeline 2 etapes fait/opinion, FP -67% | Realise |
| J16 — Documentation | Mai 2026 | Rapport, guide utilisateur, CDC, RGPD | Realise |

### 5.2 Perspectives d'evolution

Le projet a atteint le pipeline V9 (cascade fait/opinion + meta-learner V8). Les axes d'amelioration identifies pour une eventuelle poursuite :

| Axe | Description | Priorite |
|-----|-------------|----------|
| API REST | FastAPI pour l'inference en temps reel (remplacer l'appel direct au pipeline) | Haute |
| Monitoring derive | Detection automatique du concept drift via weekly_score_check.py + alertes | Haute |
| Annotation complementaire | Elargir le gold test set (500 → 1 000+ posts) pour des evaluations plus robustes | Moyenne |
| Jeu de calibration independant | Separer calibration (seuil) et evaluation pour eliminer le biais d'optimisation | Moyenne |
| Migration cloud | Deploiement sur infrastructure cloud (GCP/AWS) pour la scalabilite | Faible |

---

## 6. Indicateurs de suivi (KPI projet)

### 6.1 KPI techniques

| KPI | Methode de mesure | Frequence | Cible | Actuel |
|-----|-------------------|-----------|-------|--------|
| F1-score modele | Cross-validation 5-fold | A chaque retraining | >= 0.85 | 0.913 |
| % fiable sur Bluesky | Distribution des predictions sur 2 000 posts | Mensuel | 60-85% | 73.4% |
| Volume de posts collectes | `db.raw_posts.countDocuments()` | Quotidien | > 1 000/jour | ~2 000/jour |
| Uptime collecteur | Logs de collecte | Mensuel | > 95% | ~90% (estimé) |
| Temps d'inference | Benchmark sur 1 000 textes | A chaque version | < 100ms/texte | ~50ms |
| Empreinte carbone | CodeCarbon | A chaque entrainement | < 10 g CO2 (cumul) | 6.14 g |
| Taille du dataset d'entrainement | Comptage | A chaque version | > 100 000 | 197 782 |

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
| Binome | Avancement, blocages, decisions | Point de synchronisation | Hebdomadaire | Visio / presentiel |
| Encadrement Master | Synthese, risques, jalons | Bilan d'avancement | Mensuel | Email / document |
| Utilisateurs | Nouvelles fonctionnalites, limites | Release notes | A chaque version | Email/doc |
| Regulateur | Conformite, mesures | Rapport AIPD | Annuel | Document formel |

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
   - Azelie evalue la faisabilite technique et l'impact sur les metriques
   - Discussion en binome sur l'impact planning et les risques

3. DECISION
   - Validation conjointe en point de synchronisation
   - Si impact conformite : relecture croisee
   - Si impact performance : revue de modele en binome

4. IMPLEMENTATION
   - Branche Git dediee
   - Tests de non-regression
   - Documentation mise a jour

5. VALIDATION
   - Relecture par le second membre du binome
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
| Avril 2026 | Integration 10K posts FR sociaux synthetiques (V5) | FR court F1=0.86 encore insuffisant, dataset FR sous-represente | F1 global=0.913, FR court F1=0.904, FR=0.944, EN=0.894, dataset 197 782 textes (FR=86K/43.5%, EN=112K/56.5%) |
| Fev 2026 | Seuil 0.44 (vs 0.50) | Calibration pour textes courts | +7 points de fiabilite Bluesky |
| Mars 2026 | Dashboard glassmorphism | Amelioration UX/UI | Dashboard professionnel |
| Avril 2026 | V6 style-only topic-agnostic | Biais thematique detecte sur gold set | FP reduits, approche complementaire |
| Avril 2026 | V7 ensemble hybride + SHAP | Combiner V5+V6 avec explicabilite | Meta-learner, transparence totale |
| Avril 2026 | V8 integration CamemBERT | 3e signal semantique pour le francais | F1 suspect +28% |
| Mai 2026 | V9 pipeline 2 etapes fait/opinion | Distinction fait/opinion comme facteur discriminant | FP -67%, kappa 3x |
| Mai 2026 | Reequilibrage collecte (V3 collecteur) | Biais emotionnel (75% joie) et desequilibre FR/EN (87.5% EN) | 28 termes FR + 16 EN, inference auto |
| Mai 2026 | Refactoring Docker professionnel | Architecture non robuste (pas de healthcheck, demarrage non ordonne) | Healthchecks, depends_on conditionnel, PYTHONPATH unifie |
| Mai 2026 | Pipeline XAI complet (`src/explainability/`) | Challenge "explicabilite pourrait aller plus loin" identifie en revue equipe | 6 modules (1 450 LoC) : SHAP global beeswarm/dependence, attention CamemBERT, IG via Captum, decomposition exacte meta-learner V8, faithfulness AOPC. **Uplift AOPC = +0.21 vs random** (cible >+0.10). Model Card formelle MC-THUM-2026-001 |

---

*Document valide par la Direction Projet — Avril 2026*
*Reference : METH-THUM-2026-001 — Version 1.0*
