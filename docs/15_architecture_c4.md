# Architecture C4 -- ThumaCheck

Ce document presente l'architecture de ThumaCheck selon le modele C4 (Context, Containers, Components) de Simon Brown, illustree par des diagrammes Mermaid.

---

## Niveau 1 -- Contexte systeme

Le diagramme de contexte montre ThumaCheck dans son environnement : les utilisateurs humains et les systemes externes avec lesquels il interagit.

- **Analyste Thumalien** : utilisateur principal, consulte le dashboard et lance des analyses.
- **Administrateur** : gere l'infrastructure (Docker, MongoDB, backups).
- **Bluesky (AT Protocol)** : reseau social source des posts collectes.
- **MongoDB** : base de donnees persistante.
- **Kafka** : bus evenementiel (prototype, roadmap scalabilite).

```mermaid
C4Context
    title ThumaCheck -- Diagramme de contexte (Niveau 1)

    Person(analyste, "Analyste Thumalien", "Consulte le dashboard, soumet des textes a analyser")
    Person(admin, "Administrateur", "Deploie, configure et supervise l'infrastructure")

    System(thumacheck, "ThumaCheck", "Pipeline NLP de detection de desinformation bilingue FR/EN")

    System_Ext(bluesky, "Bluesky", "Reseau social decentralise (AT Protocol)")
    SystemDb_Ext(mongodb, "MongoDB", "Base de donnees NoSQL pour le stockage des posts et predictions")
    System_Ext(kafka, "Apache Kafka", "Bus evenementiel pour ingestion scalable (prototype)")

    Rel(analyste, thumacheck, "Consulte via navigateur", "HTTPS")
    Rel(admin, thumacheck, "Administre via CLI / Docker", "SSH / Docker")
    Rel(thumacheck, bluesky, "Collecte des posts", "AT Protocol / HTTPS")
    Rel(thumacheck, mongodb, "Lit et ecrit posts + predictions", "MongoDB Wire Protocol")
    Rel(thumacheck, kafka, "Consomme des messages texte", "Kafka Protocol")
```

---

## Niveau 2 -- Conteneurs

Le diagramme de conteneurs detaille les briques techniques deployables de ThumaCheck, telles que definies dans le `docker-compose.yml`.

| Conteneur | Technologie | Port | Role |
|---|---|---|---|
| Dashboard | Streamlit (Python) | 8501 | Interface utilisateur, 5 pages |
| API | FastAPI (Python) | 8000 | Endpoint REST `/predict`, `/explain`, `/energy` |
| Pipeline NLP | Python (scikit-learn, PyTorch) | -- | Moteur d'inference, modeles V5/V8 |
| Collector Bluesky | Python | -- | Collecte automatique des posts via AT Protocol |
| MongoDB | Mongo 8 | 27017 | Stockage posts, predictions, metriques |
| Backup | Mongo 8 (cron script) | -- | Sauvegarde quotidienne automatisee |
| Kafka Consumer | Python (confluent-kafka) | -- | Prototype d'ingestion evenementielle |
| Jupyter Notebooks | JupyterLab | 8888 | Experimentation ML, entraînement des modeles |

```mermaid
C4Container
    title ThumaCheck -- Diagramme de conteneurs (Niveau 2)

    Person(analyste, "Analyste Thumalien")

    System_Boundary(thumacheck, "ThumaCheck") {
        Container(dashboard, "Dashboard Streamlit", "Python / Streamlit", "Interface utilisateur : 5 pages (Dashboard, Analyse IA, Explorateur, Performance, A propos)")
        Container(api, "API FastAPI", "Python / FastAPI", "Endpoints REST : /predict, /explain, /emotions, /energy")
        Container(pipeline, "Pipeline NLP", "Python / scikit-learn / PyTorch", "Moteur d'inference bilingue FR/EN, modeles V5 + CamemBERT + Meta-learner V8")
        Container(collector, "Collector Bluesky", "Python", "Collecte automatique des posts via AT Protocol + inference IA")
        ContainerDb(mongo, "MongoDB", "Mongo 8", "Stockage des posts, predictions, metriques")
        Container(backup, "Backup MongoDB", "Bash / cron", "Sauvegarde quotidienne avec retention 7 jours")
        Container(kafka_consumer, "Kafka Consumer", "Python / confluent-kafka", "Prototype : consomme des textes depuis un topic Kafka")
        Container(notebooks, "Jupyter Notebooks", "JupyterLab", "Experimentation, entraînement et evaluation des modeles ML")
    }

    System_Ext(bluesky, "Bluesky", "Reseau social (AT Protocol)")
    System_Ext(kafka, "Apache Kafka", "Bus evenementiel (roadmap)")

    Rel(analyste, dashboard, "Consulte", "HTTPS :8501")
    Rel(analyste, api, "Soumet des textes", "HTTPS :8000")
    Rel(dashboard, pipeline, "Appelle les fonctions de prediction", "Import Python")
    Rel(dashboard, mongo, "Lit les statistiques et posts", "PyMongo")
    Rel(api, pipeline, "Appelle ExpertFakeNewsDetector", "Import Python")
    Rel(collector, bluesky, "Collecte les posts", "AT Protocol")
    Rel(collector, pipeline, "Lance l'inference", "Import Python")
    Rel(collector, mongo, "Ecrit posts + predictions", "PyMongo")
    Rel(kafka_consumer, kafka, "Consomme les messages", "Kafka Protocol")
    Rel(kafka_consumer, pipeline, "Lance l'inference", "Import Python")
    Rel(backup, mongo, "Dump quotidien", "mongodump")
    Rel(notebooks, mongo, "Analyse exploratoire", "PyMongo")
```

---

## Niveau 3 -- Composants du Pipeline NLP

Le pipeline NLP est le coeur de ThumaCheck. Il orchestre la detection de langue, l'extraction de features, l'inference par plusieurs modeles, et la production d'explications.

### Architecture du routage

1. Le texte entre dans le `LanguageRouter` qui detecte la langue (FR ou EN).
2. Les features sont extraites en parallele : TF-IDF, linguistiques (17 features), emotions (7 classes).
3. Le modele principal `LogisticRegression V5` produit une prediction de base.
4. Pour les textes courts, un modele Transformer prend le relais en cascade :
   - **CamemBERT** pour les textes FR ultra-courts
   - **RoBERTa EN** pour les textes EN courts
5. Le **Meta-learner V8** combine les signaux (ensemble hybride).
6. Le module **ExplainPrediction** decompose les coefficients pour fournir une explication humaine.

```mermaid
C4Component
    title Pipeline NLP -- Diagramme de composants (Niveau 3)

    Container_Boundary(pipeline, "Pipeline NLP") {
        Component(router, "LanguageRouter", "langdetect", "Detecte la langue du texte : FR ou EN")
        Component(tfidf, "TF-IDF Vectorizer", "scikit-learn", "Vectorisation n-grams avec TF sublineaire")
        Component(linguistic, "LinguisticFeatureExtractor", "Python", "17 features : ponctuation, majuscules, longueur, entites, etc.")
        Component(emotion, "EmotionFeatureExtractor", "PyTorch MLP bilingue", "7 classes d'emotions : colere, degout, joie, neutre, peur, surprise, tristesse")
        Component(logreg, "LogisticRegression V5", "scikit-learn", "Modele principal TF-IDF + features linguistiques")
        Component(camembert, "CamemBERT", "Transformers / PyTorch", "Modele FR pour textes ultra-courts (cascade)")
        Component(roberta, "RoBERTa EN", "Transformers / PyTorch", "Modele EN pour textes courts (cascade)")
        Component(metalearner, "Meta-learner V8", "scikit-learn", "Ensemble hybride : combine V5 + Transformers + emotions")
        Component(explain, "ExplainPrediction", "Python", "Decomposition des coefficients pour explicabilite")
    }

    Rel(router, tfidf, "Texte preprocesse")
    Rel(router, linguistic, "Texte brut")
    Rel(router, emotion, "Texte brut")
    Rel(tfidf, logreg, "Vecteur TF-IDF")
    Rel(linguistic, logreg, "17 features")
    Rel(emotion, metalearner, "7 scores emotions")
    Rel(logreg, metalearner, "Score V5")
    Rel(router, camembert, "Texte FR court")
    Rel(router, roberta, "Texte EN court")
    Rel(camembert, metalearner, "Score CamemBERT")
    Rel(roberta, metalearner, "Score RoBERTa")
    Rel(logreg, explain, "Coefficients + features")
    Rel(metalearner, explain, "Decision finale")
```

---

## Diagramme de flux -- Donnees de bout en bout

Ce diagramme illustre le chemin complet d'un post Bluesky, de sa collecte jusqu'a son affichage dans le dashboard.

```mermaid
flowchart LR
    subgraph Sources
        BS[Bluesky<br/>AT Protocol]
        KF[Kafka Topic<br/>prototype]
        API_IN[API /predict<br/>soumission manuelle]
    end

    subgraph Collection
        COL[Collector Bluesky<br/>Python]
        KC[Kafka Consumer<br/>Python]
    end

    subgraph Inference["Pipeline NLP"]
        direction TB
        LANG[Detection langue<br/>FR / EN]
        FEAT[Extraction features<br/>TF-IDF + 17 linguistiques<br/>+ 7 emotions]
        MOD[Modeles<br/>LogReg V5 + CamemBERT<br/>+ RoBERTa + Meta-learner V8]
        EXP[Explication<br/>decomposition coefficients]
        LANG --> FEAT --> MOD --> EXP
    end

    subgraph Stockage
        MDB[(MongoDB<br/>thumalien_db)]
        BKP[Backup quotidien<br/>retention 7 jours]
    end

    subgraph Presentation
        DASH[Dashboard Streamlit<br/>5 pages<br/>port 8501]
        API_OUT[API FastAPI<br/>JSON response<br/>port 8000]
        NB[Jupyter Notebooks<br/>port 8888]
    end

    BS -->|collecte auto| COL
    KF -->|consommation| KC
    API_IN -->|requete HTTP| API_OUT

    COL --> LANG
    KC --> LANG
    API_OUT --> LANG

    EXP -->|post + prediction| MDB
    MDB --> BKP
    MDB -->|lecture stats| DASH
    MDB -->|analyse exploratoire| NB
    EXP -->|reponse JSON| API_OUT
```

---

## Resume des correspondances code source

| Composant | Chemin source |
|---|---|
| Pipeline NLP | `src/pipeline/expert_detector.py` |
| API FastAPI | `src/api/main.py` |
| Dashboard Streamlit | `dashboard/app.py` |
| Collector Bluesky | `src/collection/` |
| Kafka Consumer | `src/scalability/kafka_consumer.py` |
| Explicabilite | `src/explainability/` |
| Monitoring | `src/monitoring/` |
| Notebooks ML | `notebooks/` |
| Docker Compose | `docker-compose.yml` |
