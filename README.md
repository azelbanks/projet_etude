# Thumalien — Social Media Intelligence & AI Monitor

![Python](https://img.shields.io/badge/Python-3.13+-blue?style=for-the-badge&logo=python)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker)
![MongoDB](https://img.shields.io/badge/MongoDB-Database-47A248?style=for-the-badge&logo=mongodb)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)

## Description du Projet

**Thumalien** est une solution complete de surveillance et d'analyse des reseaux sociaux (Bluesky) en temps reel. Le projet integre un pipeline Data Engineering complet et deux modeles d'Intelligence Artificielle pour qualifier l'information.

L'objectif est de detecter les potentiels signaux faibles, les **Fake News** et d'analyser l'**ambiance emotionnelle** des discussions en ligne.

### Fonctionnalites Cles
* **Collecte en temps reel :** Ingestion continue des posts Bluesky via l'API AT Protocol.
* **Detection de Fake News :** Pipeline NLP bilingue FR/EN (Regression Logistique + TF-IDF + 12 features linguistiques) pour evaluer la credibilite (Score 0 a 1).
* **Analyse Emotionnelle (Deep Learning) :** Reseau de neurones MLP (PyTorch) classifiant les textes selon 7 emotions (Colere, Degout, Joie, Neutre, Peur, Surprise, Tristesse).
* **Dashboard Interactif :** Visualisation des donnees et KPI en temps reel via Streamlit.
* **Green IT :** Monitoring de l'empreinte carbone des calculs IA via CodeCarbon.

### Metriques cles (V2)
* **188 553 posts** collectes depuis decembre 2025
* **145 703 textes** d'entrainement (6 datasets, FR+EN)
* **F1-score** : 0.90 (holdout), seuil de decision : 0.44
* **73.4%** des posts Bluesky classes fiables (vs 23% en V1.5)

---

## Architecture Technique

Le projet repose sur une architecture micro-services conteneurisee avec Docker.

```mermaid
graph LR
    A[Bluesky Network] -->|AT Protocol| B(Container: Collector)
    B -->|JSON| C[(Container: MongoDB)]
    D[Container: Jupyter/AI] -->|Training & Inference| C
    E[Container: Streamlit App] -->|Read & Visualize| C
    style C fill:#47A248,stroke:#333,stroke-width:2px
    style E fill:#FF4B4B,stroke:#333,stroke-width:2px