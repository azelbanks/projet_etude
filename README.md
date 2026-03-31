# 🕵️ Thumalien : Social Media Intelligence & AI Monitor

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker)
![MongoDB](https://img.shields.io/badge/MongoDB-Database-47A248?style=for-the-badge&logo=mongodb)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow)

## 📖 Description du Projet

**Thumalien** est une solution complète de surveillance et d'analyse des réseaux sociaux (Bluesky) en temps réel. Le projet intègre un pipeline Data Engineering complet et deux modèles d'Intelligence Artificielle pour qualifier l'information.

L'objectif est de détecter les potentiels signaux faibles, les **Fake News** et d'analyser l'**ambiance émotionnelle** des discussions en ligne.

### 🚀 Fonctionnalités Clés
* **Collecte en temps réel :** Ingestion continue des posts Bluesky via Websocket.
* **Détection de Fake News :** IA entraînée (Régression Logistique + TF-IDF) pour évaluer la crédibilité (Score 0 à 1).
* **Analyse Émotionnelle (Deep Learning) :** Réseau de neurones (MLP/Keras) classifiant les textes selon 6 émotions (Joie, Tristesse, Colère, Peur, Amour, Surprise).
* **Dashboard Interactif :** Visualisation des données et KPI en temps réel via Streamlit.
* **Green IT :** Monitoring de l'empreinte carbone des calculs IA via CodeCarbon.

---

## 🏗️ Architecture Technique

Le projet repose sur une architecture micro-services conteneurisée avec Docker.

```mermaid
graph LR
    A[📡 Bluesky Network] -->|Stream| B(Container: Collector)
    B -->|JSON| C[(Container: MongoDB)]
    D[Container: Jupyter/AI] -->|Training & Inference| C
    E[Container: Streamlit App] -->|Read & Visualize| C
    style C fill:#47A248,stroke:#333,stroke-width:2px
    style E fill:#FF4B4B,stroke:#333,stroke-width:2px