# Guide Utilisateur — Thumalien
## Systeme de Detection de Fake News Bilingue FR/EN

---

### 1. Presentation

Thumalien est un systeme d'analyse automatisee de contenus textuels pour la detection de fake news. Il traite les posts publies sur le reseau social Bluesky en francais et en anglais, et fournit pour chaque texte un score de credibilite, une emotion dominante et une classification fiable/suspect.

Le systeme repose sur un pipeline NLP bilingue (V2) combinant des features TF-IDF, linguistiques et emotionnelles, entraine sur 145 703 textes (articles + tweets) pour atteindre un F1-score de 0.90 avec une calibration optimisee pour les textes courts (seuil 0.44).

---

### 2. Prerequis

- Python 3.13+
- pip (gestionnaire de paquets)
- MongoDB (optionnel — le dashboard fonctionne en mode demo sans base de donnees)
- Environ 2 Go d'espace disque (modeles + donnees)

---

### 3. Installation

```bash
# Cloner le depot
git clone https://github.com/azelbanks/projet_etude.git
cd projet_etude

# Installer les dependances
pip install -r requirements.txt

# Verifier l'installation
python -c "from src.pipeline.expert_detector import ExpertFakeNewsDetector; print('OK')"
```

---

### 4. Structure du projet

```
projet_etude/
|-- dashboard/
|   |-- app.py                  # Dashboard Streamlit (interface principale)
|-- data/
|   |-- training/               # Datasets d'entrainement (non versiones)
|   |   |-- Fake.csv            # ISOT Fake News (anglais)
|   |   |-- True.csv            # ISOT True News (anglais)
|   |   |-- kaggle_fr/          # Kaggle FrenchFakeNewsDetector
|   |   |-- train.csv           # Emotions EN (entrainement)
|   |   |-- training.csv        # Emotions FR (entrainement)
|-- docs/
|   |-- guide_utilisateur.md    # Ce fichier
|   |-- architecture.png        # Schema d'architecture du pipeline
|   |-- gantt_planning.png      # Diagramme de Gantt retrospectif
|-- models/
|   |-- model_expert.pkl        # Modele LogReg bilingue
|   |-- tfidf_expert.pkl        # Vectoriseur TF-IDF (30K features)
|   |-- metrics_expert.pkl      # Metriques d'entrainement
|   |-- emotion_bilingual.pt    # MLP PyTorch (7 emotions)
|   |-- emotion_vocab_bilingual.pickle
|   |-- emotion_label_encoder_bilingual.pickle
|-- notebooks/
|   |-- 00_Audit_Qualite_Donnees.ipynb
|   |-- 02_Analyse_Emotions_MLP.ipynb
|   |-- 03_Mise_a_jour_Quotidienne.ipynb
|   |-- 05_Detection_Expert_Bilingue.ipynb
|   |-- 06_Documentation_Technique.ipynb
|   |-- 07_Analyse_Modele_GridSearch.ipynb
|   |-- 08_Integration_Datasets_V2.ipynb
|-- src/
|   |-- pipeline/
|   |   |-- expert_detector.py  # Pipeline complet (classes principales)
|   |-- collection/
|   |   |-- collect_bluesky.py  # Collecte AT Protocol
|   |-- app/
|       |-- main.py             # Point d'entree applicatif
|-- .streamlit/
|   |-- config.toml             # Theme dark + config serveur
|-- docker-compose.yml
|-- dockerfile
|-- requirements.txt
|-- emissions.csv               # Bilan carbone CodeCarbon
```

---

### 5. Lancer le dashboard

```bash
# Depuis la racine du projet
streamlit run dashboard/app.py
```

Le dashboard s'ouvre automatiquement dans le navigateur a l'adresse `http://localhost:8501`.

**Mode demo** : si MongoDB n'est pas connecte, le dashboard affiche des donnees d'exemple (15 posts FR/EN) avec un bandeau informatif.

---

### 6. Pages du dashboard

#### 6.1. Vue Globale

La page d'accueil presente une vision synthetique de l'ensemble des posts analyses :

- **Metriques cles** : nombre total de posts, pourcentage de posts fiables, score de credibilite moyen, repartition FR/EN.
- **Profil emotionnel** : radar chart des 7 emotions moyennes sur l'ensemble du corpus (colere, degout, joie, neutre, peur, surprise, tristesse).
- **Fiabilite par langue** : diagramme en barres horizontales comparant les posts fiables et suspects pour le francais et l'anglais.
- **Tableau des posts** : liste des derniers posts analyses avec leur texte (tronque), langue, label et score de credibilite.

#### 6.2. Analyse en temps reel

Cette page permet d'analyser un texte en temps reel :

1. Collez un texte (article, post Bluesky, tweet, ou tout texte FR/EN) dans la zone de saisie.
2. Cliquez sur le bouton **Analyser**.
3. Le systeme retourne :
   - Un **score de credibilite** (jauge 0 a 1) : 0 = probablement faux, 1 = probablement fiable.
   - Un **verdict** : FIABLE (vert) ou SUSPECT (rouge).
   - L'**emotion dominante** detectee avec sa probabilite.
   - Un **radar chart** detaille des 7 probabilites emotionnelles.
   - La **langue detectee** automatiquement (FR ou EN).

**Interpretation du score** :
- Score > 0.7 : le texte presente des caracteristiques de contenu fiable.
- Score entre 0.4 et 0.7 : zone d'incertitude, verification manuelle recommandee.
- Score < 0.4 : le texte presente des marqueurs de desinformation.

> **Avertissement** : le score est un indicateur probabiliste base sur des patterns statistiques. Il ne constitue pas une verification factuelle et ne doit pas etre utilise comme seul critere de decision.

#### 6.3. Metriques & Transparence

Cette page fournit les indicateurs de performance et de conformite :

- **Ablation study** : tableau et graphique des F1-scores pour les 7 conditions experimentales testees (EN seul, FR seul, bilingue, bilingue + emotions, etc.).
- **Bilan carbone** : emissions CO2 totales du projet mesurees par CodeCarbon.
- **Roadmap** : les 4 versions planifiees du pipeline (V1 a V3) avec leurs objectifs.
- **Conformite** : fiches RGPD et IA Act resumant les mesures de conformite.

---

### 7. Utiliser le pipeline en Python

Le pipeline peut etre utilise directement en Python sans le dashboard :

```python
import sys
sys.path.insert(0, 'src')

from pipeline.expert_detector import ExpertFakeNewsDetector
import pandas as pd

# Charger le modele
detector = ExpertFakeNewsDetector(model_dir='models', use_emotions=True)
detector.load(suffix='expert_v2' if os.path.exists('models/model_expert_v2.pkl') else 'expert')

# Analyser des textes
textes = pd.Series([
    "Le CNRS publie une etude sur le climat.",
    "BREAKING: Secret labs control your mind with 5G!!!",
    "La BCE maintient ses taux directeurs.",
])

resultats = detector.predict(textes)
print(resultats[['text', 'language', 'prediction_label', 'ai_score_credibility']])
```

**Colonnes retournees par `predict()`** :
| Colonne | Description |
|---------|-------------|
| `text` | Texte original |
| `language` | Langue detectee (`fr`, `en`, `other`) |
| `prediction_label` | 0 = Fiable, 1 = Suspect |
| `ai_score_credibility` | Probabilite de fiabilite (0 a 1) |
| `ai_analysis_log` | Log d'analyse lisible (ex: "[FR] Fiable (credibilite: 89%)") |

---

### 8. Utiliser l'analyse d'emotions

```python
from pipeline.expert_detector import EmotionFeatureExtractor

emo = EmotionFeatureExtractor(model_dir='models')
emo.load()

probas = emo.get_emotion_features([
    "Je suis tellement heureux de cette nouvelle !",
    "This is absolutely terrifying news.",
])

# probas.shape = (2, 7) — 7 probabilites par texte
# Classes : colere, degout, joie, neutre, peur, surprise, tristesse
print(probas)
```

---

### 9. Notebooks

Les notebooks documentent chaque etape du projet et sont executes sequentiellement :

| Notebook | Description | Sortie |
|----------|-------------|--------|
| 00 | Audit qualite des donnees d'entrainement | Statistiques, biais Reuters, distributions |
| 02 | Entrainement du MLP emotions bilingue (PyTorch) | `emotion_bilingual.pt` + vocabulaire + label encoder |
| 03 | Pipeline de mise a jour quotidienne | Collecte + inference sur posts recents |
| 05 | Pipeline expert bilingue + ablation study (7 conditions) | `model_expert.pkl` + `tfidf_expert.pkl` + metriques |
| 06 | Documentation technique complete | Limites, roadmap, PRA/PCA, Green IT, conformite |
| 07 | Analyse du modele + GridSearch (36 combinaisons) | Feature importance, learning curves, optimisation |
| 08 | Integration de datasets sociaux (V2) | `model_expert_v2.pkl`, adaptation textes courts |

**Pour re-executer un notebook** :
```bash
jupyter nbconvert --to notebook --execute notebooks/05_Detection_Expert_Bilingue.ipynb \
    --output 05_Detection_Expert_Bilingue.ipynb --ExecutePreprocessor.timeout=600
```

---

### 10. Deploiement Docker

Le projet inclut un `docker-compose.yml` pour le deploiement complet :

```bash
# Lancer l'ensemble de la stack
docker-compose up -d

# Verifier les services
docker-compose ps

# Consulter les logs
docker-compose logs -f
```

**Services** :
- **MongoDB** : stockage des posts collectes et des resultats d'analyse.
- **Collecteur Bluesky** : script de collecte automatisee via le protocole AT.
- **Dashboard Streamlit** : interface web de visualisation (port 8501).

---

### 11. Configuration

#### Variables d'environnement (fichier `.env`)

```
BLUESKY_HANDLE=votre_handle.bsky.social
BLUESKY_APP_PASSWORD=votre_app_password
MONGODB_URI=mongodb://mongodb:27017/
```

#### Theme du dashboard (`.streamlit/config.toml`)

Le theme dark est configure par defaut. Pour modifier les couleurs, editez `.streamlit/config.toml` :

```toml
[theme]
primaryColor = "#00D4FF"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1A1F2E"
textColor = "#E0E0E0"
```

---

### 12. FAQ

**Q : Le dashboard affiche "Mode demo". Comment connecter MongoDB ?**
R : Lancez MongoDB (via Docker ou en local sur le port 27017) et assurez-vous que la base `thumalien_db` contient une collection `raw_posts`. Le dashboard se connecte automatiquement au demarrage.

**Q : Comment re-entrainer le modele avec de nouvelles donnees ?**
R : Executez le notebook 05 (`05_Detection_Expert_Bilingue.ipynb`). Il recharge les CSV depuis `data/training/`, re-entraine le modele et sauvegarde les fichiers `.pkl` dans `models/`.

**Q : Le modele classe mal un texte. Que faire ?**
R : Le modele est entraine sur des articles de presse et peut mal generaliser sur des textes courts ou atypiques (posts de 10 mots, memes, satire). Le score doit etre interprete comme un indicateur, pas comme un verdict. Consultez la section "Limites" du notebook 06.

**Q : Puis-je ajouter d'autres langues ?**
R : La V2 supporte uniquement le francais et l'anglais. Les textes dans d'autres langues sont routes vers le pipeline anglais par defaut. La V3 (sentence-transformers multilingue) etendra le support a 50+ langues.

**Q : Quel est le cout carbone d'une prediction ?**
R : Une prediction sur un batch de 1 000 textes emet moins de 0.001 g de CO2. L'entrainement complet emet environ 0.01 g de CO2 (moins qu'un email).

---

### 13. Support

- **Code source** : [github.com/azelbanks/projet_etude](https://github.com/azelbanks/projet_etude)
- **Documentation technique** : notebook 06
- **Bugs et suggestions** : ouvrir une issue sur le depot GitHub

---

*Thumalien V2 — Pipeline bilingue FR/EN — Master Big Data, Sup de Vinci*
