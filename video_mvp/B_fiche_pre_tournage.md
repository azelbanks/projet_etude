# Fiche pré-tournage — Vidéo Thumalien V9

**Objectif** : tout préparer en 2-3 heures pour ne pas perdre de temps pendant la captation. Une mauvaise préparation = 4x le temps de tournage et un résultat médiocre.

---

## 1. Setup système (15 min)

### 1.1 Notifications désactivées
```bash
# Mode "Ne pas déranger" macOS pour toute la durée du tournage
# (Centre de contrôle → Concentration → Ne pas déranger pendant 4 h)
```

### 1.2 Préparation du Bureau et navigateur
```bash
# Bureau propre (cacher les fichiers)
defaults write com.apple.finder CreateDesktop false && killall Finder

# Pour réafficher après le tournage :
# defaults write com.apple.finder CreateDesktop true && killall Finder
```

### 1.3 Mode plein écran pour Chrome
- Configurer Chrome en mode "Présentation" (Vue → Plein écran : Cmd+Ctrl+F)
- Désactiver les extensions : Chrome → Préférences → Extensions → tout désactiver
- Créer un profil Chrome dédié `thumalien-demo` sans historique ni notifications
- Vider l'historique : Cmd+Shift+Suppr → "Tout l'historique"

### 1.4 Police curseur visible
```bash
# Augmenter la taille du curseur pour qu'il soit bien visible à l'écran
# Préférences Système → Accessibilité → Affichage → Taille du curseur : 2.5
```

### 1.5 Audio Mac silencieux
```bash
# Couper le son des notifications système
osascript -e 'set volume alert volume 0'
osascript -e 'set volume input volume 75'
```

---

## 2. Préparation des données et modèles (10 min)

### 2.1 Vérifier que tout est en place
```bash
cd "/Users/azeliebernard/Documents/MASTER Big data/projet_etude"

# Vérifier les modèles
ls -la models/*.pkl models/*.joblib models/*.pt

# Vérifier les figures XAI
ls docs/figures/xai/

# Vérifier les notebooks clés
ls notebooks/01_*.ipynb notebooks/24_*.py notebooks/27_*.py
```

### 2.2 Régénérer les figures XAI les plus récentes
```bash
# Si besoin de figures fraîches pour la vidéo (sinon skip)
python3 scripts/run_xai_pipeline.py
```

### 2.3 Créer les textes test pour la démo dashboard
```bash
mkdir -p video_mvp/demo_texts

# Texte clairement fiable (FR)
cat > video_mvp/demo_texts/fiable_fr.txt << 'EOF'
Le CNRS publie aujourd'hui une étude dans la revue Nature confirmant l'efficacité du nouveau traitement contre l'hépatite C, avec un taux de guérison de 95% sur 1200 patients suivis pendant 18 mois.
EOF

# Texte clairement suspect (FR)
cat > video_mvp/demo_texts/suspect_fr.txt << 'EOF'
SCANDALE : le gouvernement nous CACHE la vérité sur les chemtrails ! Les médecins refusent de parler par peur ! Partagez avant censure ! On vous ment depuis 30 ans !!!
EOF

# Texte indécis pour la démo XAI poussée (FR, P proche de 0.5)
cat > video_mvp/demo_texts/indecis_fr.txt << 'EOF'
D'après certains experts, l'inflation pourrait dépasser 4% au prochain trimestre. Les marchés réagissent déjà avec inquiétude selon plusieurs analystes.
EOF

# Texte fiable EN
cat > video_mvp/demo_texts/fiable_en.txt << 'EOF'
Researchers at MIT published in Nature a study confirming that the new carbon capture method reduces costs by 40 percent across 12 industrial sites tested over 18 months.
EOF

# Texte suspect EN
cat > video_mvp/demo_texts/suspect_en.txt << 'EOF'
EXPOSED: Big Pharma doesn't want you to know this ONE trick that cures all diseases overnight! Wake up sheeple! Share before they delete this!!!
EOF

ls -la video_mvp/demo_texts/
```

### 2.4 Tester chaque texte dans le pipeline avant tournage
```bash
# Test rapide pour confirmer que les scores sont cohérents avec ce qu'on veut montrer
python3 -c "
import sys; sys.path.insert(0, 'src')
from pipeline.expert_detector import ExpertFakeNewsDetector
import pandas as pd

det = ExpertFakeNewsDetector(model_dir='models', threshold=0.44)
det.load(suffix='expert_v5')

texts = [
    open('video_mvp/demo_texts/fiable_fr.txt').read().strip(),
    open('video_mvp/demo_texts/suspect_fr.txt').read().strip(),
    open('video_mvp/demo_texts/indecis_fr.txt').read().strip(),
]
labels = ['FIABLE_FR_attendu', 'SUSPECT_FR_attendu', 'INDECIS_FR_attendu']

for text, label in zip(texts, labels):
    res = det.predict(pd.Series([text]))
    score = float(res['ai_score_credibility'].iloc[0])
    pred = 'FIABLE' if score > 0.44 else 'SUSPECT'
    print(f'{label:25s} score={score:.3f} -> {pred}')
"
```

**Résultat attendu** :
- fiable_fr : score > 0.7 → FIABLE ✅
- suspect_fr : score < 0.3 → SUSPECT ✅
- indecis_fr : score entre 0.4 et 0.6

Si un texte ne donne pas le score attendu, **ajuste-le** pour amplifier les signaux. Tu ne veux pas découvrir en plein tournage que ton "exemple suspect" est classé fiable.

---

## 3. Lancement du dashboard pour les captures (5 min)

### 3.1 Démarrer MongoDB en arrière-plan
```bash
# Si MongoDB tourne via Docker
docker compose up -d mongo

# Vérifier
docker ps | grep mongo
```

### 3.2 Lancer le dashboard Streamlit
```bash
cd "/Users/azeliebernard/Documents/MASTER Big data/projet_etude"
streamlit run dashboard/app.py --server.headless=false --server.port=8501
```

### 3.3 Pré-naviguer dans les pages
1. Ouvrir Chrome plein écran sur `http://localhost:8501`
2. Naviguer une fois sur chaque page pour précharger les données :
   - Vue Globale → laisser charger 5s
   - Analyse Temps Réel → coller le texte fiable, soumettre, vérifier que SHAP s'affiche, screenshot
   - Explorateur → laisser charger
   - Performance → laisser charger
   - À propos → laisser charger
3. Revenir sur Vue Globale (page d'entrée)

**Pourquoi pré-naviguer** : la première charge de chaque page Streamlit prend 2-5s. En tournage tu ne veux pas avoir ces latences à l'écran.

---

## 4. Setup OBS Studio pour les captures (15 min)

### 4.1 Installation et premiers réglages
```bash
# Installation via Homebrew si pas déjà installé
brew install --cask obs

# Lancer OBS
open -a "OBS"
```

### 4.2 Configuration recommandée OBS

**Paramètres → Vidéo** :
- Résolution de base : 2880×1800 (résolution native MacBook Air M4)
- Résolution de sortie : 1920×1080
- Filtre downscale : Lanczos
- FPS : 30

**Paramètres → Sortie** :
- Mode : Avancé
- Encodeur : Apple VT H.264 Hardware Encoder (matériel Apple Silicon)
- Bitrate : 12000 Kbps
- Format de fichier : MP4
- Dossier de sortie : `~/Desktop/thumalien_captures/`

**Paramètres → Audio** :
- Fréquence d'échantillonnage : 48 kHz
- Désactiver tous les périphériques audio par défaut (on enregistrera la voix off séparément)

### 4.3 Créer les scènes OBS

**Scène 1 — Capture écran complète**
- Source : Capture d'affichage (sélectionner ton écran principal)
- Filtre : Aucun, sortie native

**Scène 2 — Capture fenêtre Chrome (zoom dashboard)**
- Source : Capture de fenêtre → Chrome plein écran
- Cadrer pour montrer juste la zone dashboard sans la barre URL

**Scène 3 — Capture caméra (pour les inserts)**
- Source : Périphérique de capture vidéo → caméra Mac (FaceTime HD)
- Filtre : Color Correction → augmenter saturation +10, contraste +5

### 4.4 Test de capture (1 min)
```bash
mkdir -p ~/Desktop/thumalien_captures
# Lancer une capture de 30s avec OBS pour valider le setup
# Vérifier que le fichier MP4 généré pèse ~50 MB et joue correctement
```

---

## 5. Plan d'enregistrement des captures écran (60 min total)

Suivre cet ordre **strict** pour ne pas perdre de temps :

### Capture 1 — Notebook V1 avec biais Reuters (5 min)
**But** : illustrer la slide 4 et le passage à 02:30

```bash
# Ouvrir le notebook 01_Exploration_Bluesky.ipynb dans Jupyter
jupyter notebook notebooks/01_Exploration_Bluesky.ipynb &

# Naviguer dans le notebook jusqu'à la cellule qui montre :
# 1. La sortie F1 = 0.99 en cross-validation
# 2. Les top mots LogReg avec Reuters/AFP/AP en tête
```

**Action OBS** : Scène 1, démarrer enregistrement, scroller lentement de la sortie F1 vers les top mots, arrêter (~30s de capture). Sauvegarde : `01_notebook_biais_reuters.mp4`.

### Capture 2 — Architecture C4 (slide statique, pas besoin de capture)

Skip — utiliser directement la slide 5 exportée.

### Capture 3 — Dashboard Vue Globale (3 min)
**But** : illustrer le début de la démo à 07:00

**Action OBS** : Scène 2, démarrer, parcourir la page Vue Globale lentement, hover sur les KPI pour montrer les tooltips, scroller jusqu'en bas. Durée : 30s. Sauvegarde : `03_dashboard_vue_globale.mp4`.

### Capture 4 — Dashboard Analyse Temps Réel — texte FIABLE (5 min)

**Action OBS** : Scène 2, démarrer, action :
1. Cliquer sur "Analyse Temps Réel" (1s)
2. Coller `fiable_fr.txt` dans la zone texte (3s — utiliser Cmd+V)
3. Cliquer "Analyser" (1s)
4. Attendre l'affichage du score (2s)
5. Scroller pour montrer la décomposition SHAP V6 (5s)
6. Scroller pour montrer la décomposition méta-learner V8 (5s)
7. Hover sur la barre top contributeur pour montrer la valeur exacte (3s)

Durée : 20s. Sauvegarde : `04_dashboard_realtime_fiable.mp4`.

### Capture 5 — Dashboard Analyse Temps Réel — texte SUSPECT (5 min)
Même procédure que capture 4 mais avec `suspect_fr.txt`. Sauvegarde : `05_dashboard_realtime_suspect.mp4`.

### Capture 6 — Heatmap attention CamemBERT (3 min)
**But** : illustrer la phrase "voici la heatmap d'attention CamemBERT" à 09:00

**Action** : ouvrir l'image `docs/figures/xai/camembert_attention_fp_36.png` dans Aperçu, capturer un zoom progressif sur la zone "SCANDALE" qui s'illumine en rouge.

```bash
open docs/figures/xai/camembert_attention_fp_36.png
```

**Action OBS** : Scène 1, démarrer, zoomer progressivement (Cmd++), durée 8s. Sauvegarde : `06_attention_camembert.mp4`.

### Capture 7 — Courbe AOPC (5 min)
**But** : illustrer la slide 8 à 11:30

```bash
open docs/figures/xai/faithfulness_aopc_curve.png
```

**Action OBS** : Scène 1, démarrer, vue d'ensemble, puis zoomer sur la divergence rouge/gris vers k=10, durée 6s. Sauvegarde : `07_aopc_curve.mp4`.

### Capture 8 — Coverage report (3 min)
**But** : illustrer la slide 9 à 12:30

```bash
# Lancer le rapport HTML coverage
open docs/coverage_html/index.html
```

**Action OBS** : Scène 1, démarrer, vue d'ensemble du rapport coverage, scroller pour montrer 80% en haut, puis zoomer sur quelques modules à 95%+ (meta_decomposition.py 100%, faithfulness.py 97%), durée 10s. Sauvegarde : `08_coverage_report.mp4`.

### Capture 9 — Mutation testing résultats (3 min)
**But** : illustrer la slide 9 (mention 80,3% kill rate)

```bash
# Si rapport HTML mutmut existe
mutmut html  # génère html/index.html
open html/index.html
```

**Action OBS** : Scène 1, démarrer, capture de la page mutmut showing 143/178 killed = 80.3%, durée 6s. Sauvegarde : `09_mutmut_results.mp4`.

### Capture 10 — Pie chart bilan carbone (3 min)
**But** : illustrer la slide 11

Capture statique depuis Keynote slide 11.

### Capture 11 — Comparaison V1 vs V9 split-screen (3 min)
**But** : illustrer la conclusion à 17:00

**Action** : ouvrir deux fenêtres côte à côte :
- Gauche : notebook 01 avec sortie CV F1=0.99 visible
- Droite : rapport principal au § F1 V9 = 0.67 visible

**Action OBS** : Scène 1 plein écran capturant les deux fenêtres, durée 8s. Sauvegarde : `11_v1_vs_v9.mp4`.

---

## 6. Setup voix off (10 min avant captation audio)

### 6.1 Conditions environnementales
- Pièce silencieuse : couper ventilateur, climatisation, frigo si possible
- Heure : matin tôt (8h-10h) ou soir tard (21h-23h) pour éviter bruits extérieurs
- Tester avec un enregistrement de 10s en silence : vérifier le bruit de fond < -50 dB

### 6.2 Setup micro
- Yeti USB ou cravate Lavalier branché en USB
- Distance bouche-micro : 15 cm pour Yeti, attaché au revers pour Lavalier
- Pop filter (filtre anti-pop) à 5 cm devant le Yeti
- Position cardioïde sur Yeti pour rejeter le bruit de fond

### 6.3 Test audio dans Audacity
```bash
brew install --cask audacity  # si pas déjà installé
open -a Audacity
```

- Sélectionner le micro USB dans Audacity
- Enregistrer 30s en lisant un texte test
- Vérifier que le signal monte à -12 dB en moyenne, jamais au-dessus de -3 dB
- Si signal trop faible : monter le gain micro
- Si signal sature : baisser le gain micro

### 6.4 Backup en double
Pendant la captation voix off, **toujours enregistrer en parallèle sur le téléphone** comme backup. Les fichiers Audacity peuvent corrompre. Avoir une seconde source sauve une demi-journée de re-tournage.

---

## 7. Workflow de nommage des fichiers

Tous les fichiers vont dans `~/Desktop/thumalien_captures/` avec ce nommage strict :

```
01_notebook_biais_reuters.mp4
02_slide_architecture.png         (export Keynote)
03_dashboard_vue_globale.mp4
04_dashboard_realtime_fiable.mp4
05_dashboard_realtime_suspect.mp4
06_attention_camembert.mp4
07_aopc_curve.mp4
08_coverage_report.mp4
09_mutmut_results.mp4
10_pie_chart_carbon.png            (export Keynote)
11_v1_vs_v9.mp4

vo_01_intro_hook.wav               (voix off section 1)
vo_02_problematique.wav
vo_03_decouverte_biais.wav
vo_04_architecture.wav
vo_05_demo_dashboard.wav
vo_06_xai_faithfulness.wav
vo_07_qualite_industrielle.wav
vo_08_conformite.wav
vo_09_limites.wav
vo_10_conclusion.wav

cam_01_intro.mp4                   (insert caméra ouverture)
cam_02_transition_qualite.mp4      (insert caméra à 12:00)
cam_03_conclusion.mp4              (insert caméra à 17:00)
```

Ce nommage permet de tout charger dans DaVinci Resolve dans l'ordre de lecture par tri alphabétique.

---

## 8. Checklist finale avant le clap "Action"

- [ ] Notifications silencieuses (Mode "Ne pas déranger" actif)
- [ ] Bureau propre (Finder caché)
- [ ] Dashboard Streamlit lancé sur localhost:8501
- [ ] MongoDB tourne (Docker container actif)
- [ ] Chrome en plein écran sans extensions ni notifications
- [ ] Curseur agrandi pour visibilité
- [ ] Textes demo dans `video_mvp/demo_texts/` validés
- [ ] OBS configuré, scènes prêtes, dossier de sortie créé
- [ ] Micro testé, niveau -12 dB en moyenne
- [ ] Téléphone en mode avion à côté pour backup audio
- [ ] Pièce silencieuse, fenêtres fermées
- [ ] Bouteille d'eau à proximité
- [ ] Script imprimé en A4 paysage à côté du clavier

---

## 9. Pro tips qui font la différence

**Pendant les captures écran** :
- Bouger le curseur **lentement** : 50% de la vitesse normale. Le cerveau du spectateur a besoin de suivre.
- Faire des **pauses** de 1s avant chaque clic important : le spectateur anticipe l'action.
- Ne **jamais** cliquer sur un menu sans que le spectateur l'ait vu apparaître.

**Pendant la voix off** :
- Lire le script à 90% de ta vitesse naturelle de parole.
- Marquer des **micro-pauses** entre les phrases (200ms minimum) pour faciliter le montage.
- Si tu butes sur un mot, **fais une pause de 2s nette** puis reprends la phrase entière. Cette pause sert de marqueur pour le montage.

**Pendant les inserts caméra** :
- Regarder **directement l'objectif** de la caméra, pas l'écran de retour.
- Sourire imperceptiblement entre les phrases : ça transparaît dans le ton.
- Si tu rates une prise, attends 3 secondes en silence puis recommence — encore un marqueur de coupe pour le montage.

**Sauvegardes** :
- À la fin de chaque session de tournage, **copier immédiatement le dossier `thumalien_captures/` sur disque externe ou cloud**. Une carte SD qui meurt = un week-end perdu.

---

*Fiche pré-tournage Thumalien V9 — Mai 2026*
