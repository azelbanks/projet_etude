# Guide de Production Video MVP — Thumalien
## VI.2 - VIDEO & MVP - SAVOIR CONVAINCRE

**Projet** : Thumalien — Social Media Intelligence & AI Monitor
**Equipe** : Azelie Bernard (Lead Technique) / Sebastien Lazcanotegui (Optimisation ML & Documentation)
**Formation** : Master Big Data — Sup de Vinci
**Duree cible** : 15-18 minutes
**Livrable** : MP4 1920x1080 ou lien YouTube non repertorie

---

## 1. Strategie de differenciation

### Ce que font les autres groupes
- PowerPoint filme avec voix off monotone
- Demo basique en fin de video
- Pas de storytelling, pas d'emotion

### Ce que NOUS allons faire
- **Storytelling "enquete"** : commencer par le probleme (vrais posts suspects Bluesky), pas par la solution
- **Plot twist V1** : raconter comment notre V1 a 99.6% etait en realite cassee — montre la maturite intellectuelle
- **Demo live interactive** : taper des textes en direct (FR et EN) et voir le verdict en temps reel
- **Radar emotionnel** : exploiter le radar chart 7 emotions du dashboard (visuellement tres fort)
- **Chiffres d'impact** : 188K posts, 11 versions, 0.55g CO2 — des metriques concretes qui impressionnent

---

## 2. Moyens techniques de production

### 2.1 Capture video

| Outil | Usage | Prix | Plateforme |
|-------|-------|------|------------|
| **OBS Studio** | Capture ecran + webcam simultanee | Gratuit | Mac/Win/Linux |
| **QuickTime Player** | Capture ecran simple (Mac) | Gratuit | Mac |
| **Loom** | Capture ecran + webcam + partage facile | Gratuit (5 min) / 15$/mois | Web |

**Recommandation** : OBS Studio — permet le picture-in-picture (ecran + visage), les scenes pre-configurees et l'incrustation du bandeau nom.

### Configuration OBS recommandee :
- Resolution : 1920x1080 @ 30fps
- Encodeur : x264, CRF 18 (qualite elevee)
- Audio : micro USB ou AirPods Pro (PAS le micro integre du laptop)
- Scenes a pre-configurer :
  1. "Slide" : capture fenetre presentaion plein ecran
  2. "Demo" : capture ecran + webcam en bas a droite (300x200px)
  3. "Speaker" : webcam plein ecran avec bandeau nom en bas
  4. "Split" : ecran divise 50/50 (code a gauche, dashboard a droite)

### 2.2 Montage video

| Outil | Niveau | Prix | Avantage |
|-------|--------|------|----------|
| **iMovie** | Debutant | Gratuit (Mac) | Simple, suffisant pour couper/assembler |
| **DaVinci Resolve** | Intermediaire/Pro | Gratuit | Etalonnage, effets, titrage pro |
| **CapCut Desktop** | Debutant/Intermediaire | Gratuit | Sous-titres auto, templates modernes |

**Recommandation** : **CapCut Desktop** pour les sous-titres automatiques et les templates de bandeaux nom, ou **DaVinci Resolve** si vous voulez un rendu plus pro.

### 2.3 Slides / Visuels

| Outil | Usage | Prix |
|-------|-------|------|
| **Canva** | Slides, schemas, bandeaux | Gratuit / Pro etudiant |
| **Google Slides** | Slides collaboratifs | Gratuit |
| **Figma** | Schemas d'architecture sur mesure | Gratuit |
| **Mermaid Live Editor** | Diagrammes depuis le code existant | Gratuit (mermaid.live) |

**Recommandation** : **Canva** — templates pro, export en image haute qualite pour incrustation dans OBS.

### 2.4 Son

- **Micro-casque USB** ou **AirPods Pro** — jamais le micro integre du laptop
- Enregistrer dans une piece calme, porte fermee
- Faire un test de 30 secondes et reecouter AVANT de filmer
- Si le son est mauvais malgre tout : post-traitement avec **Audacity** (gratuit) — reduction de bruit

---

## 3. Comment l'IA peut nous aider

### 3.1 Generation de visuels avec l'IA

| Tache | Outil IA | Comment |
|-------|----------|---------|
| **Slides professionnelles** | **Canva AI** (Magic Design) | Generer des slides a partir du plan, theme dark/tech |
| **Schema d'architecture** | **Claude** (mode artefact) | Demander un schema SVG ou Mermaid du pipeline |
| **Illustrations** | **DALL-E / Midjourney** | Generer des visuels "cybersecurity", "data analysis" pour les slides |
| **Logo Thumalien** | **Canva AI / DALL-E** | Logo style tech/surveillance pour le titre |

### 3.2 Aide au script et voix

| Tache | Outil IA | Comment |
|-------|----------|---------|
| **Script detaille** | **Claude** | Generer le texte mot-a-mot de chaque partie (deja fait ci-dessous) |
| **Sous-titres automatiques** | **CapCut AI** | Genere les sous-titres FR automatiquement sur la video |
| **Corrections audio** | **Adobe Podcast AI** (gratuit) | Ameliore la qualite audio, supprime bruit de fond |

### 3.3 Remotion via Claude — Generation video programmatique

**Remotion** (remotion.dev) est un framework React pour creer des videos programmatiquement. Claude peut generer le code Remotion pour :

- Des animations de graphiques (courbe F1 qui monte version apres version)
- Des bandeaux nom animes
- Des transitions entre slides
- L'affichage progressif du tableau des 11 versions

**Avantages** :
- Rendu pixel-perfect, reproductible
- Animations fluides impossibles a faire dans PowerPoint
- Claude peut ecrire tout le code

**Inconvenients** :
- Necessite Node.js + React
- Courbe d'apprentissage si vous ne connaissez pas React
- Temps de setup : ~2h pour un projet de base

**Verdict** : Remotion est excellent pour les animations de donnees (graphiques, tableaux animes), mais **pas pour le screencast du dashboard ni les passages webcam**. L'approche recommandee est **hybride** :
- Remotion pour les sequences animees (intro, transitions, graphiques)
- OBS pour les screencasts et passages camera
- Assemblage final dans DaVinci Resolve ou CapCut

### 3.4 Alternative recommandee : Canva + CapCut (sans code)

Si Remotion est trop technique, cette combinaison donne un resultat equivalent :
1. **Canva** : creer toutes les slides avec animations integrees, exporter en video MP4
2. **CapCut** : assembler slides Canva + screencasts OBS + sous-titres auto
3. **Claude** : generer le script, les textes des slides, et les talking points

---

## 4. Script detaille — 15 minutes

### ACTE 1 — "Le Probleme" (Sebastien, 3 min)

**[0:00-0:30] ACCROCHE — Ecran noir + texte qui apparait**
Scene OBS : slide plein ecran (fond sombre)

Texte a l'ecran : "Sur Bluesky, 1 post sur 4 que nous analysons est suspect."
Puis transition vers des captures de vrais posts conspirationnistes.

Script Sebastien :
"Chaque jour, des milliers de posts circulent sur Bluesky. Certains informent.
D'autres manipulent. A l'oeil nu, la difference est parfois invisible.
Comment faire le tri dans 188 000 posts, en francais ET en anglais,
quand chaque texte ne fait que 10 mots ?"

**[0:30-1:30] LA PROBLEMATIQUE**
Scene OBS : slide "Le defi"

Script Sebastien :
"Bluesky est un reseau social decentralise base sur le protocole AT.
Contrairement a Twitter, il n'y a pas de moderation centralisee.
C'est un terrain fertile pour la desinformation.

Notre defi :
- Des textes ultra-courts : 5 a 20 mots en moyenne
- Deux langues : francais et anglais
- Un volume massif : plus de 188 000 posts collectes
- Et zero budget cloud : tout tourne en local.

Un humain met en moyenne 3 minutes pour verifier un seul post.
Pour 188 000 posts, ca represente plus d'un an de travail non-stop.
C'est pour ca qu'on a cree Thumalien."

**[1:30-2:30] PRESENTATION DE L'EQUIPE**
Scene OBS : slide equipe (photos + roles)

Script Sebastien :
"Je suis Sebastien Lazcanotegui. Mon role sur ce projet :
l'optimisation du machine learning et la coherence documentaire.
J'ai travaille sur le debiaisage des donnees, l'optimisation
des hyperparametres par GridSearch, et la consolidation de la documentation.

Ma collegue Azelie Bernard est la lead technique du projet.
C'est elle qui a concu et itere l'ensemble du pipeline :
de la collecte Bluesky au dashboard, en passant par les 11 versions
du modele, les fine-tunings CamemBERT et RoBERTa."

**[2:30-3:00] TEASER DE LA SOLUTION**
Scene OBS : slide architecture

Script Sebastien :
"Voici Thumalien, en un schema : Bluesky, collecteur, MongoDB,
pipeline NLP, dashboard. Tout conteneurise avec Docker.
Une commande, et le systeme entier demarre."

---

### ACTE 2 — "Le Parcours" (Sebastien, 4 min)

**[3:00-4:00] METHODOLOGIE**
Scene OBS : slide CRISP-DM

Script Sebastien :
"On ne construit pas un modele d'IA en une fois.
On suit le cycle CRISP-DM adapte a l'IA :
comprendre le probleme, explorer les donnees,
preparer, modeliser, evaluer, deployer.
Et surtout : iterer. Encore et encore."

**[4:00-5:30] LE PLOT TWIST — La V1 a 99.6%**
Scene OBS : slide avec metriques V1 (99.6% en GRAND), puis animation qui barre en rouge

Script Sebastien :
"Notre premier modele affichait 99.6% de precision.
(pause)
99.6%. On pourrait se feliciter et rentrer chez nous.

Sauf que quand on a regarde les mots les plus predictifs,
on a trouve... 'reuters', 'reporting by', 'editing by'.

Le modele ne detectait pas les fake news.
Il detectait les articles Reuters. C'etait un biais de donnees.

C'est LA lecon de ce projet : des metriques parfaites
peuvent cacher un modele completement inutile.
Il faut toujours tester sur des donnees reelles."

**[5:30-7:00] 11 VERSIONS EN 5 MOIS**
Scene OBS : tableau des versions qui se remplit ligne par ligne

Script Sebastien :
"A partir de ce constat, on a itere. 11 fois.

V1.5 : on debiaise Reuters, on ajoute le francais.
V2 : on integre des tweets et titres courts. Le taux de posts fiables
sur Bluesky passe de 23% a 73%.
V3 : on decouvre que 5 de nos 12 features etaient a zero — un bug de preprocessing.
V4 : on augmente les donnees FR courtes. Le F1 francais court fait +32%.
V5 : 10 000 posts sociaux synthetiques, le test bilingue passe a 12 sur 12.

Et ensuite, les transformers :
CamemBERT V2 pour le francais : F1 de 0.957 sur les textes ultra-courts.
RoBERTa V2 pour l'anglais : F1 de 0.874, test 16 sur 18.

De mon cote, j'ai optimise les hyperparametres par GridSearch —
C passe de 1.0 a 5.0, min_df de 3 a 5, n-grams de (1,3) a (1,2) —
et j'ai ajoute le debiaisage residuel : suppression des mentions
d'agences dans le corps du texte et des annees artefacts 2015-2020.

Le tout pour 0.55 gramme de CO2. C'est l'IA responsable, en actes."

---

### ACTE 3 — "La Demo" (Azelie, 6 min)

**[7:00-8:00] ARCHITECTURE TECHNIQUE**
Scene OBS : terminal + slide architecture

Script Azelie :
"Voici l'architecture : 4 containers Docker.
Le collecteur se connecte a Bluesky via le protocole AT,
stocke les posts dans MongoDB,
le pipeline NLP analyse chaque texte,
et le dashboard Streamlit affiche tout en temps reel.

On lance tout avec une seule commande."
(montrer docker-compose up dans le terminal)

**[8:00-9:30] COLLECTE + MONGODB**
Scene OBS : ecran + webcam picture-in-picture

Script Azelie :
"Le collecteur tourne en continu. Chaque post est enrichi :
texte, auteur, timestamp, langue detectee.
En 6 mois, on a collecte plus de 245 000 posts, et la collecte continue."
(montrer MongoDB Compass ou shell : les documents, le count)
"On a mis en place des indexes, de la validation de schema,
et un monitoring de la qualite des donnees."

**[9:30-11:30] DEMO DASHBOARD — Le moment cle**
Scene OBS : dashboard plein ecran + webcam en petit

Script Azelie :
"Voici le dashboard Thumalien."
(ouvrir le dashboard Streamlit — le design glassmorphism dark va impressionner)

"On voit les KPIs en temps reel : nombre de posts, pourcentage fiable,
distribution des emotions.

Maintenant, le test en direct. Je vais taper des textes et on va voir
ce que le modele en pense."

DEMO LIVE :
1. "SCANDALE ! Le gouvernement cache la verite sur les vaccins !"
   -> Score 0.02 -> SUSPECT (montrer le verdict rouge)
   -> Radar emotion : colere + peur dominant

2. "Le CNRS publie une etude sur le changement climatique."
   -> Score 0.95 -> FIABLE (montrer le verdict vert)
   -> Radar emotion : neutre dominant

3. "SHARE before they DELETE this!! The truth about 5G!"
   -> Score 0.02 -> SUSPECT
   "Le modele detecte aussi bien en anglais."

4. "The city council approved the new budget."
   -> Score 0.81 -> FIABLE

"Le modele ne fait pas que classer vrai ou faux.
Il analyse l'emotion derriere chaque texte : colere, peur, joie, surprise...
C'est ce radar que vous voyez ici."
(montrer le radar chart en gros plan)

**[11:30-13:00] SOUS LE CAPOT**
Scene OBS : slide pipeline + notebook

Script Azelie :
"Comment ca marche sous le capot ?
Le pipeline combine 3 types de features :
- TF-IDF : 30 000 n-grammes, le vocabulaire du texte
- 12 features linguistiques : majuscules, ponctuation, sensationnalisme
- 7 probabilites emotionnelles : notre MLP PyTorch

Le tout alimente une regression logistique. Pourquoi pas un gros Transformer ?
Parce qu'on peut EXPLIQUER chaque prediction.
On sait quels mots, quelles features ont fait basculer le verdict.
C'est essentiel pour un outil de lutte contre la desinformation."

(montrer rapidement un notebook : courbe d'apprentissage CamemBERT ou RoBERTa)

---

### ACTE 4 — "L'Impact" (Azelie + Sebastien, 2 min)

**[13:00-14:00] ROI (Azelie)**
Scene OBS : slide ROI

Script Azelie :
"L'impact concret :
- 188 000 posts analyses automatiquement — ce qui prendrait 1 an a un humain
- Zero euro de cloud GPU : tout tourne sur un MacBook avec Apple Silicon
- 0.55 gramme de CO2 pour l'ensemble des entrainements
- 100% open source, 100% reproductible
- Scalable : de Docker Compose a Kubernetes en production"

**[14:00-15:00] PERSPECTIVES ET CONCLUSION (Sebastien)**
Scene OBS : slide perspectives + les deux noms

Script Sebastien :
"Les prochaines etapes pour Thumalien :
- Une API REST avec FastAPI pour integrer la detection dans d'autres outils
- La detection multimodale : analyser les images, pas seulement le texte
- L'extension a d'autres langues

Ce projet montre qu'avec les bonnes donnees, la bonne methodologie,
et un esprit critique sur ses propres resultats,
on peut construire un outil de lutte contre la desinformation
qui est a la fois efficace, interpretable, et eco-responsable.

Merci pour votre attention."

(Slide finale : "Thumalien — Azelie Bernard & Sebastien Lazcanotegui — Master Big Data — Sup de Vinci")

---

## 5. Planning de production

| Etape | Qui | Duree estimee | Livrable |
|-------|-----|---------------|----------|
| 1. Creer les slides (Canva) | Les deux | 3h | 10 slides exportees en PNG/MP4 |
| 2. Repeter le script | Chacun | 2x 30min | Fluide, < 15 min chrono |
| 3. Configurer OBS (scenes, bandeau) | Azelie | 1h | 4 scenes pre-configurees |
| 4. Enregistrer Actes 1+2 (Sebastien) | Sebastien | 1-2h | Rushes Sebastien |
| 5. Enregistrer Actes 3+4 (Azelie) | Azelie | 1-2h | Rushes Azelie + screencasts |
| 6. Montage + sous-titres (CapCut) | Les deux | 3h | Video assemblee |
| 7. Relecture + corrections | Les deux | 1h | Version finale |
| 8. Export MP4 + upload YouTube | Azelie | 30min | Livrable final |
| **TOTAL** | | **~12-15h** | |

---

## 6. Checklist avant tournage

- [ ] Dashboard Streamlit demarre et fonctionnel (docker-compose up)
- [ ] MongoDB avec des donnees interessantes (188K posts)
- [ ] OBS Studio installe + 4 scenes configurees
- [ ] Micro teste (enregistrement 30s + reecoute)
- [ ] Slides Canva finalisees et exportees
- [ ] Script imprime / sur un ecran secondaire
- [ ] Piece calme, porte fermee, notifications desactivees
- [ ] Webcam bien cadree (visage centre, fond neutre ou range)
- [ ] Bandeau nom prepare dans OBS (texte blanc sur fond semi-transparent noir)

---

## 7. Nomenclature du livrable

### Solution 1 — Fichier ZIP
```
PE_2526_[codepromo]_Bernard_Lazcanotegui.zip
  └── PE-2526_[codepromo]_BernardAzelie.mp4
```

### Solution 2 — Lien YouTube
```
PE_2526_[codepromo]_Bernard_Lazcanotegui.txt
  └── contient : https://youtube.com/watch?v=XXXXX (non repertorie)
```

Remplacer `[codepromo]` par votre code promo (ex: M1DEVA, M2BIGDATA, etc.)
