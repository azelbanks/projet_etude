# 17 Slides Keynote — Video ThumaCheck (client : Thumalien)
**Format** : 16:9, 1920x1080, exporte en MP4 ou PNG haute resolution pour incrustation montage
**Palette** : fond `#0e1117` (noir bleute dashboard), primaire `#00D4FF` (cyan), succes `#00E676` (vert), danger `#FF1744` (rouge), accent `#FFD600` (jaune), texte `#E8E8E8`
**Police** : Inter ou SF Pro Display — titres 48-64pt, corps 28-32pt, legendes 18-22pt
**Regle absolue** : un message par slide, max 3 puces, jamais plus de 30 mots a l'ecran simultanement

---

## SLIDE 1 — Titre projet (00:30)

**Composition** : centree, fond degrade noir-bleu, logo ThumaCheck grand format en haut

```
                    [LOGO THUMACHECK — 200px hauteur]

              THUMACHECK V9
   Detection de desinformation Bluesky — bilingue, explicable, frugale

              ──────────────────────────

           Azelie Bernard . Sebastien Lazcanotegui
              Master 1 Big Data IA — 2025/2026

                         [QR code -> repo GitHub]
```

**Notes mise en page** :
- Titre principal `THUMACHECK V9` en cyan #00D4FF, font-weight 700, 72pt
- Sous-titre en gris #B0B0B0, 28pt
- Filet horizontal cyan 2px largeur 60% centre
- Noms en blanc 24pt, formation en gris 18pt
- QR code 120x120px en bas droite (URL : repo GitHub public)

**Animation** : logo en fade-in, puis contenu en wipe vertical descendant 1.2s.

---

## SLIDE 2 — Equipe Niamato Consulting (01:30)

**Titre** : `NIAMATO CONSULTING — L'EQUIPE` (cyan, 44pt)

**Composition** : 2 colonnes — un bloc par membre avec photo placeholder + role

```
   ┌─────────────────────────┬─────────────────────────┐
   │      [Photo Azelie]     │    [Photo Sebastien]     │
   │                         │                          │
   │   AZELIE BERNARD        │  SEBASTIEN LAZCANOTEGUI  │
   │   Lead Technique &      │  Validation &            │
   │   Architecture          │  Qualite ML              │
   │                         │                          │
   │   • Pipeline NLP V1-V9  │  • Annotation gold set   │
   │   • Dashboard Streamlit │  • Debiaisage Reuters    │
   │   • XAI 3 niveaux       │  • Hyperparametres       │
   │   • Industrialisation   │  • Qualite donnees       │
   └─────────────────────────┴─────────────────────────┘
```

**Notes mise en page** :
- Photos en cercle 150x150px avec bordure cyan 2px (ou placeholder initiales)
- Noms en blanc bold 28pt
- Roles en cyan 22pt
- Bullets en gris 18pt
- Fond cartes #1a1f2e, gap central 40px

**Footer** : *"Niamato Consulting — Expertise Data & Intelligence Artificielle"*

---

## SLIDE 3 — Problematique chiffree (02:00)

**Titre** : `LA DESINFORMATION SUR BLUESKY` (cyan, 44pt, en haut gauche)

**Composition** : 4 chiffres en grille 2x2, chacun avec icone SF Symbol

```
   ┌─────────────────────┬─────────────────────┐
   │     35 millions     │     60 000+/jour    │
   │  utilisateurs Bluesky│   posts publics FR/EN│
   ├─────────────────────┼─────────────────────┤
   │      0 equipe       │     Angle mort      │
   │   moderation centr. │   regulation EU     │
   └─────────────────────┴─────────────────────┘
```

**Notes mise en page** :
- Chiffres en cyan 80pt bold
- Legendes en gris 22pt sous chaque chiffre
- Bordures cellules : trait gris #2a2f3e 1px
- Padding interne 40px

**Animation** : les 4 cellules apparaissent en cascade haut-gauche -> bas-droite, intervalle 0.3s.

---

## SLIDE 4 — Les 4 exigences du cahier des charges (02:30)

**Titre** : `LE CAHIER DES CHARGES` (cyan, 44pt)

**Composition** : 4 cartes horizontales avec icone a gauche + texte a droite

```
   TRANSPARENCE          Chaque score doit etre explicable et auditable
   BILINGUE              Francais + anglais avec performance equivalente
   FRUGALITE             < 5ms par texte, empreinte CO2 mesuree
   CONFORMITE            RGPD art. 22 + AI Act art. 13/14/50 — applicable 2 aout 2026
```

**Notes mise en page** :
- Icones 64x64px en couleur (cyan, vert, jaune, rouge selon theme)
- Mots-cles en blanc bold 32pt
- Descriptions en gris 22pt
- Hauteur uniforme 100px par carte, gap 20px

**Phrase basse (footer 16pt cyan)** : *"Quatre exigences non-negociables pour passer du prototype au systeme deployable"*

---

## SLIDE 5 — La fausse victoire et sa decouverte (03:00)

**Titre** : `F1 = 0.99 — LE PIEGE` (rouge #FF1744, 44pt, dramatique)

**Composition** : split horizontal 50/50

A gauche :
```
   CROSS-VALIDATION 5-FOLD
   ┌──────────────────────┐
   │   F1 macro:  0.987  │
   │   Precision: 0.992  │
   │   Recall:    0.984  │
   │   Accuracy:  0.989  │
   └──────────────────────┘
   Tous les voyants verts
```

A droite :
```
   TOP MOTS LOGREG (XAI)
   ┌──────────────────────┐
   │  reuters       +0.84 │
   │  afp           +0.71 │
   │  ap_news       +0.69 │
   │  associated    +0.63 │
   │  press         +0.58 │
   └──────────────────────┘
   Le modele apprend le STYLE
       d'agence de presse
```

**Notes mise en page** :
- Tableaux gauche/droite encadres, fond #1a1f2e
- Ligne separatrice verticale rouge 2px au centre
- Cocher vert + fleche d'attention rouge avec animation arrivee sequentielle

**Phrase basse (footer 18pt rouge)** : *"Sans XAI, on aurait livre un modele qui collapse en production"*

**Animation cruciale** : le tableau de gauche apparait immediatement. Le tableau de droite arrive 3 secondes apres (revelation visuelle).

---

## SLIDE 6 — Architecture C4 niveau 2 (04:30)

**Titre** : `ARCHITECTURE — 8 CONTENEURS` (cyan, 44pt)

**Composition** : diagramme C4 complet, exporte depuis Mermaid ou structure

```
   ┌────────────┐    ┌────────────┐    ┌──────────────┐
   │  Bluesky   │───>│ Collector  │───>│   MongoDB    │
   │  AT Proto  │    │  (Python)  │    │  (storage)   │
   └────────────┘    └────────────┘    └──────────────┘
                                               │
                                               v
   ┌────────────┐    ┌────────────┐    ┌──────────────┐
   │ Streamlit  │<───│ Pipeline   │<───│  Detector    │
   │ Dashboard  │    │  Cascade   │    │  V5 + V6     │
   └────────────┘    │  Stage 1+2 │    │  + CamemBERT │
         │           └────────────┘    └──────────────┘
         │                  │
         v                  v
   ┌────────────┐    ┌──────────────┐
   │ XAI Engine │    │  Monitoring  │
   │ SHAP+Captum│    │  (CodeCarbon)│
   └────────────┘    └──────────────┘
```

**Notes mise en page** :
- Conteneurs en boites #1a1f2e bordure cyan 1px
- Fleches cyan 2px avec pointe
- Legende : "C4 Model — Niveau 2 (Container)" en bas droite 16pt gris

---

## SLIDE 7 — Diagramme de sequence cascade V9 (06:00)

**Titre** : `PIPELINE CASCADE — INFERENCE` (cyan, 44pt)

**Composition** : diagramme de sequence horizontal, 7 acteurs

```
  User    Stage1     V5      V6      CamBERT   V8     XAI
   │       │         │       │        │        │      │
   │ text  │         │       │        │        │      │
   ├──────>│         │       │        │        │      │
   │       │ opinion?│       │        │        │      │
   │       ├────╗    │       │        │        │      │
   │       │    ║    │       │        │        │      │
   │       │<───╝    │       │        │        │      │
   │       │         │       │        │        │      │
   │       │ if FAIT │       │        │        │      │
   │       ├────────>│       │        │        │      │
   │       │         │ score │        │        │      │
   │       │         ├──────>│        │        │      │
   │       │         │       │ style  │        │      │
   │       │         │       ├───────>│        │      │
   │       │         │       │        │ embed  │      │
   │       │         │       │        ├───────>│      │
   │       │ score V8                          ├─────>│
   │       │<──────────────────────────────────│      │
   │ score+│         │       │        │        │      │
   │ XAI   │         │       │        │        │      │
   │<──────│         │       │        │        │      │
```

**Notes mise en page** :
- Acteurs en haut sur ligne cyan 2px
- Lignes de vie verticales en gris pointille
- Messages fleches cyan
- Annotation "1.5ms total" en bas avec encadre jaune
- Symbole AI Act a cote de Stage 1 avec tooltip *"opinion != fake news"*

**Animation possible** : faire defiler les fleches une par une avec timing 0.3s.

---

## SLIDE 8 — Faithfulness : pourquoi et comment (09:30)

**Titre** : `LA FIDELITE DES EXPLICATIONS` (cyan 44pt) avec sous-titre `Au-dela de SHAP : la mesurer` (gris 24pt)

**Composition** : moitie haute = question, moitie basse = methode

```
   ┌────────────────────────────────────────────────┐
   │   Comment savez-vous que vos explications      │
   │   refletent vraiment le comportement           │
   │   du modele ?                                  │
   └────────────────────────────────────────────────┘

                         |
                         v

   ┌────────────────────────────────────────────────┐
   │   PROTOCOLE ERASER (DeYoung et al., ACL 2020) │
   │                                                │
   │   1. Identifier les top-k features (SHAP)     │
   │   2. Masquer ces features (= 0)               │
   │   3. Mesurer la chute de P(suspect)            │
   │   4. Comparer a un masquage aleatoire          │
   │                                                │
   │   -> Si l'explication est fidele :             │
   │     chute_attribution >> chute_random          │
   └────────────────────────────────────────────────┘
```

**Notes mise en page** :
- Question en haut en cyan 32pt, encadree
- Fleche descendante grosse cyan
- Methode en cadre #1a1f2e avec 4 etapes numerotees
- Conclusion en bas en jaune bold 24pt

---

## SLIDE 9 — Courbe AOPC (10:30)

**Titre** : `RESULTAT FAITHFULNESS — UPLIFT +21%` (vert #00E676 44pt)

**Composition** : graphique pleine largeur + chiffres cles a droite

A gauche (70%) : la figure `faithfulness_aopc_curve.png` agrandie

A droite (30%) : tableau recap

```
   AOPC attribution  : 0.253
   AOPC random       : 0.045
   ─────────────────────────
   Uplift            : +0.21
   Ratio             : 5.6x

   Comprehensiveness@5 : 0.232
   Sufficiency@5       : 0.058

   Cible >+0.10
   Atteinte (x2)
```

**Notes mise en page** :
- Figure encadree subtilement
- Chiffres cles alignes a droite avec espacement aere
- Ligne separatrice horizontale entre AOPC et Compr/Suff
- Cocher vert pour signaler l'atteinte de cible

---

## SLIDE 10 — Qualite industrielle (12:00)

**Titre** : `QUALITE INDUSTRIELLE` (cyan 44pt) sous-titre `537 tests . 80% coverage . 80,3% mutation kill rate`

**Composition** : 3 panneaux verticaux

```
   ┌─────────────┬─────────────┬─────────────┐
   │   PYTEST    │  COVERAGE   │  MUTMUT     │
   │             │             │             │
   │    537      │    80%      │   80,3%     │
   │   tests     │  line cov   │ kill rate   │
   │             │  77,9%      │  143/178    │
   │  passing    │ branch cov  │ mutations   │
   │             │             │             │
   │             │ Quality gate│  > Google   │
   │             │ --fail-under│  benchmark  │
   │             │ =75 sur CI  │  (60-75%)   │
   └─────────────┴─────────────┴─────────────┘
```

**Notes mise en page** :
- 3 cellules largeur egale, hauteur 500px
- Chiffres principaux en 90pt cyan/vert/jaune
- Sous-textes en gris 20pt
- Mention "> Google benchmark (60-75%)" en italique blanc bold (l'argument qui frappe)

**Footer (en bas, 18pt gris)** : *"De la quantite a la qualite — chaque ligne de production est validee"*

---

## SLIDE 11 — Conformite AI Act et RGPD (13:30)

**Titre** : `CONFORMITE REGLEMENTAIRE` (cyan 44pt)

**Composition** : timeline horizontale + chips de conformite

```
   AI ACT (UE 2024/1689) ────────┬───── pleinement applicable 2 aout 2026
                                 │
   Art. 13 (transparence)        │  -> Model Card MC-THUM-2026-001
   Art. 14 (supervision)         │  -> Decomposition B.x dashboard
   Art. 50 (transparence IA)     │  -> Banniere IA visible dans le dashboard
   Risque limite classifie       │  -> Doc 02 § 4.1
                                 │
   RGPD (UE 2016/679)           │
                                 │
   Art. 22 (decision auto)       │  -> Droit explication SHAP+Captum
   Art. 35 (AIPD)               │  -> Document RGPD-THUM-2026-001
   Base legale art. 6.1.f       │  -> Posts publics, interet legitime
```

**Notes mise en page** :
- Coches vertes 24px
- Articles en monospace bleu pale pour effet "code juridique"
- Trait vertical cyan separant articles et preuves

**Footer (en bas, 16pt gris italique)** : *"Positionnement FLI AI Safety Index : supervision humaine, explications par defaut, empreinte mesuree"*

---

## SLIDE 12 — GreenIT et arbitrage CamemBERT (14:00)

**Titre** : `GREEN IT — ~ 6,88 g CO2 TOTAL` (vert 44pt)

**Composition** : pie chart a gauche + tableau de decision a droite

A gauche : pie chart de repartition

```
   Pie chart 350x350px :
   - RoBERTa EN fine-tune  : 45% (cyan fonce)
   - V5 LogReg training    : 33% (cyan)
   - CamemBERT FR fine-tune: 11% (cyan clair)
   - V6 + XAI + inference  : 11% (jaune/vert)
```

A droite :

```
   DECISION ARCHITECTURALE

   Production :
   -> V5 (1.5 ms, 0.6 g CO2/jour)

   Analyse offline :
   -> V8 incluant CamemBERT
     (entrainement unique)

   ROI GreenIT :
   -> Frugalite prod + puissance analyse
   -> Documente dans Model Card § 8

   ~ 26 m en voiture equivalent
```

**Notes mise en page** :
- Pie chart avec legende integree
- Tableau decision en cadre vert pale
- Mention "26 m en voiture equivalent" en footer pour vulgariser

---

## SLIDE 13 — Methodologie et organisation (15:00)

**Titre** : `METHODOLOGIE & ORGANISATION` (cyan 44pt)

**Composition** : 2 blocs — CRISP-DM a gauche, gestion de projet a droite

A gauche :
```
   CRISP-DM ADAPTE ML

   1. Comprendre le probleme
   2. Explorer les donnees
   3. Preparer (debiaisage, augmentation)
   4. Modeliser (V1 -> V9, 9 iterations)
   5. Evaluer (gold set, bootstrap IC)
   6. Deployer (Docker Compose)

   Cycle iteratif : 9 versions en 6 mois
```

A droite :
```
   GESTION DE PROJET

   Gantt : 16 work packages, 28 jalons
   Versionning : Git + GitHub Actions CI/CD
   Conteneurisation : Docker Compose
   Base de donnees : MongoDB
   API : FastAPI
   Monitoring : CodeCarbon

   Reproductible en 1 commande :
   docker compose up
```

**Notes mise en page** :
- 2 cadres #1a1f2e cote a cote, bordure cyan 1px
- Titres de blocs en cyan bold 28pt
- Contenu en gris 20pt
- Commande docker en monospace vert #00E676

---

## SLIDE 14 — ROI et budget (15:45)

**Titre** : `ROI & BUDGET — NIAMATO CONSULTING` (cyan 44pt)

**Composition** : 3 zones — couts projet, couts exploitation, gains

```
   ┌─────────────────┬─────────────────┬─────────────────┐
   │  COUT PROJET    │  EXPLOITATION   │  GAINS CLIENT   │
   │                 │                 │                 │
   │   ~50 000 EUR   │   ~930 EUR/mois │  x10 productivite│
   │                 │                 │                 │
   │  110 jours-homme│  Serveur 30 EUR │  60 000 posts/j │
   │  0 EUR licence  │  + 2j maint.   │  vs 300 humain  │
   │  0 EUR cloud    │                 │                 │
   │  750 EUR annot. │  1,8M posts/mois│  -67% faux pos. │
   │                 │  0,0005 c/post  │                 │
   └─────────────────┴─────────────────┴─────────────────┘
```

**Notes mise en page** :
- 3 panneaux verticaux, fond #1a1f2e
- Montants principaux en 60pt (cyan / jaune / vert)
- Details en gris 18pt
- "x10 productivite" et "-67% faux positifs" en vert bold

---

## SLIDE 15 — Roadmap V10-V12 (16:30)

**Titre** : `ROADMAP — VERS V12` (cyan 44pt) sous-titre `Limites assumees et perspectives`

**Composition** : 3 colonnes verticales

```
   ┌───────────────┬───────────────┬───────────────┐
   │      V10      │      V11      │      V12      │
   │   Q3 2026     │   Q4 2026     │   2027        │
   ├───────────────┼───────────────┼───────────────┤
   │               │               │               │
   │ . MLflow      │ . ClaimBuster │ . Mistral /   │
   │   tracking    │  + LLM open-w.│   souverain   │
   │ . Drift       │ . Monitoring  │ . Annotation  │
   │   monitoring  │   Grafana     │   communautaire│
   │ . Streamlit   │ . Tests E2E   │ . Federated   │
   │   AppTest     │   nightly     │   learning    │
   │               │               │               │
   ├───────────────┼───────────────┼───────────────┤
   │ Industrialis. │ Verif factuel │  Inclusion    │
   └───────────────┴───────────────┴───────────────┘
```

**Notes mise en page** :
- Colonnes hauteur egale, fond legerement bleute
- Versions en cyan 36pt
- Dates en gris 22pt
- Bullets a puces cyan
- Etiquettes theme en bas en jaune

---

## SLIDE 16 — Citation finale (17:00)

**Composition** : slide noire avec citation centree

```
   ┌────────────────────────────────────────────────┐
   │                                                │
   │                                                │
   │           << Un score sans explication          │
   │             est un verdict sans proces. >>      │
   │                                                │
   │                                                │
   │                                                │
   └────────────────────────────────────────────────┘
                  Fond #000000 . Texte #FFFFFF
                       Inter italique 56pt
```

**Animation** : la citation reste 5 secondes completes en silence (pendant la phrase de conclusion), puis fondu enchaine 1.5s vers slide 17.

---

## SLIDE 17 — Remerciements (17:30)

**Composition** : centree, fond noir bleute

```
                    [LOGO THUMACHECK]

                       MERCI

         Azelie Bernard . Sebastien Lazcanotegui
              Master 1 Big Data IA — 2026

   ─────────────────────────────────────────

   Repository      [QR code -> github.com/...]
   Rapport         [QR code -> docs/pdf/...]
   Model Card      [QR code -> docs/12_model_card.md]

       Questions ? Disponible dans le chat de soutenance
```

**Notes mise en page** :
- Logo et "MERCI" centres haut
- Noms et formation en blanc 24pt
- 3 QR codes alignes horizontalement, 100x100px chacun
- Filets de separation cyan
- Phrase finale en italique gris 18pt

---

## Notes generales sur l'export

**Format final** : exporter chaque slide en **PNG transparent 1920x1080** pour incrustation propre au montage DaVinci. Garder aussi un export MP4 du fichier Keynote complet en backup.

**Animations** : utiliser uniquement Keynote Magic Move ou wipe/fade, JAMAIS les transitions agressives (cube, flip, etc.) qui font tutoriel-debutant.

**Coherence** : toutes les slides doivent partager le meme footer en bas a droite : `Niamato Consulting . ThumaCheck . Client : Thumalien . M1 BDIA 2026` en gris 14pt — repere visuel constant.

**Test rapide de lisibilite** : ouvrir la slide sur un telephone a 50 cm — si tu ne lis pas tout, augmente la police.

---

## Correspondance slides / script video v3

| Slide | Titre | Temps script | Speaker |
|-------|-------|-------------|---------|
| 1 | Titre projet | 00:30 | Azelie |
| 2 | Equipe Niamato | 01:30 | Sebastien |
| 3 | Desinformation Bluesky | 02:00 | Sebastien |
| 4 | Cahier des charges | 02:30 | Sebastien |
| 5 | F1 = 0.99 — Le piege | 03:00 | Sebastien |
| 6 | Architecture C4 | 04:30 | Azelie |
| 7 | Pipeline cascade | 06:00 | Azelie |
| — | Demo dashboard (screencast) | 06:30-09:30 | Azelie |
| 8 | Faithfulness methode | 09:30 | Azelie |
| 9 | Courbe AOPC | 10:30 | Azelie |
| 10 | Qualite industrielle | 12:00 | Sebastien |
| 11 | Conformite reglementaire | 13:30 | Sebastien |
| 12 | Green IT | 14:00 | Sebastien |
| 13 | Methodologie & organisation | 15:00 | Sebastien |
| 14 | ROI & Budget | 15:45 | Sebastien |
| 15 | Roadmap V10-V12 | 16:30 | Sebastien |
| 16 | Citation finale | 17:00 | Azelie |
| 17 | Merci + QR codes | 17:30 | Azelie |
