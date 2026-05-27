# 13 Slides Keynote — Vidéo Thumalien V9
**Format** : 16:9, 1920×1080, exporté en MP4 ou PNG haute résolution pour incrustation montage
**Palette** : fond `#0e1117` (noir bleuté dashboard), primaire `#00D4FF` (cyan), succès `#00E676` (vert), danger `#FF1744` (rouge), accent `#FFD600` (jaune), texte `#E8E8E8`
**Police** : Inter ou SF Pro Display — titres 48-64pt, corps 28-32pt, légendes 18-22pt
**Règle absolue** : un message par slide, max 3 puces, jamais plus de 30 mots à l'écran simultanément

---

## SLIDE 1 — Titre projet (00:30)

**Composition** : centrée, fond dégradé noir-bleu, logo Thumalien grand format en haut

```
                    [LOGO THUMALIEN — 200px hauteur]

              THUMALIEN V9
   Détection de désinformation Bluesky — bilingue, explicable, frugale

              ──────────────────────────

           Azélie Bernard · Sébastien Lazcanotegui
              Master 1 Big Data IA — 2025/2026

                         [QR code → repo GitHub]
```

**Notes mise en page** :
- Titre principal `THUMALIEN V9` en cyan #00D4FF, font-weight 700, 72pt
- Sous-titre en gris #B0B0B0, 28pt
- Filet horizontal cyan 2px largeur 60% centré
- Noms en blanc 24pt, formation en gris 18pt
- QR code 120×120px en bas droite (URL : ton repo GitHub public)

**Visuel innovation** : faire apparaître le logo en fade-in sur les premiers mots de l'intro caméra, puis tout le reste qui se compose en wipe vertical descendant en 1.2s.

---

## SLIDE 2 — Problématique chiffrée (01:00)

**Titre** : `LA DÉSINFORMATION SUR BLUESKY` (cyan, 44pt, en haut gauche)

**Composition** : 4 chiffres en grille 2×2, chacun avec icône SF Symbol

```
   ┌─────────────────────┬─────────────────────┐
   │     35 millions     │     60 000+/jour    │
   │  utilisateurs Bluesky│   posts publics FR/EN│
   ├─────────────────────┼─────────────────────┤
   │      0 équipe       │     Angle mort      │
   │   modération centr. │   régulation EU     │
   └─────────────────────┴─────────────────────┘
```

**Notes mise en page** :
- Chiffres en cyan 80pt bold
- Légendes en gris 22pt sous chaque chiffre
- Bordures cellules : trait gris #2a2f3e 1px
- Padding interne 40px

**Animation** : les 4 cellules apparaissent en cascade haut-gauche → bas-droite, intervalle 0.3s.

---

## SLIDE 3 — Les 4 exigences du cahier des charges (02:00)

**Titre** : `LE CAHIER DES CHARGES` (cyan, 44pt)

**Composition** : 4 cartes horizontales avec icône à gauche + texte à droite

```
   🔍  TRANSPARENCE          Chaque score doit être explicable et auditable
   🌐  BILINGUE              Français + anglais avec performance équivalente
   🌱  FRUGALITÉ             < 5ms par texte, empreinte CO2 mesurée
   ⚖️  CONFORMITÉ            RGPD art. 22 + AI Act art. 13/14
```

**Notes mise en page** :
- Icônes 64×64px en couleur (cyan, vert, jaune, rouge selon thème)
- Mots-clés en blanc bold 32pt
- Descriptions en gris 22pt
- Hauteur uniforme 100px par carte, gap 20px

**Phrase basse de slide (en footer 16pt cyan)** : *"Quatre exigences non-négociables pour passer du prototype au système déployable"*

---

## SLIDE 4 — La fausse victoire et sa découverte (03:00)

**Titre** : `F1 = 0.99 — LE PIÈGE` (rouge #FF1744, 44pt, dramatique)

**Composition** : split horizontal 50/50

À gauche :
```
   CROSS-VALIDATION 5-FOLD
   ┌──────────────────────┐
   │   F1 macro:  0.987  │
   │   Précision: 0.992  │
   │   Recall:    0.984  │
   │   Accuracy:  0.989  │
   └──────────────────────┘
   ✅ Tous les voyants verts
```

À droite :
```
   TOP MOTS LOGREG (XAI)
   ┌──────────────────────┐
   │  reuters       +0.84 │
   │  afp           +0.71 │
   │  ap_news       +0.69 │
   │  associated    +0.63 │
   │  press         +0.58 │
   └──────────────────────┘
   ⚠️ Le modèle apprend le STYLE
       d'agence de presse
```

**Notes mise en page** :
- Tableaux gauche/droite encadrés, fond #1a1f2e
- Ligne séparatrice verticale rouge 2px au centre
- Cocher vert + flèche d'attention rouge avec animation arrivée séquentielle

**Phrase basse (footer 18pt rouge)** : *"Sans XAI, on aurait livré un modèle qui collapse en production"*

**Animation cruciale** : le tableau de gauche apparaît immédiatement avec la slide. Le tableau de droite arrive 3 secondes après (au moment où tu dis "on a fait un test simple"). C'est la révélation visuelle.

---

## SLIDE 5 — Architecture C4 niveau 2 (05:00)

**Titre** : `ARCHITECTURE — 8 CONTENEURS` (cyan, 44pt)

**Composition** : diagramme C4 complet, exporté depuis Mermaid ou structure

```
   ┌────────────┐    ┌────────────┐    ┌──────────────┐
   │  Bluesky   │───▶│ Collector  │───▶│   MongoDB    │
   │  AT Proto  │    │  (Python)  │    │  (storage)   │
   └────────────┘    └────────────┘    └──────────────┘
                                               │
                                               ▼
   ┌────────────┐    ┌────────────┐    ┌──────────────┐
   │ Streamlit  │◀───│ Pipeline   │◀───│  Detector    │
   │ Dashboard  │    │  Cascade   │    │  V5 + V6     │
   └────────────┘    │  Stage 1+2 │    │  + CamemBERT │
         │           └────────────┘    └──────────────┘
         │                  │
         ▼                  ▼
   ┌────────────┐    ┌──────────────┐
   │ XAI Engine │    │  Monitoring  │
   │ SHAP+Captum│    │  (CodeCarbon)│
   └────────────┘    └──────────────┘
```

**Notes mise en page** :
- Conteneurs en boîtes #1a1f2e bordure cyan 1px
- Flèches cyan 2px avec pointe
- Légende : "C4 Model — Niveau 2 (Container)" en bas droite 16pt gris

**Pourquoi C4 et pas un random schéma** : C4 est le standard de fait pour la documentation d'architecture en industrie. Le mentionner par son nom signale ta connaissance des conventions pro.

---

## SLIDE 6 — Diagramme de séquence cascade V9 (06:00)

**Titre** : `PIPELINE CASCADE — INFÉRENCE` (cyan, 44pt)

**Composition** : diagramme de séquence horizontal, 6 acteurs

```
  User    Stage1     V5      V6      CamBERT   V8     XAI
   │       │         │       │        │        │      │
   │ text  │         │       │        │        │      │
   ├──────▶│         │       │        │        │      │
   │       │ opinion?│       │        │        │      │
   │       ├────╗    │       │        │        │      │
   │       │    ║    │       │        │        │      │
   │       │◀───╝    │       │        │        │      │
   │       │         │       │        │        │      │
   │       │ if FAIT │       │        │        │      │
   │       ├────────▶│       │        │        │      │
   │       │         │ score │        │        │      │
   │       │         ├──────▶│        │        │      │
   │       │         │       │ style  │        │      │
   │       │         │       ├───────▶│        │      │
   │       │         │       │        │ embed  │      │
   │       │         │       │        ├───────▶│      │
   │       │ score V8                          ├─────▶│
   │       │◀──────────────────────────────────│      │
   │ score+│         │       │        │        │      │
   │ XAI   │         │       │        │        │      │
   │◀──────│         │       │        │        │      │
```

**Notes mise en page** :
- Acteurs en haut sur ligne cyan 2px
- Lignes de vie verticales en gris pointillé
- Messages flèches cyan
- Annotation "1.5ms total" en bas avec encadré jaune
- Symbole AI Act : `⚖️` à côté de Stage 1 avec tooltip *"opinion ≠ fake news"*

**Animation possible** : faire défiler les flèches une par une avec un timing 0.3s pendant que la voix off décrit le processus.

---

## SLIDE 7 — Faithfulness : pourquoi et comment (10:30)

**Titre** : `LA FIDÉLITÉ DES EXPLICATIONS` (cyan 44pt) avec sous-titre `Au-delà de SHAP : la mesurer` (gris 24pt)

**Composition** : moitié haute = question, moitié basse = méthode

```
   ┌────────────────────────────────────────────────┐
   │   ❓ Comment savez-vous que vos explications   │
   │     reflètent vraiment le comportement         │
   │     du modèle ?                                │
   └────────────────────────────────────────────────┘

                         ↓

   ┌────────────────────────────────────────────────┐
   │   PROTOCOLE ERASER (DeYoung et al., ACL 2020) │
   │                                                │
   │   1. Identifier les top-k features (SHAP)     │
   │   2. Masquer ces features (= 0)               │
   │   3. Mesurer la chute de P(suspect)           │
   │   4. Comparer à un masquage aléatoire         │
   │                                                │
   │   → Si l'explication est fidèle :             │
   │     chute_attribution >> chute_random         │
   └────────────────────────────────────────────────┘
```

**Notes mise en page** :
- Question en haut en cyan 32pt, encadrée
- Flèche descendante grosse cyan
- Méthode en cadre #1a1f2e avec 4 étapes numérotées
- Conclusion en bas en jaune bold 24pt

---

## SLIDE 8 — Courbe AOPC (11:30)

**Titre** : `RÉSULTAT FAITHFULNESS — UPLIFT +21%` (vert #00E676 44pt)

**Composition** : graphique pleine largeur + chiffres clés à droite

À gauche (70%) : la figure `faithfulness_aopc_curve.png` agrandie

À droite (30%) : tableau récap

```
   AOPC attribution  : 0.253
   AOPC random       : 0.045
   ─────────────────────────
   Uplift            : +0.21
   Ratio             : 5.6×

   Comprehensiveness@5 : 0.232
   Sufficiency@5       : 0.058

   ✅ Cible >+0.10
   ✅ Atteinte (×2)
```

**Notes mise en page** :
- Figure encadrée subtilement
- Chiffres clés alignés à droite avec espacement aéré
- Ligne séparatrice horizontale entre AOPC et Compr/Suff
- Cocher vert pour signaler l'atteinte de cible

---

## SLIDE 9 — Qualité industrielle (12:30)

**Titre** : `QUALITÉ INDUSTRIELLE` (cyan 44pt) sous-titre `501 tests · 80% coverage · 80,3% mutation kill rate`

**Composition** : 3 panneaux verticaux

```
   ┌─────────────┬─────────────┬─────────────┐
   │   PYTEST    │  COVERAGE   │  MUTMUT     │
   │             │             │             │
   │    501      │    80%      │   80,3%     │
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
- 3 cellules largeur égale, hauteur 500px
- Chiffres principaux en 90pt cyan/vert/jaune
- Sous-textes en gris 20pt
- Mention "> Google benchmark (60-75%)" en italique blanc bold (l'argument qui frappe)

**Footer (en bas, 18pt gris)** : *"De la quantité à la qualité — chaque ligne de production est validée"*

---

## SLIDE 10 — Conformité AI Act et RGPD (14:00)

**Titre** : `CONFORMITÉ RÉGLEMENTAIRE` (cyan 44pt)

**Composition** : timeline horizontale + chips de conformité

```
   AI ACT (UE 2024/1689) ────────┬───── pleinement applicable août 2026
                                 │
   ✅ Art. 13 (transparence)     │  → Model Card MC-THUM-2026-001
   ✅ Art. 14 (supervision)      │  → Décomposition β·x dashboard
   ✅ Risque limité classifié    │  → Doc 02 § 4.1
                                 │
   RGPD (UE 2016/679)            │
                                 │
   ✅ Art. 22 (décision auto)    │  → Droit explication SHAP+Captum
   ✅ Art. 35 (AIPD)             │  → Document RGPD-THUM-2026-001
   ✅ Base légale art. 6.1.f     │  → Posts publics, intérêt légitime
```

**Notes mise en page** :
- Coches vertes 24px
- Articles en monospace bleu pâle pour effet "code juridique"
- Trait vertical cyan séparant articles et preuves

---

## SLIDE 11 — GreenIT et arbitrage CamemBERT (14:30)

**Titre** : `GREEN IT — 6,14 g CO2 TOTAL` (vert 44pt)

**Composition** : pie chart à gauche + tableau de décision à droite

À gauche : pie chart de répartition

```
   Pie chart 350×350px :
   - CamemBERT fine-tune : 64% (cyan foncé)
   - V5 LogReg training  : 18% (cyan)
   - V6 GradientBoost    : 12% (cyan clair)
   - Pipeline XAI run    :  4% (jaune)
   - Inference cumulée   :  2% (vert)
```

À droite :

```
   DÉCISION ARCHITECTURALE

   Production :
   → V5 (1.5 ms, 0.6 g CO2/jour)

   Analyse offline :
   → V8 incluant CamemBERT
     (3.9 g CO2 entraînement unique)

   ROI GreenIT :
   → Frugalité prod + puissance analyse
   → Documenté dans Model Card § 8
```

**Notes mise en page** :
- Pie chart avec légende intégrée
- Tableau décision en cadre vert pâle
- Mention "26 m en voiture équivalent" en footer pour vulgariser

---

## SLIDE 12 — Roadmap V10-V12 (16:30)

**Titre** : `ROADMAP — VERS V12` (cyan 44pt) sous-titre `Limites assumées et perspectives`

**Composition** : 3 colonnes verticales

```
   ┌───────────────┬───────────────┬───────────────┐
   │      V10      │      V11      │      V12      │
   │   Q3 2026     │   Q4 2026     │   2027        │
   ├───────────────┼───────────────┼───────────────┤
   │               │               │               │
   │ • MLflow      │ • ClaimBuster │ • Modèle FR   │
   │   tracking    │   integration │   spécialisé  │
   │ • Drift       │ • Monitoring  │ • Annotation  │
   │   monitoring  │   Grafana     │   communautaire│
   │ • Streamlit   │ • Tests E2E   │ • Federated   │
   │   AppTest     │   nightly     │   learning    │
   │               │               │               │
   ├───────────────┼───────────────┼───────────────┤
   │ Industrialis. │ Vérif factuel│  Inclusion    │
   └───────────────┴───────────────┴───────────────┘
```

**Notes mise en page** :
- Colonnes hauteur égale, fond légèrement bleuté
- Versions en cyan 36pt
- Dates en gris 22pt
- Bullets à puces cyan
- Étiquettes thème en bas en jaune

---

## SLIDE 13 — Citation finale + remerciements (17:30)

**Composition phase 1 (slide noire avec citation)** :

```
   ┌────────────────────────────────────────────────┐
   │                                                │
   │                                                │
   │           « Un score sans explication          │
   │             est un verdict sans procès. »      │
   │                                                │
   │                                                │
   │                                                │
   └────────────────────────────────────────────────┘
                  Fond #000000 · Texte #FFFFFF
                       Inter italique 56pt
```

**Composition phase 2 (transition douce vers remerciements)** :

```
                    [LOGO THUMALIEN]

                       MERCI

         Azélie Bernard · Sébastien Lazcanotegui
              Master 1 Big Data IA — 2026

   ─────────────────────────────────────────

   Repository      [QR code → github.com/...]
   Rapport         [QR code → docs/pdf/...]
   Model Card      [QR code → docs/12_model_card.md]

       Questions ? Disponible dans le chat de soutenance
```

**Notes mise en page phase 2** :
- Logo et "MERCI" centrés haut
- Noms et formation en blanc 24pt
- 3 QR codes alignés horizontalement, 100×100px chacun
- Filets de séparation cyan
- Phrase finale en italique gris 18pt

**Animation transition phase 1 → 2** : la citation reste 5 secondes complètes en silence (pendant que tu prononces la phrase), puis fondu enchaîné de 1.5s vers la slide remerciements. Ce silence renforce l'impact.

---

## Notes générales sur l'export

**Format final** : exporter chaque slide en **PNG transparent 1920×1080** pour incrustation propre au montage DaVinci. Garder aussi un export MP4 du fichier Keynote complet en backup au cas où une slide doit être projetée live pendant la soutenance.

**Animations** : utiliser uniquement Keynote Magic Move ou wipe/fade, JAMAIS les transitions agressives (cube, flip, etc.) qui font tutoriel-débutant.

**Cohérence** : toutes les slides doivent partager le même footer en bas à droite : `Thumalien V9 · M1 BDIA 2026` en gris 14pt — repère visuel constant.

**Test rapide de lisibilité** : ouvrir la slide sur un téléphone à 50 cm — si tu ne lis pas tout, augmente la police.
