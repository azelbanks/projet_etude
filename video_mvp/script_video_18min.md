# Script vidéo MVP — ThumaCheck (client : Thumalien)
**Durée cible** : 18 min (15-20 min couverts par la grille MASTERE)
**Format** : screencast + voix off + 3 inserts caméra (intro, transition centrale, conclusion)
**Audience** : jury MASTERE M1 BDIA — exigeant, technique, sensible à la rigueur méthodologique
**Objectif stratégique** : démontrer en 18 min que le projet n'est pas un *exercice scolaire avancé* mais un **système de décision auditable de niveau industriel** avec une **rigueur scientifique**.

---

## 1. Stratégie narrative — l'arc en 3 actes

Plutôt qu'une démo linéaire "voici notre projet, voici nos features, voici notre dashboard", on bâtit la vidéo autour d'un **arc narratif d'enquête** qui transforme la présentation en histoire :

**Acte 1 — La chute (2-5 min)** : on montre le triomphe initial (F1 = 0.99 en cross-validation), puis on révèle qu'il s'agit d'une illusion (biais Reuters découvert grâce au XAI). Cet acte capte l'attention et établit la crédibilité scientifique.

**Acte 2 — La reconstruction (5-13 min)** : itérations V5 → V9, ajout de l'XAI, méta-learner, validation par faithfulness. C'est le cœur technique avec démos live.

**Acte 3 — La validation (13-18 min)** : conformité réglementaire, GreenIT, limites assumées, perspectives. On clôt par une **affirmation forte** plutôt qu'un récap mou.

**Pourquoi cet arc fonctionne** : il transforme l'évaluateur d'un *spectateur passif* en *enquêteur engagé*. Il intègre naturellement les critiques anticipées (kappa modéré, CamemBERT non-prod, asymétrie de contribution) comme des **éléments du récit** plutôt que des aveux défensifs.

---

## 2. Script chronométré — version finale

> **Notation** :
> - `[CAM]` = plan caméra Azélie face caméra
> - `[SCREEN]` = capture écran ou démo live
> - `[SLIDE N]` = slide projetée
> - `[B-ROLL]` = images d'illustration (logo, code qui défile, dashboard)
> - `[MUSIQUE]` = niveau musical (off, doux, montant)
> - `[CUT]` = changement de scène

---

### **00:00 — 00:30** — Hook (CAM, intro forte)

`[CAM]` `[MUSIQUE: off, son ambiant]`

> *"En décembre 2025, on collecte 100 000 posts Bluesky.*
> *On entraîne un modèle de détection de fake news.*
> *On obtient un F1-score de 0,99 en cross-validation.*
> *Et c'est précisément à ce moment-là qu'on aurait dû s'inquiéter."*

`[CUT]` `[SLIDE 1: Titre projet]`

> *"Bonjour, je suis Azélie Bernard. Avec Sébastien Lazcanotegui, on a passé six mois sur ThumaCheck — un système de détection de désinformation bilingue sur Bluesky.*
> *En 18 minutes, je vais vous raconter pourquoi ce 0,99 était un piège, et comment on a construit un système qui, lui, est défendable devant un régulateur, devant un utilisateur, et devant vous."*

**Intention** : ouvrir avec une affirmation contre-intuitive qui crée une attente. Pas de "bonjour je vais vous présenter mon projet". Direct au sujet.

---

### **00:30 — 02:30** — Acte 1, scène 1 : La problématique

`[SLIDE 2: chiffres clés désinformation]`
`[B-ROLL: scroll Bluesky en speed-up, posts d'actualité]`

> *"La désinformation sur les réseaux sociaux décentralisés est un angle mort de la régulation européenne. Bluesky a 35 millions d'utilisateurs et un protocole AT entièrement ouvert — donc plus de 60 000 posts publics par jour analysables, mais aucune équipe de modération centralisée comme Meta ou X.*
>
> *Notre objectif : construire un pipeline NLP capable de classer un post comme 'fiable' ou 'suspect' en moins de 5 millisecondes, en français comme en anglais, avec une explication auditable de chaque décision.*
>
> *Le cahier des charges fixe quatre exigences non-négociables : transparence — on doit pouvoir expliquer chaque score ; multilinguisme FR-EN ; frugalité énergétique ; et conformité au RGPD et à l'AI Act."*

`[SLIDE 3: les 4 exigences en icônes]`

**Intention** : poser le contexte sans lourdeur, citer le cahier des charges explicitement (ça compte sur la grille).

---

### **02:30 — 04:30** — Acte 1, scène 2 : La fausse victoire et sa découverte

`[SCREEN: notebook Jupyter, sortie F1=0.99]`

> *"Voici la première version du pipeline. Logistic Regression sur du TF-IDF avec 30 000 features, entraînée sur 197 782 articles. Cross-validation à 5 plis : F1-score macro de 0,99. Sur le papier, le projet est terminé.*
>
> *Sauf qu'on a fait un test simple : on a regardé quels mots avaient le plus de poids dans les coefficients de la régression."*

`[SCREEN: visualisation des top mots LogReg avec "Reuters", "AFP", "AP" en tête]`

> *"Et c'est là qu'on a vu le problème. Le modèle n'apprenait pas à détecter de la désinformation. Il apprenait à détecter le **style d'agence de presse**. Si un post commençait par 'Reuters', le modèle disait 'fiable'. Si un post avait un style emphatique typique de réseau social, il disait 'suspect'.*
>
> *Cette découverte n'a été possible que parce qu'on a appliqué de l'**explicabilité** dès la version 1. Sans XAI, on aurait livré un modèle à 0,99 de F1 qui aurait collapsé en production. Cette anecdote, c'est la raison pour laquelle l'XAI n'est pas une feature dans ThumaCheck — c'est l'épine dorsale méthodologique."*

`[CUT vers SLIDE 4: "F1=0.99 → biais Reuters → V2"]`

**Intention** : positionner l'XAI comme le **héros du récit**, pas comme un module annexe. Cette scène justifie tout le travail XAI qui suivra.

---

### **04:30 — 07:00** — Acte 2, scène 1 : Itération V2 → V8, l'architecture finale

`[SLIDE 5: Architecture C4 niveau 2 - les 8 conteneurs]`

> *"De la version 2 à la version 9, on a fait neuf itérations majeures, chacune corrigeant une faiblesse identifiée par évaluation sur un gold set annoté manuellement.*
>
> *La version 5, c'est notre baseline frugale : LogReg + TF-IDF + 15 features linguistiques + 7 émotions issues d'un MLP PyTorch. 1,5 milliseconde par texte, F1 macro 0,91.*
>
> *La version 6 ajoute un classifieur **style-only** : 28 features purement stylistiques sans lexique — densité de ponctuation, ratio de majuscules, présence de citations. C'est topic-agnostique : il classe un post de complot sur le climat de la même manière qu'un post de complot sur la politique, parce qu'il ne regarde pas le sujet, il regarde la **forme**.*
>
> *La version 8, c'est l'ensemble : un méta-learner Logistic Regression qui combine V5, V6 et un CamemBERT fine-tuné sur des textes courts français. Sept features d'entrée, dont deux signaux de désaccord entre les modèles."*

`[SLIDE 6: diagramme de séquence inférence — 8 acteurs]`

> *"En production, le pipeline cascade fonctionne ainsi : un premier filtre sépare les opinions des faits, parce que l'AI Act et la jurisprudence européenne convergent sur un point — on ne peut pas qualifier une opinion de 'fake news'. Ensuite, le méta-learner V8 analyse uniquement les contenus factuels.*
>
> *Cette cascade nous fait passer de 57 faux positifs sur le gold set à 21 — une réduction de 67% statistiquement significative au test exact de Fisher, p inférieur à 0,000001."*

**Intention** : compresser six mois de travail en deux minutes en gardant les chiffres qui claquent. Citer la statistique avec son test (Fisher) et son p-value.

---

### **07:00 — 09:30** — Acte 2, scène 2 : Démo dashboard live (le moment fort)

`[CUT]` `[SCREEN: dashboard Streamlit en plein écran]`

> *"Voici le dashboard, qui est l'interface de validation pour un utilisateur opérationnel — un modérateur, un journaliste, un chercheur. Cinq pages.*
>
> *Vue Globale : 245 000 posts collectés, 67% classés fiables, distribution émotionnelle dominée par la neutralité — ce qui, soit dit en passant, contredit le narratif médiatique selon lequel les réseaux sociaux sont saturés de colère.*
>
> *Mais le cœur, c'est cette page : Analyse Temps Réel."*

`[SCREEN: Analyse Temps Réel, soumettre un texte fiable connu]`

> *"Premier test : un communiqué officiel d'une institution scientifique. Score 0,89 fiable. Le modèle est confiant. Mais surtout — et c'est ce qui distingue ThumaCheck d'un classifieur boîte noire — voici **trois niveaux d'explication simultanés**.*
>
> *Niveau 1 : les top mots qui poussent vers fiable, calculés à partir des coefficients exacts de la LogReg. Pas une approximation, c'est la formule fermée du modèle."*

`[SCREEN: pointer chaque section explicabilité]`

> *"Niveau 2 : SHAP appliqué au modèle de style. On voit que la présence de citations de sources et la diversité lexicale poussent vers fiable, alors qu'un score de sensationnalisme moyen pousse légèrement vers suspect.*
>
> *Niveau 3 : et c'est ce que personne ne fait habituellement — la **décomposition exacte du méta-learner**. On voit que CamemBERT contribue pour minus 0,42, V5 pour minus 0,33, et le désaccord entre les modèles pour minus 0,18. La somme plus l'intercept donne le logit final, et on lit littéralement la décision du modèle."*

`[SCREEN: changer pour un texte clairement suspect]`

> *"Deuxième test : un texte sensationnaliste avec 'SCANDALE', 'on vous ment', triple point d'exclamation. Score 0,12 — donc 0,88 suspect. La décomposition est inversée : le score de sensationnalisme est désormais en rouge à plus 0,71, le score de citation à plus 0,54.*
>
> *Et voici l'élément innovant : la heatmap d'attention CamemBERT qui montre **token par token** ce que le modèle 'regarde'. Les mots 'SCANDALE' et 'mentent' s'illuminent en rouge. C'est exactement ce que vous voudriez voir comme modérateur pour défendre votre décision."*

**Intention** : démontrer le dashboard en moins de 3 minutes en montrant **un cas fiable, un cas suspect, et les 4 niveaux d'explication**. Pas de blabla, du concret.

---

### **09:30 — 12:00** — Acte 2, scène 3 : XAI niveau publication et faithfulness

`[CUT]` `[SLIDE 7: titre "Au-delà de l'explicabilité : la fidélité des explications"]`

> *"À ce stade, vous pourriez vous demander : comment savez-vous que ces explications reflètent **vraiment** ce que fait votre modèle, et pas une rationalisation a posteriori ?*
>
> *Cette question — la fidélité des explications, en anglais 'faithfulness' — est la frontière actuelle de la recherche en XAI. La réponse standard est le **protocole ERASER** publié par DeYoung et collègues à ACL 2020. On masque progressivement les top-k features identifiées comme importantes, et on mesure la chute de probabilité prédite. Si l'explication est fidèle, cette chute doit être bien plus rapide qu'avec un masquage aléatoire.*
>
> *On a implémenté ce protocole."*

`[SLIDE 8: courbe AOPC — attribution vs random]`

> *"Voici la courbe sur notre gold set. En rouge, l'attribution SHAP. En gris pointillé, cinq baselines aléatoires. AOPC attribution : 0,253. AOPC random : 0,045. Uplift : +0,21. Autrement dit, **nos explications SHAP sont 5,6 fois plus prédictives** qu'une attribution au hasard.*
>
> *Pour les transformers, on est allés plus loin avec **Layer Integrated Gradients via Captum**, qui donne une attribution causale par token avec garantie axiomatique de Sundararajan, ICML 2017. Sur les cas où le modèle hésite, on atteint un Δ_convergence de 0,04 — niveau axiomatique.*
>
> *Et on a découvert quelque chose d'intéressant. Sur les cas où CamemBERT est très confiant, le ReLU du head MLP sature et bloque les gradients, ce qui rend l'attribution gradient-based difficile. Ce n'est pas un bug — c'est une **signature de la rigidité décisionnelle** du modèle, qu'on documente honnêtement dans la Model Card."*

**Intention** : démontrer une rigueur scientifique de niveau publication. Citer trois articles de référence (DeYoung 2020, Sundararajan 2017, Mitchell 2019) prouve la maîtrise littéraire du domaine.

---

### **12:00 — 13:30** — Transition centrale : qualité industrielle

`[CAM 2 — insert face caméra court]`

> *"Quelqu'un peut me dire à ce stade : 'tout ça c'est très bien, mais quelle est la qualité de votre code ? Ce n'est pas parce que vous avez de jolies courbes que le système est robuste.' C'est une question légitime."*

`[CUT]` `[SLIDE 9: dashboard de qualité — coverage, mutation testing, CI gate]`

> *"Donc on s'est imposé les standards de l'industrie.*
> *501 tests pytest. 80% de line coverage, 77,9% de branch coverage. Quality gate sur GitHub Actions qui rejette toute pull request descendant sous 75%.*
> *Mais ce qui compte vraiment, c'est la qualité des tests, pas leur quantité. Donc on a fait du **mutation testing** avec mutmut sur le module critique de décomposition méta-learner. 178 mutations introduites artificiellement. 143 détectées par nos tests. **Kill rate : 80,3%** — au-dessus de la moyenne Google qui est de 60 à 75%.*
>
> *Cette dimension qualité n'est pas un ajout cosmétique. C'est ce qui sépare un prototype académique d'un système qu'une équipe DevOps peut prendre en main."*

**Intention** : couper court à l'éventuelle critique "joli mais pas robuste". Donner les chiffres bruts et la comparaison Google qui frappe.

---

### **13:30 — 15:30** — Acte 3, scène 1 : Conformité réglementaire et GreenIT

`[CUT]` `[SLIDE 10: panorama Model Card + AI Act + RGPD + CodeCarbon]`

> *"Sur la dimension réglementaire, on a anticipé le règlement sur l'IA européen, l'AI Act, qui sera pleinement applicable en août 2026.*
>
> *On a produit une **Model Card formelle** au format Mitchell 2019, avec une section dédiée à l'explicabilité incluant les méthodes, l'audience par persona, les limites observées et les niveaux de Completeness atteints. Un commissaire CNIL pourrait reprendre ce document tel quel dans un audit.*
>
> *On a réalisé une **AIPD complète** au sens de l'article 35 du RGPD, avec analyse des risques, mesures de mitigation et journal de décision."*

`[SLIDE 11: bilan carbone 6.14g, comparatif V5 vs V8]`

> *"Côté GreenIT, on a tracé chaque entraînement avec CodeCarbon. Total : 6,14 grammes de CO2-équivalent — l'équivalent de 26 mètres en voiture. CamemBERT à lui seul représente 64% de ce bilan, ce qui justifie qu'en production on exécute la version 5 frugale, et que CamemBERT serve uniquement de signal complémentaire dans le méta-learner V8 d'analyse offline.*
>
> *Cette décision — **avoir un modèle puissant entraîné mais pas servi en chaud** — est un choix architectural assumé que la Model Card documente dans la section limites."*

**Intention** : couvrir réglementaire et GreenIT en 2 minutes en montrant que **chaque décision technique est défendable**.

---

### **15:30 — 17:00** — Acte 3, scène 2 : Limites assumées et perspectives

`[CAM 3 — insert face caméra]`

> *"Maintenant la partie qu'aucun étudiant n'aime mettre dans une vidéo de présentation : les limites. Parce que si je vous dis que tout est parfait, vous ne me croyez pas.*
>
> *Limite numéro 1 : notre kappa de Cohen entre annotateurs sur le gold set est de 0,498 sur la version 2 — modéré-faible. La frontière fiable / suspect est intrinsèquement subjective. On atténue avec un bootstrap d'intervalle de confiance à 95% sur la réduction des faux positifs : intervalle de minus 73% à minus 60%, donc l'effet reste robuste, mais on ne peut pas prétendre à une précision quasi-parfaite.*
>
> *Limite numéro 2 : ThumaCheck détecte des **signaux de désinformation**, pas la vérité factuelle. On classe la forme du discours, pas son contenu. C'est documenté dans le guide utilisateur et affiché en clair sur le dashboard.*
>
> *Limite numéro 3 : on est un binôme étudiant. Dans une équipe de quatre comme prévu par le cadre pédagogique, on aurait probablement développé une couche de monitoring drift plus sophistiquée et une intégration MLflow pour le tracking d'expériences. On les a placées dans la roadmap V10 à V12."*

`[SLIDE 12: roadmap V10-V12]`

> *"Roadmap qui inclut MLflow, integration ClaimBuster pour le fact-checking factuel, et un travail sur l'équité algorithmique pour vérifier que le modèle ne discrimine pas certains sujets ou certaines communautés."*

**Intention** : assumer trois limites factuelles désamorce 80% des questions critiques en Q&R. Cite explicitement le cadre pédagogique (équipe de 4) pour montrer qu'on l'a lu et compris.

---

### **17:00 — 18:00** — Conclusion forte (CAM)

`[CUT]` `[CAM]` `[MUSIQUE: doux, montant]`

> *"Pour conclure, je voudrais revenir au F1-score de 0,99 du début.*
>
> *Aujourd'hui, en V9, notre F1 macro sur gold est de 0,67. C'est plus bas. C'est plus honnête. C'est validé par 501 tests, 80% de mutation kill rate, une AIPD, une Model Card, et une explicabilité qui prouve sa propre fidélité avec un uplift de 0,21 contre baseline aléatoire.*
>
> *ThumaCheck n'est pas un classifieur. C'est un **système de décision auditable** — défendable devant un utilisateur, devant un régulateur, et devant vous.*
>
> *La désinformation ne se règle pas par un modèle. Elle se règle par un dispositif technique, méthodologique et éthique articulé. C'est ce qu'on a essayé de construire en six mois.*
>
> *Merci pour votre attention. Je suis disponible pour les questions."*

`[CUT]` `[SLIDE 13: Merci + équipe + lien GitHub + lien rapport]`
`[MUSIQUE: fade out]`

**Intention** : boucler narrativement sur l'ouverture (F1=0.99). La phrase "défendable devant un utilisateur, devant un régulateur, et devant vous" cible explicitement les trois auditoires de l'AI Act et inclut le jury, ce qui est élégant.

---

## 3. Méthode de production senior

### 3.1 Pré-production (1 jour)

**Storyboard détaillé** : pour chaque section du script, dessiner sur papier la composition visuelle. Cela évite les surprises au montage.

**Slides** : Keynote ou Google Slides. **Une règle stricte** : un message par slide, pas de bullet points à plus de 3 lignes, pas de transitions animées agressives. Police sans-serif (Inter, Helvetica) à 32-44pt minimum. Fond sombre cohérent avec le dashboard glassmorphism (couleurs `#0e1117` ou `#1a1f2e`).

**Décor pour les inserts caméra** : fond neutre (mur uni), lumière naturelle de côté ou ring light à 5500K, microphone-cravate ou Yeti USB (jamais le micro intégré du Mac). Tenue professionnelle sobre.

**Captures dashboard préparées** : ouvrir le dashboard avec **deux textes pré-sélectionnés** dans des onglets prêts, pour ne pas perdre 30 secondes en taping pendant la démo. Effacer l'historique de navigation.

**Timer visible** : avoir un chronomètre à l'écran pendant l'enregistrement pour respecter les durées par section. Discipliner les transitions à 2-3 secondes maximum.

### 3.2 Outils techniques recommandés

| Catégorie | Outil | Coût | Pourquoi |
|---|---|---|---|
| Capture écran | OBS Studio | Gratuit | Standard, multi-source, multi-piste audio |
| Montage vidéo | DaVinci Resolve | Gratuit | Niveau pro, color grading, étalonnage audio |
| Slides | Keynote (macOS) | Gratuit | Export 4K propre, transitions clean |
| Audio | Audacity ou Adobe Audition | Gratuit / payant | Réduction de bruit, normalisation à -16 LUFS |
| Sous-titres | Whisper (OpenAI) ou Descript | Gratuit / payant | Transcription auto + édition synchronisée |
| Hébergement | YouTube non listé | Gratuit | Standard MASTERE, qualité 1080p garantie |

### 3.3 Standards de qualité audiovisuelle

**Audio** : c'est 70% du ressenti perçu. Enregistrer la voix off **séparément** des captures écran. Cible : -16 LUFS intégrés (norme broadcast YouTube). Compresser légèrement (ratio 3:1, threshold -18 dB). Aucun bruit de fond audible au-dessus de -50 dB.

**Vidéo** : 1080p 30fps minimum, idéalement 4K 30fps si la captation OBS le permet. Bitrate cible 10-12 Mbps en H.264. Aspect 16:9 strict.

**Cohérence visuelle** : palette de couleurs alignée avec le dashboard (bleu cyan #00D4FF, rouge danger #FF1744, vert succès #00E676, fond #0e1117). Police Inter ou SF Pro. **Logo ThumaCheck** en watermark discret en bas à droite tout au long.

**Rythme** : viser un changement de plan ou un focus de mouvement **toutes les 8-12 secondes**. Pas de plan fixe de plus de 20 secondes — le cerveau décroche.

### 3.4 Innovation visuelle (ce qui va te démarquer)

**1. Animations de "découverte"** : quand tu révèles le biais Reuters à 2:30, fais apparaître les mots un par un avec une animation `wipe` plutôt que tout d'un bloc. Ça crée une tension dramatique.

**2. Picture-in-picture pendant les démos** : pendant la démo dashboard à 7:00, garde un petit insert de toi en bas à droite (15% de l'écran). Ça humanise la démo et évite l'effet "tutoriel YouTube anonyme".

**3. Dashboard XAI animé** : enregistre le clic sur "Analyser" du dashboard → la décomposition apparaît en 2 secondes. Ralentis cette portion à 0,5x au montage pour que le jury voie chaque élément se construire. Effet "machine qui pense" sans être gadget.

**4. Comparaison V1 vs V9 en split-screen** : à 15:30 quand tu mentionnes "F1=0.99 → F1=0.67", split l'écran en deux : à gauche le notebook V1 avec sortie CV, à droite le rapport V9 avec gold set. C'est visuellement frappant et raconte l'humilité scientifique.

**5. Zoom sur les chiffres clés** : à chaque mention d'un nombre important (245k posts, AOPC +0,21, 80,3% mutation, 6,14g CO2), grossir le chiffre à l'écran avec une animation rapide. Le cerveau retient les chiffres frappés visuellement.

**6. Transitions de chapitres avec un compteur** : "01/03 — La chute", "02/03 — La reconstruction", "03/03 — La validation". Ça structure mentalement le jury et signale ton arc narratif explicitement.

### 3.5 Sous-titres et accessibilité

**Sous-titres incrustés en français** sur toute la vidéo. Génération automatique via Whisper, puis correction manuelle (compter 30 min de relecture pour 18 min de vidéo). Format SRT téléchargeable séparément si demandé.

**Description YouTube structurée** :
```
ThumaCheck V9 — Système de détection de désinformation Bluesky
M1 BDIA - Projet d'étude 2025/2026 - Azélie Bernard & Sébastien Lazcanotegui

00:00 Introduction
00:30 Problématique et cadre réglementaire
02:30 Découverte du biais Reuters
04:30 Architecture V5-V8 et pipeline cascade V9
07:00 Démonstration dashboard live
09:30 XAI : SHAP, Captum, validation faithfulness
12:00 Qualité industrielle : tests, coverage, mutation
13:30 Conformité AI Act, RGPD, GreenIT
15:30 Limites assumées et perspectives
17:00 Conclusion

Repository : [lien GitHub]
Rapport complet : [lien PDF]
Model Card : [lien Model Card]
```

Cette description est elle-même un livrable montrant ta rigueur — utilise-la.

---

## 4. Plan de tournage opérationnel (3 demi-journées)

### Jour 1 — Préparation (matin)

- 2h : finaliser slides Keynote (13 slides au total)
- 1h : préparer dashboard avec textes test pré-chargés et onglets prêts
- 30 min : tester l'enregistrement OBS sur 2 minutes pour valider audio/vidéo
- 30 min : impression du script chronométré au format A4 paysage pour relecture pendant le tournage

### Jour 1 — Captation captures écran (après-midi)

Enregistrer **toutes les sections SCREEN en une session** sans voix off, juste captures écran avec curseur :
- 3 démos dashboard (fiable, suspect, comparaison)
- 1 capture notebook V1 avec biais Reuters
- 1 capture courbe AOPC
- 1 capture coverage report
- 1 capture résultats mutmut

Format : MP4 1080p 30fps, fichier par séquence nommé `screen_XX_description.mp4`.

### Jour 2 — Captation voix off (matin, voix au mieux)

Lire le script chronométré section par section, **deux prises minimum par section** pour avoir le choix au montage. Conditions : pièce silencieuse, micro à 15 cm de la bouche, pop filter. Boire de l'eau toutes les 10 minutes pour éviter le claquement de bouche audible.

Format : WAV 48 kHz 24-bit (qualité maximale, le compromis se fera au final mix). Fichier par section nommé `vo_XX_description.wav`.

### Jour 2 — Captation caméra (après-midi)

3 inserts caméra à enregistrer :
- Hook intro (00:00-00:30) — 5 prises
- Transition centrale qualité (12:00-12:15) — 3 prises
- Conclusion (17:00-18:00) — 5 prises

Format : MP4 1080p 30fps, 16:9, plan poitrine. Fond neutre, regard caméra.

### Jour 3 — Montage et finition

- 2h : montage chronologique (DaVinci Resolve), assembler captures + voix off + inserts caméra
- 1h : ajouter slides en surimpression aux moments indiqués
- 1h : color grading homogène, normalisation audio à -16 LUFS
- 1h : génération sous-titres Whisper + corrections
- 1h : ajout titres de chapitres, animations chiffres, transitions
- 30 min : export final 1080p H.264 (~ 1,5 GB pour 18 min)
- 30 min : upload YouTube non listé, description, miniature personnalisée

**Total : 6 heures de montage** pour un résultat senior. Si tu rushes, tu auras une vidéo correcte. Si tu respectes ces 6 heures, tu auras une vidéo qui sort du lot.

---

## 5. Innovation finale — la touche qui peut transformer la note

**Idée 1 : ouvrir avec un faux écran de news**

Les 5 premières secondes de la vidéo, montrer un écran simulant un téléphone Bluesky avec un post viral suspect en train d'être partagé. Animation "Scroll → Post → Like → Repost x1000". Voix off : *"En décembre 2025, on collecte 100 000 posts comme celui-ci..."*. Coût : 30 min sur Figma + After Effects ou Keynote magic move. Effet : cinématographique.

**Idée 2 : intégrer une démo en direct du XAI dans une réponse à une question fictive**

À 9:30, au lieu de présenter SHAP comme une fonctionnalité, mets en scène une question : *"Imaginez qu'un journaliste vous appelle et vous dit : 'votre modèle a classé mon article comme suspect, expliquez-moi pourquoi'. Voici comment on répond."* — puis démo. Ça transforme une explication technique en cas d'usage humain.

**Idée 3 : finir par une citation forte qui résume ton positionnement**

Avant le merci final, intercaler une slide noire avec une seule phrase blanche au centre :

> *"Un score sans explication est un verdict sans procès."*

Citer à l'oral : *"Un score sans explication est un verdict sans procès. C'est la phrase qui a guidé chacune de nos décisions techniques."*

C'est puissant, c'est mémorable, ça résume ta thèse en une phrase. Le jury va te citer cette phrase en délibération.

**Idée 4 : easter egg pour les jurys techniques**

Dans la description YouTube, ajouter en dernier paragraphe :

> *"Pour les évaluateurs techniques : `git clone` puis `python scripts/run_xai_pipeline.py` reproduit toutes les figures de cette vidéo en moins de 3 minutes. Quality gate `--fail-under=75` actif sur CI."*

Petite phrase qui montre que tu sais à qui tu t'adresses et que tu as confiance dans la reproductibilité de ton travail.

---

## 6. Checklist de validation avant publication

Avant de cliquer sur "publier" :

- [ ] Audio normalisé à -16 LUFS, aucun pic au-dessus de -1 dB
- [ ] Sous-titres français corrigés, sans erreur factuelle
- [ ] Tous les chiffres cités à l'oral correspondent à ceux du rapport et de la Model Card
- [ ] Logo ThumaCheck présent en watermark sur 100% de la vidéo
- [ ] Aucune information personnelle visible (notification email, chat Slack) dans les captures
- [ ] Lien GitHub testé, mène bien au repo public
- [ ] Lien rapport testé, le PDF s'ouvre correctement
- [ ] Vidéo lue en entier sans pause pour vérifier la cohérence narrative
- [ ] Test sur smartphone : la vidéo reste lisible et audible sur petit écran
- [ ] Description YouTube structurée avec timestamps cliquables
- [ ] Miniature YouTube personnalisée (1280x720, contraste fort, lisible)
- [ ] Confidentialité : "non listé" (pas "privé" qui demanderait un compte autorisé)
- [ ] Lien YouTube placé dans `PE-2526_M1BDIA_BernardAzelie_LazcanoteguiSebastien_video.txt` à la racine du livrable

---

## 7. Résumé exécutif

| Axe | Decision |
|---|---|
| **Durée** | 18 min (cible mid-range de la grille 15-20) |
| **Structure** | Arc narratif 3 actes : chute → reconstruction → validation |
| **Innovation** | Ouverture sur un échec (F1=0.99 piège), démo dashboard avec 4 niveaux XAI, mention explicite du mutation testing 80,3% |
| **Différenciation** | Storytelling enquête + comparaison Google + références scientifiques (DeYoung, Sundararajan, Mitchell) |
| **Qualité audio** | -16 LUFS, micro dédié, voix off séparée |
| **Qualité vidéo** | 1080p 30fps, palette cohérente avec dashboard |
| **Sous-titres** | Français incrustés + SRT téléchargeable |
| **Production** | 3 demi-journées (préparation, captation, montage) |
| **Publication** | YouTube non listé + lien dans le ZIP livrable |

**Phrase d'ancrage** : *"Un score sans explication est un verdict sans procès."*

---

*Script vidéo ThumaCheck (client : Thumalien) — version finale Mai 2026 — Azélie Bernard*
