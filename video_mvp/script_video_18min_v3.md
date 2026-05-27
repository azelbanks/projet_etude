# Script video 18 min — Thumalien V9 (v3 client)

**Duree cible** : 18 min (15-20 min cadre MASTERE)
**Format** : screencast + voix off + inserts camera (intro, transitions, conclusion)
**Speakers** : Azelie Bernard (A) / Sebastien Lazcanotegui (S) — repartition 50/50
**Entreprise** : Niamato Consulting (expertise Data & IA)
**Audience** : client (entreprise media / fact-checking / plateforme) + jury evaluateur
**Objectif** : convaincre un decideur que Thumalien resout un probleme business reel, avec ROI mesurable

---

## Repartition du temps de parole

| Speaker | Sections | Duree |
|---------|----------|-------|
| **Azelie** | Hook + Contexte client + Architecture + Demo dashboard + XAI + Conclusion | ~9 min |
| **Sebastien** | Equipe + Problematique + Iterations + Qualite industrielle + Conformite + Green IT + Methodologie + Limites + ROI | ~9 min |

---

## Arc narratif en 3 actes

**Acte 1 — Le besoin (0:00 - 4:30)** : le probleme client, l'equipe, le piege F1=0.99
**Acte 2 — La solution (4:30 - 12:00)** : architecture, demo live, explicabilite
**Acte 3 — La valeur (12:00 - 18:00)** : qualite, conformite, ROI, methodologie, roadmap

---

## Script chronometre

> **Notation** :
> - `[A-CAM]` = Azelie face camera
> - `[S-CAM]` = Sebastien face camera
> - `[SCREEN]` = capture ecran ou demo live
> - `[SLIDE N]` = slide projetee (numerotation PPTX 1-17)
> - Bandeau nom affiche pour chaque prise de parole camera

---

### 00:00 - 00:30 — Hook (Azelie, camera)

`[A-CAM]` *Bandeau : Azelie Bernard — Lead Technique & Architecture*

> *"En decembre 2025, on collecte 100 000 posts Bluesky.
> On entraine un modele de detection de fake news.
> On obtient un F1-score de 0,99 en cross-validation.
> Et c'est precisement a ce moment-la qu'on aurait du s'inquieter."*

---

### 00:30 - 01:30 — Contexte client et besoin (Azelie)

`[SLIDE 1 : Titre projet]`

> *"Bonjour, je suis Azelie Bernard, lead technique chez Niamato Consulting.
> Nous sommes un cabinet d'expertise Data et Intelligence Artificielle.
> Nous presentons aujourd'hui Thumalien : une solution de detection
> de desinformation pour les plateformes de reseaux sociaux decentralises.
>
> Notre client cible : toute organisation qui analyse des contenus publics —
> media, fact-checker, observatoire de la desinformation, ou plateforme
> qui souhaite outiller ses moderateurs.
>
> Le probleme business est simple : un moderateur humain met en moyenne
> 3 a 5 minutes par post pour evaluer sa fiabilite.
> A 60 000 posts publics par jour sur Bluesky, c'est intenable.
> Thumalien analyse un texte en 1,5 milliseconde, avec une explication
> auditable a chaque decision."*

---

### 01:30 - 03:00 — Equipe + problematique (Sebastien, camera)

`[S-CAM]` *Bandeau : Sebastien Lazcanotegui — Validation & Qualite ML*

`[SLIDE 2 : Equipe et roles]`

> *"Bonjour, je suis Sebastien Lazcanotegui, consultant chez Niamato Consulting.
>
> Mon role : la validation et la qualite du machine learning.
> J'ai pilote l'annotation du gold set — 473 posts annotes
> par deux annotateurs humains — le debiaisage des donnees,
> et l'optimisation des hyperparametres.
>
> Azelie a concu l'architecture du pipeline, le dashboard
> et les trois niveaux d'explicabilite."*

`[SLIDE 3 : Chiffres cles]`

> *"Bluesky : 35 millions d'utilisateurs, protocole AT entierement ouvert,
> plus de 60 000 posts publics par jour analysables.
> Aucune equipe de moderation centralisee.
>
> Notre promesse au client : classifier un post comme fiable ou suspect
> en moins de 5 millisecondes, en francais comme en anglais,
> avec une explication qui permet au moderateur de defendre sa decision."*

`[SLIDE 4 : Les 4 exigences]`

> *"Le cahier des charges fixe quatre exigences non-negociables.
> Transparence : chaque score est explicable.
> Bilinguisme : francais et anglais a performance equivalente.
> Frugalite : moins de 5 ms par texte, empreinte CO2 mesuree.
> Et conformite au RGPD et a l'AI Act.
> Ces quatre exigences structurent toute la solution."*

---

### 03:00 - 04:30 — La fausse victoire et les iterations (Sebastien)

`[SCREEN : notebook Jupyter, sortie F1=0.99]`

> *"Voici notre premiere version du pipeline.
> Logistic Regression sur du TF-IDF, 30 000 features,
> entrainee sur 197 782 articles.
> Cross-validation a 5 plis : F1 macro de 0,99.
> Sur le papier, le projet est termine."*

`[SLIDE 5 : F1=0.99 — LE PIEGE]`

> *"Sauf qu'on a regarde quels mots avaient le plus de poids.
> Et on a trouve : reuters, afp, associated press.
>
> Le modele n'apprenait pas a detecter la desinformation.
> Il apprenait a reconnaitre le style des agences de presse.
>
> C'est moi qui ai pilote le debiaisage : creation de la liste
> BODY_AGENCY_TERMS pour neutraliser les signatures d'agences,
> filtrage des annees artefacts 2015-2020, et les tests de non-regression
> apres chaque correction.
>
> En parallele, j'ai lance un GridSearch systematique :
> le passage de C=1 a C=5 et de min_df=3 a min_df=5
> a confirme les hyperparametres optimaux.
>
> De la version 2 a la version 5, on a corrige ce biais
> et ajoute des donnees synthetiques bilingues.
> Le F1 en cross-validation est descendu a 0,91.
> Mais cette fois, c'est un F1 honnete."*

---

### 04:30 - 06:30 — Architecture et pipeline cascade (Azelie)

`[A-CAM]` *Bandeau : Azelie Bernard — Lead Technique & Architecture*

> *"A partir de cette base saine, j'ai itere l'architecture
> jusqu'a la version 9 actuelle."*

`[SLIDE 6 : Architecture C4]`

> *"Voici notre architecture, modelisee en C4 niveau 2.
> Le flux est simple : donnees, intelligence artificielle, decision.
>
> Le collecteur se connecte a Bluesky via le protocole AT
> et stocke les posts dans MongoDB — c'est la couche donnees.
> Le pipeline NLP analyse chaque texte avec trois modeles —
> c'est la couche IA.
> Le dashboard Streamlit expose les resultats avec trois niveaux
> d'explication — c'est la couche decision.
> Le tout est conteneurise avec Docker Compose.
>
> Trois modeles travaillent en ensemble :
> V5 — LogReg + TF-IDF + 15 features linguistiques + 7 emotions,
> c'est notre baseline frugale a 1,5 milliseconde.
> V6 — un classifieur style-only avec 28 features purement stylistiques,
> topic-agnostique.
> CamemBERT fine-tune sur des textes courts francais.
> Le meta-learner V8 combine les trois."*

`[SLIDE 7 : Pipeline cascade]`

> *"En production, le pipeline cascade fonctionne en deux etapes.
> Le Stage 1 separe les opinions des faits — parce que l'AI Act
> interdit de qualifier une opinion de fake news.
> Le Stage 2 analyse uniquement les contenus factuels.
>
> Cette cascade fait passer nos faux positifs de 57 a 21
> sur le gold set — une reduction de 67 %,
> statistiquement significative, p < 0.000001."*

---

### 06:30 - 09:30 — Demo dashboard live (Azelie)

`[SCREEN : dashboard Streamlit plein ecran]`

> *"Voici le dashboard — l'interface que le client utilise.
> Cinq pages. Vue Globale : 245 000 posts collectes,
> 67 % classes fiables.
>
> Chaque visualisation a ete pensee pour la decision :
> la jauge de score utilise un code couleur vert-rouge intuitif,
> les barres SHAP montrent la contribution positive ou negative
> de chaque feature, et la heatmap d'attention utilise un gradient
> qui guide l'oeil du moderateur vers les tokens critiques."*

`[SCREEN : soumettre texte fiable]`

> *"Premier test : un communique scientifique.
> Score 0,89 fiable. Mais ce qui distingue Thumalien
> d'un classifieur boite noire — trois niveaux d'explication.
>
> Niveau 1 : les top mots de la LogReg.
> Pas une approximation — c'est la formule fermee exacte du modele.
>
> Niveau 2 : SHAP applique au modele de style V6.
> La presence de citations et la diversite lexicale poussent vers fiable.
>
> Niveau 3 : la decomposition exacte du meta-learner V8.
> On lit litteralement la decision du modele,
> coefficient par coefficient.
>
> En tant que moderateur, vous avez tout ce qu'il faut
> pour justifier votre decision aupres de votre hierarchie."*

`[SCREEN : soumettre texte suspect]`

> *"Deuxieme test : un texte sensationnaliste.
> Score 0,12, donc 0,88 suspect.
> La decomposition est inversee : le sensationnalisme
> passe en rouge a +0,71.
>
> Et voici la heatmap d'attention CamemBERT :
> les tokens 'SCANDALE' et 'mentent' s'illuminent en rouge.
> C'est exactement ce qu'un moderateur veut voir
> pour defendre sa decision."*

---

### 09:30 - 12:00 — XAI et faithfulness (Azelie)

`[SLIDE 8 : Faithfulness methode]`

> *"Question legitime du client : comment savez-vous
> que ces explications refletent vraiment le comportement du modele ?
>
> On a implemente le protocole ERASER de DeYoung et collegues, ACL 2020.
> On masque les top-k features identifiees par SHAP,
> et on mesure si la prediction chute plus vite
> qu'avec un masquage aleatoire."*

`[SLIDE 9 : courbe AOPC]`

> *"Le resultat : AOPC attribution 0,253 contre 0,045 pour le random.
> Uplift de +0,21. Nos explications sont 5,6 fois plus predictives
> qu'une attribution au hasard.
>
> Pour les transformers, on est alles plus loin avec
> Layer Integrated Gradients via Captum.
>
> Et on a decouvert que sur les cas ou CamemBERT est tres confiant,
> le ReLU du head sature et bloque les gradients.
> C'est une signature documentee dans la Model Card —
> on sait ou nos explications sont fiables et ou elles ne le sont pas."*

---

### 12:00 - 13:30 — Qualite industrielle (Sebastien)

`[S-CAM]` *Bandeau : Sebastien Lazcanotegui — Validation & Qualite ML*

`[SLIDE 10 : Qualite industrielle]`

> *"Quand un client evalue une solution, il regarde aussi
> la qualite du code — est-ce que ca tiendra en production ?
>
> 501 tests pytest. 80 % de couverture de code.
> 77,9 % de branch coverage.
> Quality gate sur GitHub Actions qui rejette toute PR
> descendant sous 75 %.
>
> On a fait du mutation testing avec mutmut
> sur le module critique de decomposition meta-learner.
> 178 mutations artificielles. 143 detectees.
> Kill rate : 80,3 % — au-dessus de la moyenne Google
> qui se situe entre 60 et 75 %.
>
> C'est ce qui separe un prototype
> d'un systeme qu'une equipe DevOps peut reprendre."*

---

### 13:30 - 15:00 — Conformite + Green IT (Sebastien)

`[SLIDE 11 : Conformite]`

> *"Sur la conformite reglementaire — un point crucial pour tout client
> qui deploie de l'IA en Europe a partir d'aout 2026.
>
> AI Act article 13 transparence : couvert par notre Model Card
> au format Mitchell 2019.
> Article 14 supervision humaine : la decomposition dans le dashboard
> permet a l'operateur de comprendre et contester chaque decision.
>
> RGPD article 22 : le droit d'explication est couvert
> par SHAP et Captum.
> L'AIPD est documentee. Base legale : interet legitime
> sur des posts publics."*

`[SLIDE 12 : Green IT]`

> *"Le bilan carbone total est d'environ 6,9 grammes de CO2.
> 6,14 grammes mesures par CodeCarbon sur 6 entrainements,
> plus environ 0,7 gramme estime pour V6, le pipeline XAI et l'inference.
> RoBERTa represente la moitie du bilan, LogReg un tiers,
> et le reste se repartit entre CamemBERT, V6 et l'explicabilite.
>
> En production, V5 seul : 1,5 milliseconde, 0,6 gramme par jour.
> CamemBERT sert uniquement en analyse offline.
> C'est un choix architectural documente dans la Model Card."*

---

### 15:00 - 15:45 — Methodologie et organisation (Sebastien)

`[S-CAM]` *Bandeau : Sebastien Lazcanotegui — Validation & Qualite ML*

`[SLIDE 13 : Methodologie]`

> *"Notre methodologie suit le cycle CRISP-DM adapte au ML :
> comprendre, explorer, preparer, modeliser, evaluer, deployer.
> 9 versions en 6 mois, chacune documentee avec metriques avant/apres.
>
> La gestion de projet : un Gantt avec 16 work packages
> et 28 jalons, versionne sur GitHub.
> Nos outils : Git avec CI/CD sur GitHub Actions,
> Docker Compose, MongoDB, FastAPI, CodeCarbon.
> Le tout reproductible en une commande docker compose up."*

---

### 15:45 - 16:30 — ROI, budget et valeur business (Sebastien)

`[SLIDE 14 : ROI & Budget]`

> *"Parlons chiffres. Le projet Thumalien a coute environ 50 000 euros,
> essentiellement en ressources humaines — 110 jours-homme repartis
> entre le developpement technique et la validation.
>
> Zero euro de licence : notre stack est 100 % open source.
> Zero euro de cloud : tout tourne en local sur Docker.
> Le seul cout additionnel : 750 euros pour l'annotation manuelle
> du gold set de validation.
>
> En exploitation, le cout mensuel est d'environ 930 euros :
> un petit serveur a 30 euros et 2 jours de maintenance.
> Pour 1,8 million de posts analyses par mois,
> ca revient a 0,0005 centime par post.
>
> En retour : un gain de productivite x10 pour les moderateurs,
> une couverture de 60 000 posts par jour — 200 fois plus qu'un humain —
> et une reduction de 67 % des faux positifs."*

---

### 16:30 - 17:00 — Limites et roadmap (Sebastien)

`[SLIDE 15 : Roadmap V10-V12]`

> *"En toute transparence, les limites.
>
> Le kappa de Cohen entre annotateurs est de 0,498 — modere.
> La frontiere fiable/suspect est intrinsequement subjective.
> On attenue avec un intervalle de confiance bootstrap a 95 % :
> la reduction des faux positifs reste entre -73 % et -60 %.
>
> Thumalien detecte des signaux de desinformation,
> pas la verite factuelle. On classe la forme, pas le contenu.
>
> La roadmap V10 inclut MLflow pour le tracking,
> V11 ClaimBuster pour la verification factuelle,
> et V12 un audit d'equite algorithmique."*

---

### 17:00 - 18:00 — Conclusion (Azelie, camera)

`[A-CAM]` *Bandeau : Azelie Bernard — Lead Technique & Architecture*

`[SLIDE 16 : Citation]`

> *"Pour conclure, je reviens au F1 de 0,99 du debut.
>
> Aujourd'hui, en V9, notre F1 macro sur gold est de 0,67.
> C'est plus bas. C'est plus honnete.
> C'est valide par 501 tests, 80 % de mutation kill rate,
> une AIPD, une Model Card, et une explicabilite
> qui prouve sa propre fidelite.
>
> Thumalien n'est pas un classifieur.
> C'est un systeme de decision auditable — defendable
> devant un moderateur, devant un regulateur, et devant vous.
>
> Un score sans explication est un verdict sans proces.
> C'est la phrase qui a guide chacune de nos decisions techniques."*

`[SLIDE 17 : Remerciements + QR codes]`

> *"Merci pour votre attention.
> Le repository, le rapport et la Model Card
> sont accessibles via les QR codes a l'ecran.
> Nous sommes disponibles pour vos questions."*

---

## Tableau recapitulatif du temps de parole

| Section | Speaker | Duree | Slides |
|---------|---------|-------|--------|
| Hook intro | **Azelie** | 0:30 | — (camera) |
| Contexte client + besoin | **Azelie** | 1:00 | 1 |
| Equipe + problematique + exigences | **Sebastien** | 1:30 | 2, 3, 4 |
| Biais Reuters + iterations | **Sebastien** | 1:30 | 5 |
| Architecture + pipeline cascade | **Azelie** | 2:00 | 6, 7 |
| Demo dashboard live | **Azelie** | 3:00 | — (screencast) |
| XAI + faithfulness | **Azelie** | 2:30 | 8, 9 |
| Qualite industrielle | **Sebastien** | 1:30 | 10 |
| Conformite + Green IT | **Sebastien** | 1:30 | 11, 12 |
| Methodologie + organisation | **Sebastien** | 0:45 | 13 |
| ROI et valeur business | **Sebastien** | 0:45 | 14 |
| Limites + roadmap | **Sebastien** | 0:30 | 15 |
| Conclusion | **Azelie** | 1:00 | 16, 17 |
| **TOTAL Azelie** | | **~9:00** | |
| **TOTAL Sebastien** | | **~9:00** | |

---

## Structure besoin - solution - demo (conformite cahier des charges)

| Phase | Temps | Contenu |
|-------|-------|---------|
| **BESOIN** | 0:00-4:30 | Hook, contexte client, equipe, problematique, le piege F1=0.99 |
| **SOLUTION** | 4:30-12:00 | Architecture, pipeline cascade, demo dashboard, XAI+faithfulness |
| **DEMO** | 6:30-9:30 | Screencast live : texte fiable, texte suspect, 3 niveaux XAI |
| **VALEUR** | 12:00-18:00 | Qualite, conformite, Green IT, ROI, methodologie, roadmap, conclusion |

---

## Checklist conformite cadre pedagogique

- [x] 15-20 minutes (cible 18 min)
- [x] Prise de parole des deux membres (50/50)
- [x] Bandeau nom affiche pour chaque speaker
- [x] Structure besoin - solution - demo
- [x] Presentation de l'entreprise/contexte client et de l'equipe
- [x] Analyse de la problematique et introduction a la solution
- [x] Organisation et methodologies (CRISP-DM, Gantt, CI/CD)
- [x] Presentation de la solution technique
- [x] Pipeline donnees - IA - decision (explicitement nomme)
- [x] Metriques d'evaluation detaillees
- [x] Demo live du dashboard (screencast)
- [x] ROI et impact quantifie (temps, volume, risque, cout)
- [x] Conformite reglementaire (AI Act, RGPD)
- [x] Limites assumees + perspectives
- [x] Qualite industrielle (tests, coverage, mutation)
- [x] Posture professionnelle (ton client, pas academique)
- [x] Valorisation dataviz (choix visuels justifies dans la demo)
- [x] Green IT avec donnees reelles + estimations

---

## Cours associes couverts

| Cours | Ou dans la video |
|-------|-----------------|
| Machine Learning | V1-V9, LogReg, GradientBoosting, CamemBERT, meta-learner |
| NLP | TF-IDF, tokenization, CamemBERT, RoBERTa, embeddings |
| Visualisation de donnees | Demo dashboard : jauges, barres SHAP, heatmaps, choix de design |
| DataScience | Pipeline complet, metriques, cross-validation, gold set |
| Personal Branding | Bandeaux pro, posture client, structuration narrative |

---

## Bandeau OBS a configurer

**Bandeau Azelie** :
- Texte : `Azelie Bernard — Lead Technique & Architecture | Niamato Consulting`
- Position : bas gauche, fond semi-transparent noir 70%, texte blanc 20pt
- Dimensions : 640x40px

**Bandeau Sebastien** :
- Texte : `Sebastien Lazcanotegui — Validation & Qualite ML | Niamato Consulting`
- Position : bas gauche, fond semi-transparent noir 70%, texte blanc 20pt
- Dimensions : 640x40px

---

## Transitions entre speakers

| Temps | Transition | Type |
|-------|-----------|------|
| 0:30 | Azelie (hook) continue en voix-off | Slide titre |
| 1:30 | Azelie - Sebastien | Cut vers S-CAM |
| 4:30 | Sebastien - Azelie | Cut vers A-CAM |
| 12:00 | Azelie - Sebastien | Cut vers S-CAM |
| 17:00 | Sebastien - Azelie | Cut vers A-CAM |

5 transitions en tout. Blocs de 2-5 min par speaker.

---

## Ordre d'enregistrement recommande

**Session 1 — Sebastien (toutes ses sections)**
1. Section equipe + problematique (01:30-03:00) — 2 prises
2. Section biais Reuters + iterations (03:00-04:30) — 2 prises
3. Section qualite industrielle (12:00-13:30) — 2 prises
4. Section conformite + Green IT (13:30-15:00) — 2 prises
5. Section methodologie (15:00-15:45) — 2 prises
6. Section ROI (15:45-16:30) — 2 prises
7. Section limites + roadmap (16:30-17:00) — 2 prises

**Session 2 — Azelie (toutes ses sections)**
1. Hook intro (00:00-00:30) — 5 prises (c'est l'ouverture, soigner)
2. Contexte client (00:30-01:30) — 2 prises
3. Section architecture (04:30-06:30) — 2 prises
4. Section demo dashboard (06:30-09:30) — 3 prises (la demo peut rater)
5. Section XAI + faithfulness (09:30-12:00) — 2 prises
6. Conclusion (17:00-18:00) — 5 prises (c'est la fermeture, soigner)

**Session 3 — Captures ecran (sans voix)**
Enregistrer toutes les navigations dashboard separement,
puis synchroniser au montage.

---

*Script video Thumalien V9 — v3 client — Mai 2026*
