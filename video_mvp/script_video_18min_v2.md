# Script video 18 min — ThumaCheck (client : Thumalien) (v2 duo)

**Duree cible** : 18 min (15-20 min cadre MASTERE)
**Format** : screencast + voix off + inserts camera (intro, transitions, conclusion)
**Speakers** : Azelie Bernard (A) / Sebastien Lazcanotegui (S) — repartition 50/50
**Audience** : jury MASTERE M1 BDIA
**Objectif** : demontrer un systeme de decision auditable, pas un exercice scolaire

---

## Repartition du temps de parole

| Speaker | Sections | Duree |
|---------|----------|-------|
| **Azelie** | Hook + Architecture + Demo dashboard + Faithfulness + Conclusion | ~9 min |
| **Sebastien** | Problematique + Cahier des charges + Biais Reuters + Qualite industrielle + Conformite + Green IT + Roadmap | ~9 min |

---

## Arc narratif en 3 actes

**Acte 1 — La chute (0:00 - 4:30)** : le piege F1=0.99
**Acte 2 — La reconstruction (4:30 - 12:00)** : architecture, demo, XAI
**Acte 3 — La validation (12:00 - 18:00)** : qualite, conformite, limites, conclusion

---

## Script chronometre

> **Notation** :
> - `[A-CAM]` = Azelie face camera
> - `[S-CAM]` = Sebastien face camera
> - `[SCREEN]` = capture ecran ou demo live
> - `[SLIDE N]` = slide projetee
> - Bandeau nom affiche pour chaque prise de parole camera

---

### 00:00 - 00:30 — Hook (Azelie, camera)

`[A-CAM]` *Bandeau : Azelie Bernard — Lead Technique*

> *"En decembre 2025, on collecte 100 000 posts Bluesky.
> On entraine un modele de detection de fake news.
> On obtient un F1-score de 0,99 en cross-validation.
> Et c'est precisement a ce moment-la qu'on aurait du s'inquieter."*

`[SLIDE 1 : Titre projet]`

> *"Bonjour, je suis Azelie Bernard, lead technique sur ThumaCheck.
> Avec Sebastien Lazcanotegui, on a passe six mois a construire un systeme
> de detection de desinformation bilingue sur Bluesky.
> En 18 minutes, on va vous raconter pourquoi ce 0,99 etait un piege,
> et comment on a construit un systeme defendable."*

---

### 00:30 - 02:00 — Problematique + equipe (Sebastien, camera)

`[S-CAM]` *Bandeau : Sebastien Lazcanotegui — Validation & Qualite ML*

`[SLIDE 2 : chiffres cles]`

> *"Bonjour, je suis Sebastien Lazcanotegui.
> Mon role sur ce projet : la validation et la qualite du machine learning.
> J'ai pilote l'annotation du gold set — 473 posts annotes
> par deux annotateurs humains — le debiaisage du biais Reuters,
> et l'optimisation des hyperparametres par GridSearch.
>
> Bluesky compte 35 millions d'utilisateurs et un protocole AT
> entierement ouvert — plus de 60 000 posts publics par jour
> analysables, mais aucune equipe de moderation centralisee.
>
> Notre objectif : classifier un post comme fiable ou suspect
> en moins de 5 millisecondes, en francais comme en anglais,
> avec une explication auditable de chaque decision."*

`[SLIDE 3 : les 4 exigences]`

> *"Le cahier des charges fixe quatre exigences non-negociables.
> Transparence : chaque score doit etre explicable.
> Bilinguisme : francais et anglais a performance equivalente.
> Frugalite : moins de 5 ms par texte, empreinte CO2 mesuree.
> Et conformite au RGPD et a l'AI Act.
> Ces quatre exigences structurent toute notre demarche."*

---

### 02:00 - 04:30 — La fausse victoire et les iterations (Sebastien)

`[SCREEN : notebook Jupyter, sortie F1=0.99]`

> *"Voici notre premiere version du pipeline.
> Logistic Regression sur du TF-IDF, 30 000 features,
> entrainee sur 197 782 articles.
> Cross-validation a 5 plis : F1 macro de 0,99.
> Sur le papier, le projet est termine."*

`[SLIDE 4 : F1=0.99 — LE PIEGE]`

> *"Sauf qu'on a fait un test simple : on a regarde quels mots
> avaient le plus de poids dans la regression.
> Et on a trouve : reuters, afp, associated press.
>
> Le modele n'apprenait pas a detecter la desinformation.
> Il apprenait a reconnaitre le style des agences de presse.
>
> C'est moi qui ai pilote le debiaisage : creation de la liste
> BODY_AGENCY_TERMS pour neutraliser les signatures d'agences,
> filtrage des annees artefacts 2015-2020, et surtout
> les tests de non-regression apres chaque correction
> pour s'assurer qu'on ne cassait pas les performances.
>
> En parallele, j'ai lance un GridSearch systematique :
> le passage de C=1 a C=5 et de min_df=3 a min_df=5
> a confirme et ameliore les hyperparametres du pipeline.
>
> De la version 2 a la version 5, on a corrige ce biais
> et ajoute des donnees synthetiques francaises et anglaises.
> Le F1 en cross-validation est descendu a 0,91.
> Mais cette fois, c'est un F1 honnete."*

---

### 04:30 - 06:30 — Architecture et pipeline cascade (Azelie)

`[A-CAM]` *Bandeau : Azelie Bernard — Lead Technique*

> *"A partir de cette base saine, j'ai itere l'architecture
> jusqu'a la version 9 actuelle."*

`[SLIDE 5 : Architecture C4]`

> *"Voici notre architecture en 8 conteneurs, modele C4 niveau 2.
> Le collecteur se connecte a Bluesky via le protocole AT,
> stocke les posts dans MongoDB. Le pipeline NLP analyse chaque texte.
> Le dashboard Streamlit expose les resultats.
> Le tout est conteneurise avec Docker Compose.
>
> Trois modeles travaillent en ensemble :
> V5 — LogReg + TF-IDF + 15 features linguistiques + 7 emotions,
> c'est notre baseline frugale a 1,5 milliseconde.
> V6 — un classifieur style-only avec 28 features purement stylistiques,
> topic-agnostique : il ne regarde pas le sujet, il regarde la forme.
> CamemBERT fine-tune sur des textes courts francais.
> Le meta-learner V8 combine les trois."*

`[SLIDE 6 : Pipeline cascade]`

> *"En production, le pipeline cascade fonctionne en deux etapes.
> Le Stage 1 separe les opinions des faits — parce que l'AI Act
> interdit de qualifier une opinion de fake news.
> Le Stage 2 analyse uniquement les contenus factuels
> avec notre meta-learner V8.
>
> Cette cascade fait passer nos faux positifs de 57 a 21
> sur le gold set — une reduction de 67 %,
> statistiquement significative au test exact de Fisher, p < 0.000001."*

---

### 06:30 - 09:30 — Demo dashboard live (Azelie)

`[SCREEN : dashboard Streamlit plein ecran]`

> *"Voici le dashboard, l'interface de validation operationnelle.
> Cinq pages. Vue Globale : 245 000 posts collectes,
> 67 % classes fiables.
>
> Le coeur, c'est l'Analyse Temps Reel."*

`[SCREEN : soumettre texte fiable]`

> *"Premier test : un communique d'institution scientifique.
> Score 0,89 fiable. Mais surtout — et c'est ce qui distingue
> ThumaCheck d'un classifieur boite noire — trois niveaux d'explication.
>
> Niveau 1 : les top mots de la LogReg.
> Pas une approximation, c'est la formule fermee exacte du modele.
>
> Niveau 2 : SHAP applique au modele de style V6.
> La presence de citations et la diversite lexicale poussent vers fiable.
>
> Niveau 3 : la decomposition exacte du meta-learner V8.
> On lit litteralement la decision du modele,
> coefficient par coefficient."*

`[SCREEN : soumettre texte suspect]`

> *"Deuxieme test : un texte sensationnaliste.
> Score 0,12, donc 0,88 suspect.
> La decomposition est inversee : le sensationnalisme
> passe en rouge a +0,71.
>
> Et voici la heatmap d'attention CamemBERT qui montre
> token par token ce que le modele regarde.
> Les mots 'SCANDALE' et 'mentent' s'illuminent en rouge.
> C'est exactement ce qu'un moderateur veut voir
> pour defendre sa decision."*

---

### 09:30 - 12:00 — XAI et faithfulness (Azelie)

`[SLIDE 7 : Faithfulness methode]`

> *"A ce stade, une question legitime : comment savez-vous
> que ces explications refletent vraiment le comportement du modele ?
>
> On a implemente le protocole ERASER de DeYoung et collegues, ACL 2020.
> On masque les top-k features identifiees par SHAP,
> et on mesure si la prediction chute plus vite
> qu'avec un masquage aleatoire."*

`[SLIDE 8 : courbe AOPC]`

> *"Le resultat : AOPC attribution 0,253 contre 0,045 pour le random.
> Uplift de +0,21. Nos explications sont 5,6 fois plus predictives
> qu'une attribution au hasard.
>
> Pour les transformers, on est alles plus loin avec
> Layer Integrated Gradients via Captum, qui donne une attribution
> causale par token avec garantie axiomatique de Sundararajan.
>
> Et on a decouvert que sur les cas ou CamemBERT est tres confiant,
> le ReLU du head sature et bloque les gradients.
> Ce n'est pas un bug — c'est une signature documentee
> dans la Model Card."*

---

### 12:00 - 13:30 — Qualite industrielle (Sebastien)

`[S-CAM]` *Bandeau : Sebastien Lazcanotegui — Validation & Qualite ML*

> *"A ce stade vous pourriez vous demander :
> tout ca c'est bien, mais quelle est la qualite du code ?
> C'est une question que je me suis posee egalement."*

`[SLIDE 9 : Qualite industrielle]`

> *"On s'est impose les standards de l'industrie.
> 501 tests pytest. 80 % de couverture de code.
> 77,9 % de branch coverage.
> Quality gate sur GitHub Actions qui rejette toute PR
> descendant sous 75 %.
>
> Mais ce qui compte vraiment, c'est la qualite des tests.
> Donc on a fait du mutation testing avec mutmut
> sur le module critique de decomposition meta-learner.
> 178 mutations artificielles. 143 detectees.
> Kill rate : 80,3 % — au-dessus de la moyenne Google
> qui se situe entre 60 et 75 %.
>
> Cette dimension qualite n'est pas cosmetique.
> C'est ce qui separe un prototype academique
> d'un systeme qu'une equipe DevOps peut reprendre."*

---

### 13:30 - 15:00 — Conformite reglementaire et Green IT (Sebastien)

`[SLIDE 10 : Conformite]`

> *"Sur la dimension reglementaire, on a anticipe l'AI Act
> qui sera pleinement applicable en aout 2026.
>
> Article 13 transparence : couvert par notre Model Card
> au format Mitchell 2019, reference MC-THUM-2026-001.
> Article 14 supervision humaine : la decomposition beta x
> dans le dashboard permet a un operateur de comprendre
> et contester chaque decision.
> Notre systeme est classe risque limite.
>
> Cote RGPD : l'article 22 sur les decisions automatisees
> est couvert par nos explications SHAP et Captum.
> L'AIPD est documentee. Notre base legale est l'interet
> legitime sur des posts publics."*

`[SLIDE 11 : Green IT]`

> *"Le bilan carbone total est d'environ 6,9 grammes de CO2.
> 6,14 grammes mesures par CodeCarbon sur 6 entrainements,
> plus environ 0,7 gramme estime pour V6, le pipeline XAI et l'inference.
> RoBERTa represente la moitie du bilan, LogReg un tiers,
> et le reste se repartit entre CamemBERT, V6 et l'explicabilite.
>
> En production, on utilise V5 seul : 1,5 milliseconde,
> 0,6 gramme par jour. CamemBERT sert uniquement
> de signal complementaire en analyse offline.
>
> Cette decision — avoir un modele puissant entraine
> mais pas servi en chaud — est un choix architectural
> assume et documente dans la Model Card."*

---

### 15:00 - 16:00 — Methodologie et organisation (Sebastien)

`[S-CAM]` *Bandeau : Sebastien Lazcanotegui*

> *"Un mot sur notre methodologie.
> On a suivi le cycle CRISP-DM adapte au machine learning :
> comprendre, explorer, preparer, modeliser, evaluer, deployer.
> Et surtout : iterer. 9 versions en 6 mois.
>
> La gestion de projet s'est appuyee sur un planning Gantt
> avec 16 work packages et 28 jalons.
> Chaque iteration etait documentee dans un notebook dedie
> avec metriques avant/apres.
>
> Nos outils : Git avec CI/CD sur GitHub Actions,
> Docker Compose pour le deploiement,
> MongoDB pour le stockage, CodeCarbon pour le suivi carbone.
> Le tout versionne et reproductible."*

---

### 16:00 - 17:00 — Limites et roadmap (Sebastien)

`[SLIDE 12 : Roadmap V10-V12]`

> *"Maintenant la partie que personne n'aime mettre
> dans une video : les limites.
>
> Limite 1 : notre kappa de Cohen entre annotateurs
> est de 0,498 — modere. La frontiere fiable/suspect
> est intrinsequement subjective. On attenue avec un bootstrap
> d'intervalle de confiance a 95 % : la reduction des faux positifs
> reste entre -73 % et -60 %, donc l'effet est robuste.
>
> Limite 2 : ThumaCheck detecte des signaux de desinformation,
> pas la verite factuelle. On classe la forme, pas le contenu.
>
> Limite 3 : on est un binome. Dans une equipe de quatre
> comme prevu par le cadre pedagogique,
> on aurait developpe un monitoring drift plus sophistique.
> C'est dans la roadmap V10.
>
> Cette roadmap inclut MLflow, ClaimBuster pour
> la verification factuelle, et un audit d'equite algorithmique."*

---

### 17:00 - 18:00 — Conclusion (Azelie, camera)

`[A-CAM]` *Bandeau : Azelie Bernard — Lead Technique*

`[SLIDE 13 : Citation]`

> *"Pour conclure, je reviens au F1 de 0,99 du debut.
>
> Aujourd'hui, en V9, notre F1 macro sur gold est de 0,67.
> C'est plus bas. C'est plus honnete.
> C'est valide par 501 tests, 80 % de mutation kill rate,
> une AIPD, une Model Card, et une explicabilite
> qui prouve sa propre fidelite.
>
> ThumaCheck n'est pas un classifieur.
> C'est un systeme de decision auditable — defendable
> devant un utilisateur, devant un regulateur, et devant vous.
>
> Un score sans explication est un verdict sans proces.
> C'est la phrase qui a guide chacune de nos decisions techniques."*

`[SLIDE 14 : Remerciements]`

> *"Merci pour votre attention.
> Le repository, le rapport et la Model Card
> sont accessibles via les QR codes a l'ecran.
> Nous sommes disponibles pour vos questions."*

---

## Tableau recapitulatif du temps de parole

| Section | Speaker | Duree | Slides |
|---------|---------|-------|--------|
| Hook intro | **Azelie** | 0:30 | 1 |
| Problematique + equipe | **Sebastien** | 1:30 | 2, 3 |
| Biais Reuters + iterations | **Sebastien** | 2:30 | 4 |
| Architecture + pipeline cascade | **Azelie** | 2:00 | 5, 6 |
| Demo dashboard live | **Azelie** | 3:00 | — (screencast) |
| XAI + faithfulness | **Azelie** | 2:30 | 7, 8 |
| Qualite industrielle | **Sebastien** | 1:30 | 9 |
| Conformite + Green IT | **Sebastien** | 1:30 | 10, 11 |
| Methodologie + organisation | **Sebastien** | 1:00 | — (camera) |
| Limites + roadmap | **Sebastien** | 1:00 | 12 |
| Conclusion | **Azelie** | 1:00 | 13, 14 |
| **TOTAL Azelie** | | **~9:00** | |
| **TOTAL Sebastien** | | **~9:00** | |

---

## Checklist conformite cadre pedagogique

- [x] 15-20 minutes (cible 18 min)
- [x] Prise de parole des deux membres
- [x] Bandeau nom affiche pour chaque speaker
- [x] Structure besoin → solution → demo
- [x] Presentation de l'equipe et des roles
- [x] Methodologie appliquee (CRISP-DM, Gantt, CI/CD)
- [x] Pipeline data → IA → visualisation
- [x] Metriques d'evaluation detaillees
- [x] Demo live du dashboard (screencast)
- [x] ROI et impact (Green IT, 245K posts, latence)
- [x] Conformite reglementaire
- [x] Limites assumees + perspectives
- [x] Qualite industrielle (tests, coverage, mutation)
- [x] Posture professionnelle

---

## Bandeau OBS a configurer

Creer 2 bandeaux dans OBS Studio :

**Bandeau Azelie** :
- Texte : `Azelie Bernard — Lead Technique`
- Position : bas gauche, fond semi-transparent noir 70%, texte blanc 20pt
- Dimensions : 400x40px

**Bandeau Sebastien** :
- Texte : `Sebastien Lazcanotegui — Validation & Qualite ML`
- Position : bas gauche, fond semi-transparent noir 70%, texte blanc 20pt
- Dimensions : 520x40px

Switcher entre les deux bandeaux a chaque changement de speaker dans OBS.

---

## Transitions entre speakers

Les transitions speaker se font aux moments suivants :

| Temps | Transition | Type |
|-------|-----------|------|
| 0:30 | Azelie → Sebastien | Cut vers S-CAM |
| 4:30 | Sebastien → Azelie | Cut vers A-CAM |
| 12:00 | Azelie → Sebastien | Cut vers S-CAM |
| 17:00 | Sebastien → Azelie | Cut vers A-CAM |

4 transitions en tout. Chaque bloc est assez long (2-5 min) pour eviter l'effet ping-pong et laisser chaque speaker developper son propos.

---

## Ordre d'enregistrement recommande

Pour minimiser les changements de setup :

**Session 1 — Sebastien (toutes ses sections)**
1. Section problematique + equipe (00:30-02:00) — 2 prises
2. Section biais Reuters (02:00-04:30) — 2 prises
3. Section qualite industrielle (12:00-13:30) — 2 prises
4. Section conformite + Green IT (13:30-15:00) — 2 prises
5. Section methodologie (15:00-16:00) — 2 prises
6. Section limites + roadmap (16:00-17:00) — 2 prises

**Session 2 — Azelie (toutes ses sections)**
1. Hook intro (00:00-00:30) — 5 prises (c'est l'ouverture, soigner)
2. Section architecture (04:30-06:30) — 2 prises
3. Section demo dashboard (06:30-09:30) — 3 prises (la demo peut rater)
4. Section XAI + faithfulness (09:30-12:00) — 2 prises
5. Conclusion (17:00-18:00) — 5 prises (c'est la fermeture, soigner)

**Session 3 — Captures ecran (sans voix)**
Enregistrer toutes les navigations dashboard separement,
puis synchroniser au montage.

---

*Script video ThumaCheck (client : Thumalien) — v2 duo — Mai 2026*
