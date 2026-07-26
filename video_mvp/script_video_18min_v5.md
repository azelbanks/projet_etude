# Script video 18 min — ThumaCheck (v5 solo)

**Duree cible** : 18 min (15-20 min cadre MASTERE)
**Format** : screencast + voix off + inserts camera (intro, transitions, conclusion)
**Speaker** : Azelie Bernard (presentation solo)
**Entreprise** : Niamato Consulting (expertise Data & IA)
**Client** : Thumalien (societe de fact-checking et veille mediatique)
**Solution** : ThumaCheck — pipeline NLP + dashboard de detection de desinformation
**Audience** : equipe dirigeante Thumalien + jury evaluateur
**Objectif** : convaincre Thumalien que ThumaCheck resout son probleme de moderation, avec ROI mesurable

> **Changements v4 -> v5** :
> - Presentation solo Azelie (Sebastien absent pour l'enregistrement)
> - Sections de Sebastien reecrites a la 3e personne, presentees par Azelie
> - Credits explicites a Sebastien sur ses contributions (debiaisage, annotation, mutation testing)
> - Directions camera simplifiees (plus de [S-CAM])
> - Conclusion adaptee ("je suis a votre disposition" au singulier)

---

## Arc narratif en 3 actes

**Acte 1 — Le besoin (0:00 - 4:30)** : le probleme client, l'equipe, le piege F1=0.99
**Acte 2 — La solution (4:30 - 12:00)** : architecture, demo live, emotions, explicabilite
**Acte 3 — La valeur (12:00 - 18:00)** : qualite, securite, conformite, ROI, methodologie, roadmap

---

## Script chronometre

> **Notation** :
> - `[A-CAM]` = Azelie face camera
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
> Nous sommes un cabinet d'expertise Data et Intelligence Artificielle,
> et vous nous avez mandates pour developper une solution
> de detection de desinformation sur Bluesky.
>
> Vos equipes de fact-checking passent en moyenne
> 3 a 5 minutes par post pour evaluer sa fiabilite.
> A 60 000 posts publics par jour sur Bluesky, c'est intenable.
>
> ThumaCheck est la solution que nous avons construite pour vous.
> Elle analyse un texte en 1,5 milliseconde,
> avec une explication auditable a chaque decision.
>
> Et si votre base de donnees est temporairement indisponible,
> le dashboard continue de fonctionner en mode demonstration —
> vos equipes ne sont jamais bloquees."*

---

### 01:30 - 03:00 — Equipe + problematique (Azelie)

`[A-CAM]` *Bandeau : Azelie Bernard — Lead Technique & Architecture*

`[SLIDE 2 : Equipe et roles]`

> *"Ce projet a ete realise en binome.
>
> Mon collegue Sebastien Lazcanotegui a pris en charge
> la validation et la qualite du machine learning.
> C'est lui qui a pilote l'annotation du gold set —
> 465 textes Bluesky annotes par deux annotateurs humains —
> le debiaisage des donnees,
> et l'optimisation des hyperparametres.
>
> De mon cote, j'ai concu l'architecture du pipeline,
> le dashboard et les trois niveaux d'explicabilite."*

`[SLIDE 3 : Chiffres cles]`

> *"Bluesky : 35 millions d'utilisateurs, protocole AT entierement ouvert,
> plus de 60 000 posts publics par jour analysables.
> Aucune equipe de moderation centralisee.
>
> Ce que nous vous livrons :
> classifier un post comme fiable ou suspect
> en moins de 5 millisecondes, en francais comme en anglais,
> avec une explication que vos equipes peuvent defendre
> aupres de votre direction et de vos regulateurs."*

`[SLIDE 4 : Les 4 exigences]`

> *"Des le depart, nous avons fixe avec vous
> quatre exigences non-negociables.
> Transparence : chaque score est explicable.
> Bilinguisme : francais et anglais a performance equivalente.
> Frugalite : moins de 5 ms par texte, empreinte carbone mesuree.
> Et conformite au RGPD et a l'AI Act europeen,
> applicable en aout 2026.
> Ces quatre exigences structurent toute la solution
> que nous vous presentons aujourd'hui."*

---

### 03:00 - 04:30 — La fausse victoire et les iterations (Azelie)

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
> Sebastien a pilote le debiaisage : creation de la liste
> de termes d'agences pour neutraliser ces signatures,
> filtrage des annees artefacts 2015-2020, et les tests de non-regression
> apres chaque correction.
>
> En parallele, il a lance un GridSearch systematique
> sur les hyperparametres du modele.
>
> De la version 2 a la version 5, on a corrige ce biais
> et ajoute des donnees bilingues.
> Le F1 en cross-validation est descendu a 0,91.
> Mais cette fois, c'est un F1 honnete."*

---

### 04:30 - 06:30 — Architecture et pipeline cascade (Azelie)

`[A-CAM]` *Bandeau : Azelie Bernard — Lead Technique & Architecture*

> *"A partir de cette base saine, j'ai itere l'architecture
> jusqu'a la version 9 actuelle."*

`[SLIDE 6 : Architecture C4]`

> *"Voici notre architecture.
> Le flux suit trois couches : donnees, intelligence artificielle, decision.
>
> Premiere couche — les donnees.
> Le collecteur se connecte a Bluesky via le protocole AT
> et stocke les posts dans MongoDB.
> 537 000 posts collectes a ce jour,
> avec deduplication automatique et reprise sur erreur reseau.
>
> Deuxieme couche — l'intelligence artificielle.
> Le pipeline analyse chaque texte avec trois modeles complementaires.
> V5 — notre modele principal : il combine TF-IDF
> avec 17 features linguistiques et 7 emotions detectees automatiquement.
> C'est lui qui repond en 1,5 milliseconde.
> V6 — un classifieur qui analyse uniquement le style d'ecriture,
> independamment du sujet.
> Et CamemBERT — un modele de deep learning
> specialise sur les textes courts en francais.
> Le meta-learner V8 combine les trois pour la decision finale.
>
> Troisieme couche — la decision.
> Le dashboard expose les resultats avec cinq pages dediees
> a vos equipes et trois niveaux d'explication.
> Le tout est conteneurise avec Docker Compose —
> une seule commande pour tout demarrer."*

`[SLIDE 7 : Pipeline cascade]`

> *"En production, le pipeline fonctionne en deux etapes.
> La premiere etape separe les opinions des faits —
> parce que l'AI Act interdit de qualifier une opinion de desinformation.
> La seconde etape analyse uniquement les contenus factuels.
>
> Cette cascade a reduit vos faux positifs de 57 a 21
> sur notre jeu de test — soit une baisse de 67 %,
> validee statistiquement.
> Concretement pour vous :
> moins d'alertes inutiles dans les files de moderation
> de vos equipes."*

---

### 06:30 - 09:30 — Demo dashboard live + Emotions (Azelie)

`[SCREEN : dashboard Streamlit plein ecran]`

> *"Voici le dashboard ThumaCheck —
> l'interface que vos equipes utiliseront au quotidien.
> Cinq pages. Vue Globale : 537 000 posts collectes,
> 67 % classes fiables.
>
> Chaque visualisation a ete pensee pour la decision :
> la jauge de score utilise un code couleur vert-rouge intuitif,
> les barres d'explication montrent ce qui pousse vers fiable ou suspect,
> et la heatmap de mots-cles guide l'oeil de vos analystes
> vers les passages critiques."*

`[SCREEN : soumettre texte fiable]`

> *"Premier test : une depeche scientifique du CNRS publiee dans Nature.
>
> Avant meme d'evaluer la fiabilite, le pipeline effectue un premier filtre.
> Il verifie si le texte exprime un fait ou une opinion —
> parce que l'AI Act interdit de qualifier une opinion de desinformation.
> Ici, le contenu est bien factuel. On passe a l'etape d'analyse.
>
> Regardons maintenant les trois modeles travailler en parallele.
> V5 — notre modele TF-IDF principal — donne un score de 0,94.
> Tres fiable.
> V6 — le detecteur de style — affiche 0,05 de suspicion.
> Le style d'ecriture est neutre et informatif.
> Le meta-learner V8 combine ces deux signaux avec CamemBERT
> et rend son verdict : FIABLE.
>
> Le desaccord entre V5 et V6 est de 0,01 — quasi nul.
> Les trois modeles sont en accord. C'est un signal fort de coherence.
>
> L'emotion dominante : neutre a 65,8 %.
> Pas de charge emotionnelle excessive. Coherent avec un contenu scientifique.
>
> Maintenant les trois niveaux d'explication.
>
> Premier niveau : les contributions SHAP par feature de style.
> Ce ne sont pas des approximations — ce sont les contributions exactes
> de chaque caracteristique linguistique au score V6.
>
> Deuxieme niveau : la heatmap d'attention CamemBERT.
> CamemBERT decoupe le texte en sous-unites de mots.
> Les tokens en rouge intense sont ceux sur lesquels
> le modele a concentre son attention —
> ce sont les ancres semantiques qui ont guide sa decision.
>
> Troisieme niveau : la decomposition complete du meta-learner V8.
> Le logit z vaut -0,918. Transforme en probabilite : 28,5 % de suspicion.
> Donc 71,5 % de fiabilite. Decision FIABLE, seuil franchi.
> On lit le poids exact de chaque modele dans la decision finale,
> coefficient par coefficient.
>
> Vos analystes ont tout ce qu'il faut
> pour justifier chaque decision aupres de votre hierarchie —
> et aupres de vos regulateurs."*

`[SCREEN : soumettre texte suspect]`

> *"Deuxieme test : un texte sensationnaliste typique des reseaux sociaux.
>
> Stage 1 : le pipeline le classe factuel a 48 % — on passe a l'analyse.
> Et la, les signaux sont sans appel.
>
> V5 — le TF-IDF — donne 0,00 de fiabilite. Zero.
> Il ne detecte aucun signal de credibilite dans ce texte.
> V6 — le detecteur de style — affiche 0,98 de suspicion.
> Le style crie la desinformation : majuscules, points d'exclamation,
> appel au partage viral, injonction a la peur.
> Le meta-learner V8 combine les deux :
> logit z = +0,991, soit 72,9 % de suspicion. Decision SUSPECT.
>
> Le desaccord V5/V6 est de 0,02 — quasi nul.
> Les deux modeles sont en accord total.
> C'est le cas contraire du texte fiable :
> ici, tout converge vers le meme verdict.
>
> Le dashboard detecte et liste automatiquement
> les mots sensationnalistes du texte :
> 'scandale', 'on vous ment', 'partagez avant censure'.
> Ce sont exactement les marqueurs rhetorique
> que vos analystes cherchent manuellement aujourd'hui.
>
> La heatmap CamemBERT visualise l'attention du modele token par token.
> CamemBERT decoupe les mots en sous-unites —
> SCANDALE devient SC-AND-ALE, CACHE devient CA-CHE.
> Les sous-tokens en rouge intense sont ceux
> sur lesquels le modele a concentre son attention.
> Les termes les plus alarmants ressortent clairement.
>
> Et le premier contributeur dans la decomposition V8 :
> score_v6_suspect avec +0,567 vers SUSPECT.
> C'est le style qui a fait basculer la decision.
> Pas le sujet — le style."*

`[SCREEN : page emotions ou analyse]`

> *"Dernier element de cette demo : l'emotion dominante.
>
> Sur le texte fiable, l'emotion etait neutre a 65,8 %.
> Sur ce texte suspect : surprise a 74,6 %.
>
> Ce n'est pas anodin. Les contenus de desinformation
> utilisent deliberement la surprise et la peur
> pour contourner le sens critique.
>
> Pour vos equipes, c'est un signal complementaire puissant.
> Un post SUSPECT avec une emotion de surprise elevee
> peut etre priorise en tete de votre file de moderation.
>
> ThumaCheck analyse 7 dimensions emotionnelles :
> colere, degout, joie, neutre, peur, surprise, tristesse.
> C'est ce que votre cahier des charges demandait
> sous le nom d'analyse emotionnelle."*

---

### 09:30 - 12:00 — XAI et faithfulness (Azelie)

`[SLIDE 8 : Faithfulness methode]`

> *"Vous pourriez nous demander :
> comment savez-vous que ces explications
> refletent vraiment le comportement du modele ?
>
> C'est une question legitime.
> Un outil qui explique mal est pire
> qu'un outil qui n'explique pas.
>
> On a mis en place un protocole de verification.
> Le principe est simple :
> on identifie les criteres que l'explication designe
> comme les plus importants, on les masque dans le texte,
> et on regarde si la prediction du modele change.
> Si elle change fortement, c'est que l'explication
> pointait les bons criteres."*

`[SLIDE 9 : courbe AOPC]`

> *"Le resultat est visible sur cette slide.
>
> On a masque les mots que l'outil jugeait les plus decisifs,
> et on a mesure combien la prediction changeait.
> On a fait la meme chose avec des mots choisis au hasard.
>
> Resultat : 5,6 fois plus d'impact avec nos explications
> qu'avec une selection aleatoire.
>
> Ce chiffre signifie une seule chose :
> quand notre outil vous dit 'ce mot a fait basculer la decision',
> ce n'est pas une intuition — c'est mesure et verifie.
>
> Pour CamemBERT, on a applique la meme logique mot par mot.
> Et quand nos explications ont des limites, on les documente
> dans notre Model Card.
> Cette transparence sur les limites, c'est exactement
> ce que l'AI Act attend d'un systeme responsable."*

---

### 12:00 - 13:30 — Qualite industrielle + Securite (Azelie)

`[A-CAM]` *Bandeau : Azelie Bernard — Lead Technique & Architecture*

`[SLIDE 10 : Qualite industrielle]`

> *"Quand vous evaluez une solution,
> vous regardez aussi la qualite du code —
> est-ce que ca tiendra en production ?
>
> 537 tests automatises. 80 % de couverture de code.
> Un quality gate sur notre CI/CD
> qui rejette automatiquement toute modification
> descendant sous 75 %.
>
> Sebastien a egalement mis en place du mutation testing —
> on introduit volontairement des erreurs dans le code
> pour verifier que nos tests les detectent.
> Sur le module le plus critique :
> 178 erreurs injectees, 143 detectees.
> Taux de detection de 80,3 % —
> au-dessus de la moyenne des equipes Google.
>
> En complement, un scan de securite Trivy
> tourne a chaque commit pour detecter les vulnerabilites connues.
> Vos identifiants et mots de passe sont isoles
> dans des variables d'environnement,
> jamais exposes dans le code source.
>
> C'est ce qui separe un prototype
> d'un systeme que votre equipe peut reprendre et maintenir."*

---

### 13:30 - 15:00 — Conformite + Green IT (Azelie)

`[SLIDE 11 : Conformite]`

> *"Sur la conformite reglementaire —
> un point crucial pour vous si vous deployez de l'IA en Europe.
> L'AI Act entre en application le 2 aout 2026.
>
> Nous avons classe ThumaCheck en risque limite selon l'article 50.
> Ca veut dire : obligation de transparence,
> mais pas les contraintes des systemes a haut risque.
>
> Article 13 — transparence : couvert par notre Model Card,
> un document standardise qui decrit le modele,
> ses performances et ses limites.
> Article 14 — supervision humaine :
> vos operateurs peuvent comprendre et contester chaque decision
> grace aux trois niveaux d'explication.
>
> Cote RGPD : article 22 — le droit d'explication
> est couvert par nos outils d'explicabilite.
> L'analyse d'impact sur la vie privee est documentee.
> Base legale : interet legitime sur des posts publics.
> Et si un utilisateur Bluesky exerce son droit a l'effacement,
> nous pouvons supprimer ses donnees en moins de 5 secondes."*

`[SLIDE 12 : Green IT]`

> *"Le bilan carbone total de tout le projet —
> douze entrainements mesures avec CodeCarbon,
> de V3 a CamemBERT en passant par RoBERTa —
> represente 8,9 grammes de CO2.
> C'est l'equivalent de 52 metres en voiture essence,
> sur la base de 170 grammes de CO2 par kilometre,
> reference ADEME 2024 — une empreinte quasi nulle.
>
> En production, le modele principal V5 :
> 1,5 milliseconde par texte, 0,6 gramme de CO2 par jour.
> Les modeles lourds comme CamemBERT ne servent qu'en analyse approfondie,
> pas en traitement temps reel.
> C'est un choix d'architecture documente et assume :
> performance et sobriete ne sont pas incompatibles."*

---

### 15:00 - 15:45 — Methodologie et organisation (Azelie)

`[A-CAM]` *Bandeau : Azelie Bernard — Lead Technique & Architecture*

`[SLIDE 13 : Methodologie]`

> *"Notre methodologie suit le cycle CRISP-DM adapte au ML :
> comprendre, explorer, preparer, modeliser, evaluer, deployer.
> 9 versions en 6 mois, chacune documentee avec metriques avant et apres.
>
> La gestion de projet : un planning detaille avec 16 lots de travail
> et 28 jalons, versionne sur GitHub.
> Nos outils : Git avec integration continue,
> Docker Compose, MongoDB, FastAPI pour l'API,
> CodeCarbon pour le suivi carbone.
> Le tout reproductible en une seule commande."*

---

### 15:45 - 16:30 — ROI, budget et valeur business (Azelie)

`[SLIDE 14 : ROI & Budget]`

> *"Parlons chiffres. Le projet ThumaCheck a coute environ 50 000 euros,
> essentiellement en ressources humaines — 110 jours-homme repartis
> entre le developpement et la validation.
>
> Zero euro de licence : notre stack est 100 % open source.
> Zero euro de cloud : tout tourne en local sur Docker.
> Le seul cout additionnel : 750 euros pour l'annotation
> du jeu de test de reference.
>
> En exploitation, le cout mensuel est d'environ 930 euros :
> un serveur a 30 euros et 2 jours de maintenance.
> Pour 1,8 million de posts analyses par mois,
> ca revient a 0,0005 centime par post.
>
> Ce que vous gagnez : un facteur x10 de productivite
> pour vos moderateurs,
> une couverture de 60 000 posts par jour —
> 200 fois plus qu'un humain seul —
> et une reduction de 67 % des faux positifs
> qui encombrent aujourd'hui vos files de moderation."*

---

### 16:30 - 17:00 — Limites et roadmap (Azelie)

`[SLIDE 15 : Roadmap V10-V12]`

> *"En toute transparence, les limites.
>
> L'accord entre nos deux annotateurs humains est de 0,498
> sur l'echelle de Cohen — c'est un accord modere.
> Ca ne veut pas dire que le modele est mauvais.
> Ca veut dire que la frontiere entre fiable et suspect
> est intrinsequement subjective, meme pour des humains.
>
> Notre F1 macro sur le jeu de test est de 0,67.
> En cross-validation sur les donnees d'entrainement, il est de 0,91.
> L'ecart s'explique par cette subjectivite.
> C'est pourquoi ThumaCheck donne un score de confiance,
> pas un verdict definitif. Vos analystes gardent le dernier mot.
>
> ThumaCheck detecte des signaux de desinformation —
> le style, le ton, le sensationnalisme — pas la verite factuelle.
> C'est un outil d'aide a la decision, pas un juge.
>
> La roadmap : V10 integre le suivi de modeles
> pour detecter les baisses de performance.
> V11 ajoute la verification factuelle avec des outils
> comme ClaimBuster.
> Et V12 prevoit un audit d'equite
> pour garantir que le modele traite tous les sujets
> de maniere equilibree."*

---

### 17:00 - 18:00 — Conclusion (Azelie, camera)

`[A-CAM]` *Bandeau : Azelie Bernard — Lead Technique & Architecture*

`[SLIDE 16 : Citation]`

> *"Pour conclure, je reviens au F1 de 0,99 du debut.
> Ce chiffre flatteur cachait un biais.
> Nous l'avons detecte, corrige, et documente.
>
> Aujourd'hui, ThumaCheck V9
> est valide par 537 tests automatises,
> 80 % de taux de detection en mutation testing,
> une analyse d'impact vie privee, une Model Card,
> et une explicabilite dont on a prouve la fidelite.
>
> ThumaCheck n'est pas un classifieur.
> C'est un systeme de decision auditable —
> defendable devant vos analystes,
> devant vos regulateurs, et devant votre direction.
>
> Un score sans explication est un verdict sans proces.
> C'est la phrase qui a guide
> chacune de nos decisions techniques tout au long de ce projet."*

`[SLIDE 17 : Remerciements + QR codes]`

> *"Merci pour votre attention.
> Le repository, le rapport technique et la Model Card
> sont accessibles via les QR codes a l'ecran.
> Je suis a votre disposition pour vos questions."*

---

## Tableau recapitulatif

| Section | Duree | Slides |
|---------|-------|--------|
| Hook intro | 0:30 | — (camera) |
| Contexte client + besoin + mode demo | 1:00 | 1 |
| Equipe + problematique + exigences | 1:30 | 2, 3, 4 |
| Biais Reuters + iterations | 1:30 | 5 |
| Architecture + pipeline cascade | 2:00 | 6, 7 |
| Demo dashboard + emotions | 3:00 | — (screencast) |
| XAI + faithfulness | 2:30 | 8, 9 |
| Qualite industrielle + securite | 1:30 | 10 |
| Conformite + Green IT | 1:30 | 11, 12 |
| Methodologie + organisation | 0:45 | 13 |
| ROI et valeur business | 0:45 | 14 |
| Limites + roadmap | 0:30 | 15 |
| Conclusion | 1:00 | 16, 17 |
| **TOTAL** | **~18:00** | |

---

## Couverture des exigences CDC (CDC-THUM-2026-001)

| Module CDC | Exigences | Couvert dans la video | Timing |
|------------|-----------|----------------------|--------|
| **COL** | COL-01 a COL-08 | COL-01 (AT Protocol), COL-04 (deduplication), COL-05 (reprise erreur) | 04:30 |
| **STO** | STO-01 a STO-06 | STO-01 (MongoDB), STO-02 (537K docs), STO-06 (enrichissements) | 04:30 |
| **DET** | DET-01 a DET-12 | Tous couverts (F1, bilinguisme, seuil, explicabilite, debiaisage) | 03:00-06:30 |
| **EMO** | EMO-01 a EMO-05 | EMO-01 (7 emotions), EMO-04 (bilingue), EMO-05 (features pipeline) | 08:30 |
| **DASH** | DASH-01 a DASH-09 | DASH-01 (Streamlit), DASH-02 (5 pages), DASH-04 (analyse), DASH-05 (mode demo), DASH-06 (XAI), DASH-07 (dark theme) | 06:30-09:30 |
| **XAI** | XAI-01 a XAI-05 | Tous couverts (SHAP, faithfulness, IG, attention, audience) | 09:30-12:00 |
| **PERF** | PERF-01 a PERF-06 | PERF-02 (< 5ms), PERF-05 (modele leger) | 04:30, 14:30 |
| **SEC** | SEC-01 a SEC-04 | SEC-01/02 (.env), SEC-04 (Trivy) | 12:00-13:30 |
| **FIA** | FIA-01 a FIA-05 | FIA-04 (Docker restart), FIA-05 (volumes) | 04:30, 15:00 |
| **MAI** | MAI-01 a MAI-05 | MAI-01 (modules), MAI-02 (versioning), MAI-05 (CodeCarbon) | 15:00 |

---

## Checklist conformite cadre pedagogique

- [x] 15-20 minutes (cible 18 min)
- [x] Presentation de l'equipe et repartition des roles (credit Sebastien)
- [x] Bandeau nom affiche pour chaque prise de parole camera
- [x] Structure besoin - solution - demo
- [x] Presentation de l'entreprise/contexte client et de l'equipe
- [x] Analyse de la problematique et introduction a la solution
- [x] Organisation et methodologies (CRISP-DM, Gantt, CI/CD)
- [x] Presentation de la solution technique
- [x] Pipeline donnees - IA - decision (explicitement nomme)
- [x] Metriques d'evaluation detaillees
- [x] Demo live du dashboard (screencast)
- [x] Module emotions demontre (7 classes, signal complementaire)
- [x] Securite (variables env, scan Trivy, isolation)
- [x] ROI et impact quantifie (temps, volume, risque, cout)
- [x] Conformite reglementaire (AI Act art. 50, RGPD art. 22, AIPD)
- [x] Limites assumees + contextualisation (kappa, subjectivite)
- [x] Perspectives et roadmap (V10-V12)
- [x] Qualite industrielle (537 tests, coverage, mutation)
- [x] Posture professionnelle (ton client, pas academique)
- [x] Valorisation dataviz (choix visuels justifies dans la demo)
- [x] Green IT avec donnees reelles + comparaison tangible (30 m en voiture)
- [x] Mode demo mentionne (DASH-05)
- [x] Droit a l'effacement RGPD operationnel (< 5s)

---

## Ordre d'enregistrement recommande (solo)

**Session 1 — Sections camera (face Azelie)**
1. Hook intro (00:00-00:30) — 5 prises (c'est l'ouverture, soigner)
2. Contexte client (00:30-01:30) — 2 prises
3. Equipe + problematique (01:30-03:00) — 2 prises
4. Architecture intro (04:30) — 2 prises
5. Qualite industrielle intro (12:00) — 2 prises
6. Methodologie intro (15:00) — 2 prises
7. Conclusion (17:00-18:00) — 5 prises (c'est la fermeture, soigner)

**Session 2 — Sections screencast + slides (voix off)**
1. Biais Reuters + iterations (03:00-04:30) — 2 prises
2. Architecture + pipeline cascade (04:30-06:30) — 2 prises
3. Demo dashboard + emotions (06:30-09:30) — 3 prises (la demo peut rater)
4. XAI + faithfulness (09:30-12:00) — 2 prises
5. Qualite + securite (12:00-13:30) — 2 prises
6. Conformite + Green IT (13:30-15:00) — 2 prises
7. Methodologie (15:00-15:45) — 2 prises
8. ROI (15:45-16:30) — 2 prises
9. Limites + roadmap (16:30-17:00) — 2 prises

**Session 3 — Captures ecran (sans voix)**
Enregistrer toutes les navigations dashboard separement,
puis synchroniser au montage.

---

*Script video Thumalien V9 — v5 solo — Juillet 2026*
