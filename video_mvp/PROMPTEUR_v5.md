# PROMPTEUR v5 — THUMACHECK
## Niamato Consulting pour Thumalien
### Presentation solo Azelie — Juillet 2026

---
---
---

# ========================================
# AZELIE — HOOK (00:00 - 00:30)
# ========================================
# [CAMERA] Bandeau : Azelie Bernard — Lead Technique & Architecture

En decembre 2025,
on collecte 100 000 posts Bluesky.

On entraine un modele
de detection de fake news.

On obtient un F1-score de 0,99
en cross-validation.

Et c'est precisement a ce moment-la
qu'on aurait du s'inquieter.



---
---
---

# ========================================
# AZELIE — CONTEXTE CLIENT (00:30 - 01:30)
# ========================================
# [SLIDE 1 : Titre projet]

Bonjour, je suis Azelie Bernard,
lead technique chez Niamato Consulting.

Nous sommes un cabinet d'expertise
Data et Intelligence Artificielle,
et vous nous avez mandates
pour developper une solution
de detection de desinformation sur Bluesky.

---

Vos equipes de fact-checking passent
en moyenne 3 a 5 minutes par post
pour evaluer sa fiabilite.

A 60 000 posts publics par jour
sur Bluesky, c'est intenable.

---

ThumaCheck est la solution
que nous avons construite pour vous.

Elle analyse un texte
en 1,5 milliseconde,
avec une explication auditable
a chaque decision.

Et si votre base de donnees
est temporairement indisponible,
le dashboard continue de fonctionner
en mode demonstration —
vos equipes ne sont jamais bloquees.



---
---
---

# ========================================
# AZELIE — EQUIPE + PROBLEMATIQUE (01:30 - 03:00)
# ========================================
# [SLIDE 2 : Equipe et roles]

Ce projet a ete realise
en binome.

Mon collegue Sebastien Lazcanotegui
a pris en charge la validation
et la qualite du machine learning.

C'est lui qui a pilote
l'annotation du gold set —
465 textes Bluesky annotes
par deux annotateurs humains —
le debiaisage des donnees,
et l'optimisation des hyperparametres.

De mon cote,
j'ai concu l'architecture du pipeline,
le dashboard
et les trois niveaux d'explicabilite.

---

# [SLIDE 3 : Chiffres cles]

Bluesky :
35 millions d'utilisateurs,
protocole AT entierement ouvert,
plus de 60 000 posts publics par jour
analysables.

Aucune equipe de moderation centralisee.

---

Ce que nous vous livrons :
classifier un post
comme fiable ou suspect
en moins de 5 millisecondes,
en francais comme en anglais,
avec une explication
que vos equipes peuvent defendre
aupres de votre direction
et de vos regulateurs.

---

# [SLIDE 4 : Les 4 exigences]

Des le depart,
nous avons fixe avec vous
quatre exigences non-negociables.

Transparence :
chaque score est explicable.

Bilinguisme :
francais et anglais
a performance equivalente.

Frugalite :
moins de 5 ms par texte,
empreinte carbone mesuree.

Et conformite au RGPD
et a l'AI Act europeen,
applicable en aout 2026.

Ces quatre exigences
structurent toute la solution
que nous vous presentons aujourd'hui.



---
---
---

# ========================================
# AZELIE — BIAIS REUTERS + ITERATIONS (03:00 - 04:30)
# ========================================
# [ECRAN : notebook Jupyter, sortie F1=0.99]

Voici notre premiere version du pipeline.
Logistic Regression sur du TF-IDF,
30 000 features,
entrainee sur 197 782 articles.

Cross-validation a 5 plis :
F1 macro de 0,99.
Sur le papier, le projet est termine.

---

# [SLIDE 5 : F1=0.99 — LE PIEGE]

Sauf qu'on a regarde
quels mots avaient le plus de poids.

Et on a trouve :
reuters, afp, associated press.

Le modele n'apprenait pas
a detecter la desinformation.
Il apprenait a reconnaitre
le style des agences de presse.

---

Sebastien a pilote le debiaisage :
creation de la liste de termes d'agences
pour neutraliser ces signatures,
filtrage des annees artefacts 2015-2020,
et les tests de non-regression
apres chaque correction.

En parallele, il a lance
un GridSearch systematique
sur les hyperparametres du modele.

---

De la version 2 a la version 5,
on a corrige ce biais
et ajoute des donnees bilingues.

Le F1 en cross-validation
est descendu a 0,91.

Mais cette fois,
c'est un F1 honnete.



---
---
---

# ========================================
# AZELIE — ARCHITECTURE (04:30 - 06:30)
# ========================================
# [SLIDE 6 : Architecture C4]

A partir de cette base saine,
j'ai itere l'architecture
jusqu'a la version 9 actuelle.

Voici notre architecture.
Le flux suit trois couches :
donnees, intelligence artificielle, decision.

---

Premiere couche — les donnees.
Le collecteur se connecte a Bluesky
via le protocole AT
et stocke les posts dans MongoDB.
537 000 posts collectes a ce jour,
avec deduplication automatique
et reprise sur erreur reseau.

---

Deuxieme couche — l'intelligence artificielle.
Le pipeline analyse chaque texte
avec trois modeles complementaires.

V5 — notre modele principal :
il combine TF-IDF avec 17 features linguistiques
et 7 emotions detectees automatiquement.
C'est lui qui repond en 1,5 milliseconde.

V6 — un classifieur qui analyse
uniquement le style d'ecriture,
independamment du sujet.

Et CamemBERT — un modele de deep learning
specialise sur les textes courts en francais.

Le meta-learner V8 combine les trois
pour la decision finale.

---

Troisieme couche — la decision.
Le dashboard expose les resultats
avec cinq pages dediees a vos equipes
et trois niveaux d'explication.

Le tout est conteneurise avec Docker Compose —
une seule commande pour tout demarrer.

---

# [SLIDE 7 : Pipeline cascade]

En production,
le pipeline fonctionne
en deux etapes.

La premiere etape separe
les opinions des faits —
parce que l'AI Act interdit
de qualifier une opinion de desinformation.

La seconde etape analyse uniquement
les contenus factuels.

---

Cette cascade a reduit
vos faux positifs de 57 a 21
sur notre jeu de test —
soit une baisse de 67 %,
validee statistiquement.

Concretement pour vous :
moins d'alertes inutiles
dans les files de moderation
de vos equipes.



---
---
---

# ========================================
# AZELIE — DEMO DASHBOARD (06:30 - 09:30)
# ========================================
# [ECRAN : dashboard Streamlit plein ecran]

Voici le dashboard ThumaCheck —
l'interface que vos equipes
utiliseront au quotidien.

Cinq pages.
Vue Globale :
537 000 posts collectes,
67 % classes fiables.

---

Chaque visualisation
a ete pensee pour la decision :

la jauge de score utilise
un code couleur vert-rouge intuitif,

les barres d'explication montrent
ce qui pousse vers fiable ou suspect,

et la heatmap de mots-cles
guide l'oeil de vos analystes
vers les passages critiques.

---

# [ECRAN : soumettre texte fiable]

Premier test :
une depeche scientifique du CNRS
publiee dans Nature.

---

Avant meme d'evaluer la fiabilite,
le pipeline effectue un premier filtre.

Il verifie si le texte exprime un fait
ou une opinion —
parce que l'AI Act interdit
de qualifier une opinion de desinformation.

Ici, le contenu est factuel.
On passe a l'etape d'analyse.

---

Regardons les trois modeles
travailler en parallele.

V5 — TF-IDF principal —
score 0,94. Tres fiable.

V6 — detecteur de style —
0,05 de suspicion.
Le style est neutre et informatif.

Le meta-learner V8 combine les deux
avec CamemBERT :
verdict FIABLE.

---

Le desaccord V5 / V6 : 0,01.
Quasi nul.
Les trois modeles sont en accord.

Emotion dominante : neutre a 65,8 %.
Pas de charge emotionnelle excessive.
Coherent avec un contenu scientifique.

---

Maintenant les trois niveaux
d'explication.

Premier niveau :
contributions SHAP par feature de style.
Ce sont les contributions exactes
de chaque caracteristique linguistique
au score V6.

Deuxieme niveau :
la heatmap d'attention CamemBERT.

CamemBERT decoupe le texte
en sous-unites de mots.

Les tokens en rouge intense —
ce sont les ancres semantiques
qui ont guide la decision du modele.

Troisieme niveau :
la decomposition complete du meta-learner.

Le logit z vaut -0,918.
En probabilite : 28,5 % de suspicion.
Donc 71,5 % de fiabilite.
Decision FIABLE, seuil franchi.

On lit le poids exact de chaque modele,
coefficient par coefficient.

---

Vos analystes ont tout ce qu'il faut
pour justifier chaque decision
aupres de votre hierarchie —
et aupres de vos regulateurs.

---

# [ECRAN : soumettre texte suspect]

Deuxieme test :
un texte sensationnaliste
typique des reseaux sociaux.

---

Stage 1 : factuel a 48 %.
On passe a l'analyse.

---

V5 — TF-IDF principal —
score 0,00 de fiabilite.
Zero. Aucun signal de credibilite.

V6 — detecteur de style —
0,98 de suspicion.
Majuscules, points d'exclamation,
appel au partage, injonction a la peur.

Le meta-learner V8 :
logit z = +0,991.
72,9 % de suspicion.
Decision SUSPECT.

---

Desaccord V5/V6 : 0,02.
Les deux modeles sont en accord total.

---

Le dashboard liste automatiquement
les mots sensationnalistes :
"scandale", "on vous ment",
"partagez avant censure".

Ce sont exactement les marqueurs
que vos analystes cherchent
manuellement aujourd'hui.

---

Heatmap CamemBERT :
CamemBERT decoupe les mots
en sous-unites —
SCANDALE → SC-AND-ALE,
CACHE → CA-CHE.

Les sous-tokens en rouge
sont ceux sur lesquels
le modele concentre son attention.
Les termes alarmants ressortent.

---

Premier contributeur dans V8 :
score_v6_suspect a +0,567 vers SUSPECT.

C'est le style qui a fait basculer
la decision.
Pas le sujet — le style.

---

# [ECRAN : page emotions ou analyse]

Dernier element de cette demo :
l'emotion dominante.

---

Sur le texte fiable :
neutre a 65,8 %.

Sur ce texte suspect :
surprise a 74,6 %.

---

Les contenus de desinformation
utilisent deliberement la surprise
et la peur
pour contourner le sens critique.

---

Pour vos equipes,
un post SUSPECT
avec surprise elevee :
a prioriser en tete de file.

---

ThumaCheck analyse
7 dimensions emotionnelles :
colere, degout, joie, neutre,
peur, surprise, tristesse.

C'est ce que votre cahier des charges
demandait sous le nom
d'analyse emotionnelle.



---
---
---

# ========================================
# AZELIE — XAI + FAITHFULNESS (09:30 - 12:00)
# ========================================
# [SLIDE 8 : Faithfulness methode]

Vous pourriez nous demander :
comment savez-vous que ces explications
refletent vraiment
le comportement du modele ?

C'est une question legitime.
Un outil qui explique mal
est pire qu'un outil qui n'explique pas.

---

On a mis en place un protocole de verification.
Le principe est simple :

on identifie les criteres
que l'explication designe comme les plus importants,
on les masque dans le texte,
et on regarde si la prediction du modele change.

Si elle change fortement,
c'est que l'explication pointait
les bons criteres.

---

# [SLIDE 9 : courbe AOPC]

Le resultat est sur cette slide.

On a masque les mots que l'outil jugeait decisifs,
puis mesure combien la prediction changeait.
Comparaison : meme chose avec des mots pris au hasard.

Resultat : 5,6 fois plus d'impact
avec nos explications qu'avec le hasard.

Dit autrement :
quand notre outil vous dit
"ce mot a fait basculer la decision",
ce n'est pas une intuition.
C'est mesure, verifie, documente.

---

Pour les modeles de deep learning
comme CamemBERT,
on a utilise une technique d'attribution
au niveau de chaque mot.

On a aussi identifie les cas limites :
quand le modele est tres confiant,
les gradients saturent
et l'explication perd en precision.

C'est documente dans notre Model Card —
on sait ou nos explications sont fiables
et ou elles doivent etre lues
avec precaution.

Cette transparence sur les limites,
c'est exactement ce que l'AI Act
attend d'un systeme responsable.



---
---
---

# ========================================
# AZELIE — QUALITE INDUSTRIELLE (12:00 - 13:30)
# ========================================
# [SLIDE 10 : Qualite industrielle]

Quand vous evaluez une solution,
vous regardez aussi la qualite du code —
est-ce que ca tiendra en production ?

---

537 tests automatises.
80 % de couverture de code.

Un quality gate sur notre CI/CD
qui rejette automatiquement
toute modification
descendant sous 75 %.

---

Sebastien a egalement mis en place
du mutation testing —
on introduit volontairement
des erreurs dans le code
pour verifier que nos tests les detectent.

Sur le module le plus critique :
178 erreurs injectees,
143 detectees.
Taux de detection de 80,3 % —
au-dessus de la moyenne
des equipes Google.

---

En complement, un scan de securite
Trivy tourne a chaque commit
pour detecter les vulnerabilites connues.

Vos identifiants et mots de passe
sont isoles dans des variables d'environnement,
jamais exposes dans le code source.

---

C'est ce qui separe un prototype
d'un systeme que votre equipe
peut reprendre et maintenir.



---
---
---

# ========================================
# AZELIE — CONFORMITE + GREEN IT (13:30 - 15:00)
# ========================================
# [SLIDE 11 : Conformite]

Sur la conformite reglementaire —
un point crucial pour vous
si vous deployez de l'IA en Europe.

L'AI Act entre en application
le 2 aout 2026.

---

Nous avons classe ThumaCheck
en risque limite selon l'article 50.
Ca veut dire : obligation de transparence,
mais pas les contraintes des systemes a haut risque.

Article 13 — transparence :
couvert par notre Model Card,
un document standardise
qui decrit le modele, ses performances
et ses limites.

Article 14 — supervision humaine :
vos operateurs peuvent comprendre
et contester chaque decision
grace aux trois niveaux d'explication.

---

Cote RGPD :
article 22 — le droit d'explication
est couvert par nos outils d'explicabilite.

L'analyse d'impact sur la vie privee
est documentee.

Base legale : interet legitime
sur des posts publics.

Et si un utilisateur Bluesky
exerce son droit a l'effacement,
nous pouvons supprimer ses donnees
en moins de 5 secondes.

---

# [SLIDE 12 : Green IT]

Le bilan carbone total de tout le projet —
12 entrainements mesures avec CodeCarbon —
represente 8,9 grammes de CO2.

C'est l'equivalent de 52 metres
en voiture essence.
Reference ADEME 2024 : 170 g CO2/km.
Empreinte quasi nulle.

---

En production,
le modele principal V5 :
1,5 milliseconde par texte,
0,6 gramme de CO2 par jour.

Les modeles lourds comme CamemBERT
ne servent qu'en analyse approfondie,
pas en traitement temps reel.

C'est un choix d'architecture
documente et assume :
performance et sobriete
ne sont pas incompatibles.



---
---
---

# ========================================
# AZELIE — METHODOLOGIE (15:00 - 15:45)
# ========================================
# [SLIDE 13 : Methodologie]

Notre methodologie suit le cycle CRISP-DM
adapte au machine learning :
comprendre, explorer, preparer,
modeliser, evaluer, deployer.

9 versions en 6 mois,
chacune documentee
avec metriques avant et apres.

---

La gestion de projet :
un planning detaille avec 16 lots de travail
et 28 jalons,
versionne sur GitHub.

Nos outils :
Git avec integration continue,
Docker Compose, MongoDB,
FastAPI pour l'API,
CodeCarbon pour le suivi carbone.

Le tout reproductible
en une seule commande.



---
---
---

# ========================================
# AZELIE — ROI & BUDGET (15:45 - 16:30)
# ========================================
# [SLIDE 14 : ROI & Budget]

Parlons chiffres.

Le projet ThumaCheck
a coute environ 50 000 euros,
essentiellement en ressources humaines —
110 jours-homme repartis
entre le developpement
et la validation.

---

Zero euro de licence :
notre stack est 100 % open source.

Zero euro de cloud :
tout tourne en local sur Docker.

Le seul cout additionnel :
750 euros pour l'annotation
du jeu de test de reference.

---

En exploitation,
le cout mensuel
est d'environ 930 euros :
un serveur a 30 euros
et 2 jours de maintenance.

Pour 1,8 million de posts
analyses par mois,
ca revient a 0,0005 centime par post.

---

Ce que vous gagnez :
un facteur x10 de productivite
pour vos moderateurs,

une couverture de 60 000 posts par jour —
200 fois plus qu'un humain seul —

et une reduction de 67 %
des faux positifs
qui encombrent aujourd'hui
vos files de moderation.



---
---
---

# ========================================
# AZELIE — LIMITES + ROADMAP (16:30 - 17:00)
# ========================================
# [SLIDE 15 : Roadmap V10-V12]

En toute transparence, les limites.

L'accord entre nos deux annotateurs humains
est de 0,498 sur l'echelle de Cohen —
c'est un accord modere.

Ca ne veut pas dire
que le modele est mauvais.
Ca veut dire que la frontiere
entre fiable et suspect
est intrinsequement subjective,
meme pour des humains.

---

Notre F1 macro sur le jeu de test
est de 0,67.
En cross-validation sur les donnees
d'entrainement, il est de 0,91.

L'ecart s'explique par cette subjectivite.
C'est pourquoi ThumaCheck
donne un score de confiance,
pas un verdict definitif.

Vos analystes gardent le dernier mot.

---

ThumaCheck detecte
des signaux de desinformation —
le style, le ton, le sensationnalisme —
pas la verite factuelle.

C'est un outil d'aide a la decision,
pas un juge.

---

La roadmap :
V10 integre le suivi de modeles
pour detecter les baisses de performance.

V11 ajoute la verification factuelle
avec des outils comme ClaimBuster.

Et V12 prevoit un audit d'equite
pour garantir que le modele
traite tous les sujets de maniere equilibree.



---
---
---

# ========================================
# AZELIE — CONCLUSION (17:00 - 18:00)
# ========================================
# [SLIDE 16 : Citation]

Pour conclure,
je reviens au F1 de 0,99 du debut.

Ce chiffre flatteur cachait un biais.
Nous l'avons detecte,
corrige,
et documente.

---

Aujourd'hui, ThumaCheck V9
est valide par 537 tests automatises,
80 % de taux de detection en mutation testing,
une analyse d'impact vie privee,
une Model Card,
et une explicabilite
dont on a prouve la fidelite.

---

ThumaCheck n'est pas un classifieur.

C'est un systeme de decision auditable —
defendable devant vos analystes,
devant vos regulateurs,
et devant votre direction.

---

Un score sans explication
est un verdict sans proces.

C'est la phrase qui a guide
chacune de nos decisions techniques
tout au long de ce projet.

---

# [SLIDE 17 : Remerciements + QR codes]

Merci pour votre attention.

Le repository, le rapport technique
et la Model Card
sont accessibles via les QR codes
a l'ecran.

Je suis a votre disposition
pour vos questions.



---
---
---

# FIN DU SCRIPT
