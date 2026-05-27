# PROMPTEUR — THUMACHECK
## Niamato Consulting pour Thumalien

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
et Thumalien nous a mandates
pour developper une solution
de detection de desinformation sur Bluesky.

---

Vous le savez :
vos equipes de fact-checking passent
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



---
---
---

# ========================================
# SEBASTIEN — EQUIPE + PROBLEMATIQUE (01:30 - 03:00)
# ========================================
# [CAMERA] Bandeau : Sebastien Lazcanotegui — Validation & Qualite ML
# [SLIDE 2 : Equipe et roles]

Bonjour, je suis Sebastien Lazcanotegui,
consultant chez Niamato Consulting.

Mon role :
la validation et la qualite
du machine learning.

J'ai pilote l'annotation du gold set —
473 posts annotes
par deux annotateurs humains —
le debiaisage des donnees,
et l'optimisation des hyperparametres.

Azelie a concu l'architecture du pipeline,
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

Ce que nous vous garantissons :
classifier un post
comme fiable ou suspect
en moins de 5 millisecondes,
en francais comme en anglais,
avec une explication qui permet
a vos equipes de defendre chaque decision.

---

# [SLIDE 4 : Les 4 exigences]

Nous avons fixe
quatre exigences non-negociables.

Transparence :
chaque score est explicable.

Bilinguisme :
francais et anglais
a performance equivalente.

Frugalite :
moins de 5 ms par texte,
empreinte CO2 mesuree.

Et conformite au RGPD et a l'AI Act.

Ces quatre exigences
structurent toute la solution
que nous vous presentons aujourd'hui.



---
---
---

# ========================================
# SEBASTIEN — BIAIS REUTERS + ITERATIONS (03:00 - 04:30)
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

C'est moi qui ai pilote le debiaisage :
creation de la liste BODY_AGENCY_TERMS
pour neutraliser les signatures d'agences,
filtrage des annees artefacts 2015-2020,
et les tests de non-regression
apres chaque correction.

En parallele, j'ai lance
un GridSearch systematique :
le passage de C=1 a C=5
et de min_df=3 a min_df=5
a confirme les hyperparametres optimaux.

---

De la version 2 a la version 5,
on a corrige ce biais
et ajoute des donnees synthetiques bilingues.

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
# [CAMERA] Bandeau : Azelie Bernard — Lead Technique & Architecture

A partir de cette base saine,
j'ai itere l'architecture
jusqu'a la version 9 actuelle.

---

# [SLIDE 6 : Architecture C4]

Voici notre architecture,
modelisee en C4 niveau 2.

Le flux est simple :
donnees, intelligence artificielle, decision.

---

Le collecteur se connecte a Bluesky
via le protocole AT
et stocke les posts dans MongoDB —
c'est la couche donnees.

Le pipeline NLP analyse chaque texte
avec trois modeles —
c'est la couche IA.

Le dashboard Streamlit
expose les resultats
avec trois niveaux d'explication —
c'est la couche decision.

Le tout est conteneurise
avec Docker Compose.

---

Trois modeles travaillent en ensemble :

V5 — LogReg + TF-IDF
+ 15 features linguistiques + 7 emotions,
c'est notre baseline frugale
a 1,5 milliseconde.

V6 — un classifieur style-only
avec 28 features purement stylistiques,
topic-agnostique.

CamemBERT fine-tune
sur des textes courts francais.

Le meta-learner V8 combine les trois.

---

# [SLIDE 7 : Pipeline cascade]

En production,
le pipeline cascade fonctionne
en deux etapes.

Le Stage 1 separe les opinions des faits —
parce que l'AI Act interdit de qualifier
une opinion de fake news.

Le Stage 2 analyse uniquement
les contenus factuels.

---

Cette cascade fait passer
nos faux positifs de 57 a 21
sur le gold set —
une reduction de 67 %,
statistiquement significative,
p < 0.000001.



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
245 000 posts collectes,
67 % classes fiables.

---

Chaque visualisation
a ete pensee pour la decision :

la jauge de score utilise
un code couleur vert-rouge intuitif,

les barres SHAP montrent
la contribution positive ou negative
de chaque feature,

et la heatmap d'attention utilise un gradient
qui guide l'oeil de vos analystes
vers les tokens critiques.

---

# [ECRAN : soumettre texte fiable]

Premier test : un communique scientifique.
Score 0,89 fiable.

Mais ce qui distingue ThumaCheck
d'un classifieur boite noire —
trois niveaux d'explication.

---

Niveau 1 : les top mots de la LogReg.
Pas une approximation —
c'est la formule fermee exacte du modele.

Niveau 2 : SHAP applique
au modele de style V6.
La presence de citations
et la diversite lexicale
poussent vers fiable.

Niveau 3 : la decomposition exacte
du meta-learner V8.
On lit litteralement la decision du modele,
coefficient par coefficient.

---

Vos analystes ont tout ce qu'il faut
pour justifier chaque decision
aupres de votre hierarchie —
et aupres de vos regulateurs.

---

# [ECRAN : soumettre texte suspect]

Deuxieme test :
un texte sensationnaliste.

Score 0,12, donc 0,88 suspect.

La decomposition est inversee :
le sensationnalisme passe en rouge
a +0,71.

---

Et voici la heatmap d'attention CamemBERT :
les tokens "SCANDALE" et "mentent"
s'illuminent en rouge.

C'est exactement ce que vos analystes
veulent voir pour defendre leur decision.



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

On a implemente le protocole ERASER
de DeYoung et collegues, ACL 2020.

On masque les top-k features
identifiees par SHAP,
et on mesure si la prediction chute
plus vite qu'avec un masquage aleatoire.

---

# [SLIDE 9 : courbe AOPC]

Le resultat :
AOPC attribution 0,253
contre 0,045 pour le random.

Uplift de +0,21.

Nos explications sont 5,6 fois
plus predictives
qu'une attribution au hasard.

---

Pour les transformers,
on est alles plus loin
avec Layer Integrated Gradients
via Captum.

Et on a decouvert que
sur les cas ou CamemBERT est tres confiant,
le ReLU du head sature
et bloque les gradients.

C'est une signature documentee
dans la Model Card —
on sait ou nos explications sont fiables
et ou elles ne le sont pas.



---
---
---

# ========================================
# SEBASTIEN — QUALITE INDUSTRIELLE (12:00 - 13:30)
# ========================================
# [CAMERA] Bandeau : Sebastien Lazcanotegui — Validation & Qualite ML
# [SLIDE 10 : Qualite industrielle]

Quand vous evaluez une solution,
vous regardez aussi la qualite du code —
est-ce que ca tiendra en production ?

---

501 tests pytest.
80 % de couverture de code.
77,9 % de branch coverage.

Quality gate sur GitHub Actions
qui rejette toute PR
descendant sous 75 %.

---

On a fait du mutation testing
avec mutmut
sur le module critique
de decomposition meta-learner.

178 mutations artificielles.
143 detectees.
Kill rate : 80,3 % —
au-dessus de la moyenne Google
qui se situe entre 60 et 75 %.

---

C'est ce qui separe un prototype
d'un systeme que votre equipe DevOps
peut reprendre des demain.



---
---
---

# ========================================
# SEBASTIEN — CONFORMITE + GREEN IT (13:30 - 15:00)
# ========================================
# [SLIDE 11 : Conformite]

Sur la conformite reglementaire —
un point crucial pour vous
si vous deployez de l'IA en Europe
a partir d'aout 2026.

---

AI Act article 13 transparence :
couvert par notre Model Card
au format Mitchell 2019.

Article 14 supervision humaine :
la decomposition dans le dashboard
permet a vos operateurs de comprendre
et contester chaque decision.

---

RGPD article 22 :
le droit d'explication est couvert
par SHAP et Captum.

L'AIPD est documentee.
Base legale : interet legitime
sur des posts publics.

---

# [SLIDE 12 : Green IT]

Le bilan carbone total
est d'environ 6,9 grammes de CO2.

6,14 grammes mesures par CodeCarbon
sur 6 entrainements,
plus environ 0,7 gramme estime
pour V6, le pipeline XAI
et l'inference.

---

RoBERTa represente
la moitie du bilan,
LogReg un tiers,
et le reste se repartit
entre CamemBERT, V6
et l'explicabilite.

---

En production, V5 seul :
1,5 milliseconde,
0,6 gramme par jour.

CamemBERT sert uniquement
en analyse offline.

C'est un choix architectural
documente dans la Model Card.



---
---
---

# ========================================
# SEBASTIEN — METHODOLOGIE (15:00 - 15:45)
# ========================================
# [CAMERA] Bandeau : Sebastien Lazcanotegui — Validation & Qualite ML
# [SLIDE 13 : Methodologie]

Notre methodologie suit le cycle CRISP-DM
adapte au ML :
comprendre, explorer, preparer,
modeliser, evaluer, deployer.

9 versions en 6 mois,
chacune documentee
avec metriques avant/apres.

---

La gestion de projet :
un Gantt avec 16 work packages
et 28 jalons,
versionne sur GitHub.

Nos outils :
Git avec CI/CD sur GitHub Actions,
Docker Compose, MongoDB,
FastAPI, CodeCarbon.

Le tout reproductible
en une commande docker compose up.



---
---
---

# ========================================
# SEBASTIEN — ROI & BUDGET (15:45 - 16:30)
# ========================================
# [SLIDE 14 : ROI & Budget]

Parlons chiffres.

Le projet ThumaCheck
a coute environ 50 000 euros,
essentiellement en ressources humaines —
110 jours-homme repartis
entre le developpement technique
et la validation.

---

Zero euro de licence :
notre stack est 100 % open source.

Zero euro de cloud :
tout tourne en local sur Docker.

Le seul cout additionnel :
750 euros pour l'annotation manuelle
du gold set de validation.

---

En exploitation,
le cout mensuel
est d'environ 930 euros :
un petit serveur a 30 euros
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
qui polluent aujourd'hui vos files de moderation.



---
---
---

# ========================================
# SEBASTIEN — LIMITES + ROADMAP (16:30 - 17:00)
# ========================================
# [SLIDE 15 : Roadmap V10-V12]

En toute transparence, les limites.

Le kappa de Cohen entre annotateurs
est de 0,498 — modere.

La frontiere fiable/suspect
est intrinsequement subjective.

On attenue avec un intervalle de confiance
bootstrap a 95 % :
la reduction des faux positifs
reste entre -73 % et -60 %.

---

ThumaCheck detecte
des signaux de desinformation,
pas la verite factuelle.

On classe la forme,
pas le contenu.

---

La roadmap V10 inclut MLflow
pour le tracking,
V11 ClaimBuster
pour la verification factuelle,
et V12 un audit d'equite algorithmique.



---
---
---

# ========================================
# AZELIE — CONCLUSION (17:00 - 18:00)
# ========================================
# [CAMERA] Bandeau : Azelie Bernard — Lead Technique & Architecture
# [SLIDE 16 : Citation]

Pour conclure,
je reviens au F1 de 0,99 du debut.

Aujourd'hui, en V9,
notre F1 macro sur gold est de 0,67.

C'est plus bas.
C'est plus honnete.

---

C'est valide par 501 tests,
80 % de mutation kill rate,
une AIPD, une Model Card,
et une explicabilite
qui prouve sa propre fidelite.

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
chacune de nos decisions techniques.

---

# [SLIDE 17 : Remerciements + QR codes]

Merci pour votre attention.

Le repository, le rapport
et la Model Card
sont accessibles via les QR codes
a l'ecran.

Nous sommes disponibles
pour vos questions.



---
---
---

# FIN DU SCRIPT
