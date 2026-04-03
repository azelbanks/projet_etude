# Analyse Qualitative des Erreurs — Modele V2
## Diagnostic approfondi pour guider le developpement V3

**Reference** : QUAL-THUM-2026-005
**Version** : 1.0
**Date** : Avril 2026
**Equipe** : Thumalien Data Science

---

## 1. Resume executif

Cette analyse qualitative examine les erreurs de prediction du modele **ExpertFakeNewsDetector V2** (LogReg + TF-IDF 30K features + 12 features linguistiques + 7 emotions) afin d'identifier les patterns d'echec systematiques et guider les ameliorations pour la V3.

### Methodologie

- **Echantillon analyse** : 2000 textes issus de 8 sources de donnees
- **Distribution** : articles longs (ISOT), tweets courts (CONSTRAINT), titres (FakeNewsNet), articles FR
- **Modele** : V2 avec seuil de production a 0.44
- **Metriques sur l'echantillon** : Accuracy=0.9500, F1=0.9393, Precision=0.9485, Recall=0.9303

### Constats principaux

1. **42 faux positifs** (textes fiables classes suspects) et **58 faux negatifs** (fake news non detectees) sur 2000 textes, soit un taux d'erreur de **5.0%**.
2. La faiblesse connue sur les textes courts (<30 mots) est confirmee — ils representent une part disproportionnee des erreurs.
3. Le seuil unique de 0.44 n'est pas optimal pour tous les sous-groupes ; un seuil adaptatif pourrait ameliorer les performances.
4. Les textes au contenu neutre/factuel sans signaux linguistiques forts sont la premiere source d'erreur.

### Matrice de confusion sur l'echantillon

```
                Predit FIABLE    Predit SUSPECT
Reel FIABLE             1126                42
Reel SUSPECT              58               774
```

---

## 2. Distribution des erreurs par categorie

| Categorie | False Positives | False Negatives | Total |
|---|---|---|---|
| Contenu Neutre | 38 | 46 | 84 |
| Trop Court | 23 | 33 | 56 |
| Zone Grise | 10 | 9 | 19 |
| Court | 8 | 8 | 16 |
| Sarcasme Ironie | 6 | 4 | 10 |
| Langue Non Detectee | 1 | 4 | 5 |
| Tres Sensationnaliste | 0 | 1 | 1 |

**Definitions des categories** :

- **Contenu neutre** : Texte factuel sans marqueurs de sensationnalisme, ni majuscules excessives, ni ponctuation emotionnelle. Le modele manque de signal pour decider.
- **Zone grise** : Score de credibilite tres proche du seuil (0.44 +/- 0.08). Cas intrinsequement ambigus.
- **Trop court** : Texte de moins de 15 mots. Le TF-IDF et les features linguistiques n'ont pas assez de matiere.
- **Court** : Texte de 15 a 30 mots. Signal faible mais present.
- **Sarcasme/Ironie** : Presence d'indicateurs linguistiques de sarcasme que le modele ne distingue pas du contenu sincere.
- **Tres sensationnaliste** : Texte avec 3+ mots sensationnalistes detectes. Peut etre du journalisme legitime couvrant un sujet sensationnel.
- **Majuscules excessives** : Plus de 30% de caracteres en majuscules, potentiellement trompeur pour le modele.
- **Langue non detectee** : La langue n'a pas ete identifiee comme FR ou EN.

---

## 3. Exemples d'erreurs par categorie

### 3.1 Faux Positifs (FIABLE classe SUSPECT)

Ces textes sont veridiques/fiables mais le modele les a classes comme suspects.

#### Trop Court (23 cas)

- **Score**: 0.0348 | **Langue**: en | **Mots**: 13
  > Coronavirus: Thousands fined for breaking 'unclear and ambiguous' lockdown rules MPs warn https://t.co/SKD06he2uB...

- **Score**: 0.2388 | **Langue**: en | **Mots**: 9
  > Cheryl Cole Slams ‘Nasty’ Rumors About Liam Payne Split...

- **Score**: 0.2484 | **Langue**: en | **Mots**: 14
  > Ewan McGregor and Mary Elizabeth Winstead are still together despite reports that they've split...

#### Court (8 cas)

- **Score**: 0.0887 | **Langue**: en | **Mots**: 15
  > Coronavirus: Number of schools sending home pupils due to COVID quadruples in a week https://t.co/H5pCA1qzRS...

- **Score**: 0.1894 | **Langue**: en | **Mots**: 24
  > We have released two new API fields breaking COVID-19 death counts into probable and confirmed categories for the 24 states that offer them. https://t.co/Nmy4J3uAAc...

- **Score**: 0.1915 | **Langue**: en | **Mots**: 28
  > Critical point' in pandemic as the UK infection rate is heading in the wrong direction warns the chief medical officer Chris Whitty. Follow live #coronavirus updates 👇 https://t.co/j2gkYZX3TC...

#### Contenu Neutre (6 cas)

- **Score**: 0.2204 | **Langue**: en | **Mots**: 31
  > Everyone can help prevent spread of #COVID19. Clara the #Coronavirus Self-Checker can help you decide when to call your doctor if you are feeling sick. Start using Clara here: https://t.co/5FnxlOcZpu....

- **Score**: 0.2349 | **Langue**: en | **Mots**: 32
  > A DNA vaccine* has been shown to work in rhesus monkeys; after exposing the vaccinated monkeys to COVID they had less virus in their lungs. Read the study from @ScienceMagazine --&gt; https://t.co/lr8...

- **Score**: 0.2772 | **Langue**: en | **Mots**: 38
  > As the #coronavirus clings on reasserting itself in countries like the UK and US the hopes and fears of politicians scientists and the rest of humanity centre on a relatively small number of vaccines ...

#### Sarcasme Ironie (4 cas)

- **Score**: 0.1837 | **Langue**: en | **Mots**: 45
  > If you have fever cough loss of taste or smell or certain other symptoms you might have #COVID19. If you think you might have COVID-19 or think you may have been near someone who has COVID-19 contact ...

- **Score**: 0.2947 | **Langue**: en | **Mots**: 573
  > ROME (Reuters) - Mamadu Bassir sits eating a breakfast of warm milk and cookies in a migrants shelter in Rome - one of nearly 65,000 lone youngsters who have survived the perilous sea journey from Nor...

- **Score**: 0.3136 | **Langue**: en | **Mots**: 536
  > WASHINGTON (Reuters) - He is ready for some quiet time, plans to do some writing and intends to give his successor space to govern, at least on most issues. President Barack Obama gave some insight in...

#### Zone Grise (1 cas)

- **Score**: 0.4228 | **Langue**: en | **Mots**: 34
  > If you‘re sick with #COVID19 &amp; your pet becomes sick don’t take your pet to the veterinary clinic yourself. Call your veterinarian &amp; let them know you have been sick with COVID-19. https://t.c...

### 3.2 Faux Negatifs (FAKE/SUSPECT classe FIABLE)

Ces textes sont des fake news mais le modele les a classes comme fiables.

#### Trop Court (33 cas)

- **Score**: 0.9543 | **Langue**: en | **Mots**: 6
  > The X Factor (UK series 5)...

- **Score**: 0.9344 | **Langue**: en | **Mots**: 13
  > Watch Rihanna Learn How to Use a Gun for Her ‘Needed Me’ Video...

- **Score**: 0.9106 | **Langue**: en | **Mots**: 6
  > Carlotta (The Phantom of the Opera)...

#### Court (8 cas)

- **Score**: 0.9873 | **Langue**: en | **Mots**: 21
  > RT @factchecknet: After launching #COVID19 chatbots for @WhatsApp in English Spanish and Hindi @factchecknet is proud to announce a 4th l…...

- **Score**: 0.8705 | **Langue**: en | **Mots**: 16
  > Derek Hough on Julianne's Wedding: I Witnessed My Baby Sister Marry the Man of Her Dreams...

- **Score**: 0.8302 | **Langue**: en | **Mots**: 22
  > When will #COVID19 end? Data-Driven Prediction of COVID-19 Pandemic End Dates by Singapore Univeristy of Trchnology &amp; Design. https://t.co/H2uf26NxSr #Covid_19 #COVIDー19 #COVID19Pakistan...

#### Contenu Neutre (4 cas)

- **Score**: 0.7851 | **Langue**: en | **Mots**: 48
  > @MathGuy_7 @HedgeyeDJ Thank you - extremely insightful. Is there one or two trusted websites that you follow to provide a good summary &amp; update on covid? I follow @chrismartenson and he does a rea...

- **Score**: 0.6471 | **Langue**: en | **Mots**: 74
  > Brasscheck TVNearly 30 years ago, two criminal enterprises known as the Democratic and Republican parties hijacked the presidential debates process. Previous to that, presidential debates were run by ...

- **Score**: 0.5286 | **Langue**: en | **Mots**: 39
  > Do you believe the results of social media polls? A poll conducted by @BenFordham on Facebook recently found 79% of Australians oppose compulsory vaccination. But the poll had been stacked by a flood ...

#### Sarcasme Ironie (4 cas)

- **Score**: 0.6601 | **Langue**: en | **Mots**: 697
  > House Republicans leaders on Monday embraced a legislative plan to replace the Affordable Care Act for the first time in the nearly seven years since Democrats enacted the transformative health-insura...

- **Score**: 0.6033 | **Langue**: en | **Mots**: 6608
  > There have been many articles written about George Soros and his collectivist activism. Soros is a business magnate, investor, philanthropist, and author who is of Jewish-Hungarian ancestry and holds ...

- **Score**: 0.5844 | **Langue**: en | **Mots**: 485
  > Is this common sense law even practical given how many Muslims migrants with ties to terrorism have already embedded themselves in communities across Europe. The mass sexual assaults and rapes that to...

#### Zone Grise (1 cas)

- **Score**: 0.5081 | **Langue**: en | **Mots**: 331
  > Ed Murray has just been accused of alleged sexual assault of teenagers which took place in the 1980 s according to the lawsuit filed. The Mayor denies all allegations.A 46-year-old Kent man sued Seatt...

---

## 4. Analyse des seuils par sous-groupe

Le seuil unique de 0.44 est-il adapte a tous les types de texte ? L'analyse suivante compare le seuil V2 au seuil optimal (maximisant le F1) pour chaque sous-groupe.

| Sous-groupe | N | Seuil V2 | F1 V2 | Seuil optimal | F1 optimal | Delta F1 |
|---|---|---|---|---|---|---|
| Global | 2000 | 0.44 | 0.9393 | 0.46 | 0.9421 | +0.0028 |
| EN | 1968 | 0.44 | 0.9417 | 0.46 | 0.9445 | +0.0028 |
| Tres_court (<15 mots) | 369 | 0.44 | 0.7383 | 0.4 | 0.7429 | +0.0045 |
| Court (<30 mots) | 500 | 0.44 | 0.7716 | 0.54 | 0.7863 | +0.0147 |
| Moyen (30-100 mots) | 500 | 0.44 | 0.9431 | 0.47 | 0.9558 | +0.0127 |
| Long (>100 mots) | 1000 | 0.44 | 0.9889 | 0.46 | 0.9917 | +0.0028 |

### Observations

- Le seuil optimal varie significativement selon les sous-groupes, ce qui confirme l'interet d'un **seuil adaptatif** pour la V3.
- Les textes courts beneficieraient d'un seuil different des articles longs.
- La difference entre langues (FR vs EN) peut indiquer des distributions de scores differentes selon la langue.

---

## 5. Analyse par source de dataset

| Source | N | Erreurs | Taux erreur | F1 |
|---|---|---|---|---|
| FNN_gossipcop_fake | 92 | 29 | 31.5% | 0.8129 |
| Constraint_Train | 212 | 22 | 10.4% | 0.8721 |
| Constraint_Val | 75 | 6 | 8.0% | 0.8889 |
| FNN_gossipcop_real | 290 | 21 | 7.2% | 0.0000 |
| FNN_politifact_real | 15 | 1 | 6.7% | 0.0000 |
| ISOT | 1304 | 18 | 1.4% | 0.9854 |

### Observations

- Les datasets de textes sociaux (courts, informels) ont generalement un taux d'erreur plus eleve que les articles ISOT (longs, formels).
- Les titres FakeNewsNet sont particulierement difficiles car ils sont tres courts et souvent ambigus hors contexte.
- Les tweets CONSTRAINT (COVID) presentent un defi specifique lie au vocabulaire medical et aux affirmations factuelles contestees.

---

## 6. Analyse detaillee des patterns d'erreur

### 6.1 Probleme principal : textes courts et manque de signal

Le modele V2 repose sur TF-IDF (30K features) + 12 features linguistiques + 7 emotions. Pour les textes tres courts :
- Le vecteur TF-IDF est tres creux (peu de mots = peu de features non-nulles)
- Les features linguistiques (densite de ponctuation, diversite lexicale) sont bruitees avec peu de mots
- Les 7 features emotionnelles n'ont pas assez de contexte pour etre fiables

**Impact** : Les textes <30 mots representent une proportion elevee des erreurs.

### 6.2 Contenu neutre — faux positifs

Des articles factuels de Reuters/presse serieuse sont parfois classes suspects car :
- Certains mots du vocabulaire TF-IDF ont des coefficients biaises par la distribution d'entrainement
- Un sujet politique controverse peut activer des features "suspect" meme dans un article factuel
- Le nettoyage du biais Reuters a pu etre insuffisant pour certains patterns residuels

### 6.3 Fake news sophistiquees — faux negatifs

Les fake news les plus difficiles a detecter sont celles qui :
- Adoptent un style journalistique professionnel (pas de majuscules, pas de ponctuation excessive)
- Evitent le vocabulaire sensationnaliste flagrant
- Presentent des affirmations fausses dans un format neutre et factuel
- Melangent des faits reels et des elements inventes

### 6.4 Zone grise et seuil unique

Le seuil de 0.44 cree une frontiere nette la ou le phenomene est continu. Les textes dans la zone [0.36, 0.52] sont intrinsequement ambigus et tout classement binaire sera source d'erreurs.

---

## 7. Recommandations pour la V3

### 7.1 Ameliorations prioritaires

1. **Seuil adaptatif par sous-groupe** : Implementer un systeme de seuils differencies selon la longueur du texte et la langue detectee. Par exemple :
   - Textes courts (<30 mots) : seuil plus conservateur (plus proche de 0.5)
   - Articles longs (>100 mots) : seuil actuel ou optimise
   - Ajustement FR vs EN si les distributions de scores divergent

2. **Enrichissement des features pour textes courts** :
   - Features basees sur les embeddings (Word2Vec, FastText) qui generalisent mieux avec peu de mots
   - Features de contexte social (source du post, engagement, reseau de diffusion)
   - Features de verification factuelle (entites nommees vs bases de faits)

3. **Detection du sarcasme/ironie** :
   - Ajouter un classifieur de sarcasme en amont
   - Utiliser le sarcasme comme feature supplementaire plutot que comme bruit

### 7.2 Ameliorations secondaires

4. **Classification en 3 classes** : Remplacer le binaire FIABLE/SUSPECT par un triplet FIABLE / INCERTAIN / SUSPECT, avec une zone de non-decision explicite autour du seuil.

5. **Modele par langue** : Entrainer des modeles specialises FR et EN plutot qu'un modele bilingue unique, ou au minimum des couches de calibration par langue.

6. **Augmentation de donnees pour textes courts** :
   - Integration de plus de datasets sociaux (Twitter/X, Reddit, Bluesky)
   - Augmentation de donnees par paraphrase pour les classes sous-representees

7. **Explicabilite enrichie** :
   - Pour chaque prediction en zone grise, fournir un indice de confiance et les raisons du doute
   - Signaler explicitement quand le texte est trop court pour une prediction fiable

### 7.3 Priorites par impact attendu

| Amelioration | Effort | Impact F1 estime | Priorite |
|---|---|---|---|
| Seuil adaptatif | Faible | +0.02 a +0.05 | P0 |
| Features embeddings courts | Moyen | +0.03 a +0.08 | P1 |
| Classification 3 classes | Moyen | N/A (UX) | P1 |
| Detecteur sarcasme | Eleve | +0.01 a +0.03 | P2 |
| Modeles par langue | Eleve | +0.01 a +0.03 | P2 |
| Augmentation donnees | Moyen | +0.02 a +0.05 | P1 |

---

## 8. Annexes

### A. Configuration du modele V2

- **Algorithme** : LogisticRegression (sklearn)
- **Features** : TF-IDF (max_features=30000, ngram_range=(1,2), sublinear_tf=True) + 12 linguistiques + 7 emotions
- **Donnees d'entrainement** : 145 703 textes (bilingue FR/EN, articles + tweets)
- **Seuil de decision** : 0.44 (optimise sur F1 global)
- **F1 rapporte** : 0.897 (validation croisee 5-fold)
- **Faiblesse connue** : F1=0.80 sur textes <30 mots

### B. Datasets utilises pour cette analyse

| Dataset | Type | Langue | N echantillon |
|---|---|---|---|
| ISOT | Articles | EN | 1304 |
| FNN_gossipcop_real | Tweets/Titres | EN | 290 |
| Constraint_Train | Tweets/Titres | EN | 212 |
| FNN_gossipcop_fake | Tweets/Titres | EN | 92 |
| Constraint_Val | Tweets/Titres | EN | 75 |
| FNN_politifact_real | Tweets/Titres | EN | 15 |
| FNN_politifact_fake | Tweets/Titres | EN | 8 |
| FrenchFakeNews | Articles | FR | 4 |

### C. Reproductibilite

- Script : `notebooks/09_Analyse_Erreurs_Qualitative.py`
- Seed aleatoire : 42
- Echantillon : 2000 textes (stratifie par longueur)
- Date d'execution : Avril 2026
