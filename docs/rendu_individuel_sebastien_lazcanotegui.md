# Rendu Individuel - Sebastien Lazcanotegui
## Projet Thumalien - Master Big Data, Sup de Vinci

**Etudiant** : Sebastien Lazcanotegui
**Role principal** : Consolidation ML, optimisation hyperparametres, revue de code
**Date** : Avril 2026

---

## 1. Mon role dans le projet

J'ai contribue au projet Thumalien en tant que **specialiste consolidation ML**, avec un focus sur l'optimisation des hyperparametres, le debiaisage du pipeline et la revue critique des choix techniques. Mon apport principal a ete la consolidation du modele NLP pour en ameliorer la robustesse.

### Responsabilites principales
- Optimisation des hyperparametres du pipeline (GridSearch)
- Implementation du debiaisage des datasets (Reuters, agences de presse)
- Revue et challenge des choix techniques
- Contribution a la video MVP

---

## 2. Contributions techniques detaillees

### 2.1 Optimisation des hyperparametres

J'ai mene une campagne de GridSearch systematique pour identifier les hyperparametres optimaux du pipeline NLP :

| Parametre | Avant | Apres | Justification |
|-----------|:-----:|:-----:|---------------|
| C (regularisation) | 1.0 | 5.0 | Meilleur F1 sur la grille exploree |
| min_df (frequence minimale) | 3 | 5 | Reduction du bruit, meilleure generalisation |
| ngram_range | (1,3) | (1,2) | Les trigrammes n'apportaient pas de valeur significative |

Cette optimisation a contribue a l'amelioration globale du pipeline de la V2 a la V3.

### 2.2 Debiaisage du pipeline

J'ai identifie et implemente des mesures de debiaisage dans le pipeline NLP :

- **BODY_AGENCY_TERMS** : liste de patterns regex pour neutraliser les signatures d'agences de presse (Reuters, Associated Press, AFP) qui permettaient au modele de "tricher"
- **ARTIFACT_YEAR_PATTERN** : suppression des annees (2015-2020) qui etaient correlees aux articles de presse mais pas a la veracite
- Integration dans la fonction `clean_for_ml()` du pipeline expert

### 2.3 Simplification du pipeline

J'ai propose la suppression de blocs de code qui ajoutaient de la complexite sans gain mesurable :
- Suppression du bloc d'augmentation FR court (redondant avec V5)
- Suppression du chargement de donnees sociales supplementaires
- Nettoyage des parametres inutilises dans `prepare_bilingual_dataset()`

### 2.4 Revue critique

Participation aux revues de modele pour valider les choix techniques :
- Validation de la pertinence du seuil 0.44
- Revue des metriques par segment (FR/EN, court/long)
- Challenge sur les risques de surapprentissage

---

## 3. Competences mobilisees et acquises

### 3.1 Competences techniques

| Competence | Niveau avant | Niveau apres | Contexte d'application |
|-----------|:------------:|:------------:|----------------------|
| Machine Learning (scikit-learn) | Intermediaire | Avance | GridSearch, regularisation, evaluation |
| NLP / Text Mining | Debutant | Intermediaire | TF-IDF, debiaisage, features linguistiques |
| Python | Intermediaire | Intermediaire | Modification du pipeline, scripting |
| Git / GitHub | Debutant | Intermediaire | Commits, merge, collaboration |
| Analyse de donnees | Intermediaire | Intermediaire | Identification des biais, analyse des distributions |

### 3.2 Competences transversales

| Competence | Mise en pratique |
|-----------|-----------------|
| Esprit critique | Identification des biais Reuters, remise en question des metriques |
| Rigueur scientifique | Protocole de GridSearch, comparaison systematique |
| Communication | Presentation des resultats d'optimisation a l'equipe |
| Travail en equipe | Coordination avec la lead technique, revues de code |

---

## 4. Defis rencontres et solutions

| Defi | Contexte | Solution | Apprentissage |
|------|----------|----------|---------------|
| Comprendre le pipeline existant | Code complexe avec 22 notebooks, architecture evoluant rapidement (V1 a V9) | Lecture methodique du code et des notebooks, sessions de pair-review avec Azelie, prise de notes structurees | L'importance de la documentation et de l'onboarding dans un projet data |
| GridSearch couteux en temps | Espace de recherche large (3 hyperparametres x 5 valeurs), 145K textes d'entrainement | Reduction de l'espace par analyse de sensibilite prealable, parallelisation des jobs | Equilibrer exhaustivite et pragmatisme — une recherche exhaustive n'est pas toujours necessaire |
| Merge Git catastrophique | Depot recree par erreur, divergence d'historique, conflits massifs sur 15+ fichiers | Resolution avec `--allow-unrelated-histories`, resolution manuelle des conflits, verification post-merge | Toujours travailler sur une branche dediee, ne jamais reinitialiser un depot partage |
| Debiaisage sans casser les performances | Retirer les marqueurs Reuters/AP sans degrader le F1 sur les textes non-biaises | Approche incrementale : retrait progressif, tests de non-regression apres chaque modification | Le debiaisage est un equilibre delicat — chaque correction doit etre validee |
| Annotation des 500 posts | Volume important (500 posts), criteres de classification ambigus sur certains cas limites (satire, opinion forte) | Suivre rigoureusement le guide d'annotation, discuter les cas limites avec Azelie pour aligner les criteres | La subjectivite est inherente a l'annotation — le kappa mesure cette subjectivite, il ne l'elimine pas |
| Adapter mon rythme au projet en cours | Integration en cours de projet, pipeline deja avance (V5), necessite de contribuer sans casser l'existant | Cibler les taches a forte valeur ajoutee (GridSearch, debiaisage) plutot que de toucher a l'architecture globale | Il vaut mieux faire peu mais bien que de tout changer sans comprendre l'impact |

---

## 5. Analyse critique et recul

### Ce qui a bien fonctionne
- **La specialisation des roles** : le binome lead technique / validation-qualite a ete efficace. Azelie assurait le developpement continu, je fournissais un regard exterieur critique qui a permis d'identifier des problemes (biais Reuters, hyperparametres sous-optimaux) qu'une personne seule aurait pu manquer
- **Le GridSearch a confirme et ameliore les choix d'hyperparametres** : le passage de C=1.0 a C=5.0 et min_df=3 a min_df=5 a ameliore le F1 de 0.3 points, prouvant l'interet d'une recherche systematique
- **Le debiaisage a rendu le modele plus robuste** : la neutralisation des marqueurs Reuters a elimine le data leakage, rendant les metriques fiables et le modele generaliste
- **L'annotation a double aveugle** : ma participation comme second annotateur sur les 500 posts a permis de mesurer un kappa inter-annotateurs (0.498), donnant une base quantitative pour evaluer la difficulte de la tache

### Forces identifiees
- Capacite d'analyse critique : identifier les biais et les faiblesses methodologiques
- Rigueur dans l'optimisation : approche systematique par GridSearch avec documentation des resultats
- Adaptabilite : integration dans un projet en cours sans perturber le flux de developpement
- Honnetete intellectuelle : reconnatre les limites de ma contribution plutot que de la surestimer

### Faiblesses identifiees
- **Implication tardive** : mon arrivee en cours de projet (V3) a reduit mon impact global. Une integration plus precoce aurait permis d'influencer les choix architecturaux
- **Volume de code produit limite** : ma contribution se concentre sur l'optimisation et la qualite plutot que sur le developpement de nouvelles fonctionnalites
- **Dependance a la lead technique** : pour les decisions d'architecture et de pipeline, j'aurais pu prendre plus d'initiative sur des modules autonomes

### Ce que j'aurais fait differemment
- M'impliquer des le debut du projet (V1) pour mieux comprendre l'evolution de l'architecture
- Proposer et implementer des tests automatises pour le debiaisage (test de regression automatique apres chaque modification du pipeline)
- Documenter mes experimentations dans des notebooks dedies (au lieu de modifications directes dans le code)
- Prendre en charge un module complet de bout en bout (par exemple le monitoring de derive ou une feature du dashboard)
- Creer une branche Git dediee a mes experimentations au lieu de travailler sur main

---

## 6. Competences developpees — avant/apres

| Domaine | Competence specifique | Niveau debut | Niveau fin | Preuve concrete |
|---------|----------------------|:------------:|:----------:|-----------------|
| ML | Optimisation hyperparametres (GridSearch) | Intermediaire | Avance | Campagne systematique C/min_df/ngram, resultats documentes |
| ML | Debiaisage de pipeline NLP | Debutant | Intermediaire | BODY_AGENCY_TERMS, ARTIFACT_YEAR_PATTERN dans clean_for_ml() |
| NLP | Comprehension TF-IDF + LogReg | Debutant | Intermediaire | Analyse de l'impact des ngrams, regularisation |
| Data | Annotation manuelle et accord inter-annotateurs | Debutant | Intermediaire | 500 posts annotes, comprehension du kappa Cohen/Gwet AC1 |
| DevOps | Git collaboratif | Debutant | Intermediaire | Resolution de conflits complexes, `--allow-unrelated-histories` |
| Soft skills | Revue de code et challenge technique | Intermediaire | Avance | Identification du biais Reuters, validation du seuil 0.44 |
| Soft skills | Travail en binome sur projet technique | Intermediaire | Avance | Communication continue, gestion de la complementarite des roles |

---

## 7. Axes d'amelioration personnels pour de futurs projets

1. **Contribuer au code des le jour 1** : meme une contribution modeste (documentation, tests) des le depart cree une dynamique d'implication
2. **Maitriser Git avant de commencer** : le merge catastrophique aurait ete evite avec une meilleure connaissance des branches et du rebase
3. **Documenter en temps reel** : chaque experimentation devrait etre tracee dans un notebook dedie, pas seulement dans le code
4. **Prendre en charge un module de A a Z** : pour developper une responsabilite complete sur un livrable
5. **Apprendre les tests automatises** : pytest, fixtures, mocking — c'est un investissement qui evite les regressions silencieuses

---

## 8. Bilan personnel

Ce projet m'a permis de comprendre la complexite d'un pipeline NLP complet, de la collecte au deploiement. Ma contribution sur l'optimisation des hyperparametres et le debiaisage m'a appris que les ameliorations les plus impactantes ne sont pas toujours les plus visibles : retirer un biais ou ajuster un hyperparametre peut avoir plus d'impact qu'ajouter une nouvelle fonctionnalite.

Le travail d'annotation (500 posts a double aveugle) m'a confronte a la difficulte reelle de la tache : distinguer un fait d'une opinion, un texte sensationnaliste d'un texte factuellement inexact, n'est pas trivial. Le kappa de 0.498 en temoigne — meme deux humains ne s'accordent que moderement sur cette classification. Cela m'a appris que la performance d'un modele ML est bornee par la qualite de la tache d'annotation, et que cette qualite doit etre quantifiee (kappa, AC1), pas supposee.

J'ai egalement pris conscience de l'importance de la collaboration continue dans un projet technique. Mon implication tardive a ete un frein que je ne repeterai pas. Pour mes futurs projets, je m'engagerai des le cadrage initial pour maximiser mon impact et eviter la frustration de "rattraper le train en marche".

L'apprentissage le plus marquant est la difference entre **optimiser un modele** et **comprendre pourquoi il echoue**. Le GridSearch optimise, mais c'est l'analyse des erreurs (FP, FN sur le gold set) et le debiaisage qui transforment reellement un pipeline.

---

*Rendu individuel - Sebastien Lazcanotegui - Mai 2026*
