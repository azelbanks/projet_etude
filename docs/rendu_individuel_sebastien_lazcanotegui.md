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
| Comprendre le pipeline existant | Code complexe avec 22 notebooks | Lecture methodique, discussions avec Azelie | L'importance de la documentation |
| GridSearch couteux en temps | Espace de recherche large, 145K textes | Reduction de l'espace, parallelisation | Equilibrer exhaustivite et pragmatisme |
| Merge Git catastrophique | Depot recree par erreur, conflits massifs | Resolution avec `--allow-unrelated-histories` | Toujours travailler sur une branche dediee |
| Debiaisage sans casser les performances | Retirer les signaux biaises sans degrader le F1 | Approche incrementale, tests de non-regression | Le debiaisage est un equilibre delicat |

---

## 5. Analyse critique et recul

### Ce qui a bien fonctionne
- La specialisation des roles (lead technique vs consolidation ML) a ete efficace
- Le GridSearch a confirme et ameliore les choix d'hyperparametres
- Le debiaisage a rendu le modele plus robuste et generaliste

### Ce que j'aurais fait differemment
- M'impliquer plus tot dans le projet pour avoir un impact plus large
- Proposer des tests automatises pour le debiaisage
- Documenter mes experimentations dans des notebooks dedies
- Contribuer davantage au code du collecteur et du dashboard

### Limites de ma contribution
- Nombre de commits limite (1 commit principal de consolidation)
- Implication tardive dans le cycle de developpement
- Contribution concentree sur l'optimisation plutot que le developpement

---

## 6. Bilan personnel

Ce projet m'a permis de comprendre la complexite d'un pipeline NLP complet, de la collecte au deploiement. Ma contribution sur l'optimisation des hyperparametres et le debiaisage m'a appris que les ameliorations les plus impactantes ne sont pas toujours les plus visibles : retirer un biais ou ajuster un hyperparametre peut avoir plus d'impact qu'ajouter une nouvelle fonctionnalite.

J'ai egalement pris conscience de l'importance de la collaboration continue dans un projet technique, et de la necessite de s'impliquer des le debut pour maximiser son apport.

---

*Rendu individuel - Sebastien Lazcanotegui - Avril 2026*
