# Analyse du modèle ExpertFakeNewsDetector V2 par longueur de texte

**Date** : 2026-04-03  
**Modèle** : ExpertFakeNewsDetector V2 (LogReg + TF-IDF 30k features + 12 linguistiques, sans émotions)
**Seuil de décision** : 0.44  
**Données évaluées** : 94603 textes provenant de 5 sources  

## 1. Sources de données

| Source | Nombre | Fiable (0) | Suspect (1) | Type |
|--------|--------|------------|-------------|------|
| ISOT | 44,124 | 21,416 | 22,708 | Articles longs EN |
| FakeNewsNet | 22,596 | 16,987 | 5,609 | Titres courts EN |
| CONSTRAINT | 8,559 | 4,479 | 4,080 | Tweets COVID EN |
| KaggleFR | 9,483 | 4,837 | 4,646 | Articles FR |
| CredibilityCorpus | 9,841 | 7,522 | 2,319 | Tweets FR+EN |
| **Total** | **94,603** | **55,241** | **39,362** | |

## 2. Performance par segment de longueur

| Segment | N | Accuracy | F1 | Precision | Recall | TP | FP | TN | FN |
|---------|---|----------|-----|-----------|--------|----|----|----|----|
| Ultra-court (<15 mots) | 24,533 | 0.8462 | 0.7472 | 0.8168 | 0.6885 | 5,574 | 1,250 | 15,187 | 2,522 |
| Court (15-30 mots) | 16,175 | 0.8915 | 0.8442 | 0.9153 | 0.7834 | 4,755 | 440 | 9,665 | 1,315 |
| Moyen (30-100 mots) | 13,228 | 0.9746 | 0.9415 | 0.9455 | 0.9376 | 2,704 | 156 | 10,188 | 180 |
| Long (100-300 mots) | 10,516 | 0.9835 | 0.9835 | 0.9960 | 0.9713 | 5,170 | 21 | 5,172 | 153 |
| Très long (>300 mots) | 30,151 | 0.9884 | 0.9897 | 0.9880 | 0.9915 | 16,844 | 204 | 12,958 | 145 |

### Scores de crédibilité moyens par segment

| Segment | Score moyen TP (vrais suspects) | Score moyen TN (vrais fiables) | Écart |
|---------|--------------------------------|-------------------------------|-------|
| Ultra-court (<15 mots) | 0.1548 | 0.8260 | 0.6712 |
| Court (15-30 mots) | 0.1276 | 0.8782 | 0.7507 |
| Moyen (30-100 mots) | 0.1114 | 0.9297 | 0.8182 |
| Long (100-300 mots) | 0.0761 | 0.9276 | 0.8516 |
| Très long (>300 mots) | 0.0395 | 0.9024 | 0.8628 |

> **Interprétation** : Le score de crédibilité est P(fiable). Un vrai suspect (TP) devrait avoir un score bas, un vrai fiable (TN) un score élevé. L'écart mesure la séparation entre les deux classes.

## 3. Performance par langue

| Langue | N | Accuracy | F1 | Precision | Recall | TP | FP | TN | FN |
|--------|---|----------|-----|-----------|--------|----|----|----|----|
| FR | 13,216 | 0.8955 | 0.8463 | 0.9927 | 0.7375 | 3,801 | 28 | 8,034 | 1,353 |
| EN | 79,994 | 0.9404 | 0.9283 | 0.9401 | 0.9169 | 30,867 | 1,967 | 44,361 | 2,799 |
| OTHER | 1,393 | 0.8284 | 0.7603 | 0.8330 | 0.6993 | 379 | 76 | 775 | 163 |

### Croisement longueur x langue

| Langue | Segment | N | F1 | Accuracy |
|--------|---------|----:|------:|---------:|
| FR | Ultra-court (<15 mots) | 2 322 | 0.6498 | 0.7145 |
| FR | Court (15-30 mots) | 4 203 | 0.7389 | 0.8344 |
| FR | Moyen (30-100 mots) | 4 760 | 0.9880 | 0.9958 |
| FR | Long (100-300 mots) | 218 | 1.0000 | 1.0000 |
| FR | Tres long (>300 mots) | 1 713 | 0.9992 | 0.9988 |
| EN | Ultra-court (<15 mots) | 20 956 | 0.7626 | 0.8624 |
| EN | Court (15-30 mots) | 11 853 | 0.8771 | 0.9118 |
| EN | Moyen (30-100 mots) | 8 451 | 0.9221 | 0.9626 |
| EN | Long (100-300 mots) | 10 297 | 0.9828 | 0.9831 |
| EN | Tres long (>300 mots) | 28 437 | 0.9890 | 0.9878 |

> **Point critique** : Le croisement FR + ultra-court affiche un F1 de 0.6498, le pire resultat de toute l'analyse. C'est precisement le cas d'usage Bluesky (posts courts en francais).

## 4. Analyse de calibration

**ECE (Expected Calibration Error) = 0.0492**

> L'ECE mesure l'écart moyen entre la confiance du modèle et la précision réelle. 
> Un ECE < 0.05 indique un modèle bien calibré. Un ECE > 0.10 suggère une sur-confiance ou sous-confiance.

| Bin P(suspect) | N | P(suspect) prédit | Taux réel suspect | Écart |
|----------------|---|-------------------|-------------------|-------|
| 0.0-0.1 | 32,429 | 0.0354 | 0.0101 | 0.0253 |
| 0.1-0.2 | 10,621 | 0.1449 | 0.0538 | 0.0911 |
| 0.2-0.3 | 6,044 | 0.2461 | 0.1191 | 0.1270 |
| 0.3-0.4 | 3,974 | 0.3471 | 0.2262 | 0.1209 |
| 0.4-0.5 | 2,918 | 0.4481 | 0.3578 | 0.0903 |
| 0.5-0.6 | 2,467 | 0.5492 | 0.5440 | 0.0052 |
| 0.6-0.7 | 2,618 | 0.6514 | 0.7086 | 0.0571 |
| 0.7-0.8 | 2,967 | 0.7524 | 0.8362 | 0.0838 |
| 0.8-0.9 | 4,900 | 0.8558 | 0.9373 | 0.0816 |
| 0.9-1.0 | 25,665 | 0.9736 | 0.9947 | 0.0211 |

## 5. Mots les plus discriminants (coefficients TF-IDF)

### Top 30 mots poussant vers SUSPECT (coef positif)

| Rang | Mot | Coefficient |
|------|-----|-------------|
| 1 | `citron` | +10.0172 |
| 2 | `coronavirus` | +9.2530 |
| 3 | `trump` | +7.6968 |
| 4 | `2017` | +6.6059 |
| 5 | `2017 0` | +6.1206 |
| 6 | `hollande` | +6.0468 |
| 7 | `image` | +5.9633 |
| 8 | `corona` | +5.8379 |
| 9 | `obama` | +5.6744 |
| 10 | `hidalgo` | +5.6566 |
| 11 | `swineflu` | +5.6353 |
| 12 | `machin` | +5.3370 |
| 13 | `corse machin` | +5.3259 |
| 14 | `south africa` | +5.1263 |
| 15 | `swine` | +5.0298 |
| 16 | `hillary` | +4.9111 |
| 17 | `atm` | +4.8533 |
| 18 | `swine flu` | +4.7781 |
| 19 | `flu` | +4.7700 |
| 20 | `lockdown` | +4.6963 |
| 21 | `virus` | +4.6444 |
| 22 | `covid 19` | +4.5942 |
| 23 | `septembre` | +4.3850 |
| 24 | `cancer` | +4.3826 |
| 25 | `fact` | +4.3230 |
| 26 | `le cancer` | +4.2697 |
| 27 | `exclusif` | +4.2410 |
| 28 | `janvier 2018` | +4.2374 |
| 29 | `africa` | +4.1622 |
| 30 | `read more` | +4.0435 |

### Top 30 mots poussant vers FIABLE (coef négatif)

| Rang | Mot | Coefficient |
|------|-----|-------------|
| 1 | `rt` | -13.2763 |
| 2 | `said on` | -6.9920 |
| 3 | `facebook twitter` | -6.7568 |
| 4 | `partage facebook` | -6.2593 |
| 5 | `partage facebook twitter` | -6.2593 |
| 6 | `reuters` | -5.7967 |
| 7 | `partage` | -5.7754 |
| 8 | `said` | -5.7056 |
| 9 | `u` | -4.9355 |
| 10 | `on tuesday` | -4.9224 |
| 11 | `on wednesday` | -4.8582 |
| 12 | `â` | -4.4788 |
| 13 | `on thursday` | -4.4429 |
| 14 | `uefa` | -4.3099 |
| 15 | `le` | -4.2271 |
| 16 | `libération` | -4.2146 |
| 17 | `on friday` | -4.2088 |
| 18 | `on monday` | -4.0033 |
| 19 | `said in` | -3.9993 |
| 20 | `mardi` | -3.9114 |
| 21 | `amp` | -3.9105 |
| 22 | `and` | -3.7984 |
| 23 | `u s` | -3.7665 |
| 24 | `indiafightscorona` | -3.6472 |
| 25 | `data` | -3.6304 |
| 26 | `coronavirusupdates` | -3.6156 |
| 27 | `said the` | -3.5891 |
| 28 | `u s president` | -3.5012 |
| 29 | `told reuters` | -3.4785 |
| 30 | `libé` | -3.4715 |

### Coefficients des features linguistiques

| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| `lexical_diversity` | -3.6434 | FIABLE |
| `numeric_density` | +1.7577 | SUSPECT |
| `sensationalism_score` | +0.3290 | SUSPECT |
| `has_url` | +0.3277 | SUSPECT |
| `sentence_count` | +0.2072 | SUSPECT |
| `avg_word_length` | +0.1794 | SUSPECT |
| `word_count` | -0.0004 | FIABLE |
| `avg_sentence_length` | -0.0004 | FIABLE |
| `caps_ratio` | +0.0000 | FIABLE |
| `exclamation_count` | +0.0000 | FIABLE |
| `question_count` | +0.0000 | FIABLE |
| `punct_density` | +0.0000 | FIABLE |

## 6. Mise à jour recommandée des listes de mots sensationnalistes

### Analyse critique des coefficients TF-IDF

**Constat important** : Les mots avec les plus forts coefficients SUSPECT ne sont pas des mots sensationnalistes au sens rhétorique. Ce sont principalement des **marqueurs thématiques** liés aux sujets sur-représentés dans les fake news du corpus d'entraînement :

- **Santé/COVID** : `coronavirus` (+9.25), `corona` (+5.84), `virus` (+4.64), `covid 19` (+4.59), `cancer` (+4.38), `flu` (+4.77)
- **Politique US** : `trump` (+7.70), `obama` (+5.67), `hillary` (+4.91)
- **Politique FR** : `hollande` (+6.05), `hidalgo` (+5.66), `machin` (+5.34)
- **Rumeurs spécifiques** : `citron` (+10.02, remède miracle), `swineflu` (+5.64)

Ces mots ne doivent **pas** etre ajoutés aux listes sensationnalistes car ils reflètent un biais thématique du corpus, pas un signal rhétorique de désinformation.

### Efficacité des listes actuelles

**`SENSATIONALIST_EN`** (22 termes) -- 17/22 présents dans le vocabulaire TF-IDF :

| Mot | Coefficient | Pertinence |
|-----|-------------|------------|
| `hoax` | +2.8695 | Fort signal SUSPECT |
| `breaking` | +2.6089 | Fort signal SUSPECT |
| `exclusive` | +1.2259 | Signal SUSPECT |
| `banned` | +1.1625 | Signal SUSPECT |
| `secret` | +0.9561 | Signal SUSPECT |
| `conspiracy` | +0.6930 | Signal SUSPECT |
| `bombshell` | +0.6364 | Signal SUSPECT |
| `mainstream media` | +0.5707 | Signal SUSPECT |
| `shocking` | +0.5023 | Signal SUSPECT |
| `deep state` | +0.3555 | Signal SUSPECT |
| `exposed` | +0.0471 | Faible signal |
| `alert` | -0.0156 | Non pertinent (neutre) |
| `unbelievable` | -0.0465 | Non pertinent |
| `wake up` | -0.2124 | Inversé (FIABLE) |
| `revealed` | -0.9083 | Inversé (FIABLE) |
| `scandal` | -1.1914 | Inversé (FIABLE) |
| `urgent` | -1.2988 | Inversé (FIABLE) |

Mots absents du vocabulaire TF-IDF : `big pharma`, `censored`, `cover-up`, `coverup`, `they dont want`

**`SENSATIONALIST_FR`** (52 termes) -- seulement 10/52 dans le vocabulaire TF-IDF :

| Mot | Coefficient | Pertinence |
|-----|-------------|------------|
| `exclusif` | +4.2410 | Tres fort signal SUSPECT |
| `révélation` | +2.6530 | Fort signal SUSPECT |
| `choc` | +2.3483 | Fort signal SUSPECT |
| `complot` | +2.2587 | Fort signal SUSPECT |
| `incroyable` | +2.0868 | Fort signal SUSPECT |
| `alerte` | +1.0243 | Signal SUSPECT |
| `corruption` | -0.1841 | Non pertinent |
| `collusion` | -0.7360 | Inversé |
| `urgent` | -1.2988 | Inversé |
| `scandale` | -1.5133 | Inversé |

42 termes FR absents du vocabulaire (expressions multi-mots, termes rares).

### Recommandations concrètes

**Mots à retirer ou reclasser** (coefficient négatif = poussent vers FIABLE) :
- EN : `scandal`, `revealed`, `urgent`, `wake up` -- utilisés aussi en journalisme sérieux
- FR : `scandale`, `urgent` -- meme constat

**Mots à ajouter (vrais signaux rhétoriques manquants)** :

Pour `SENSATIONALIST_EN` :
- `read more` (coef +4.04) -- clickbait typique
- `lockdown` (coef +4.70) -- contexte COVID alarmiste

Pour `SENSATIONALIST_FR` :
- `citron` (coef +10.02) -- associé aux remèdes miracles
- Expressions de type "remède miracle", "guérison naturelle"

**Restructuration recommandée** : Transformer les listes en catégories pondérées plutôt qu'un simple comptage binaire :
- **Clickbait** : `breaking`, `exclusive`, `read more`, `exclusif`
- **Complotisme** : `conspiracy`, `hoax`, `deep state`, `complot`
- **Alarmisme santé** : `virus`, `cancer`, `citron` (remède miracle)

### État actuel des listes

- `SENSATIONALIST_EN` : 22 termes (12 pertinents, 5 inversés, 5 absents du vocabulaire)
- `SENSATIONALIST_FR` : 52 termes (6 pertinents, 4 inversés/neutres, 42 absents du vocabulaire)

## 7. Recommandations pour la V3

### Constats principaux

1. **Ecart massif par longueur** : F1 passe de 0.7472 (ultra-court) a 0.9897 (tres long), soit un ecart de 24 points. Le modele perd significativement en performance sous 30 mots.
2. **FR sous-performe vs EN** : F1 FR = 0.8463 vs F1 EN = 0.9283 (-8 points). Le recall FR (0.7375) est le point faible : le modele manque 26% des fake news francaises.
3. **FR ultra-court est le pire cas** : F1 = 0.6498 avec seulement 71% d'accuracy. C'est le cas d'usage le plus frequent sur Bluesky (posts courts en francais).
4. **Calibration correcte** : ECE = 0.0492 (juste sous le seuil de 0.05). Les bins intermediaires (0.2-0.4) montrent une sur-confiance notable (ecart > 0.12).
5. **Biais thematique dans le TF-IDF** : Les plus forts coefficients sont des topics (COVID, politique) et non des signaux rheoriques, ce qui indique un apprentissage partiel du style vs le sujet.
6. **Features linguistiques sous-exploitees** : `lexical_diversity` (-3.64) est la plus forte feature linguistique ; `caps_ratio`, `exclamation_count`, `question_count` ont des coefficients quasi-nuls, suggerant que le nettoyage ML les a neutralisees.
7. **Mots fiables = biais residuel** : `rt` (-13.28), `reuters` (-5.80), `said on` (-6.99) montrent que le modele utilise encore des marqueurs de source journalistique comme signal de fiabilite.

### Pistes d'amelioration (priorite decroissante)

1. **Architecture a deux etages pour textes courts** : Implementer un systeme de routing par longueur (< 30 mots vs >= 30 mots) avec un modele specialise court-texte. Envisager des embeddings pre-entraines (sentence-transformers) pour les textes courts ou le TF-IDF manque de signal.

2. **Supprimer le biais residuel de source** : Les tokens `rt`, `reuters`, `said on`, `on tuesday/wednesday/...` ne sont pas des signaux de veracite. Les ajouter aux patterns de nettoyage dans `DatasetCleaner.AGENCY_PATTERNS` ou les exclure du vocabulaire TF-IDF via `stop_words`.

3. **Calibration post-hoc** : Appliquer une regression isotonique sur les bins 0.1-0.4 ou l'ecart predit/reel depasse 0.09. Alternative : `CalibratedClassifierCV(method='isotonic')` sur un jeu de calibration separe.

4. **Renforcer les features linguistiques** : Les features `caps_ratio`, `exclamation_count`, `question_count` sont neutralisees par le nettoyage `clean_for_ml()` qui convertit en minuscules et supprime la ponctuation. Calculer ces features **avant** le nettoyage (sur `text_original`) et non sur `text_clean`.

5. **Augmentation des donnees FR courtes** : Le corpus FR est domine par des articles longs (KaggleFR). Ajouter des sources de tweets/posts FR (factuel vs desinformation) pour equilibrer. Le CredibilityCorpus FR est trop petit (3 799 textes).

6. **Seuil adaptatif par longueur** : Utiliser un seuil plus conservateur (ex: 0.50) pour les textes < 30 mots ou la precision est plus elevee que le recall, et un seuil plus agressif (0.40) pour les textes longs.

7. **Restructurer les listes sensationnalistes** : Passer d'un comptage binaire a un score pondere par categorie (clickbait, complotisme, alarmisme sante). Retirer les mots a coefficient negatif (`scandal`, `urgent`, `scandale`).

8. **Activer les features emotionnelles** : Le modele V2 a ete entraine sans le module d'emotions (`use_emotions=False`). Tester l'impact des 7 features emotionnelles sur les textes courts ou les signaux TF-IDF sont faibles.
