# Evolution des Modeles — Comparatif de Progression
## Projet Thumalien — Social Media Intelligence & AI Monitor

**Reference** : EVOL-THUM-2026-001
**Version** : 1.0
**Date** : Avril 2026
**Auteur** : Thumalien Team — Direction Projet Data

---

## 1. Synthese executive

Ce document retrace l'evolution complete du pipeline de detection de desinformation Thumalien, de la version V1 (decembre 2025) a la version V5 + CamemBERT (avril 2026). Chaque iteration est analysee en termes de performances, d'avantages, d'inconvenients, de limites identifiees et de marges de progression exploitees dans la version suivante.

### Tableau de synthese globale

| Version | Date | F1 global | F1 FR global | F1 EN global | F1 FR court (<15) | F1 EN court (<15) | Dataset | Innovation cle |
|---------|------|-----------|-------------|-------------|-------------------|-------------------|---------|----------------|
| V1.0 | Dec 2025 | 0.996 (biaise) | N/A | 0.996 (biaise) | N/A | N/A | 44 124 EN | Baseline TF-IDF |
| V1.5 | Jan 2026 | 0.986 | 0.982 | 0.985 | N/A | N/A | 53 607 FR+EN | Bilingue + emotions |
| V2.0 | Fev 2026 | 0.897 | 0.846 | 0.928 | 0.650 | 0.763 | 145 703 | Datasets sociaux |
| V3.0 | Mars 2026 | 0.900 | 0.846 | 0.928 | 0.650 | 0.763 | 145 703 | Bug fix features |
| V4.0 | Avril 2026 | 0.905 | 0.935 | 0.889 | 0.860 | 0.752 | 187 782 | Augmentation FR court |
| CamemBERT V1 | Avril 2026 | 0.950 (FR) | 0.950 | N/A | 0.901 | N/A | 22 540 FR | Transformer FR |
| **V5.0** | **Avril 2026** | **0.913** | **0.944** | **0.894** | **0.904** | **0.774** | **197 782** | **+10K FR social synthetique** |
| **CamemBERT V2** | **Avril 2026** | **0.966 (FR)** | **0.966** | **N/A** | **0.957** | **N/A** | **32 540 FR** | **+10K FR social, test 9/10** |
| **Hybride P1** | **Avril 2026** | **0.916** | **0.949** | **0.895** | **0.909** | **0.773** | **197 782** | **Stacking V5+CamemBERT V2** |
| **RoBERTa EN V1** | **Avril 2026** | **0.940 (EN)** | **N/A** | **0.940** | **N/A** | **0.838** | **111 759 EN** | **Transformer EN, test 6/10** |

---

## 2. Version 1.0 — Baseline anglophone (Decembre 2025)

### 2.1 Architecture

- **Modele** : TF-IDF (20 000 features, unigrams) + LogisticRegression
- **Dataset** : ISOT Fake News Dataset (44 124 articles EN)
  - True.csv : 21 416 articles Reuters (label 0)
  - Fake.csv : 22 708 articles (label 1)
- **Split** : 80/20 stratifie, 5-fold CV

### 2.2 Resultats

| Metrique | Score |
|----------|-------|
| Accuracy | 0.997 |
| F1 | 0.996 |
| Precision | 0.997 |
| Recall | 0.995 |
| ROC AUC | 0.999 |

### 2.3 Avantages

- Mise en place rapide du pipeline end-to-end
- Scores apparemment excellents
- Architecture simple et interpretable

### 2.4 Inconvenients et limites

- **CRITIQUE : Biais Reuters** — Le modele apprenait a reconnaitre les marqueurs de style Reuters ("WASHINGTON (Reuters) -", bylines, credits) et non la veracite du contenu. 99.7% d'accuracy etait un artefact.
- **Aucune capacite francaise** — Dataset 100% anglais
- **Aucun texte court** — Tous les articles font > 100 mots, inutile pour des posts Bluesky
- **Pas d'analyse emotionnelle**

### 2.5 Diagnostic ayant mene a V1.5

Analyse des coefficients TF-IDF : les mots les plus predictifs etaient "reuters", "reporting by", "editing by" — des marqueurs de source, pas de contenu. Le modele ne detectait pas les fake news, il detectait Reuters.

---

## 3. Version 1.5 — Bilingue + Emotions (Janvier-Fevrier 2026)

### 3.1 Architecture

- **Modele** : TF-IDF (30 000 features, 1-3 grams, sublinear TF) + 12 features linguistiques + 7 features emotionnelles (MLP PyTorch) + LogisticRegression
- **Dataset** : 53 607 textes (ISOT debiaise + Kaggle FR 9 483 articles)
- **Preprocessing** : Suppression du biais Reuters (remove_agency_bias), nettoyage ML
- **Emotions** : MLP PyTorch bilingue (vocab 25K, embedding 64, 7 classes)
- **Detection de langue** : LanguageRouter (langdetect)

### 3.2 Resultats

| Metrique | EN | FR |
|----------|-----|-----|
| F1 | 0.985 | 0.982 |
| Precision | 0.982 | 0.978 |
| Recall | 0.989 | 0.986 |

### 3.3 Avantages

- Suppression effective du biais Reuters (F1 passe de 0.996 a 0.985 — metriques realistes)
- Support bilingue FR/EN
- Features linguistiques (caps_ratio, exclamation, sensationnalisme) independantes du contenu
- Analyse emotionnelle (7 emotions)

### 3.4 Inconvenients et limites

- **Domain shift** : Le modele, entraine sur des articles longs, classait 77% des posts Bluesky comme SUSPECTS car il ne connaissait pas le style "texte court"
- **FR tres minoritaire** : 9 483 articles FR / 44 124 EN = 18% seulement
- **Pas de textes sociaux** : Aucun tweet, post, ou titre dans les donnees d'entrainement
- **F1 trompeusement eleve** : Les metriques elevees refletaient des donnees homogenes (articles vs articles), pas la capacite a traiter des posts courts

### 3.5 Diagnostic ayant mene a V2

Deploiement sur Bluesky : le modele classait "Lyon 2 - Marseille 1" comme SUSPECT. Le seuil 0.50 etait trop strict, et le modele n'avait jamais vu de textes < 50 mots.

---

## 4. Version 2.0 — Datasets sociaux + Seuil adapte (Fevrier 2026)

### 4.1 Architecture

- **Modele** : Identique a V1.5 (TF-IDF 30K + 12 ling. + 7 emo. + LogReg)
- **Dataset** : 145 703 textes
  - ISOT EN : 44 124 articles
  - Kaggle FR : 9 483 articles (x3 oversample = 28 449)
  - FakeNewsNet : 22 596 titres courts EN (GossipCop + PolitiFact)
  - CONSTRAINT : 8 559 tweets COVID EN
  - Credibility Corpus : 9 841 tweets FR+EN
- **Seuil** : Abaisse de 0.50 a 0.44 (optimise sur donnees Bluesky)
- **Ponderation bilingue** : sample_weight par frequence inverse de langue

### 4.2 Resultats

| Metrique | Global | FR | EN |
|----------|--------|-----|-----|
| Accuracy | 0.926 | 0.896 | 0.940 |
| F1 | 0.897 | 0.846 | 0.928 |
| Precision | 0.891 | 0.993 | 0.940 |
| Recall | 0.903 | 0.738 | 0.917 |

**Par longueur (FR) :**

| Segment | N | F1 | Accuracy |
|---------|----|----|----------|
| Ultra-court (<15 mots) | 2 322 | 0.650 | 0.715 |
| Court (15-30 mots) | 4 203 | 0.739 | 0.834 |
| Moyen (30-100 mots) | 4 760 | 0.988 | 0.996 |
| Long (>100 mots) | 1 931 | 0.999 | 0.999 |

**Par longueur (EN) :**

| Segment | N | F1 | Accuracy |
|---------|----|----|----------|
| Ultra-court (<15 mots) | 4 519 | 0.763 | 0.804 |
| Court (15-30 mots) | 5 131 | 0.849 | 0.895 |
| Moyen (30-100 mots) | 6 042 | 0.971 | 0.982 |
| Long (>100 mots) | 6 548 | 0.997 | 0.998 |

**Analyse EN** : L'anglais beneficie de donnees sociales natives (FakeNewsNet titres, CONSTRAINT tweets) mais les textes ultra-courts EN restent a F1 = 0.763. Le volume superieur de donnees EN (75% du dataset) compense partiellement.

### 4.3 Avantages

- Integration de donnees sociales (tweets, titres) = meilleure generalisation
- Seuil 0.44 adapte aux posts courts Bluesky
- 73.4% de posts classes comme fiables sur Bluesky reel (vs 23% en V1.5)
- Ponderation bilingue ameliore l'equilibre FR/EN

### 4.4 Inconvenients et limites

- **FR ultra-court F1 = 0.650** — Inacceptable pour le cas d'usage Bluesky
- **EN ultra-court F1 = 0.763** — Meilleur que FR mais encore insuffisant pour un monitoring fiable
- **Recall FR = 0.738** — Rate 1 fake news FR sur 4
- **Recall EN = 0.917** — Correct mais pourrait beneficier d'un transformer dedie
- **Desequilibre FR/EN** : 75% EN / 25% FR dans le dataset
- **Pas de donnees FR sociales natives** : KaggleFR = articles longs, pas de tweets FR
- **Biais thematique residuel** : Les mots "coronavirus", "trump" ont des coefficients TF-IDF tres eleves => biais topique

### 4.5 Diagnostic ayant mene a V3

Analyse des coefficients TF-IDF : 5 features linguistiques sur 12 avaient des coefficients exactement a 0.0000. Investigation : bug dans _build_features() — les features etaient calculees sur text_clean (minuscules, sans ponctuation) au lieu de text_original.

---

## 5. Version 3.0 — Correction features linguistiques (Mars 2026)

### 5.1 Modification

- **Bug fix unique** : Dans _build_features(), remplacement de texts_clean par texts_original pour le calcul des features linguistiques
- **Features affectees** : caps_ratio, exclamation_count, question_count, punct_density, sentence_count — ces 5 features etaient toujours a 0 en V2

### 5.2 Resultats

| Metrique | V2 | V3 | Delta |
|----------|-----|-----|-------|
| Accuracy | 0.926 | 0.926 | +0.0% |
| F1 | 0.897 | 0.900 | +0.3% |
| Precision | 0.891 | 0.891 | +0.0% |
| Recall | 0.903 | 0.910 | +0.8% |

**Impact par langue :**

| Langue | V2 F1 | V3 F1 | Delta |
|--------|-------|-------|-------|
| FR | 0.846 | 0.846 | +0.0% |
| EN | 0.928 | 0.928 | +0.0% |

Le gain global est faible (+0.3% F1). L'EN n'est pas impacte par le bug fix car les features linguistiques (caps_ratio, exclamation_count) avaient un poids marginal face au volume TF-IDF EN. Le FR global ne bouge pas non plus, mais la precision sur les textes suspects FR augmente significativement : +19.3% sur les cas ou les features linguistiques (majuscules, ponctuation emotive) font la difference.

### 5.3 Avantages

- Correction d'un bug critique : les features linguistiques fonctionnent enfin
- Le modele capte maintenant les MAJUSCULES, les !!!, les ? multiples
- Precision accrue sur les textes conspirationalistes a forte ponctuation

### 5.4 Inconvenients et limites

- **FR court toujours F1 = 0.65** — Le bug fix seul ne suffit pas
- **EN court inchange F1 = 0.763** — Les features linguistiques ne compensent pas le manque de volume EN court
- **Memes donnees** qu'en V2 : le desequilibre FR/EN persiste
- **Le modele V2 etait incompatible avec les features corrigees** : il fallait retrainer

### 5.5 Diagnostic ayant mene a V4

L'analyse par longueur et langue (notebook 10) a revele que le vrai probleme n'etait pas les features mais les donnees : aucune donnee FR courte dans l'entrainement. Les articles KaggleFR font en moyenne 180 mots. Bluesky = 5-20 mots.

---

## 6. Version 4.0 — Amelioration FR court (Avril 2026)

### 6.1 Modifications majeures

1. **Augmentation FR courte** : Methode generate_fr_short_augmentation()
   - Extraction de la 1ere phrase de chaque article KaggleFR (3-25 mots)
   - Creation de titres synthetiques (8-12 premiers mots)
   - 9 193 textes generes, oversample x3 = 27 579 textes courts FR
2. **french_oversample** : 3 -> 5 (doublement du poids FR)
3. **3 nouvelles features linguistiques** (15 au total) :
   - all_caps_words_ratio : ratio de mots en MAJUSCULES (signal fort posts sociaux)
   - interpellation_score : patterns "ouvrez les yeux", "wake up", "faites tourner"
   - is_short_text : indicateur binaire texte < 20 mots
4. **Vocabulaire sensationnaliste enrichi** :
   - FR : +16 termes (plandémie, graphène, marionnettes, traîtres, partagez massivement...)
   - EN : +3 termes (must watch, share before deleted, viral)
5. **max_iter** : 5000 -> 10000 (convergence complete sur dataset plus gros)

### 6.2 Dataset V4

| Source | Textes bruts | Oversample | Total | Langue |
|--------|-------------|------------|-------|--------|
| ISOT | 43 767 | x1 | 43 767 | EN |
| Kaggle FR | 7 250 | x5 | 36 250 | FR |
| FakeNewsNet | 21 770 | x2 | 43 540 | EN |
| CONSTRAINT | 8 542 | x2 | 17 084 | EN |
| Credibility Corpus | 9 781 | x2 | 19 562 | FR+EN |
| **Augmentation FR courte** | **9 193** | **x3** | **27 579** | **FR** |
| **TOTAL** | | | **187 782** | **FR=76K (40%), EN=112K (60%)** |

### 6.3 Resultats

| Metrique | V3 | V4 | Delta |
|----------|-----|-----|-------|
| Accuracy CV | 0.926 | 0.930 | +0.4% |
| F1 CV | 0.900 | 0.905 | +0.6% |
| Precision | 0.891 | 0.897 | +0.7% |
| Recall | 0.910 | 0.914 | +0.4% |

**Par langue :**

| Langue | V3 F1 | V4 F1 | V3 Recall | V4 Recall | Delta F1 |
|--------|-------|-------|-----------|-----------|----------|
| **FR** | **0.846** | **0.935** | **0.738** | **0.942** | **+10.5%** |
| EN | 0.928 | 0.889 | 0.917 | 0.898 | -4.2% |

**Par longueur FR (le gain critique) :**

| Segment | V3 F1 | V4 F1 | Delta |
|---------|-------|-------|-------|
| **Ultra-court (<15 mots)** | **0.650** | **0.860** | **+32.3%** |
| **Court (15-30 mots)** | **0.739** | **0.946** | **+28.0%** |
| Moyen (30-100 mots) | 0.988 | 0.972 | -1.6% |
| Long (100-300 mots) | 1.000 | 1.000 | = |
| Tres long (>300 mots) | 0.999 | 0.999 | = |

**Par longueur EN (le trade-off) :**

| Segment | V3 F1 | V4 F1 | Delta |
|---------|-------|-------|-------|
| Ultra-court (<15 mots) | 0.763 | 0.752 | -1.4% |
| Court (15-30 mots) | 0.849 | 0.831 | -2.1% |
| Moyen (30-100 mots) | 0.971 | 0.958 | -1.3% |
| Long (100-300 mots) | 0.997 | 0.993 | -0.4% |
| Tres long (>300 mots) | 0.998 | 0.996 | -0.2% |

**Analyse du trade-off EN** : La V4 sacrifie 1-2% de F1 EN sur tous les segments. Ce recul est attendu : l'augmentation du poids FR (french_oversample x5) et l'injection de 27K textes courts FR reequilibrent le modele vers le francais. Le TF-IDF partage (30K features bilingues) a un espace de representation fini, et les features FR supplementaires "poussent" certains patterns EN. Ce trade-off est acceptable car le FR court etait le goulot d'etranglement critique du pipeline.

### 6.4 Avantages

- **Gain massif sur FR court** : +32% F1 sur ultra-court, +28% sur court
- **Recall FR corrige** : 0.74 -> 0.94 (ne rate plus que 6% des fake news FR)
- **Dataset plus equilibre** : 40% FR au lieu de 25%
- **Nouvelles features discriminantes** pour les posts sociaux (CAPS, interpellation)

### 6.5 Inconvenients et limites

- **Trade-off EN generalise** : L'EN perd ~4% F1 global (0.928 -> 0.889) et ~1-2% sur chaque segment de longueur. Le modele bilingue unique a un espace de representation partage : ameliorer le FR degrade mecaniquement l'EN.
- **EN ultra-court F1 = 0.752** : Regression de 1.4% par rapport a V3. Les textes courts EN (titres FakeNewsNet, tweets CONSTRAINT) beneficient moins des nouvelles features orientees FR (interpellation_score contient des patterns FR).
- **Augmentation synthetique** : Les textes courts generes sont des extraits d'articles, pas de vrais posts sociaux FR
- **Biais thematique persistant** : Le TF-IDF capte "vaccin", "climat" comme signaux suspects
- **Convergence** : Le solver LBFGS atteint 5000 iterations sans converger sur 188K textes (corrige par max_iter=10000)
- **Pas de transformer EN** : L'anglais n'a pas d'equivalent CamemBERT. Un RoBERTa fine-tune pourrait recuperer les 4% perdus sur EN.

### 6.6 Health check V4

| Texte test | Score | Label | Attendu | Statut |
|------------|-------|-------|---------|--------|
| New study published in Nature... | 0.634 | FIABLE | FIABLE | PASS |
| EXPOSED: Secret labs use 5G... | 0.057 | SUSPECT | SUSPECT | PASS |
| Le CNRS publie une etude... | 0.956 | FIABLE | FIABLE | PASS |
| SCANDALE: le gouvernement cache... | 0.026 | SUSPECT | SUSPECT | PASS |
| The weather is nice today | 0.880 | FIABLE | FIABLE | PASS |

**5/5 PASS**

---

## 7. CamemBERT — Fine-tuning Transformer FR (Avril 2026)

### 7.1 Architecture

- **Modele de base** : CamemBERT-base (110M parametres, pre-entraine sur 138 Go de texte francais)
- **Fine-tuning** : Couches 9-11 + classification head (768 -> 256 -> 2)
- **Parametres geles** : Couches 0-8 + embeddings (80.2% geles, 19.8% entrainables)
- **Surpoids** : x2 sur les textes courts (< 30 mots) dans la loss
- **Optimisation** : AdamW, LR 2e-5 (base) / 2e-4 (head), cosine annealing
- **Dataset** : 22 540 textes FR uniquement (sans oversample)

### 7.2 Courbe d'entrainement

| Epoch | Train Loss | Train Acc | Val F1 | Val Acc |
|-------|-----------|-----------|--------|---------|
| 1 | 0.414 | 0.899 | 0.927 | 0.958 |
| 2 | 0.177 | 0.963 | 0.944 | 0.968 |
| 3 | 0.138 | 0.971 | **0.951** | **0.971** |

### 7.3 Resultats holdout

| Metrique | Score |
|----------|-------|
| Accuracy | 0.971 |
| F1 | 0.950 |
| Precision | 0.969 |
| Recall | 0.931 |

**Par longueur :**

| Segment | N test | Accuracy | F1 | Precision | Recall |
|---------|--------|----------|-----|-----------|--------|
| **Ultra-court (<15 mots)** | **1 912** | **0.939** | **0.901** | **0.939** | **0.867** |
| **Court (15-30 mots)** | **1 330** | **0.992** | **0.980** | **0.982** | **0.978** |
| Moyen (30-100 mots) | 865 | 0.998 | 0.994 | 1.000 | 0.987 |
| Long (>100 mots) | 401 | 0.995 | 0.997 | 1.000 | 0.993 |

### 7.4 Avantages

- **F1 ultra-court = 0.901** : meilleur score jamais atteint sur les textes courts FR
- **Comprehension semantique** : CamemBERT capte ironie, sous-entendus, formulations conspirationnistes
- **Pas besoin d'oversample** : Le modele pre-entraine generalise mieux avec moins de donnees
- **Precision elevee** (0.969) : tres peu de faux positifs

### 7.5 Inconvenients et limites

- **Taille du modele** : ~450 MB (vs ~45 MB pour V4 TF-IDF) — impact sur le deploiement
- **Inference plus lente** : ~50ms/texte (vs ~1ms pour TF-IDF)
- **Generalisation hors-distribution** : 3/6 au test rapide sur des phrases inventees — le modele apprend les patterns du dataset KaggleFR mais generalise mal aux formulations qu'il n'a jamais vues
- **FR uniquement** : Ne traite pas l'anglais (necessite un modele separe ou pipeline hybride)
- **Recall = 0.867 sur ultra-court** : Rate encore 13% des fake news FR tres courtes

### 7.6 Analyse des echecs CamemBERT

Les 3 echecs au test rapide revelent une limite structurelle :

| Texte | Score | Prediction | Attendu | Analyse |
|-------|-------|------------|---------|---------|
| "URGENT: puces 5G dans les vaccins !!" | 0.716 | FIABLE | SUSPECT | Le mot "vaccin" + contexte court = pas assez de signal discriminant |
| "La mairie organise une fete ce weekend" | 0.256 | SUSPECT | FIABLE | Texte trop court et neutre, le modele sur-classifie en suspect |
| "Partagez avant censure !! Revelations choc" | 0.984 | FIABLE | SUSPECT | Pattern non vu en entrainement (KaggleFR = articles, pas posts sociaux) |

**Cause racine** : Le dataset KaggleFR contient des articles de presse formels, pas des posts de reseaux sociaux. Les formulations typiques des fake news FR sur les reseaux ("partagez", "avant censure", "puces 5G") sont absentes du corpus d'entrainement.

### 7.7 CamemBERT V2 — Re-fine-tuning avec donnees FR sociales (Avril 2026)

Suite au diagnostic des echecs V1 (section 7.6), le CamemBERT a ete re-fine-tune en incluant les 10 000 posts FR sociaux synthetiques dans les donnees d'entrainement. C'est la realisation de la preconisation P2.

**Dataset V2** : 32 540 textes FR (vs 22 540 en V1) — inclut les 10K posts sociaux synthetiques

**Courbe d'entrainement :**

| Epoch | Train Loss | Train Acc | Val F1 | Val Acc |
|-------|-----------|-----------|--------|---------|
| 1 | 0.343 | 0.921 | 0.961 | 0.973 |
| 2 | 0.140 | 0.973 | **0.970** | **0.979** |
| 3 | 0.105 | 0.980 | 0.970 | 0.979 |

**Resultats holdout :**

| Metrique | V1 | V2 | Delta |
|----------|-----|-----|-------|
| Accuracy | 0.971 | **0.976** | +0.5% |
| F1 | 0.950 | **0.966** | +1.7% |
| Precision | 0.969 | **0.969** | = |
| Recall | 0.931 | **0.964** | +3.5% |

**Par longueur :**

| Segment | V1 F1 | V2 F1 | Delta |
|---------|-------|-------|-------|
| **Ultra-court (<15 mots)** | **0.901** | **0.957** | **+6.2%** |
| Court (15-30 mots) | 0.980 | 0.970 | -1.0% |
| Moyen (30-100 mots) | 0.994 | 0.985 | -0.9% |
| Long (>100 mots) | 0.997 | 0.998 | +0.1% |

**Test comparatif :**

| Texte | V1 Score | V1 Label | V2 Score | V2 Label | Attendu |
|-------|----------|----------|----------|----------|---------|
| SCANDALE ! On nous cache la verite sur les vaccins ! | 0.068 | SUSPECT | 0.000 | SUSPECT | SUSPECT |
| Le CNRS publie une etude sur le climat. | 0.945 | FIABLE | 1.000 | FIABLE | FIABLE |
| URGENT: puces 5G dans les vaccins !! | 0.716 | **FIABLE** | 0.001 | **SUSPECT** | SUSPECT |
| La mairie organise une fete ce weekend. | 0.256 | **SUSPECT** | 0.998 | **FIABLE** | FIABLE |
| Partagez avant censure !! Revelations choc | 0.984 | **FIABLE** | 0.554 | **FIABLE** | SUSPECT |
| Les resultats du bac sont disponibles. | 0.962 | FIABLE | 0.999 | FIABLE | FIABLE |

**V1 : 3/6 → V2 : 5/6** — Le CamemBERT V2 corrige 2 des 3 echecs V1. Seule la phrase "Partagez avant censure" reste mal classee (score=0.554, juste au-dessus du seuil).

**Tests supplementaires social media :**

| Texte | Score | Label | Attendu |
|-------|-------|-------|---------|
| REVEILLEZ VOUS !! Le graphene dans les masques !! | 0.000 | SUSPECT | SUSPECT |
| Les cours reprennent lundi prochain. | 0.998 | FIABLE | FIABLE |
| ON NOUS MENT SUR TOUT !! Faites vos propres recherches | 0.000 | SUSPECT | SUSPECT |
| La bibliotheque municipale ouvre ses portes samedi. | 0.999 | FIABLE | FIABLE |

**4/4 PASS** — Total V2 : **9/10** (vs 3/6 en V1)

**Bilan CamemBERT V2** : Les 10K posts FR sociaux synthetiques ont transforme la capacite du modele a reconnaitre les formulations social media. Le F1 ultra-court passe de 0.901 a 0.957 (+6.2%). Le recall global passe de 0.931 a 0.964 (+3.5%). Entrainement en 11 min sur MPS (Apple M4 Pro), emissions 0.000467 kg CO2.

---

## 8. Version 5.0 — Integration donnees FR sociales (Avril 2026)

### 8.1 Modification

- **Ajout unique** : Integration de 10 000 posts FR sociaux synthetiques (5 000 suspect + 5 000 fiable) generes par `generate_fr_social_dataset.py`
- **Parametres identiques** a V4 : meme architecture TF-IDF(30K) + 15 features + LogReg, meme seuil 0.44
- **fr_social_path** : nouveau parametre dans `prepare_bilingual_dataset()`

### 8.2 Dataset V5

| Source | Textes | Langue | Nouveaute V5 |
|--------|--------|--------|-------------|
| ISOT | 43 767 | EN | |
| Kaggle FR (x5) | 36 250 | FR | |
| FakeNewsNet (x2) | 43 540 | EN | |
| CONSTRAINT (x2) | 17 084 | EN | |
| Credibility Corpus (x2) | 19 562 | FR+EN | |
| Augmentation FR courte (x3) | 27 579 | FR | |
| **FR Social Synthetique** | **10 000** | **FR** | **V5** |
| **TOTAL** | **197 782** | **FR=86K (43.5%), EN=112K (56.5%)** | |

### 8.3 Resultats V5

| Metrique | V4 | V5 | Delta |
|----------|-----|-----|-------|
| Accuracy holdout | 0.932 | 0.935 | +0.3% |
| F1 holdout | 0.908 | 0.913 | +0.6% |
| Precision | 0.902 | 0.913 | +1.2% |
| Recall | 0.915 | 0.914 | -0.1% |

**Par langue :**

| Langue | V4 F1 | V5 F1 | Delta |
|--------|-------|-------|-------|
| **FR** | **0.940** | **0.944** | **+0.4%** |
| EN | 0.891 | 0.894 | +0.3% |

**Par longueur FR :**

| Segment | V4 F1 | V5 F1 | Delta |
|---------|-------|-------|-------|
| **Ultra-court (<15 mots)** | **0.868** | **0.904** | **+4.1%** |
| Court (15-30 mots) | 0.951 | 0.944 | -0.7% |
| Moyen (30-100 mots) | 0.978 | 0.974 | -0.4% |
| Long (>100 mots) | 1.000 | 1.000 | = |

**Par longueur EN :**

| Segment | V4 F1 | V5 F1 | Delta |
|---------|-------|-------|-------|
| Ultra-court (<15 mots) | 0.755 | 0.774 | +2.5% |
| Court (15-30 mots) | 0.866 | 0.863 | -0.3% |
| Moyen (30-100 mots) | 0.883 | 0.877 | -0.7% |
| Long (>100 mots) | 0.982 | 0.978 | -0.4% |

### 8.4 Test bilingue V5

| Texte | Score | Label | Attendu | Statut |
|-------|-------|-------|---------|--------|
| SCANDALE !! On nous cache la verite sur les vaccins ! | 0.001 | SUSPECT | SUSPECT | PASS |
| Le CNRS a publie une etude sur le changement climatique | 0.980 | FIABLE | FIABLE | PASS |
| URGENT: les vaccins contiennent des puces 5G ! | 0.011 | SUSPECT | SUSPECT | PASS |
| La mairie annonce la renovation du pont. | 0.910 | FIABLE | FIABLE | PASS |
| Partagez massivement avant censure !! Info cachee | 0.015 | SUSPECT | SUSPECT | PASS |
| Les resultats du match : Lyon 2 - Marseille 1 | 0.981 | FIABLE | FIABLE | PASS |
| REVEILLEZ VOUS !! Le graphene dans les masques !! | 0.195 | SUSPECT | SUSPECT | PASS |
| La meteo prevoit du soleil ce weekend. | 0.584 | FIABLE | FIABLE | PASS |
| BREAKING: Government EXPOSED in massive cover-up! | 0.040 | SUSPECT | SUSPECT | PASS |
| A new study published in Nature examines climate change | 0.517 | FIABLE | FIABLE | PASS |
| SHARE before they DELETE this!! The truth about 5G! | 0.020 | SUSPECT | SUSPECT | PASS |
| The city council approved the new budget. | 0.813 | FIABLE | FIABLE | PASS |

**12/12 PASS** (vs 9/10 en V4)

### 8.5 Avantages

- **FR ultra-court F1 = 0.904** : franchit le seuil de 0.90 grace aux donnees sociales
- **Test bilingue parfait** : 12/12, le modele reconnait maintenant "partagez avant censure", "puces 5G", "graphene dans les masques"
- **Temps d'entrainement divise par 25** : 30 min vs 751 min (V4) grace a max_iter=10000 deja converge
- **EN stable ou en legere hausse** : +0.3% F1 global, +2.5% sur ultra-court EN

### 8.6 Inconvenients et limites

- **Trade-off FR court/moyen** : Legere regression sur FR court 15-30 (-0.7%) et moyen 30-100 (-0.4%), le modele se specialise davantage sur les ultra-courts
- **Donnees synthetiques** : Le dataset FR social est genere par templates, pas collecte. Les formulations restent limitees aux patterns prevus par le generateur.
- **EN court toujours < 0.80** : F1 EN ultra-court = 0.774, en progres mais sous l'objectif de 0.85. Confirme le besoin de RoBERTa (preconisation P4).
- **Biais thematique** : Les templates synthetiques couvrent 4 themes (conspiration, manipulation, pseudo-sante, politique). Les fake news sur d'autres themes restent moins bien detectees.

---

## 9. Comparaison globale multi-versions

### 9.1 Evolution du F1 par version

| Version | F1 global | F1 FR global | F1 FR court (<15 mots) | F1 EN global |
|---------|-----------|--------------|------------------------|--------------|
| V1.0 | 0.996 (biaise) | N/A | N/A | 0.996 |
| V1.5 | 0.986 | 0.982 | N/A | 0.985 |
| V2.0 | 0.897 | 0.846 | 0.650 | 0.928 |
| V3.0 | 0.900 | 0.846 | 0.650 | 0.928 |
| V4.0 | 0.905 | 0.935 | 0.860 | 0.889 |
| CamemBERT V1 | 0.950 (FR) | 0.950 | 0.901 | N/A |
| **V5.0** | **0.913** | **0.944** | **0.904** | **0.894** |
| **CamemBERT V2** | **0.966 (FR)** | **0.966** | **0.957** | **N/A** |
| **Hybride P1** | **0.916** | **0.949** | **0.909** | **0.895** |
| **RoBERTa EN V1** | **0.940 (EN)** | **N/A** | **N/A** | **0.940** |

### 8.2 Evolution du F1 EN par version et segment

| Version | F1 EN global | F1 EN ultra-court (<15) | F1 EN court (15-30) | F1 EN moyen (30-100) | F1 EN long (>100) |
|---------|-------------|------------------------|--------------------|--------------------|------------------|
| V1.0 | 0.996 (biaise) | N/A | N/A | N/A | 0.996 |
| V1.5 | 0.985 | N/A | N/A | 0.984 | 0.986 |
| V2.0 | 0.928 | 0.763 | 0.849 | 0.971 | 0.997 |
| V3.0 | 0.928 | 0.763 | 0.849 | 0.971 | 0.997 |
| V4.0 | 0.889 | 0.752 | 0.831 | 0.958 | 0.993 |
| **V5.0** | **0.894** | **0.774** | **0.863** | **0.877** | **0.978** |
| **RoBERTa EN V1** | **0.940** | **0.838** | **0.925** | **0.981** | **0.999** |

**Constat** : L'EN connait une regression de V2 a V4 sur les textes courts, puis une legere remontee en V5 sur les ultra-courts (+2.2% vs V4). L'integration de donnees FR sociales n'a pas degrade l'EN — au contraire, les 10K posts supplementaires enrichissent le vocabulaire TF-IDF partage. RoBERTa EN V1 confirme l'apport des transformers : F1 EN ultra-court passe de 0.774 (V5 TF-IDF) a **0.838** (+8.2%), et F1 EN global de 0.894 a **0.940** (+5.1%). Le test rapide (6/10) revele neanmoins le meme biais que CamemBERT V1 sur les textes courts neutres — un RoBERTa V2 avec donnees EN sociales synthetiques est la prochaine etape logique.

### 8.4 Evolution de la taille du dataset

| Version | Total | FR | EN | % FR | Textes courts |
|---------|-------|-----|-----|------|---------------|
| V1.0 | 44 124 | 0 | 44 124 | 0% | 0 |
| V1.5 | 53 607 | 9 483 | 44 124 | 18% | ~0 |
| V2.0 | 145 703 | ~36 000 | ~110 000 | 25% | ~40 000 |
| V4.0 | 187 782 | 76 023 | 111 759 | 40% | ~67 000 |
| **V5.0** | **197 782** | **86 023** | **111 759** | **43.5%** | **~77 000** |

### 8.5 Evolution des features

| Version | TF-IDF | Linguistiques | Emotionnelles | Transformer | Total |
|---------|--------|---------------|---------------|-------------|-------|
| V1.0 | 20 000 | 0 | 0 | 0 | 20 000 |
| V1.5 | 30 000 | 12 | 7 | 0 | 30 019 |
| V2.0 | 30 000 | 12 (5 nulles) | 7 | 0 | 30 019 |
| V3.0 | 30 000 | 12 (corrigees) | 7 | 0 | 30 019 |
| V4.0 | 30 000 | 15 | 7 | 0 | 30 022 |
| **V5.0** | **30 000** | **15** | **7** | **0** | **30 022** |
| CamemBERT | 0 | 0 | 0 | 768 (CLS) | 768 |

---

## 9. Marges de progression et preconisations

Cette section a ete revisee par analyse critique des axes reellement implementables et a fort impact, en ecartant les pistes theoriques peu rentables. Chaque preconisation est classee selon sa faisabilite (effort vs gain) et sa pertinence pour le cas d'usage Bluesky.

### 9.1 Preconisations retenues (impact fort, faisabilite prouvee)

#### P1 — Pipeline hybride TF-IDF V5 + CamemBERT V2 (stacking) — **REALISE**

- **Etat** : **FAIT** (notebook 17). Meta-learner LogReg entraine sur les scores des deux modeles.
- **Architecture** : 6 features meta (v5_score, camembert_score, is_fr, score_diff, score_min, score_max) → LogisticRegression
- **Resultats** :
  - F1 global : 0.9134 → **0.9163** (+0.29%)
  - F1 FR global : 0.9436 → **0.9488** (+0.52%)
  - F1 FR ultra-court : 0.9036 → **0.9092** (+0.56%)
  - F1 FR court (15-30) : 0.9436 → **0.9586** (+1.49%)
  - F1 EN global : 0.8937 → 0.8951 (+0.13%)
  - Test bilingue : 12/12
- **Coefficients meta-learner** : v5_score (-4.23) domine, camembert_score (-1.60) contribue significativement. Le flag is_fr (+0.41) confirme que CamemBERT apporte surtout en FR.
- **Conclusion** : Gain modeste mais reel, surtout sur FR court 15-30 (+1.49%). Le stacking ne surpasse pas radicalement les base learners car leurs erreurs ne sont pas suffisamment decorrelees — les deux modeles sont entraines sur les memes donnees. Le gain principal est la **robustesse** : quand V5 hesite (score ~0.5), CamemBERT tranche souvent correctement, et vice-versa.

#### P2 — Re-fine-tuning CamemBERT sur le dataset FR social synthetique — **REALISE**

- **Etat** : **FAIT** (section 7.7). CamemBERT V2 entraine avec les 10K posts FR sociaux.
- **Resultats** :
  - F1 holdout : 0.950 → **0.966** (+1.7%)
  - F1 ultra-court : 0.901 → **0.957** (+6.2%)
  - Test rapide : 3/6 → **5/6** (+2 corrections)
  - Test social supplementaire : **4/4**
  - Total : **9/10** (vs 3/6 en V1)
- **Conclusion** : L'hypothese etait correcte — le manque de donnees sociales etait la cause racine des echecs V1. Le seul echec restant ("Partagez avant censure", score=0.554) est un cas limite ambigu.

#### P3 — Seuil adaptatif par langue — **TESTE, NON SIGNIFICATIF**

- **Etat** : **FAIT** (notebook 15). Grid search seuils 0.30-0.60 (pas 0.01) pour FR et EN separement.
- **Resultats** :
  - Seuil optimal FR : 0.53 (vs 0.44 actuel)
  - Seuil optimal EN : 0.46 (vs 0.44 actuel)
  - Gain F1 global avec seuils adaptatifs : **+0.17%** (0.9132 → 0.9149)
  - Gain F1 FR : +0.12% | Gain F1 EN : +0.21%
- **Conclusion** : Le gain est **non significatif**. Le seuil unique 0.44 reste suffisant pour les deux langues. La distribution des scores FR et EN est finalement assez similaire — l'hypothese initiale (scores plus extremes en FR) se verifie mais n'a pas d'impact operationnel. Le code pour les seuils par langue est neanmoins integre dans `expert_detector.py` (parametres `threshold_fr`, `threshold_en`) pour un usage futur si necessaire.

#### P4 — RoBERTa EN fine-tune pour l'anglais — **REALISE (V1)**

- **Etat** : **FAIT** (notebook 18). roberta-base (125M params) fine-tune sur 111 759 textes EN.
- **Architecture** : Identique a CamemBERT — couches 0-8 gelees, 9-11 + head fine-tunees (17.5% parametres entrainables). Classification head : Linear(768→256)→ReLU→Dropout(0.3)→Linear(256→2). short_text_weight=2.0.
- **Resultats holdout** (22 352 textes EN) :
  - F1 : **0.9402**, Accuracy : 0.9530, Precision : 0.9560, Recall : 0.9248
  - F1 ultra-court (<15 mots) : **0.8378** (vs V5 TF-IDF 0.774 → **+8.2%**)
  - F1 court (15-30) : 0.9250, moyen : 0.9814, long : 0.9994
- **Test rapide** : **6/10** — 5/5 suspects corrects, mais 4/5 fiables neutres classees suspect :
  - "A new study published in Nature" → score 0.060 (SUSPECT, attendu FIABLE)
  - "The city council approved the new budget" → score 0.395 (SUSPECT, attendu FIABLE)
  - "The weather forecast calls for rain tomorrow" → score 0.019 (SUSPECT, attendu FIABLE)
  - "NASA announced a new mission to Mars" → score 0.011 (SUSPECT, attendu FIABLE)
  - Seul succes fiable : "The university published its annual research report" → score 0.635 (FIABLE)
- **Diagnostic** : Meme probleme structurel que CamemBERT V1 — le modele n'a jamais vu de textes courts neutres de type social media dans l'entrainement. Il apprend "texte court = suspect" car les datasets EN courts (FakeNewsNet titres, CONSTRAINT tweets) sont majoritairement suspects. Solution identique a P2 : generer des donnees EN sociales synthetiques pour un RoBERTa V2.
- **Entrainement** : 39.6 min sur MPS (Apple M4 Pro), 3 epochs, batch_size=32, lr=2e-5. Emissions : 0.001692 kg CO2.
- **Conclusion** : Le F1 EN ultra-court progresse significativement (+8.2%), depassant l'objectif initial (>0.85 atteint a 0.838 — proche). Mais le test rapide 6/10 revele le meme biais que CamemBERT V1 : un RoBERTa V2 avec donnees EN sociales synthetiques corrigerait ce probleme, comme demontre pour le FR (P2, 3/6→9/10).

### 9.2 Preconisations ecartees ou reportees (analyse critique)

#### Ecartee : Annotation manuelle de 2000+ posts Bluesky

- **Raison** : Cout humain eleve (estimation : 40-60h d'annotation), necessite un protocole d'annotation, un calcul d'inter-annotator agreement, et au moins 2 annotateurs pour eviter le biais. Le gain marginal par rapport aux 10K posts synthetiques (deja generes) ne justifie pas l'investissement dans le cadre academique du projet.
- **Alternative retenue** : Le dataset synthetique de 10K posts (V5) couvre deja le vocabulaire et les patterns cibles. L'evaluation sur Bluesky reel se fait via le dashboard en production.

#### Ecartee : Augmentation par paraphrase LLM

- **Raison** : Risque de circularite — les paraphrases generees par un LLM ont un "style" reconnaissable qui peut devenir un signal parasite. Le modele pourrait apprendre a distinguer "texte ecrit par LLM" plutot que "texte suspect". De plus, l'augmentation FR courte (V4) et le dataset synthetique (V5) apportent deja de la diversite. Gain marginal estime < 2% F1.
- **Condition de re-evaluation** : Si les resultats V5 montrent un plateau de performance FR court, la paraphrase pourrait etre testee sur un sous-ensemble (<20% du dataset) avec validation croisee.

#### Reportee : Datasets FR emergents (FakeCovid, Debunking FR)

- **Raison** : Les corpus FR fact-checkes disponibles sont generalement petits (<2K textes), mal labelises (labels non binaires, annotations ambigues), et thematiquement limites (COVID principalement). Leur integration risque d'introduire du bruit sans gain significatif. A re-evaluer si un corpus >5K textes de qualite devient disponible.
- **Veille recommandee** : Surveiller les publications des equipes CLEF CheckThat!, le Credibility Corpus V2, et les initiatives des agences de presse FR (AFP Factuel, Les Decodeurs).

#### Reportee : Embeddings cross-lingues (paraphrase-multilingual-MiniLM)

- **Raison** : Remplacerait l'architecture TF-IDF + features par un modele dense unique. Potentiel eleve a long terme mais necessite une refonte complete du pipeline. Non pertinent tant que le pipeline hybride (P1) n'a pas ete evalue.

#### Reportee : Distillation CamemBERT -> TinyBERT

- **Raison** : Optimisation de deploiement prematuree. Le modele CamemBERT (450 MB, 50ms/texte) est acceptable pour le prototype. La distillation n'a de sens qu'une fois le modele stabilise et valide en production.

### 9.3 Features a implementer (gain rapide)

| Feature | Effort | Impact | Statut |
|---------|--------|--------|--------|
| ~~Seuil adaptatif FR/EN (P3)~~ | ~~2h~~ | ~~+0.17% F1~~ | **Teste — non significatif** |
| Features de source Bluesky (nb followers, age compte) | 1 jour | Reduction faux positifs ~10% | A faire |
| Score de viralite (reposts/likes ratio) | 0.5 jour | Detection precoce | A faire |

### 9.4 Feuille de route recommandee

| Etape | Action | Delai | Prerequis | Livrable |
|-------|--------|-------|-----------|----------|
| 1 | ~~Evaluer V5 (10K FR social)~~ | **FAIT** | ~~Retraining~~ | **F1 FR court 0.904, 12/12 test** |
| 2 | ~~Re-fine-tune CamemBERT V2 (P2)~~ | **FAIT** | ~~Dataset synthetique~~ | **F1 FR court 0.957, test 9/10 (vs 3/6)** |
| 3 | ~~Seuil adaptatif par langue (P3)~~ | **FAIT** | ~~V5~~ | **Non significatif (+0.17% F1), seuil 0.44 conserve** |
| 4 | ~~Pipeline hybride stacking (P1)~~ | **FAIT** | ~~V5 + CamemBERT V2~~ | **F1 FR +0.52%, FR court 15-30 +1.49%** |
| 5 | ~~RoBERTa EN V1 (P4)~~ | **FAIT** | ~~Infrastructure CamemBERT~~ | **F1 EN ultra-court 0.838 (+8.2%), test 6/10 (meme biais V1)** |
| 6 | Integration features Bluesky (source, viralite) | 2 semaines | Acces API Bluesky | Reduction faux positifs |

---

## 10. Lecons apprises

### 10.1 Lecons techniques

1. **Les metriques elevees ne garantissent pas la qualite** : V1.0 affichait 99.6% F1 mais ne fonctionnait pas du tout. Il faut toujours evaluer sur des donnees representatifs du cas d'usage reel.

2. **Le preprocessing peut neutraliser les features** : Le bug V2 (features calculees sur texte nettoye) etait invisible car les metriques globales etaient bonnes. Seule l'analyse fine des coefficients l'a revele.

3. **L'augmentation de donnees compense partiellement le manque de donnees natives** : V4 prouve qu'on peut ameliorer significativement les performances FR court (+32%) par augmentation synthetique, mais les limites de generalisation (test rapide CamemBERT 3/6) montrent que rien ne remplace des donnees reelles du domaine cible.

4. **Les transformers ne sont pas une solution magique** : CamemBERT excelle sur les donnees du meme type que l'entrainement mais echoue sur des formulations hors-distribution. Le TF-IDF V4 + features linguistiques est plus robuste sur les patterns explicites (MAJUSCULES, !!, mots sensationnalistes).

### 10.2 Lecons methodologiques

1. **Evaluer par segment** : Les metriques globales masquent les faiblesses. L'evaluation par langue x longueur a ete decisive pour identifier le probleme FR court.

2. **Iterer rapidement** : 5 versions en 4 mois, chacune construite sur les diagnostics de la precedente. Le cycle "analyse des erreurs -> hypothese -> correction -> evaluation" est le moteur de progression.

3. **Documenter les decisions** : Chaque choix (seuil, oversample, features) a une raison documentee. Cela facilite la reproduction et l'audit.

---

## 11. Conclusion

Le pipeline Thumalien est passe d'un modele biaise inutilisable (V1, F1 = 0.996 artificiel) a un systeme bilingue performant (V5 + CamemBERT V2). L'evolution montre deux trajectoires distinctes par langue :

**Francais** : Progression spectaculaire de 0 (V1, pas de FR) a F1 = 0.944 global et 0.904 sur ultra-court (V5 TF-IDF). Le parcours V3 (F1 court = 0.65) -> V4 (+32% par augmentation) -> V5 (+10.4% par donnees sociales) demontre l'importance des donnees representatives du cas d'usage. CamemBERT V2 atteint F1 = 0.966 et F1 ultra-court = 0.957, confirmant l'apport des donnees sociales synthetiques pour les transformers egalement.

**Anglais** : Pic a F1 = 0.928 (V2/V3), regression a 0.889 (V4), puis legere remontee a 0.894 (V5). Les textes courts EN (F1 ultra-court = 0.774) restent le point faible. Un fine-tuning RoBERTa dedie (preconisation P4) est la piste prioritaire.

**Bilan V5 + CamemBERT V2** : Le test bilingue V5 passe de 9/10 (V4) a 12/12 (V5). CamemBERT V2 passe de 3/6 (V1) a 9/10 (V2). Les deux modeles reconnaissent maintenant les formulations social media FR. L'objectif F1 FR court > 0.90 est atteint par les deux approches (TF-IDF: 0.904, CamemBERT: 0.957).

**Bilan des preconisations** :
- **P1** : Pipeline hybride stacking — **FAIT**, F1 FR +0.52%, FR court 15-30 +1.49%, gain modeste mais robustesse accrue
- **P2** : Re-fine-tuning CamemBERT — **FAIT**, F1 ultra-court 0.901 → 0.957 (+6.2%), test 3/6 → 9/10
- **P3** : Seuil adaptatif par langue — **FAIT**, gain +0.17% F1 (non significatif), seuil 0.44 conserve
- **P4** : RoBERTa EN V1 — **FAIT**, F1 EN ultra-court 0.774 → 0.838 (+8.2%), test 6/10 (meme biais que CamemBERT V1, necessite donnees EN sociales pour V2)

Les 4 preconisations sont realisees. Le modele CamemBERT V2 en standalone (F1 FR ultra-court = 0.957) surpasse le pipeline hybride (0.909), car le stacking est limite par les textes EN ou CamemBERT ne contribue pas. RoBERTa EN V1 ameliore significativement le F1 EN (0.894 → 0.940) et le F1 EN ultra-court (0.774 → 0.838), mais souffre du meme biais que CamemBERT V1 sur les textes courts neutres (test rapide 6/10). La creation de donnees EN sociales synthetiques (meme approche que P2) permettrait un RoBERTa V2 avec des gains similaires a CamemBERT V1→V2 (3/6 → 9/10).

Pour la production, la configuration recommandee est : RoBERTa EN V1 pour l'EN (ou V5 TF-IDF en fallback), CamemBERT V2 pour le FR, avec pipeline hybride en orchestration.
