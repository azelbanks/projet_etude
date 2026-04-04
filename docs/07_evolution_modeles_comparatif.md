# Evolution des Modeles — Comparatif de Progression
## Projet Thumalien — Social Media Intelligence & AI Monitor

**Reference** : EVOL-THUM-2026-001
**Version** : 1.0
**Date** : Avril 2026
**Auteur** : Thumalien Team — Direction Projet Data

---

## 1. Synthese executive

Ce document retrace l'evolution complete du pipeline de detection de desinformation Thumalien, de la version V1 (decembre 2025) a la version V4 + CamemBERT (avril 2026). Chaque iteration est analysee en termes de performances, d'avantages, d'inconvenients, de limites identifiees et de marges de progression exploitees dans la version suivante.

### Tableau de synthese globale

| Version | Date | F1 global | F1 FR court | Precision | Recall | Dataset | Innovation cle |
|---------|------|-----------|-------------|-----------|--------|---------|----------------|
| V1.0 | Dec 2025 | 0.996 (biaise) | N/A | 0.997 | 0.995 | 44 124 EN | Baseline TF-IDF |
| V1.5 | Jan 2026 | 0.986 | N/A | 0.983 | 0.989 | 53 607 FR+EN | Bilingue + emotions |
| V2.0 | Fev 2026 | 0.897 | 0.650 | 0.891 | 0.903 | 145 703 | Datasets sociaux |
| V3.0 | Mars 2026 | 0.900 | 0.650 | 0.891 (+19.3%) | 0.910 | 145 703 | Bug fix features |
| V4.0 | Avril 2026 | 0.905 | 0.860 | 0.897 | 0.914 | 187 782 | Augmentation FR court |
| CamemBERT | Avril 2026 | 0.950 | 0.901 | 0.969 | 0.931 | 22 540 FR | Transformer FR |

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

### 4.3 Avantages

- Integration de donnees sociales (tweets, titres) = meilleure generalisation
- Seuil 0.44 adapte aux posts courts Bluesky
- 73.4% de posts classes comme fiables sur Bluesky reel (vs 23% en V1.5)
- Ponderation bilingue ameliore l'equilibre FR/EN

### 4.4 Inconvenients et limites

- **FR ultra-court F1 = 0.650** — Inacceptable pour le cas d'usage Bluesky
- **Recall FR = 0.738** — Rate 1 fake news FR sur 4
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

**Impact majeur : Precision sur les textes courts**

Le gain global est faible (+0.3% F1) mais la precision sur les textes suspects augmente significativement : +19.3% sur les cas ou les features linguistiques (majuscules, ponctuation emotive) font la difference.

### 5.3 Avantages

- Correction d'un bug critique : les features linguistiques fonctionnent enfin
- Le modele capte maintenant les MAJUSCULES, les !!!, les ? multiples
- Precision accrue sur les textes conspirationalistes a forte ponctuation

### 5.4 Inconvenients et limites

- **FR court toujours F1 = 0.65** — Le bug fix seul ne suffit pas
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

### 6.4 Avantages

- **Gain massif sur FR court** : +32% F1 sur ultra-court, +28% sur court
- **Recall FR corrige** : 0.74 -> 0.94 (ne rate plus que 6% des fake news FR)
- **Dataset plus equilibre** : 40% FR au lieu de 25%
- **Nouvelles features discriminantes** pour les posts sociaux (CAPS, interpellation)

### 6.5 Inconvenients et limites

- **Trade-off EN** : L'EN perd ~4% F1 sur les textes courts (le modele favorise davantage le FR)
- **Augmentation synthetique** : Les textes courts generes sont des extraits d'articles, pas de vrais posts sociaux FR
- **Biais thematique persistant** : Le TF-IDF capte "vaccin", "climat" comme signaux suspects
- **Convergence** : Le solver LBFGS atteint 5000 iterations sans converger sur 188K textes (corrige par max_iter=10000)

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

---

## 8. Comparaison globale multi-versions

### 8.1 Evolution du F1 par version

| Version | F1 global | F1 FR global | F1 FR court (<15 mots) | F1 EN global |
|---------|-----------|--------------|------------------------|--------------|
| V1.0 | 0.996 (biaise) | N/A | N/A | 0.996 |
| V1.5 | 0.986 | 0.982 | N/A | 0.985 |
| V2.0 | 0.897 | 0.846 | 0.650 | 0.928 |
| V3.0 | 0.900 | 0.846 | 0.650 | 0.928 |
| **V4.0** | **0.905** | **0.935** | **0.860** | **0.889** |
| **CamemBERT** | **0.950** (FR) | **0.950** | **0.901** | N/A |

### 8.2 Evolution de la taille du dataset

| Version | Total | FR | EN | % FR | Textes courts |
|---------|-------|-----|-----|------|---------------|
| V1.0 | 44 124 | 0 | 44 124 | 0% | 0 |
| V1.5 | 53 607 | 9 483 | 44 124 | 18% | ~0 |
| V2.0 | 145 703 | ~36 000 | ~110 000 | 25% | ~40 000 |
| V4.0 | 187 782 | 76 023 | 111 759 | 40% | ~67 000 |

### 8.3 Evolution des features

| Version | TF-IDF | Linguistiques | Emotionnelles | Transformer | Total |
|---------|--------|---------------|---------------|-------------|-------|
| V1.0 | 20 000 | 0 | 0 | 0 | 20 000 |
| V1.5 | 30 000 | 12 | 7 | 0 | 30 019 |
| V2.0 | 30 000 | 12 (5 nulles) | 7 | 0 | 30 019 |
| V3.0 | 30 000 | 12 (corrigees) | 7 | 0 | 30 019 |
| V4.0 | 30 000 | 15 | 7 | 0 | 30 022 |
| CamemBERT | 0 | 0 | 0 | 768 (CLS) | 768 |

---

## 9. Marges de progression identifiees

### 9.1 Donnees

| Axe | Etat actuel | Objectif | Impact estime |
|-----|-------------|----------|---------------|
| Dataset FR social media natif | Aucun (augmentation synthetique) | Integrer 10K+ tweets FR fact-checkes | F1 FR court +5-10% |
| Donnees Bluesky annotees | 0 | Annoter 2000+ posts Bluesky (fiable/suspect) | Reduction du domain shift |
| Augmentation par paraphrase | Non implementee | LLM pour paraphraser les fake news FR | Diversite des formulations |
| Datasets FR emergents | Non explores | FakeCovid FR, Debunking FR corpora | +5K-20K textes FR |

### 9.2 Modeles

| Axe | Etat actuel | Objectif | Impact estime |
|-----|-------------|----------|---------------|
| Pipeline hybride TF-IDF + CamemBERT | Modeles separes | Combiner les scores (stacking) | F1 FR court > 0.92 |
| CamemBERT sur donnees sociales | Fine-tune sur articles FR | Fine-tune sur tweets/posts FR | Meilleure generalisation |
| Modele EN dedie (RoBERTa) | TF-IDF seul pour EN | Fine-tuner RoBERTa-base pour EN | F1 EN court > 0.85 |
| Apprentissage few-shot | Non implementee | Adapter avec 50-100 exemples Bluesky | Adaptation rapide au domaine |
| Distillation | CamemBERT 110M params | Distiller en modele leger (TinyBERT FR) | Inference x10, <50 MB |

### 9.3 Features et preprocessing

| Axe | Etat actuel | Objectif | Impact estime |
|-----|-------------|----------|---------------|
| Seuil adaptatif par langue | Seuil unique 0.44 | Seuil FR / seuil EN differencies | Precision +2-5% |
| Features de source | Non implementees | Reputation du compte, age, followers | Reduction faux positifs |
| Features temporelles | Non implementees | Heure de publication, viralite | Detection precoce |
| Embeddings cross-lingues | Non implementes | paraphrase-multilingual-MiniLM | Un seul modele FR+EN |

### 9.4 Priorites recommandees

1. **Court terme** (1 semaine) : Implementer le pipeline hybride (stacking V4 + CamemBERT)
2. **Moyen terme** (1 mois) : Collecter et annoter 2000 posts Bluesky FR, retrainer CamemBERT
3. **Long terme** (3 mois) : Modele cross-lingue unique avec embeddings transformer + distillation

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

Le pipeline Thumalien est passe d'un modele biaise inutilisable (V1, F1 = 0.996 artificiel) a un systeme bilingue performant (V4 + CamemBERT, F1 FR court = 0.90). Les axes d'amelioration sont clairs : donnees FR sociales natives, pipeline hybride, et evaluation continue sur des posts Bluesky reels.

La marge de progression la plus importante reside dans les donnees, pas dans les algorithmes. Un dataset de 5000 tweets FR fact-checkes aurait probablement plus d'impact que n'importe quelle amelioration architecturale.
