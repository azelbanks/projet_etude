# Executive Summary — Projet Thumalien

## Problematique

La desinformation sur les reseaux sociaux constitue un enjeu societaire majeur. Les plateformes decentralisees comme Bluesky (2M+ utilisateurs) ne disposent pas encore de mecanismes robustes de moderation automatisee. Comment detecter de maniere fiable, explicable et frugale les contenus potentiellement trompeurs dans un flux bilingue FR/EN en temps reel ?

---

## Solution

**Thumalien** est un pipeline NLP bilingue de detection de fake news en 2 etapes :

1. **Stage 1** : Filtre fait/opinion (ecarte les opinions avant analyse)
2. **Stage 2** : Meta-learner V8 combinant TF-IDF V5 (lexical) + Style V6 (28 features stylistiques) + CamemBERT (semantique FR)

Le systeme inclut un dashboard Streamlit 5 pages avec explicabilite SHAP integree, une API FastAPI, et un collecteur automatise Bluesky (AT Protocol).

---

## KPI du projet

| Indicateur | Valeur |
|-----------|--------|
| Posts collectes | 245 000+ (collecte continue depuis Dec 2025) |
| Textes d'entrainement | 197 782 (7 datasets, FR+EN) |
| F1-score V5 (production) | 0.913 (CV 5-folds) |
| Reduction faux positifs V9 vs V5 | -67% (IC 95% : [-73%, -60%], Fisher p < 0.001) |
| CamemBERT FR (ultra-court) | F1 = 0.957 |
| Latence inference | 1.5 ms/texte (728 textes/sec) |
| Tests unitaires | 342 tests, 49% couverture |
| Empreinte CO2 totale | 6.14 g (< 1 recherche Google) |
| Gold standard | 500 posts, 2 annotateurs, kappa = 0.498 |
| Versions iterees | 9 versions documentees (V1 a V9) |

---

## Livrables

| # | Livrable | Statut |
|---|----------|--------|
| 1 | Code source complet (Python 3.13, Docker Compose) | Livre |
| 2 | Pipeline NLP V9 bilingue (9 versions documentees) | Livre |
| 3 | Dashboard Streamlit 5 pages + API FastAPI | Livre |
| 4 | Documentation technique (17 PDF, 27 notebooks) | Livre |
| 5 | Gold test set annote (500 posts, 2 annotateurs) | Livre |
| 6 | Pipeline XAI complet (SHAP, Captum, faithfulness) | Livre |
| 7 | Model Card formelle (Mitchell et al., 2019) | Livre |
| 8 | Video MVP (15-20 min) | En cours |

---

## Impact et valeur ajoutee

- **Scientifique** : pipeline cascade fait/opinion original reduisant les FP de 67% (valide statistiquement par bootstrap)
- **Technique** : architecture Docker micro-services reproductible, CI/CD, 342 tests
- **Ethique** : conformite RGPD + AI Act, explicabilite multi-niveaux (SHAP, attention, IG), Model Card
- **Environnemental** : 6.14 g CO2 total, choix delibere de modeles frugaux (LogReg en prod, CamemBERT reserve au FR)
- **Pedagogique** : 9 iterations documentees avec echecs (self-training) et pivots (debiaisage Reuters)

---

*Projet d'etude M1 Big Data & IA — Sup de Vinci — Mai 2026*
*Equipe : Azelie Bernard (Lead technique), Sebastien Lazcanotegui (Consolidation ML)*
