# Accessibilite et Prise en Compte du Handicap
## Projet Thumalien

**Reference** : ACC-THUM-2026-001
**Version** : 1.0
**Date** : Avril 2026

---

## 1. Engagement accessibilite

Le projet Thumalien s'engage a rendre ses outils et interfaces accessibles au plus grand nombre, conformement aux principes du RGAA (Referentiel General d'Amelioration de l'Accessibilite) et des WCAG 2.1 (Web Content Accessibility Guidelines).

---

## 2. Dashboard Streamlit - Mesures d'accessibilite

### 2.1 Accessibilite visuelle

| Critere | Implementation | Statut |
|---------|---------------|:------:|
| Contraste suffisant (ratio >= 4.5:1) | Theme dark avec accents cyan sur fond sombre | Conforme |
| Textes lisibles (taille >= 14px) | Police systeme, taille configurable Streamlit | Conforme |
| Pas de dependance a la couleur seule | Les indicateurs utilisent texte + couleur + icone | Conforme |
| Mode daltonien | Palette choisie compatible deuteranopie (cyan/orange) | Partiel |
| Zoom navigateur (jusqu'a 200%) | Layout responsive Streamlit natif | Conforme |

### 2.2 Accessibilite motrice

| Critere | Implementation | Statut |
|---------|---------------|:------:|
| Navigation clavier | Streamlit supporte la navigation par Tab | Conforme |
| Zones cliquables suffisantes | Boutons et widgets Streamlit standards (>= 44px) | Conforme |
| Pas de double-clic requis | Toutes les interactions en simple clic | Conforme |

### 2.3 Accessibilite cognitive

| Critere | Implementation | Statut |
|---------|---------------|:------:|
| Langage clair | Labels descriptifs, tooltips explicatifs | Conforme |
| Hierarchie visuelle | Titres, sous-titres, sections bien delimitees | Conforme |
| Feedback utilisateur | Messages de succes/erreur explicites | Conforme |
| Aide contextuelle | Guide utilisateur accessible depuis le dashboard | Conforme |

---

## 3. Interpretation des resultats - Accessibilite de l'information

### 3.1 Score de fiabilite

Le score de fiabilite (0 a 1) est presente de maniere accessible :

- **Barre de progression coloree** : vert (fiable) a rouge (suspect)
- **Texte explicatif** : "Ce texte est classe FIABLE avec un score de 0.78/1.00"
- **Icone** : check vert ou warning orange
- **Pas de jargon technique** : les metriques ML (F1, precision) sont dans la page technique, pas dans l'interface utilisateur

### 3.2 Explicabilite

Le dashboard V3 inclut une page d'explicabilite qui :
- Montre les mots les plus influents dans la decision (positifs et negatifs)
- Affiche un radar chart des features linguistiques
- Utilise un langage non technique pour les utilisateurs finaux

---

## 4. Documentation - Accessibilite

| Document | Format | Accessibilite |
|----------|--------|--------------|
| Guide utilisateur | PDF + Markdown | Texte structuré avec titres, lisible par lecteur d'ecran |
| Rapport technique | PDF + Markdown | Structure hierarchique, tableaux avec en-tetes |
| README | Markdown (GitHub) | Format standard, rendu accessible par GitHub |
| Notebooks | Jupyter (.ipynb) | Cellules markdown + code, executables |

### 4.1 Bonnes pratiques appliquees

- **Structure semantique** : utilisation coherente des niveaux de titres (H1 > H2 > H3)
- **Tableaux avec en-tetes** : toutes les colonnes sont identifiees
- **Texte alternatif** : les graphiques generes incluent des titres et labels descriptifs
- **Pas d'information uniquement dans les images** : les donnees cles sont aussi dans le texte

---

## 5. API et donnees - Accessibilite technique

### 5.1 API de prediction (future)

L'API REST prevue (FastAPI) suivra les bonnes pratiques d'accessibilite technique :
- Documentation OpenAPI/Swagger interactive
- Messages d'erreur clairs et structures (JSON)
- Codes HTTP standards
- Temps de reponse < 100ms (pas de timeout frustrant)

### 5.2 Donnees ouvertes

- Les metriques du modele sont documentees et reproductibles
- Les datasets d'entrainement sont references avec leurs licences
- Les limitations du modele sont clairement communiquees

---

## 6. Axes d'amelioration identifies

| Axe | Priorite | Effort | Impact |
|-----|:--------:|:------:|:------:|
| Ajouter des textes alternatifs aux graphiques Plotly | Haute | Faible | Eleve |
| Tester avec un lecteur d'ecran (NVDA, VoiceOver) | Haute | Moyen | Eleve |
| Ajouter un mode haut contraste | Moyenne | Moyen | Moyen |
| Proposer une version texte-seul du dashboard | Basse | Eleve | Moyen |
| Internationalisation (i18n) du dashboard | Basse | Eleve | Moyen |
| Tests d'utilisabilite avec des utilisateurs en situation de handicap | Haute | Eleve | Eleve |

---

## 7. Conformite reglementaire

| Referentiel | Niveau vise | Statut actuel |
|-------------|:-----------:|:------------:|
| RGAA 4.1 | Partiellement conforme | En cours d'evaluation |
| WCAG 2.1 | Niveau AA | Partiellement conforme |
| AI Act (transparence) | Conforme | Conforme (explicabilite, limites documentees) |

Le dashboard Streamlit beneficie nativement de certaines fonctionnalites d'accessibilite (structure HTML semantique, navigation clavier). Des ameliorations specifiques sont prevues pour atteindre une conformite plus complete.

---

*Document valide par l'equipe projet - Avril 2026*
*Reference : ACC-THUM-2026-001 - Version 1.0*
