# Registre des Risques — Projet ThumaCheck

## Identification du document

| Champ | Valeur |
|-------|--------|
| **Reference** | RISK-THUM-2026-001 |
| **Version** | 1.0 |
| **Date** | Juillet 2026 |
| **Statut** | En vigueur |
| **Projet** | ThumaCheck — Detection de desinformation sur Bluesky |
| **Client** | Thumalien |
| **Equipe** | Azelie Bernard (Lead technique), Sebastien Lazcanotegui (Consolidation ML) |
| **Cadre** | Projet d'etude M1 Big Data & IA — Sup de Vinci |
| **Classification** | Confidentiel — Usage interne |
| **Documents lies** | RGPD-THUM-2026-001, PRA-THUM-2026-001 |

---

## 1. Methodologie d'evaluation

Chaque risque est evalue selon deux axes :

- **Probabilite (P)** : Faible (1), Moyenne (2), Elevee (3)
- **Impact (I)** : Faible (1), Moyen (2), Critique (3)
- **Score** : P x I (1 a 9)

| Score | Niveau | Action requise |
|:-----:|--------|----------------|
| 1-2 | Acceptable | Surveillance periodique |
| 3-4 | Modere | Plan d'attenuation defini |
| 6 | Eleve | Actions correctives prioritaires |
| 9 | Critique | Traitement immediat obligatoire |

---

## 2. Registre des risques

### 2.1 Risques techniques

| ID | Categorie | Risque | Probabilite | Impact | Score | Mesure d'attenuation | Statut | Responsable |
|----|-----------|--------|:-----------:|:------:|:-----:|----------------------|--------|-------------|
| R01 | Technique | **Fuite de credentials** — fichier `.env` commite par erreur dans le depot Git, exposant les tokens API Bluesky et les identifiants MongoDB | Elevee (3) | Critique (3) | **9** | `.env` ajoute au `.gitignore` ; variables injectees via Docker Compose `env_file` ; rotation des tokens en cas de doute ; pre-commit hook verifiant l'absence de secrets | Attenue | A. Bernard |
| R04 | Technique | **Indisponibilite MongoDB** — arret du conteneur Docker ou corruption du volume, rendant le pipeline et le dashboard inoperants | Moyenne (2) | Critique (3) | **6** | Politique `restart: always` dans Docker Compose ; volumes persistants ; procedure de restauration documentee dans le PRA (RTO 10 min) ; export JSON periodique | Attenue | S. Lazcanotegui |
| R09 | Technique | **Volumetrie Bluesky** — explosion du nombre de posts collectes (Bluesky passe de 2M a 20M+ utilisateurs), saturant le stockage MongoDB et le temps d'inference | Moyenne (2) | Moyen (2) | **4** | Echantillonnage aleatoire configurable dans le collecteur ; pagination AT Protocol ; monitoring de la taille MongoDB ; alertes sur seuils de stockage | En cours | A. Bernard |
| R14 | Technique | **Latence API en production** — depassement du SLA de reponse (objectif < 200 ms par requete), degradant l'experience utilisateur du dashboard | Faible (1) | Moyen (2) | **2** | Inference actuelle a 1.5 ms/texte (728 textes/sec) ; cache en memoire des modeles au demarrage ; endpoint `/health` pour monitoring ; load testing documente | Attenue | A. Bernard |

### 2.2 Risques lies aux donnees

| ID | Categorie | Risque | Probabilite | Impact | Score | Mesure d'attenuation | Statut | Responsable |
|----|-----------|--------|:-----------:|:------:|:-----:|----------------------|--------|-------------|
| R02 | Donnees | **Biais du modele** — surrepresentation de Reuters/AFP dans les donnees d'entrainement, conduisant le modele a associer "style agence de presse" a "fiable" | Moyenne (2) | Critique (3) | **6** | Debiaisage Reuters effectue (V5+) ; audit d'equite FR/EN (ecart F1 < 2 points) ; 7 datasets diversifies (197 782 textes) ; tests reguliers de fairness par sous-groupes | Attenue | S. Lazcanotegui |
| R03 | Donnees | **Drift du modele** — evolution de la distribution des posts Bluesky (nouveaux sujets, nouvelles strategies de desinformation), degradant les performances sans alerte | Moyenne (2) | Critique (3) | **6** | Gold test set de 500 posts annotes (kappa = 0.498) ; comparaison periodique des scores moyens ; pipeline de reentrainement documente (9 versions iterees V1-V9) ; monitoring de la distribution des scores sur le dashboard | En cours | A. Bernard |
| R11 | Donnees | **Perte de donnees MongoDB** — absence de backup automatise, risque de perte des 537 000+ posts collectes en cas de defaillance du volume Docker | Moyenne (2) | Critique (3) | **6** | Volumes Docker persistants ; procedure `mongodump` documentee dans le PRA ; RPO cible = 0 via volumes ; backup manuel hebdomadaire recommande ; export JSON des collections critiques | En cours | S. Lazcanotegui |
| R15 | Donnees | **Qualite du gold test set** — biais de selection dans les 500 posts annotes (accord inter-annotateurs kappa = 0.498, accord modere), fragilisant l'evaluation du modele | Moyenne (2) | Moyen (2) | **4** | Double annotation systematique ; calcul du kappa de Cohen documente ; identification des zones de desaccord ; enrichissement progressif du gold set ; transparence sur les limites dans la Model Card | Accepte | S. Lazcanotegui |

### 2.3 Risques de conformite

| ID | Categorie | Risque | Probabilite | Impact | Score | Mesure d'attenuation | Statut | Responsable |
|----|-----------|--------|:-----------:|:------:|:-----:|----------------------|--------|-------------|
| R05 | Conformite | **Non-conformite AI Act** — classification erronee du niveau de risque du systeme (classe comme "risque limite" alors qu'un usage en moderation pourrait etre "risque eleve") | Faible (1) | Critique (3) | **3** | Analyse AI Act formalisee dans le document RGPD-THUM-2026-001 ; systeme classe "risque limite" (Art. 52 — obligation de transparence) ; veille juridique sur les actes delegues ; pas de decision automatisee contraignante | Attenue | A. Bernard |
| R13 | Conformite | **RGPD — demande de suppression** — un utilisateur Bluesky exerce son droit a l'effacement (Art. 17), necessitant la suppression de ses posts de MongoDB et des modeles potentiellement entraines dessus | Faible (1) | Moyen (2) | **2** | Base legale : interet legitime (Art. 6.1.f) sur donnees publiques ; procedure de suppression par handle documentee ; pas de profilage par auteur ; pseudonymisation dans le dashboard ; donnees d'entrainement anonymisees | Attenue | A. Bernard |

### 2.4 Risques operationnels

| ID | Categorie | Risque | Probabilite | Impact | Score | Mesure d'attenuation | Statut | Responsable |
|----|-----------|--------|:-----------:|:------:|:-----:|----------------------|--------|-------------|
| R06 | Operationnel | **Depassement de cout API LLM** — utilisation non controlee des API de modeles de langage (OpenAI, Mistral) pour le benchmark ou l'explicabilite, generant des couts imprevisibles | Moyenne (2) | Moyen (2) | **4** | Budget API plafonne ; utilisation principale de modeles locaux (CamemBERT, LogReg) ; appels LLM reserves au benchmark et a l'explicabilite ; monitoring des quotas ; modeles open-weight privilegies | Attenue | A. Bernard |
| R08 | Operationnel | **Faux positifs** — texte fiable classe "suspect", generant de la defiance envers le systeme et potentiellement de la stigmatisation des auteurs | Moyenne (2) | Critique (3) | **6** | Pipeline cascade fait/opinion reduisant les FP de 67% (V9 vs V5, p < 0.001) ; seuil de decision calibre ; categorie "indecis" pour les cas ambigus ; explicabilite SHAP permettant de comprendre les predictions ; mention "indicatif" dans l'interface | Attenue | S. Lazcanotegui |
| R10 | Operationnel | **Obsolescence technologique** — deprecation de CamemBERT, de librairies Python (scikit-learn, transformers) ou de l'API AT Protocol Bluesky | Faible (1) | Moyen (2) | **2** | Docker Compose garantissant la reproductibilite ; versions figees dans `requirements.txt` ; architecture modulaire permettant le remplacement de composants ; veille technologique semestrielle | Accepte | A. Bernard |
| R12 | Operationnel | **Attaque adversariale** — texte specifiquement concu pour tromper le modele (ajout de mots-cles "fiables", obfuscation Unicode, paraphrase automatisee) | Moyenne (2) | Moyen (2) | **4** | Pipeline multi-etages (TF-IDF + features stylistiques + CamemBERT) rendant la manipulation plus difficile ; 28 features stylistiques complementaires au lexical ; gold test set incluant des cas limites ; monitoring des scores aberrants | En cours | S. Lazcanotegui |

### 2.5 Risques humains

| ID | Categorie | Risque | Probabilite | Impact | Score | Mesure d'attenuation | Statut | Responsable |
|----|-----------|--------|:-----------:|:------:|:-----:|----------------------|--------|-------------|
| R07 | Humain | **Dependance personne-cle** — equipe de 2 personnes seulement, risque d'indisponibilite d'un membre (maladie, abandon) paralysant le projet | Elevee (3) | Critique (3) | **9** | Documentation exhaustive (17 documents techniques, 27 notebooks) ; code versionne sur GitHub ; pair-programming regulier ; connaissance croisee des composants ; Docker facilitant la reprise par un tiers | En cours | A. Bernard, S. Lazcanotegui |

---

## 3. Matrice de chaleur (heatmap)

```
                        IMPACT
                 Faible (1)    Moyen (2)     Critique (3)
              ┌─────────────┬─────────────┬─────────────┐
  Elevee (3)  │             │             │ R01  R07    │
              │             │             │ Score: 9    │
              ├─────────────┼─────────────┼─────────────┤
  Moyenne (2) │             │ R06  R09    │ R02  R03    │
              │             │ R12  R15    │ R08  R11    │
              │             │ Score: 4    │ Score: 6    │
              ├─────────────┼─────────────┼─────────────┤
  Faible (1)  │             │ R10  R13    │ R05         │
              │             │ R14         │ Score: 3    │
              │             │ Score: 2    │             │
              └─────────────┴─────────────┴─────────────┘

PROBABILITE

  Legende :   Score 9 = CRITIQUE (action immediate)
              Score 6 = ELEVE (actions correctives prioritaires)
              Score 3-4 = MODERE (plan d'attenuation)
              Score 1-2 = ACCEPTABLE (surveillance)
```

### Synthese par niveau de risque

| Niveau | Risques | Nombre |
|--------|---------|:------:|
| Critique (9) | R01, R07 | 2 |
| Eleve (6) | R02, R03, R04, R08, R11 | 5 |
| Modere (3-4) | R05, R06, R09, R12, R15 | 5 |
| Acceptable (1-2) | R10, R13, R14 | 3 |

---

## 4. Plan de traitement — Top 5 des risques prioritaires

### 4.1 R01 — Fuite de credentials (Score 9)

**Objectif** : Reduire la probabilite de Elevee a Faible.

| Action | Echeance | Responsable | Statut |
|--------|----------|-------------|--------|
| Verifier que `.env` est bien dans `.gitignore` | Fait | A. Bernard | Termine |
| Scanner l'historique Git pour des secrets exposes (`git-secrets`, `trufflehog`) | Juillet 2026 | A. Bernard | A faire |
| Mettre en place un pre-commit hook interdisant les patterns de secrets | Juillet 2026 | A. Bernard | A faire |
| Documenter la procedure de rotation des tokens en cas d'incident | Juillet 2026 | S. Lazcanotegui | A faire |

**Risque residuel apres traitement** : Faible (Score 3 -> 1).

### 4.2 R07 — Dependance personne-cle (Score 9)

**Objectif** : Reduire l'impact en cas d'indisponibilite d'un membre.

| Action | Echeance | Responsable | Statut |
|--------|----------|-------------|--------|
| Maintenir la documentation technique a jour (17 docs) | Continu | A. Bernard | En cours |
| Sessions de transfert de connaissances bimensuelles | Continu | Les deux | En cours |
| Chaque composant critique doit pouvoir etre opere par l'autre membre | Fait | Les deux | Termine |
| Architecture Docker permettant un deploiement sans expertise specifique | Fait | A. Bernard | Termine |

**Risque residuel apres traitement** : Modere (Score 9 -> 6). Le risque reste inherent a la taille de l'equipe.

### 4.3 R03 — Drift du modele (Score 6)

**Objectif** : Detecter et corriger toute degradation de performance sous 2 semaines.

| Action | Echeance | Responsable | Statut |
|--------|----------|-------------|--------|
| Reevaluation mensuelle sur le gold test set (500 posts) | Mensuel | S. Lazcanotegui | A planifier |
| Monitoring de la distribution des scores sur le dashboard (page Metriques) | Fait | A. Bernard | Termine |
| Seuil d'alerte si F1 < 0.85 sur le gold set | Aout 2026 | S. Lazcanotegui | A faire |
| Procedure de reentrainement documentee (pipeline V1-V9 existant) | Fait | S. Lazcanotegui | Termine |

**Risque residuel apres traitement** : Modere (Score 6 -> 4).

### 4.4 R08 — Faux positifs (Score 6)

**Objectif** : Maintenir le taux de faux positifs en dessous de 5% sur le gold set.

| Action | Echeance | Responsable | Statut |
|--------|----------|-------------|--------|
| Pipeline cascade fait/opinion V9 (reduction -67% FP) | Fait | S. Lazcanotegui | Termine |
| Explicabilite SHAP integree pour chaque prediction | Fait | A. Bernard | Termine |
| Categorie "indecis" pour les cas ambigus (seuil calibre) | Fait | S. Lazcanotegui | Termine |
| Revue trimestrielle des faux positifs remontes par les utilisateurs | Trimestriel | A. Bernard | A planifier |

**Risque residuel apres traitement** : Modere (Score 6 -> 4).

### 4.5 R11 — Perte de donnees MongoDB (Score 6)

**Objectif** : Garantir un RPO = 0 et un RTO < 10 minutes.

| Action | Echeance | Responsable | Statut |
|--------|----------|-------------|--------|
| Volumes Docker persistants (`restart: always`) | Fait | A. Bernard | Termine |
| Script `mongodump` automatise (cron quotidien) | Aout 2026 | S. Lazcanotegui | A faire |
| Procedure de restauration documentee dans le PRA | Fait | A. Bernard | Termine |
| Stockage des backups sur un support externe (disque ou cloud) | Aout 2026 | S. Lazcanotegui | A faire |

**Risque residuel apres traitement** : Faible (Score 6 -> 2).

---

## 5. Suivi et revue

### 5.1 Frequence de revue

| Type de revue | Frequence | Participants | Livrable |
|---------------|-----------|--------------|----------|
| Revue operationnelle des risques | Mensuelle | A. Bernard, S. Lazcanotegui | Mise a jour du registre |
| Revue de conformite (RGPD/AI Act) | Trimestrielle | Equipe + referent juridique | Rapport de conformite |
| Reevaluation complete du registre | Semestrielle | Equipe + tuteur academique | Nouvelle version du document |
| Revue post-incident | A chaque incident | Equipe concernee | Fiche d'incident + mise a jour du registre |

### 5.2 Criteres de declenchement d'une revue extraordinaire

- Incident de securite (fuite de donnees, intrusion)
- Modification reglementaire impactant le projet (actes delegues AI Act)
- Changement significatif de l'architecture technique
- Degradation mesuree des performances du modele (F1 < 0.85)
- Depassement d'un seuil budgetaire API

### 5.3 Indicateurs de suivi

| Indicateur | Cible | Frequence de mesure |
|------------|-------|---------------------|
| Nombre de risques critiques (score 9) | 0 | Mensuelle |
| Nombre de risques eleves (score 6) non traites | 0 | Mensuelle |
| Taux de faux positifs sur gold set | < 5% | Mensuelle |
| F1-score en production | > 0.85 | Mensuelle |
| Temps moyen de reprise (RTO reel) | < RTO cible | A chaque incident |
| Couverture de tests | > 80% | A chaque release |

### 5.4 Historique des revisions

| Version | Date | Auteur | Modifications |
|---------|------|--------|---------------|
| 1.0 | Juillet 2026 | A. Bernard | Creation initiale — 15 risques identifies |

---

*Document etabli dans le cadre du projet d'etude M1 Big Data & IA — Sup de Vinci — Juillet 2026*
*Equipe Niamato Consulting : Azelie Bernard, Sebastien Lazcanotegui*
