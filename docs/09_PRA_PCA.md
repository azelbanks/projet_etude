# Plan de Reprise et de Continuite d'Activite (PRA/PCA)
## Projet Thumalien - Strategie de resilience

**Reference** : PRA-THUM-2026-001
**Version** : 1.0
**Date** : Avril 2026
**Equipe** : Azelie Bernard, Sebastien Lazcanotegui

---

## 1. Objectif du document

Ce document definit les procedures de reprise et de continuite d'activite du systeme Thumalien. Il garantit la disponibilite du service de detection de fake news et la protection des donnees en cas d'incident.

---

## 2. Perimetres couverts

| Composant | Criticite | RTO cible | RPO cible |
|-----------|:---------:|:---------:|:---------:|
| Collecteur Bluesky | Haute | 5 min | 10 min de posts |
| Base MongoDB | Critique | 10 min | 0 (volumes persistants) |
| Pipeline NLP (inference) | Haute | 5 min | 0 |
| Modeles entraines (.joblib, .pt) | Critique | 15 min | Derniere version Git |
| Dashboard Streamlit | Moyenne | 30 min | 0 |
| Code source | Critique | 5 min | Dernier commit |

**RTO** = Recovery Time Objective (temps maximal de reprise)
**RPO** = Recovery Point Objective (perte de donnees maximale acceptable)

---

## 3. Plan de Continuite d'Activite (PCA)

### 3.1 Architecture de resilience

```
                    ┌─────────────────────────────────────┐
                    │     ARCHITECTURE DE RESILIENCE       │
                    └─────────────────────────────────────┘

  [GitHub Repository]         [Docker Volumes]        [Modeles .joblib/.pt]
   Code source versionne      Donnees MongoDB          Sauvegardes Git LFS
   22 notebooks               Persistance garantie     Fallback V(n-1)
   Documentation complete     restart: always

                    ┌─────────────────────────────────────┐
                    │     MECANISMES DE PROTECTION         │
                    └─────────────────────────────────────┘

  [restart: always]           [Volumes Docker]         [Git versioning]
   Redemarrage auto           Survie aux crashes       Historique complet
   Tous les conteneurs        Donnees persistantes     Rollback possible
```

### 3.2 Mesures de continuite par composant

#### MongoDB (Donnees)
- **Volumes Docker persistants** : les donnees survivent au redemarrage des conteneurs
- **Politique `restart: always`** : redemarrage automatique en cas de crash
- **Index unique sur `uri`** : protection contre les doublons apres reprise
- **Backup recommande** : `mongodump` quotidien vers stockage externe

#### Collecteur Bluesky
- **`restart: always`** dans Docker Compose : reprise automatique
- **Gestion des erreurs reseau** : retry avec backoff exponentiel
- **Deduplication** : les posts deja collectes sont ignores au redemarrage
- **Impact d'un arret** : perte des posts publies pendant l'indisponibilite uniquement

#### Pipeline NLP
- **Modeles versionnes dans Git** : rollback instantane vers version precedente
- **Fallback automatique** : si le modele V5 echoue, chargement du V4 automatiquement
- **Inference stateless** : pas d'etat a restaurer, rechargement du modele suffit

#### Dashboard Streamlit
- **Lecture seule** : le dashboard ne modifie pas les donnees
- **Redemarrage rapide** : `docker-compose restart streamlit`
- **Cache Streamlit** : les donnees sont mises en cache pour des performances optimales

---

## 4. Plan de Reprise d'Activite (PRA)

### 4.1 Scenarios d'incidents et procedures

#### Scenario 1 : Crash d'un conteneur Docker

| Etape | Action | Commande | Temps |
|-------|--------|----------|-------|
| 1 | Verification automatique | `restart: always` dans docker-compose.yml | 0 min |
| 2 | Verification manuelle si echec | `docker-compose ps` | 1 min |
| 3 | Redemarrage manuel | `docker-compose restart <service>` | 2 min |
| 4 | Verification des logs | `docker-compose logs --tail=50 <service>` | 3 min |

#### Scenario 2 : Corruption de la base MongoDB

| Etape | Action | Commande | Temps |
|-------|--------|----------|-------|
| 1 | Arreter le conteneur | `docker-compose stop mongodb` | 1 min |
| 2 | Sauvegarder le volume corrompu | `cp -r mongodb_data mongodb_data_backup` | 5 min |
| 3 | Restaurer depuis backup | `mongorestore --db thumalien backup/` | 10 min |
| 4 | Redemarrer et verifier | `docker-compose up -d mongodb` | 2 min |
| 5 | Relancer la collecte | Le collecteur reprend automatiquement | 1 min |

#### Scenario 3 : Corruption ou perte d'un modele ML

| Etape | Action | Commande | Temps |
|-------|--------|----------|-------|
| 1 | Identifier le modele defaillant | Verifier les logs d'erreur | 2 min |
| 2 | Restaurer depuis Git | `git checkout HEAD -- models/` | 1 min |
| 3 | Si absent de Git : re-entrainer | Executer le notebook correspondant | 30-60 min |
| 4 | Valider les predictions | Tester sur un echantillon connu | 5 min |
| 5 | Fallback V(n-1) | Charger la version precedente du modele | 2 min |

#### Scenario 4 : Panne complete de la machine hote

| Etape | Action | Temps |
|-------|--------|-------|
| 1 | Provisionner nouvelle machine | Variable |
| 2 | Installer Docker + Docker Compose | 15 min |
| 3 | Cloner le depot Git | `git clone` - 5 min |
| 4 | Restaurer backup MongoDB | 30 min |
| 5 | Lancer `docker-compose up -d` | 5 min |
| 6 | Verifier tous les services | 10 min |
| **Total** | | **~1 heure** (hors provisionnement) |

#### Scenario 5 : API Bluesky indisponible

| Etape | Action | Temps |
|-------|--------|-------|
| 1 | Le collecteur detecte l'erreur | Automatique |
| 2 | Retry avec backoff exponentiel | 1-5 min |
| 3 | Dashboard continue sur donnees existantes | 0 min |
| 4 | Surveillance de l'API AT Protocol | Continu |
| 5 | Reprise automatique quand l'API revient | Automatique |

---

## 5. Politique de sauvegarde

### 5.1 Strategie de backup

| Donnee | Methode | Frequence | Retention | Stockage |
|--------|---------|-----------|-----------|----------|
| Code source | Git push | A chaque commit | Illimitee | GitHub |
| Modeles ML | Git (LFS si > 100MB) | A chaque version | Toutes versions | GitHub |
| Base MongoDB | mongodump | Quotidien (recommande) | 30 jours | Stockage externe |
| Configuration Docker | Versionnee dans Git | A chaque modification | Illimitee | GitHub |
| Documentation | Versionnee dans Git | A chaque modification | Illimitee | GitHub |

### 5.2 Procedure de backup MongoDB recommandee

```
# Backup quotidien automatise (crontab)
0 2 * * * docker exec mongodb mongodump --out /backup/$(date +%Y%m%d)

# Backup manuel avant operation risquee
docker exec mongodb mongodump --out /backup/pre_operation

# Restauration
docker exec mongodb mongorestore --drop /backup/20260415
```

---

## 6. Tests de reprise

### 6.1 Plan de tests

| Test | Frequence | Procedure | Critere de succes |
|------|-----------|-----------|-------------------|
| Redemarrage conteneur | Mensuel | `docker-compose restart` | Tous services UP en < 5 min |
| Restauration modele | A chaque nouvelle version | `git checkout` modele precedent | Predictions coherentes |
| Simulation panne MongoDB | Trimestriel | Arreter MongoDB, verifier reprise | Donnees intactes apres restart |
| Restauration complete | Semestriel | Clone + docker-compose up sur machine neuve | Systeme fonctionnel en < 1h |

### 6.2 Resultats des derniers tests

| Date | Test | Resultat | Observations |
|------|------|----------|--------------|
| Avril 2026 | Redemarrage Docker Compose | OK | Tous services UP en 45 secondes |
| Avril 2026 | Rollback modele V5 -> V4 | OK | Predictions coherentes, F1 stable |
| Mars 2026 | Restart MongoDB | OK | 188K posts intacts, index preserves |

---

## 7. Contacts et escalade

| Niveau | Responsable | Action |
|--------|-------------|--------|
| Niveau 1 (automatique) | Docker (`restart: always`) | Redemarrage automatique |
| Niveau 2 (manuel) | Azelie Bernard | Diagnostic + redemarrage manuel |
| Niveau 3 (escalade) | Azelie + Sebastien | Restauration complete, re-entrainement |
| Niveau 4 (critique) | Equipe + encadrant | Decision sur la strategie de reprise |

---

*Document valide par l'equipe projet - Avril 2026*
*Reference : PRA-THUM-2026-001 - Version 1.0*
