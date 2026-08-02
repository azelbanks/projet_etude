# Politique de sécurité

## Signaler une faille

**N'ouvrez pas d'issue publique pour une vulnérabilité.**

Utilisez l'onglet *Security → Report a vulnerability* du dépôt (GitHub Private
Vulnerability Reporting), ou écrivez à l'adresse de contact de Niamato
Consulting.

Merci d'inclure : la version concernée, les étapes de reproduction, l'impact
estimé et, si vous en avez un, un correctif proposé.

Délai de première réponse visé : 5 jours ouvrés.

## Périmètre

Sont concernés le code de `src/`, `dashboard/` et la configuration de
déploiement (`docker-compose.yml`, workflows GitHub Actions).

Ne sont pas concernés les notebooks d'exploration (`notebooks/`), qui ne sont
pas destinés à la production.

## Données et conformité

ThumaCheck collecte des publications publiques Bluesky. Le traitement des
données personnelles est décrit dans
[`docs/02_conformite_RGPD_AI_Act.md`](docs/02_conformite_RGPD_AI_Act.md).

Points saillants de l'implémentation :

- **Pseudonymisation** — les identifiants d'auteurs sont hachés en SHA-256
  tronqué avec un sel (`pseudonymize()` dans `src/collection/collect_bluesky.py`).
  Le sel est fourni par la variable d'environnement `PSEUDO_SALT` ; **modifiez
  la valeur par défaut en production**.
- **Droit d'opposition (RGPD art. 21)** — les comptes listés dans
  `data/excluded_handles.txt` sont exclus de la collecte, liste rechargée à
  chaque cycle.

## Secrets

Aucun secret ne doit être versionné. `.env` est ignoré par git ; partez de
`.env.example`.

Variables sensibles attendues à l'exécution :

```
MONGO_USER, MONGO_PASSWORD      # accès base
BLUESKY_HANDLE, BLUESKY_PASSWORD # collecte
PSEUDO_SALT                     # pseudonymisation RGPD
```

Si un secret a été exposé par mégarde, considérez-le compromis : révoquez-le
et faites-le tourner. Le retirer de l'historique git ne suffit pas.
