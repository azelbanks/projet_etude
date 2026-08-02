# Contribuer à ThumaCheck

Merci de l'intérêt porté au projet. Ce document décrit le fonctionnement
concret du dépôt.

## Mise en route

```bash
git clone https://github.com/azelbanks/thumacheck.git
cd thumacheck
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov ruff==0.16.1 mypy==2.3.0
```

Les poids CamemBERT et RoBERTa ne sont pas versionnés (limite de 100 Mo de
GitHub). Le projet démarre sans eux, en cascade dégradée — voir la section
« Modèles non versionnés » du README.

## Flux de travail

`main` est protégée : ni push direct, ni force-push. Tout passe par une pull
request dont la CI doit être verte.

```bash
git checkout -b fix/description-courte
# ... modifications ...
gh pr create --fill
gh pr merge --auto --squash     # fusionne des que la CI passe
```

## Avant de proposer une PR

Les quatre commandes que la CI exécutera, dans l'ordre :

```bash
ruff check src/ tests/           # lint
ruff format src/ tests/          # formatage (modifie les fichiers)
mypy                             # types — aucune erreur toleree
pytest tests/ -q                 # 575 tests, aucune exclusion
```

Et la barrière de couverture :

```bash
pytest tests/ --cov=src --cov-report=term-missing
coverage report --fail-under=78
```

## Conventions

**Versions épinglées.** `ruff` et `mypy` sont fixés à une version précise dans
`.github/workflows/ci.yml`. Une version flottante fait échouer la CI sans
qu'aucun code n'ait changé — c'est arrivé, ne le refaites pas.

**Typage.** `mypy` couvre l'intégralité de `src/` sans exception. Si une
nouvelle erreur apparaît, corrigez-la plutôt que d'ajouter une exclusion.

**Gestion des exceptions.** Un bloc `except` qui absorbe l'erreur doit
journaliser la trace :

```python
except Exception:
    logger.exception("message")            # chemin critique
except Exception:
    logger.debug("message", exc_info=True) # chemin best-effort
```

Jamais de `except: pass` muet — c'est ce qui a masqué un `NameError` en
production pendant plusieurs jours.

**Messages de commit.** Préfixe conventionnel (`fix:`, `feat:`, `docs:`,
`refactor:`, `test:`, `chore:`, `style:`) puis une ligne de résumé, et un corps
expliquant *pourquoi* si ce n'est pas évident.

**Formatage.** Les commits de reformatage sont isolés des commits de fond et
référencés dans `.git-blame-ignore-revs`.

## Structure des tests

Un fichier de test par module, nommé `tests/test_<module>.py`. Les tests ne
doivent dépendre ni du réseau, ni de MongoDB, ni des poids de modèles : la CI
tourne sans aucun des trois.

## Signaler un problème

Ouvrez une issue avec la version de Python, la commande lancée et la trace
complète. Pour une faille de sécurité, voir [SECURITY.md](SECURITY.md).
