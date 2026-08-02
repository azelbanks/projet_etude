# Journal des modifications — remise en état de la chaîne qualité

**Période :** 1er – 2 août 2026
**Périmètre :** `azelbanks/thumacheck`, de `22d52fc` à `f40862d`
**Volumétrie :** 86 fichiers modifiés, 5 611 insertions, 3 196 suppressions

Ce document recense chaque modification et sa justification. Il est destiné à
servir de trace : ce qui a été changé, pourquoi, et ce qui a été délibérément
laissé de côté.

---

## Sommaire

1. [Contexte et déclencheur](#1-contexte-et-déclencheur)
2. [Bugs corrigés](#2-bugs-corrigés)
3. [Infrastructure d'intégration continue](#3-infrastructure-dintégration-continue)
4. [Qualité de code et typage](#4-qualité-de-code-et-typage)
5. [Tests et couverture](#5-tests-et-couverture)
6. [Déploiement continu et exploitation](#6-déploiement-continu-et-exploitation)
7. [Reproductibilité des artefacts](#7-reproductibilité-des-artefacts)
8. [Gouvernance et licence](#8-gouvernance-et-licence)
9. [Transparence du cadre projet](#9-transparence-du-cadre-projet)
10. [Ce qui n'a pas été traité](#10-ce-qui-na-pas-été-traité)
11. [Journal des commits](#11-journal-des-commits)
12. [État avant / après](#12-état-avant--après)

---

## 1. Contexte et déclencheur

Point de départ : une accumulation de courriels d'échec de CI. Le diagnostic a
révélé **27 exécutions en échec consécutives** et deux problèmes distincts.

**Le workflow `tests` était cassé depuis le 1er août, 12h08.** Le commit
`2fde4d75` (« refactor: split God Object ») avait découpé un module monolithique
en plusieurs fichiers en omettant des imports. Le dernier CI vert datait de
11h36, juste avant ce refactor. Les commits suivants — dont
`fix: move mlflow service from networks to services block`, celui qui a motivé
l'alerte — héritaient d'une branche déjà cassée sans en être la cause.

**Le workflow `CI` n'avait jamais réussi.** Zéro succès sur onze exécutions
depuis sa création. Il avait été ajouté avec `ruff check` et
`ruff format --check` sur une base qui n'avait jamais été passée à ruff.

> **Constat de fond.** Une CI durablement rouge cesse d'être un signal : on
> apprend à ignorer les alertes. C'est ce mécanisme qui a permis au refactor
> cassé de rester en place plusieurs jours.

---

## 2. Bugs corrigés

Cinq défauts réels, dont trois que la CI ne pouvait structurellement pas
détecter.

### 2.1 `pd` non importé — package entier inutilisable

**Fichier :** `src/pipeline/language_router.py`

```python
import logging
from typing import Optional      # pandas jamais importé
...
def detect_batch(cls, texts: pd.Series) -> pd.Series:   # NameError
```

L'annotation `pd.Series` est évaluée à la définition de la classe. Comme
`src/pipeline/__init__.py` importe cette classe, **tout le package devenait
inimportable** — d'où les 14 fichiers de tests en échec de collecte.

**Correction :** ajout de `import pandas as pd`, retrait de `Optional` inutilisé.

### 2.2 `pickle` non importé — crash silencieux en production

**Fichier :** `src/pipeline/emotion_classifier.py`

```python
def load(self) -> bool:
    if not all(os.path.exists(p) for p in [pt_path, vocab_path, le_path]):
        return False                    # ← CI : sortie ici, jamais détecté
    with open(vocab_path, 'rb') as f:
        self.vocab = pickle.load(f)     # ← production : NameError
```

**Le pire profil de bug possible.** En CI les modèles sont absents, la fonction
sort avant la ligne fautive : invisible. En production les fichiers existent :
`NameError`.

Aggravé par l'appelant dans `src/api/main.py` :

```python
except Exception:
    logger.warning("Emotion extractor not available")   # ← suggère un fichier manquant
```

L'erreur de code était déguisée en indisponibilité de fichier. **L'API tournait
avec l'analyse d'émotions désactivée en permanence**, sans que rien ne le
signale correctement.

**Vérification :** le code d'avant correctif a été rejoué contre les vrais
modèles présents dans le dépôt →
`NameError: name 'pickle' is not defined`. Après correctif → `load()` retourne
`True` et produit les 7 probabilités d'émotion.

### 2.3 `load()` déclarait `-> bool` sans retourner

**Fichier :** `src/pipeline/detector.py`

```python
def load(self, suffix: str = "expert") -> bool:
    ...
    logger.info("Modèle chargé depuis %s", self.model_dir)
    # aucun return → renvoie None
```

Tout appelant écrivant `if detector.load():` recevait `None`, donc « échec »,
alors que le chargement avait réussi. Violait également `ClassifierProtocol`,
le contrat défini par le projet lui-même dans `src/pipeline/protocols.py`.

Détecté par mypy dès sa première exécution effective.

### 2.4 `validate_text()` mentait sur son type de retour

**Fichier :** `src/collection/collect_bluesky.py`

```python
def validate_text(text: str) -> bool:      # annonce bool
    return False, "empty_or_missing"       # renvoie un tuple
```

Même défaut que `extract_metadata()`, corrigé au passage (`-> dict` pour un
triplet). Les appelants dépaquetaient déjà correctement : c'est l'annotation
qui était fausse, pas le code.

### 2.5 Test pointant un emplacement obsolète

**Fichier :** `tests/test_expert_detector_extended.py`

Le test patchait `pipeline.expert_detector.joblib`, alors que le refactor avait
déplacé `joblib` dans `pipeline.detector`. Échec préexistant, sans lien avec les
autres corrections.

### 2.6 Fragilités corrigées au passage

| Fichier | Défaut |
|---|---|
| `explainability/integrated_gradients.py` | dépaquetage de `best` sans vérifier qu'il est renseigné → `TypeError` si aucune stratégie ne converge |
| `explainability/shap_global.py` | accès à `self._X` avant `fit()` |
| `pipeline/detector.py` | `min_df = 3 if bilingual else 3` — condition sans effet |

---

## 3. Infrastructure d'intégration continue

### 3.1 Ruff non épinglé et sans configuration

`pip install ruff` installait la dernière version disponible, et l'absence de
section `[tool.ruff]` laissait ruff appliquer ses règles par défaut — **qui
évoluent à chaque publication**. La CI pouvait donc échouer sans qu'aucune ligne
de code n'ait changé.

**Correction :** jeu de règles explicite dans `pyproject.toml`
(`E`, `F`, `I`, `UP`, `B`, `C4`, `SIM`, `RUF`) et version figée à `ruff==0.16.1`.

### 3.2 Une étape incapable d'échouer

```yaml
- run: pytest tests/ --tb=short -q --co 2>&1 | tail -1
```

Le code de sortie était celui de `tail`, jamais de `pytest`. L'étape
« Collect tests » passait même quand la collecte plantait.

**Correction :** `set -o pipefail` en préfixe.

### 3.3 mypy installé mais jamais exécuté

Le job s'appelait `lint-and-type-check` et installait mypy… sans jamais le
lancer. Voir section 4.

### 3.4 Exclusions de tests levées

Les workflows ignoraient cinq fichiers :

```
test_benchmark_latence, test_pipeline_integration,
test_inference_cycle, test_collection_cycle, test_expert_detector
```

Vérification faite, **les 46 tests concernés passent en environ 5 secondes**.
Ce sont précisément les tests d'intégration — donc la catégorie qui aurait
détecté le blocage du 1er août.

**Correction :** exclusions supprimées, suite complète exécutée.

### 3.5 Node.js 20 déprécié

`actions/checkout@v4` et `actions/setup-python@v5` passés en `@v5` / `@v6`.
Ce n'était pas la cause des échecs, mais c'était l'avertissement présent dans
les courriels d'alerte.

---

## 4. Qualité de code et typage

### 4.1 Lint : 408 erreurs → 0

327 corrigées automatiquement, puis un passage `ruff format` sur 56 fichiers.

**Le formatage a été isolé dans un commit distinct** (`f017b44`) : mélanger
56 fichiers de reformatage avec des corrections de bugs rend toute relecture
impossible. Un fichier `.git-blame-ignore-revs` a été ajouté pour que GitHub
saute ce commit dans `git blame`.

**Attention portée aux faux positifs.** Trois `F401` « import inutilisé »
étaient des **sondes de dépendance optionnelle** :

```python
try:
    from codecarbon import EmissionsTracker   # l'import EST le test
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
```

Les supprimer aurait cassé la détection. Celle qui le nécessitait a été
conservée avec un `# noqa: F401` commenté ; les deux réellement mortes ont été
retirées.

### 4.2 Gestion des exceptions rendue observable

14 blocs `except Exception` absorbaient l'erreur sans laisser de trace. C'est ce
motif qui a masqué le `NameError` de `pickle` pendant plusieurs jours.

**Principe retenu :** le défaut n'est pas d'attraper large — c'est de ne rien
laisser voir. Restreindre les types d'exception aurait modifié le comportement
sans bénéfice ; journaliser la trace le rend diagnosticable.

```python
except Exception:
    logger.exception("message")             # chemin critique
except Exception:
    logger.debug("message", exc_info=True)  # chemin best-effort
```

**Résultat :** plus aucun `except: pass` muet dans `src/`.

### 4.3 Typage : 135 erreurs mypy → 0

Approche en deux temps. D'abord un **cliquet** — barrière sur les 18 modules
déjà propres, dette listée explicitement — puis assainissement complet des
15 modules restants.

Le motif dominant (75 des 135 erreurs) :

```python
self.model = None          # mypy infère le type `None`
...
self.model.predict(X)      # « None n'a pas d'attribut predict »
```

**Corrections apportées :**

- attributs initialisés à `None` annotés en optionnel (`_EmotionMLP | None`) ;
- `object` remplacé par `Any` là où un attribut est réellement accédé — `object`
  interdit tout accès et ne traduisait pas l'intention ;
- **gardes portant sur les attributs réellement utilisés** plutôt que sur un
  drapeau susceptible de diverger de l'état effectif :

```python
# avant
if not self._loaded:
    raise RuntimeError(...)
oov_idx = self.vocab.get(...)          # self.vocab peut être None

# après
if not self._loaded or self.model is None or self.vocab is None:
    raise RuntimeError(...)
```

L'essentiel est déclaratif, **sans effet à l'exécution**. Les seules
modifications de comportement sont les gardes ajoutées, qui remplacent un plantage
obscur par une erreur explicite.

mypy couvre désormais **les 33 modules de `src/` sans aucune exclusion**, et
s'exécute en CI.

---

## 5. Tests et couverture

| | Avant | Après |
|---|---|---|
| Tests | 525 (5 fichiers exclus) | **626** (aucune exclusion) |
| Couverture globale | 79,7 % | **82,0 %** |
| Barrière | 75 % | **78 %** |
| `protocols.py` | 0 % | **100 %** |
| `mlflow_tracker.py` | 0 % | **100 %** |
| `roberta_en_classifier.py` | 23 % | **33 %** |

### Fichiers de tests ajoutés

| Fichier | Objet |
|---|---|
| `test_protocols.py` | contrat d'interface (Strategy pattern) — aucun test n'existait |
| `test_roberta_en_classifier.py` | dataset, tête de classification, mode dégradé, gardes d'état |
| `test_mlflow_tracker.py` | seule brique MLOps, non testée — les deux chemins couverts |
| `test_api_deployment.py` | endpoints `/version` et `/ready` |
| `test_models_manifest.py` | manifeste d'artefacts |

**Limite assumée sur RoBERTa :** `fine_tune`, `_evaluate` et la sérialisation
des checkpoints exigent le modèle transformer réel. Les couvrir avec des mocks
creux aurait fait monter le pourcentage sans rien tester.

---

## 6. Déploiement continu et exploitation

### 6.1 Workflow `release.yml`

Construit l'image de l'API et la publie sur GitHub Container Registry :

- push sur `main` → tags `main` et `sha-<court>`
- tag git `v*` → tags semver + `latest`

Un job **smoke-test** démarre ensuite l'image *publiée*, attend `/health` et
vérifie que `/version` annonce bien la révision attendue. **Une image cassée ne
peut pas atteindre un déploiement sans que la CI le signale.**

### 6.2 Endpoints d'exploitation

| Endpoint | Rôle |
|---|---|
| `/health` | **vivacité** — 200 dès que le processus tourne |
| `/ready` | **disponibilité** — 200 seulement si une prédiction est possible, sinon **503** |
| `/version` | révision git, modèle chargé, état de cascade, date de build |

La distinction `/health` / `/ready` n'est pas cosmétique : un orchestrateur qui
route sur `/health` envoie du trafic à une instance dont le modèle n'est pas
encore chargé. Le `HEALTHCHECK` de l'image sonde `/ready`.

Sans `/version`, **il est impossible de vérifier qu'un déploiement ou un
rollback a pris effet** — on ne sait pas quelle révision sert le trafic.

### 6.3 Mode dégradé rendu explicite

Les poids CamemBERT et RoBERTa dépassent la limite de 100 Mo de GitHub et sont
gitignorés. **Tout clone frais démarrait donc en cascade dégradée, en silence**,
alors que le README annonce des F1 de 0,957 et 0,874.

`/health` expose désormais l'état :

```json
{"model_loaded": true, "cascade_full": false,
 "cascade_missing": ["camembert_fr.pt", "roberta_en.pt"]}
```

Le README documente la situation et la marche à suivre.

---

## 7. Reproductibilité des artefacts

`scripts/models_manifest.py` fige l'empreinte SHA-256 et la taille des
**29 artefacts versionnés** dans `models/MANIFEST.json`, et documente les
**5 poids volontairement absents**.

```bash
python scripts/models_manifest.py status   # cascade complète ou dégradée
python scripts/models_manifest.py verify   # exécuté par la CI
python scripts/models_manifest.py generate # après modification volontaire
```

La CI exécute `verify` : tout artefact altéré, supprimé ou ajouté sans suivi
fait échouer le build.

> **Incident lors de la mise en place.** Le manifeste avait été généré et
> vérifié sur le disque de travail, mais la règle `*.json` du `.gitignore`
> l'empêchait d'être suivi par git : il n'a jamais été commité, et les trois
> workflows ont échoué. Le test `test_repo_manifest_is_current` a détecté
> l'incohérence — il a fait exactement son travail.
>
> **Leçon retenue :** vérifier sur un **clone git frais**, et non sur le
> répertoire de travail. C'est la différence entre « ça marche chez moi » et
> « c'est livrable ».

---

## 8. Gouvernance et licence

### 8.1 LICENSE (MIT)

Le dépôt était public **sans licence**, donc en « tous droits réservés » par
défaut : personne ne pouvait légalement l'utiliser, le forker ou y contribuer.
Public n'est pas synonyme d'open source.

La licence précise sa **portée** : elle couvre le code, **pas** les poids de
modèles (soumis aux licences CamemBERT/RoBERTa), ni les jeux de données, ni les
données Bluesky collectées (protocole AT et RGPD).

### 8.2 Protection de la branche `main`

```
✓ PR obligatoire            ✓ historique linéaire
✓ 3 vérifications requises  ✓ force-push et suppression interdits
✓ appliquée aux admins      ✓ 0 approbation requise (travail solo)
```

> **Point important.** La configuration par défaut (`enforce_admins: false`)
> laisse les administrateurs contourner la règle — sur un dépôt à un seul
> propriétaire, elle est purement décorative. Détecté par un push de test qui
> est passé avec le message `Bypassed rule violations`. Corrigé, puis
> **revérifié : le push est désormais rejeté**.

Deux commits parasites (`1a5f6e5` et son revert `38082b2`) subsistent dans
l'historique — ce sont ces tests. Le contenu est identique à l'état antérieur ;
seule la trace demeure.

### 8.3 CONTRIBUTING.md et SECURITY.md

`CONTRIBUTING.md` consigne les conventions apprises pendant cette session :
épingler les versions d'outils, ne jamais ajouter d'exclusion mypy, ne jamais
laisser un `except` muet.

`SECURITY.md` couvre le signalement de faille et la gestion des secrets — dont
un point relevé à la rédaction : **`PSEUDO_SALT` a une valeur par défaut en dur
dans le code**. Pour de la pseudonymisation RGPD, elle doit impérativement être
changée en production.

---

## 9. Transparence du cadre projet

Thumalien (client), Niamato Consulting (agence) et Sébastien Lazcanotegui
(co-auteur) sont des **entités fictives**.

Sur un dépôt public, la formulation d'origine — avec lien vers un domaine
externe — se lisait comme une prestation commerciale réelle.

**Le scénario est conservé** : il structure le cahier des charges, l'analyse
RGPD, la planification et la répartition des rôles. Il est désormais annoncé
comme tel :

- encadré en tête de README distinguant ce qui est fictif (le cadre) de ce qui
  est réel (code, modèles, données, métriques, analyses) ;
- rappel sous la section des contributions, où des travaux précis étaient
  attribués à un co-auteur fictif ;
- lien vers `thumalien.com` retiré : pointer vers un domaine externe tout en
  déclarant l'entité fictive serait contradictoire, et ce domaine peut
  appartenir à un tiers sans rapport.

Les 130 mentions dans `docs/` sont inchangées : l'encadré du README couvre
l'ensemble, et ces documents perdraient leur cohérence narrative si le
commanditaire en était retiré.

`LICENSE` : copyright ramené à **Azélie Bernard** seule.

---

## 10. Ce qui n'a pas été traité

| Point | Raison |
|---|---|
| **Déploiement effectif** vers un hébergeur | nécessite des identifiants. L'image est construite, publiée et testée — seule la cible manque. |
| **Publication des poids** CamemBERT / RoBERTa | absents du disque. Le manifeste les liste explicitement : l'absence est traçable plutôt que subie. |
| **Étude de transfert hors domaine** | nécessite de nouveaux jeux de données et du calcul d'entraînement. Travail de recherche, hors périmètre. |
| **Couverture de `roberta_en_classifier.py`** (33 %) | le reste exige le modèle transformer réel. |
| **Nettoyage des commits de test** `1a5f6e5` / `38082b2` | demanderait de désactiver la protection de branche pour réécrire l'historique. Coût supérieur au bénéfice. |

---

## 11. Journal des commits

| SHA | Objet |
|---|---|
| `e7fda44` | `fix:` répare la CI — imports manquants, config ruff épinglée, workflows durcis |
| `f017b44` | `style:` passe ruff format sur `src/` et `tests/` (formatage pur, isolé) |
| `7c98b1b` | `chore:` ignore le commit de formatage dans `git blame` |
| `c88008a` | `feat:` rend la CI réellement protectrice (tests complets, typage, observabilité) |
| `970b54a` | `chore:` licence MIT, typage intégral, docs de gouvernance — **PR #13** |
| `1a5f6e5` | `test:` vérification de la protection *(commit de test)* |
| `38082b2` | `Revert` du précédent |
| `91ac524` | `docs:` signale le cadre fictif, copyright à Azélie Bernard — **PR #14** |
| `f40862d` | `feat:` CD, endpoints d'exploitation et manifeste d'artefacts — **PR #15** |

---

## 12. État avant / après

| | 1er août | 2 août |
|---|---|---|
| Workflow `CI` | **0 succès / 11 exécutions** | vert, 4 étapes |
| Workflow `tests` | en échec depuis 12h08 | vert |
| Tests exécutés | 525 (5 fichiers exclus) | **626** (aucune exclusion) |
| Couverture | 79,7 % | **82,0 %** (barrière 78 %) |
| Lint | 408 erreurs | **0**, ruff épinglé `0.16.1` |
| Typage | mypy installé, jamais lancé | **33 modules, 0 erreur**, exécuté en CI |
| `except: pass` muets | 14 | **0** |
| Licence | aucune | MIT, © Azélie Bernard |
| Branche `main` | libre | protégée, admins inclus, **vérifiée** |
| Déploiement continu | aucun | image GHCR + smoke-test |
| Artefacts de modèles | non vérifiables | 29 empreintés, contrôlés en CI |
| Mode dégradé | silencieux | exposé sur `/health` et documenté |
| Cadre fictif | présenté comme réel | signalé en tête de README |

---

*Document rédigé le 2 août 2026. Chaque affirmation chiffrée a été vérifiée par
exécution — lint, typage, tests, couverture — et non estimée.*
