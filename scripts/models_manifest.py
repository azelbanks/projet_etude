#!/usr/bin/env python3
"""
Manifeste verifiable des artefacts de modeles.

Pourquoi
--------
Les poids CamemBERT et RoBERTa depassent la limite de 100 Mo de GitHub et ne
sont pas versionnes : un clone frais tourne en cascade degradee. Rien ne
permettait jusqu'ici de savoir *quels* artefacts sont presents, *lesquels*
manquent, ni de verifier que ceux presents n'ont pas ete alteres.

Ce script produit et verifie `models/MANIFEST.json` : empreinte SHA-256 et
taille de chaque artefact versionne, plus la liste explicite des poids
optionnels attendus.

Usage
-----
    python scripts/models_manifest.py generate   # (re)genere le manifeste
    python scripts/models_manifest.py verify     # verifie l'existant (exit 1 si ecart)
    python scripts/models_manifest.py status     # etat lisible de la cascade
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
MANIFEST_PATH = os.path.join(MODELS_DIR, "MANIFEST.json")

# Poids trop volumineux pour GitHub (> 100 Mo), volontairement non versionnes.
# Leur absence degrade la precision sans empecher le service de repondre.
OPTIONAL_WEIGHTS = {
    "camembert_fr.pt": "CamemBERT FR fine-tune — etage FR de la cascade V9",
    "camembert_fr_v2.pt": "CamemBERT FR v2 — variante textes ultra-courts",
    "camembert_best.pt": "CamemBERT — meilleur checkpoint d'entrainement",
    "roberta_en.pt": "RoBERTa EN fine-tune — etage EN de la cascade V9",
    "roberta_en_v2.pt": "RoBERTa EN v2 — +4.3 % F1 vs v1",
}

# Extensions considerees comme des artefacts de modele.
TRACKED_SUFFIXES = (".pkl", ".pt", ".joblib", ".pickle", ".json")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tracked_files() -> list[str]:
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(
        name
        for name in os.listdir(MODELS_DIR)
        if name.endswith(TRACKED_SUFFIXES)
        and name != os.path.basename(MANIFEST_PATH)
        and name not in OPTIONAL_WEIGHTS
        and os.path.isfile(os.path.join(MODELS_DIR, name))
    )


def build_manifest() -> dict:
    """Construit le manifeste a partir du contenu actuel de models/."""
    artifacts = {}
    for name in _tracked_files():
        path = os.path.join(MODELS_DIR, name)
        artifacts[name] = {
            "sha256": _sha256(path),
            "bytes": os.path.getsize(path),
        }
    return {
        "versioned_artifacts": artifacts,
        "optional_weights": OPTIONAL_WEIGHTS,
        "note": (
            "Les poids listes dans optional_weights depassent la limite de "
            "100 Mo de GitHub et ne sont pas versionnes. Leur absence place la "
            "cascade en mode degrade — voir la section « Modeles non "
            "versionnes » du README."
        ),
    }


def load_manifest() -> dict:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def verify() -> int:
    """Compare le disque au manifeste. Retourne un code de sortie."""
    if not os.path.exists(MANIFEST_PATH):
        print(f"ERREUR: {MANIFEST_PATH} absent. Lancez `generate`.", file=sys.stderr)
        return 1

    expected = load_manifest()["versioned_artifacts"]
    problems: list[str] = []

    for name, meta in sorted(expected.items()):
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            problems.append(f"MANQUANT   {name}")
            continue
        actual = _sha256(path)
        if actual != meta["sha256"]:
            problems.append(
                f"ALTERE     {name}\n"
                f"           attendu {meta['sha256'][:16]}…\n"
                f"           obtenu  {actual[:16]}…"
            )

    extras = set(_tracked_files()) - set(expected)
    for name in sorted(extras):
        problems.append(f"NON SUIVI  {name} (absent du manifeste)")

    if problems:
        print("Verification du manifeste — ECHEC\n", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print(
            f"\n{len(problems)} ecart(s). Si le changement est voulu : "
            "python scripts/models_manifest.py generate",
            file=sys.stderr,
        )
        return 1

    print(f"Manifeste verifie — {len(expected)} artefacts conformes.")
    return 0


def status() -> int:
    """Etat lisible : ce qui est present, ce qui manque, cascade complete ou non."""
    manifest = load_manifest() if os.path.exists(MANIFEST_PATH) else build_manifest()

    versioned = manifest["versioned_artifacts"]
    total_mb = sum(m["bytes"] for m in versioned.values()) / 1048576
    print(f"Artefacts versionnes : {len(versioned)} ({total_mb:.1f} Mo)")

    missing = []
    print("\nPoids optionnels (non versionnes) :")
    for name, desc in sorted(OPTIONAL_WEIGHTS.items()):
        present = os.path.exists(os.path.join(MODELS_DIR, name))
        print(f"  [{'x' if present else ' '}] {name:<24} {desc}")
        if not present:
            missing.append(name)

    if missing:
        print(f"\nCascade DEGRADEE — {len(missing)} poids absent(s).")
        print("Le meta-learner TF-IDF et le modele d'emotions fonctionnent ;")
        print("les etages transformer sont desactives.")
    else:
        print("\nCascade COMPLETE.")
    return 0


def generate() -> int:
    manifest = build_manifest()
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    n = len(manifest["versioned_artifacts"])
    print(f"Manifeste ecrit : {MANIFEST_PATH} ({n} artefacts)")
    return 0


COMMANDS = {"generate": generate, "verify": verify, "status": status}


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd not in COMMANDS:
        print(f"Commande inconnue : {cmd}", file=sys.stderr)
        print(f"Disponibles : {', '.join(COMMANDS)}", file=sys.stderr)
        return 2
    return COMMANDS[cmd]()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
