"""
Tests du manifeste d'artefacts.

Le manifeste rend la reproductibilite verifiable : il fige l'empreinte SHA-256
de chaque artefact versionne et documente les poids volontairement absents.
La CI l'execute, ce qui detecte tout artefact altere ou ajoute sans suivi.
"""

import importlib.util
import json
import os
import sys

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "models_manifest.py")
_spec = importlib.util.spec_from_file_location("models_manifest", SCRIPT)
assert _spec and _spec.loader
mm = importlib.util.module_from_spec(_spec)
sys.modules["models_manifest"] = mm
_spec.loader.exec_module(mm)


@pytest.fixture
def fake_models(tmp_path, monkeypatch):
    """Repertoire models/ isole avec deux artefacts connus."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "model_test.pkl").write_bytes(b"contenu-a")
    (models / "metrics_test.pkl").write_bytes(b"contenu-b")
    monkeypatch.setattr(mm, "MODELS_DIR", str(models))
    monkeypatch.setattr(mm, "MANIFEST_PATH", str(models / "MANIFEST.json"))
    return models


class TestBuildManifest:
    def test_lists_tracked_artifacts(self, fake_models):
        m = mm.build_manifest()
        assert set(m["versioned_artifacts"]) == {"model_test.pkl", "metrics_test.pkl"}

    def test_records_sha256_and_size(self, fake_models):
        entry = mm.build_manifest()["versioned_artifacts"]["model_test.pkl"]
        assert len(entry["sha256"]) == 64
        assert entry["bytes"] == len(b"contenu-a")

    def test_ignores_non_artifact_extensions(self, fake_models):
        (fake_models / "notes.txt").write_text("hors perimetre")
        assert "notes.txt" not in mm.build_manifest()["versioned_artifacts"]

    def test_optional_weights_excluded_from_artifacts(self, fake_models):
        """Un poids optionnel present ne doit pas entrer dans les artefacts suivis."""
        (fake_models / "roberta_en.pt").write_bytes(b"gros-fichier")
        assert "roberta_en.pt" not in mm.build_manifest()["versioned_artifacts"]

    def test_optional_weights_are_documented(self, fake_models):
        assert "roberta_en.pt" in mm.build_manifest()["optional_weights"]

    def test_empty_models_dir_yields_empty_manifest(self, tmp_path, monkeypatch):
        empty = tmp_path / "vide"
        empty.mkdir()
        monkeypatch.setattr(mm, "MODELS_DIR", str(empty))
        assert mm.build_manifest()["versioned_artifacts"] == {}


class TestVerify:
    def test_passes_on_freshly_generated_manifest(self, fake_models, capsys):
        mm.generate()
        assert mm.verify() == 0

    def test_detects_altered_artifact(self, fake_models, capsys):
        mm.generate()
        (fake_models / "model_test.pkl").write_bytes(b"contenu-modifie")
        assert mm.verify() == 1
        assert "ALTERE" in capsys.readouterr().err

    def test_detects_missing_artifact(self, fake_models, capsys):
        mm.generate()
        (fake_models / "model_test.pkl").unlink()
        assert mm.verify() == 1
        assert "MANQUANT" in capsys.readouterr().err

    def test_detects_untracked_artifact(self, fake_models, capsys):
        mm.generate()
        (fake_models / "model_surprise.pkl").write_bytes(b"nouveau")
        assert mm.verify() == 1
        assert "NON SUIVI" in capsys.readouterr().err

    def test_fails_when_manifest_absent(self, fake_models, capsys):
        assert mm.verify() == 1
        assert "absent" in capsys.readouterr().err


class TestGenerateAndStatus:
    def test_generate_writes_valid_json(self, fake_models):
        mm.generate()
        data = json.loads((fake_models / "MANIFEST.json").read_text(encoding="utf-8"))
        assert "versioned_artifacts" in data
        assert "optional_weights" in data

    def test_generate_is_deterministic(self, fake_models):
        mm.generate()
        first = (fake_models / "MANIFEST.json").read_text(encoding="utf-8")
        mm.generate()
        assert (fake_models / "MANIFEST.json").read_text(encoding="utf-8") == first

    def test_status_reports_degraded_cascade(self, fake_models, capsys):
        assert mm.status() == 0
        assert "DEGRADEE" in capsys.readouterr().out

    def test_status_reports_complete_cascade(self, fake_models, capsys):
        for name in mm.OPTIONAL_WEIGHTS:
            (fake_models / name).write_bytes(b"x")
        mm.status()
        assert "COMPLETE" in capsys.readouterr().out


class TestCli:
    def test_unknown_command_returns_2(self, capsys):
        assert mm.main(["prog", "inexistante"]) == 2

    def test_defaults_to_status(self, fake_models, capsys):
        assert mm.main(["prog"]) == 0
        assert "Artefacts versionnes" in capsys.readouterr().out


class TestRealRepository:
    """Le manifeste reel du depot doit etre a jour."""

    def test_repo_manifest_is_current(self):
        import subprocess

        result = subprocess.run(
            [sys.executable, SCRIPT, "verify"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
