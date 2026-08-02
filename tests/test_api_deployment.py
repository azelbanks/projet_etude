"""
Tests des endpoints d'exploitation : /version et /ready.

Ces deux endpoints rendent un deploiement pilotable :
- /version identifie la revision et le modele qui servent reellement le trafic
  (verification d'un deploiement, constat d'un rollback) ;
- /ready distingue « le processus tourne » de « l'instance peut repondre ».
  Un orchestrateur qui route sur /health envoie du trafic a une instance dont
  le modele n'est pas charge.
"""

import os
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from api import main as api_main
from api.main import app


class TestVersionEndpoint:
    def test_returns_200_without_model(self):
        """/version doit repondre meme si aucun modele n'est charge."""
        with patch.object(api_main, "detector", None):
            r = TestClient(app).get("/version")
        assert r.status_code == 200

    def test_exposes_api_version(self):
        r = TestClient(app).get("/version")
        assert r.json()["api_version"] == app.version

    def test_git_sha_falls_back_to_unknown(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GIT_SHA", None)
            r = TestClient(app).get("/version")
        assert r.json()["git_sha"] == "unknown"

    def test_git_sha_read_from_environment(self):
        with patch.dict(os.environ, {"GIT_SHA": "abc1234"}):
            r = TestClient(app).get("/version")
        assert r.json()["git_sha"] == "abc1234"

    def test_build_time_read_from_environment(self):
        with patch.dict(os.environ, {"BUILD_TIME": "2026-08-02T10:00:00Z"}):
            r = TestClient(app).get("/version")
        assert r.json()["build_time"] == "2026-08-02T10:00:00Z"

    def test_model_suffix_reflects_loaded_model(self):
        with patch.object(api_main, "_loaded_model_suffix", "expert_v5"):
            r = TestClient(app).get("/version")
        assert r.json()["model_suffix"] == "expert_v5"

    def test_cascade_full_false_when_weights_missing(self, tmp_path):
        """Poids transformer absents : la cascade est signalee incomplete."""
        with patch.dict(os.environ, {"THUMALIEN_MODEL_DIR": str(tmp_path)}):
            r = TestClient(app).get("/version")
        assert r.json()["cascade_full"] is False

    def test_cascade_full_true_when_weights_present(self, tmp_path):
        (tmp_path / "camembert_fr.pt").write_bytes(b"x")
        (tmp_path / "roberta_en.pt").write_bytes(b"x")
        with patch.dict(os.environ, {"THUMALIEN_MODEL_DIR": str(tmp_path)}):
            r = TestClient(app).get("/version")
        assert r.json()["cascade_full"] is True

    def test_response_shape_is_stable(self):
        """Le contrat est consomme par l'outillage de deploiement."""
        keys = set(TestClient(app).get("/version").json())
        assert keys == {"api_version", "git_sha", "model_suffix", "cascade_full", "build_time"}


class TestReadyEndpoint:
    def test_503_when_model_not_loaded(self):
        with patch.object(api_main, "detector", None):
            r = TestClient(app).get("/ready")
        assert r.status_code == 503
        assert r.json()["ready"] is False
        assert r.json()["reason"] == "model_not_loaded"

    def test_200_when_model_loaded(self):
        with patch.object(api_main, "detector", object()):
            r = TestClient(app).get("/ready")
        assert r.status_code == 200
        assert r.json()["ready"] is True
        assert r.json()["reason"] is None

    def test_differs_from_health_when_degraded(self):
        """/health repond 200 (le process vit) alors que /ready refuse le trafic."""
        with patch.object(api_main, "detector", None):
            c = TestClient(app)
            assert c.get("/health").status_code == 200
            assert c.get("/ready").status_code == 503
