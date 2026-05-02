"""Tests for the Thumalien FastAPI API."""

import sys
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def client():
    """Create a TestClient with a mocked detector."""
    # Import inside fixture so patching works cleanly
    from api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def mock_detector():
    """Return a mock ExpertFakeNewsDetector with a predict method."""
    det = MagicMock()
    det.is_trained = True
    det.predict.return_value = pd.DataFrame(
        {
            "ai_score_credibility": [0.82],
            "prediction_label": [0],
            "language": ["fr"],
        }
    )
    return det


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------
def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_empty_text(client):
    resp = client.post("/predict", json={"text": "", "lang": "auto"})
    assert resp.status_code == 422  # Pydantic validation (min_length=1)


def test_predict_too_long_text(client):
    resp = client.post("/predict", json={"text": "a" * 10001, "lang": "auto"})
    assert resp.status_code == 422  # Pydantic validation (max_length=10000)


def test_predict_valid_text(client, mock_detector):
    import api.main as api_module

    original = api_module.detector
    api_module.detector = mock_detector
    try:
        resp = client.post(
            "/predict", json={"text": "Ceci est un article de test.", "lang": "auto"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "fiable"
        assert data["score"] == 0.82
        assert data["language"] == "fr"
        assert "emotions" in data
        mock_detector.predict.assert_called_once()
    finally:
        api_module.detector = original


def test_predict_no_model(client):
    """When no model is loaded, /predict should return 503."""
    import api.main as api_module

    original = api_module.detector
    api_module.detector = None
    try:
        resp = client.post(
            "/predict", json={"text": "Some text.", "lang": "auto"}
        )
        assert resp.status_code == 503
    finally:
        api_module.detector = original
