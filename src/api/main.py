"""
ThumaCheck -- API FastAPI minimale pour la detection de desinformation.

Lancement :
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline.expert_detector import ExpertFakeNewsDetector, EmotionFeatureExtractor

logger = logging.getLogger(__name__)

# Emotion labels for API response
EMOTION_LABELS = ['colere', 'degout', 'joie', 'neutre', 'peur', 'surprise', 'tristesse']

# ---------------------------------------------------------------------------
#  Global detector + emotion instances
# ---------------------------------------------------------------------------
detector: Optional[ExpertFakeNewsDetector] = None
emotion_extractor: Optional[EmotionFeatureExtractor] = None


def _load_detector() -> Optional[ExpertFakeNewsDetector]:
    """Try loading the best available model (V5 -> V4 -> V3 -> expert)."""
    model_dir = os.environ.get(
        "THUMALIEN_MODEL_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "models"),
    )
    model_dir = os.path.abspath(model_dir)

    suffixes = ["expert_v5", "expert_v4", "expert_v3", "expert"]
    for suffix in suffixes:
        model_path = os.path.join(model_dir, f"model_{suffix}.pkl")
        if os.path.exists(model_path):
            try:
                det = ExpertFakeNewsDetector(model_dir=model_dir)
                det.load(suffix=suffix)
                logger.info("Model loaded: %s", suffix)
                return det
            except Exception:
                logger.exception("Failed to load model %s", suffix)
    logger.warning("No model files found in %s", model_dir)
    return None


def _load_emotion_extractor() -> Optional[EmotionFeatureExtractor]:
    """Load emotion feature extractor for API responses."""
    model_dir = os.environ.get(
        "THUMALIEN_MODEL_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "models"),
    )
    model_dir = os.path.abspath(model_dir)
    try:
        emo = EmotionFeatureExtractor(model_dir=model_dir)
        if emo.load():
            logger.info("Emotion extractor loaded")
            return emo
    except Exception:
        logger.warning("Emotion extractor not available")
    return None


# ---------------------------------------------------------------------------
#  Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, emotion_extractor
    detector = _load_detector()
    emotion_extractor = _load_emotion_extractor()
    yield


# ---------------------------------------------------------------------------
#  App
# ---------------------------------------------------------------------------
app = FastAPI(title="ThumaCheck API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
#  Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    lang: str = Field(default="auto")


class PredictResponse(BaseModel):
    score: float
    label: str
    language: str
    emotions: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    emotions_loaded: bool


# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=detector is not None,
        emotions_loaded=emotion_extractor is not None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must be non-empty")

    results = detector.predict(pd.Series([text]))

    score = float(results["ai_score_credibility"].iloc[0])
    pred_label = int(results["prediction_label"].iloc[0])
    language = str(results["language"].iloc[0])

    # Emotion probabilities
    emotions: Dict[str, float] = {}
    if emotion_extractor is not None:
        try:
            import numpy as np
            probas = emotion_extractor.get_emotion_features([text])[0]
            emotions = {
                EMOTION_LABELS[i]: round(float(probas[i]), 4)
                for i in range(len(EMOTION_LABELS))
            }
        except Exception:
            logger.warning("Emotion extraction failed for predict request")

    return PredictResponse(
        score=score,
        label="fiable" if pred_label == 0 else "suspect",
        language=language,
        emotions=emotions,
    )
