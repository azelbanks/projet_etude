"""
ThumaCheck -- API FastAPI minimale pour la detection de desinformation.

Lancement :
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import collections
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from pipeline.expert_detector import EmotionFeatureExtractor, ExpertFakeNewsDetector

try:
    from codecarbon import EmissionsTracker

    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

logger = logging.getLogger(__name__)

# Emotion labels for API response
EMOTION_LABELS = ["colere", "degout", "joie", "neutre", "peur", "surprise", "tristesse"]

# ---------------------------------------------------------------------------
#  Global detector + emotion + energy instances
# ---------------------------------------------------------------------------
detector: ExpertFakeNewsDetector | None = None
emotion_extractor: EmotionFeatureExtractor | None = None

# Cumulative energy metrics for the API session
_energy_metrics: dict[str, Any] = {
    "total_requests": 0,
    "total_predict_requests": 0,
    "total_inference_time_s": 0.0,
    "co2_emissions_kg": 0.0,
    "energy_kwh": 0.0,
    "tracker_active": False,
}
_energy_tracker: Any = None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter per client IP (sliding window)."""

    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, collections.deque] = {}

    async def dispatch(self, request: Request, call_next: Any) -> JSONResponse:
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        if client_ip not in self._requests:
            self._requests[client_ip] = collections.deque()

        dq = self._requests[client_ip]
        # Purge old entries
        while dq and dq[0] < now - self.window_seconds:
            dq.popleft()

        if len(dq) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded ({self.max_requests} req/{self.window_seconds}s)"
                },
            )

        dq.append(now)
        return await call_next(request)


class EnergyTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware that tracks inference time per request for energy accounting."""

    async def dispatch(self, request: Request, call_next: Any) -> JSONResponse:
        _energy_metrics["total_requests"] += 1
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start

        if request.url.path == "/predict":
            _energy_metrics["total_predict_requests"] += 1
            _energy_metrics["total_inference_time_s"] += elapsed

        return response


def _load_detector() -> ExpertFakeNewsDetector | None:
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


def _missing_cascade_models() -> list[str]:
    """
    Poids optionnels de la cascade absents du disque.

    CamemBERT (FR) et RoBERTa (EN) pesent plus de 100 Mo et ne sont pas
    versionnes — voir .gitignore. Leur absence degrade la precision sans
    empecher le service de repondre.
    """
    model_dir = os.path.abspath(
        os.environ.get(
            "THUMALIEN_MODEL_DIR",
            os.path.join(os.path.dirname(__file__), "..", "..", "models"),
        )
    )
    return [
        name
        for name in ("camembert_fr.pt", "roberta_en.pt")
        if not os.path.exists(os.path.join(model_dir, name))
    ]


def _load_emotion_extractor() -> EmotionFeatureExtractor | None:
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
        # exception() et non warning() : un bug de code (import manquant,
        # attribut absent) doit rester lisible dans les logs.
        logger.exception("Emotion extractor not available")
    return None


# ---------------------------------------------------------------------------
#  Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, emotion_extractor, _energy_tracker
    detector = _load_detector()
    emotion_extractor = _load_emotion_extractor()

    # Start continuous energy tracker for the API session
    if CODECARBON_AVAILABLE:
        try:
            logs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
            os.makedirs(logs_dir, exist_ok=True)
            _energy_tracker = EmissionsTracker(
                project_name="ThumaCheck_API",
                output_dir=logs_dir,
                output_file="api_emissions.csv",
                log_level="error",
            )
            _energy_tracker.start()
            _energy_metrics["tracker_active"] = True
            logger.info("CodeCarbon API tracker started")
        except Exception:
            logger.exception("CodeCarbon tracker failed to start")
            _energy_tracker = None

    yield

    # Stop energy tracker on shutdown
    if _energy_tracker is not None:
        try:
            emissions = _energy_tracker.stop()
            if emissions is not None:
                _energy_metrics["co2_emissions_kg"] = float(emissions)
            logger.info("CodeCarbon API tracker stopped: %.6f kg CO2", emissions or 0)
        except Exception:
            logger.debug("CodeCarbon tracker stop failed", exc_info=True)


# ---------------------------------------------------------------------------
#  App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ThumaCheck API",
    version="1.0.0",
    description="Real-time misinformation detection API. CamemBERT (FR F1: 0.957), RoBERTa (EN F1: 0.874), V9 cascade pipeline.",
    lifespan=lifespan,
)

# CORS restrictif — origines explicites uniquement
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("THUMACHECK_CORS_ORIGINS", "http://localhost:8501").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(EnergyTrackingMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    max_requests=int(os.environ.get("THUMACHECK_RATE_LIMIT", "60")),
    window_seconds=60,
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
    emotions: dict[str, float]


class EnergyResponse(BaseModel):
    total_requests: int
    total_predict_requests: int
    total_inference_time_s: float
    co2_emissions_kg: float
    tracker_active: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    emotions_loaded: bool
    energy_tracking: bool
    # Les poids CamemBERT/RoBERTa depassent la limite de 100 Mo de GitHub et ne
    # sont donc pas versionnes : un clone frais tourne en cascade degradee.
    # Expose ici pour qu'un operateur puisse le constater sans lire les logs.
    cascade_full: bool
    cascade_missing: list[str]


# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    missing = _missing_cascade_models()
    return HealthResponse(
        status="ok",
        model_loaded=detector is not None,
        emotions_loaded=emotion_extractor is not None,
        energy_tracking=_energy_metrics["tracker_active"],
        cascade_full=not missing,
        cascade_missing=missing,
    )


@app.get("/energy", response_model=EnergyResponse)
def energy() -> EnergyResponse:
    """Return cumulative energy metrics for the API session."""
    # Update CO2 from tracker if available
    if _energy_tracker is not None:
        try:
            interim = getattr(_energy_tracker, "_total_energy", None)
            if interim is not None:
                _energy_metrics["energy_kwh"] = float(interim)
        except Exception:
            logger.debug("Interim energy read failed", exc_info=True)
    return EnergyResponse(
        total_requests=_energy_metrics["total_requests"],
        total_predict_requests=_energy_metrics["total_predict_requests"],
        total_inference_time_s=round(_energy_metrics["total_inference_time_s"], 4),
        co2_emissions_kg=round(_energy_metrics["co2_emissions_kg"], 8),
        tracker_active=_energy_metrics["tracker_active"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
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
    emotions: dict[str, float] = {}
    if emotion_extractor is not None:
        try:
            probas = emotion_extractor.get_emotion_features([text])[0]
            emotions = {
                EMOTION_LABELS[i]: round(float(probas[i]), 4) for i in range(len(EMOTION_LABELS))
            }
        except Exception:
            logger.exception("Emotion extraction failed for predict request")

    return PredictResponse(
        score=score,
        label="fiable" if pred_label == 0 else "suspect",
        language=language,
        emotions=emotions,
    )


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class WordContribution(BaseModel):
    word: str
    contribution: float


class ExplainResponse(BaseModel):
    explainable: bool
    score: float
    label: str
    top_suspect_words: list[WordContribution]
    top_fiable_words: list[WordContribution]
    sensationalist_words: list[str]


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest) -> ExplainResponse:
    """Return word-level explainability for a given text."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must be non-empty")

    explanation = detector.explain_prediction(text)

    top_suspect = [
        WordContribution(word=w["word"], contribution=round(w["contribution"], 6))
        for w in explanation.get("top_suspect_words", [])[:10]
    ]
    top_fiable = [
        WordContribution(word=w["word"], contribution=round(w["contribution"], 6))
        for w in explanation.get("top_fiable_words", [])[:10]
    ]
    sensationalist = [w["word"] for w in explanation.get("sensationalist_words", [])]

    return ExplainResponse(
        explainable=explanation.get("explainable", False),
        score=round(explanation.get("score", 0.0), 4),
        label=explanation.get("label", "unknown"),
        top_suspect_words=top_suspect,
        top_fiable_words=top_fiable,
        sensationalist_words=sensationalist,
    )
