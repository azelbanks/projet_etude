import datetime
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from atproto import Client
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

from src.collection.pipeline_monitor import PipelineMonitor

# ---------------------------------------------------------------------------
#  Structured logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("thumacheck.collector")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "..", "..", "logs", "collector.log"),
            encoding="utf-8",
        ),
    ],
)

# Chargement de l'environnement
load_dotenv()

MONGO_HOST = os.getenv("MONGO_HOST", "mongodb")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
HANDLE = os.getenv("BLUESKY_HANDLE")
PASSWORD = os.getenv("BLUESKY_PASSWORD")

# --- CONFIGURATION EXPERTE ---
# Chargement externe depuis JSON (si disponible), sinon fallback en dur
_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "search_config.json"

_DEFAULT_SEARCH_CONFIG = {
    "en": [
        "climate change",
        "vaccine",
        "conspiracy",
        "breaking news",
        "leaked",
        "censored",
        "urgent",
        "trump",
        "election",
        "exposed",
        "they lied",
        "cover up",
        "wake up",
        "weekend",
        "art",
        "technology",
        "community",
    ],
    "fr": [
        "changement climatique",
        "vaccin",
        "complot",
        "alerte info",
        "scandale",
        "censuré",
        "urgent",
        "macron",
        "élection",
        "démission",
        "on nous cache",
        "révélation",
        "ils mentent",
        "manipulation",
        "politique",
        "santé",
        "éducation",
        "immigration",
        "retraite",
        "sécurité",
        "économie",
        "justice",
        "grève",
        "assemblée nationale",
        "weekend",
        "art",
        "technologie",
        "communauté",
    ],
}


def _load_search_config():
    """Charge la config de recherche depuis config/search_config.json ou fallback."""
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
            logger.info(
                "Search config loaded from %s (%d FR, %d EN terms)",
                _CONFIG_PATH,
                len(cfg.get("fr", [])),
                len(cfg.get("en", [])),
            )
            return cfg
        except Exception as e:
            logger.warning("Failed to load search config: %s — using defaults", e)
    return _DEFAULT_SEARCH_CONFIG


SEARCH_CONFIG = _load_search_config()

# Paramètres de résilience
SLEEP_TIME = 120  # 2 minutes (réduit pour accélérer collecte, safe anti-ban)
MAX_RETRIES = 3

# --- Circuit Breaker ---
CIRCUIT_BREAKER_THRESHOLD = 5  # consecutive failures before opening
CIRCUIT_BREAKER_TIMEOUT = 120  # seconds before retrying after circuit opens

# --- RGPD Art. 21 : Liste d'exclusion (droit d'opposition) ---
_EXCLUDED_HANDLES_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "excluded_handles.txt"
)


def load_excluded_handles():
    """Charge la liste des handles exclus depuis data/excluded_handles.txt."""
    path = os.path.abspath(_EXCLUDED_HANDLES_FILE)
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


EXCLUDED_HANDLES = load_excluded_handles()

# --- RGPD : Pseudonymisation des identifiants auteurs ---
_PSEUDO_SALT = os.environ.get("PSEUDO_SALT", "thumacheck_2026_default_salt")


def pseudonymize(value: str) -> str:
    """Hash SHA-256 tronqué (16 car.) — pseudonymisation irréversible RGPD."""
    return hashlib.sha256(f"{_PSEUDO_SALT}:{value}".encode()).hexdigest()[:16]


def reload_excluded_handles():
    """Recharge la liste d'exclusion (appelé à chaque cycle de collecte)."""
    global EXCLUDED_HANDLES
    EXCLUDED_HANDLES = load_excluded_handles()


# --- VALIDATION DU TEXTE ---
# Expression régulière pour détecter les textes composés uniquement d'URLs
_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)

# Mots courants français pour la détection heuristique de langue
_FR_COMMON_WORDS = {
    "le",
    "la",
    "les",
    "de",
    "un",
    "une",
    "est",
    "et",
    "des",
    "du",
    "en",
    "au",
    "aux",
    "ce",
    "qui",
    "que",
    "dans",
    "pour",
    "pas",
    "sur",
    "il",
    "elle",
    "je",
    "tu",
    "nous",
    "vous",
    "sont",
}


def validate_text(text: str) -> tuple[bool, str]:
    """
    Valide le texte d'un post avant insertion.
    Retourne (True, cleaned_text) si le texte est acceptable, (False, reason) sinon.
    """
    if not text or not isinstance(text, str):
        return False, "empty_or_missing"

    stripped = text.strip()
    if len(stripped) < 3:
        return False, "too_short"

    # Retirer les URLs et vérifier qu'il reste du contenu réel
    text_without_urls = _URL_PATTERN.sub("", stripped).strip()
    if len(text_without_urls) < 3:
        return False, "url_only"

    return True, stripped


def compute_word_count(text: str) -> int:
    """Retourne le nombre de mots dans le texte (hors URLs)."""
    text_without_urls = _URL_PATTERN.sub("", text).strip()
    words = text_without_urls.split()
    return len(words)


def detect_language_hint(text: str) -> str:
    """
    Detection de langue via langdetect (probabiliste) avec fallback heuristique.
    Retourne 'fr', 'en', ou 'other'.
    """
    text_without_urls = _URL_PATTERN.sub("", text).strip()
    if len(text_without_urls) < 3:
        return "en"

    # Methode principale : langdetect (probabiliste, fiable)
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        lang = detect(text_without_urls[:500])
        if lang == "fr":
            return "fr"
        elif lang == "en":
            return "en"
        return "other"
    except Exception:
        logger.debug("langdetect a echoue, bascule sur l'heuristique", exc_info=True)

    # Fallback heuristique si langdetect echoue
    words = text_without_urls.lower().split()
    if not words:
        return "en"
    fr_count = sum(1 for w in words if w in _FR_COMMON_WORDS)
    ratio = fr_count / len(words)
    return "fr" if ratio > 0.30 else "en"


def connect_db() -> object:
    if MONGO_USER and MONGO_PASSWORD:
        from urllib.parse import quote_plus

        uri = f"mongodb://{quote_plus(MONGO_USER)}:{quote_plus(MONGO_PASSWORD)}@{MONGO_HOST}:27017/?authSource=admin"
    else:
        uri = f"mongodb://{MONGO_HOST}:27017/"
    logger.info("Connexion a MongoDB : %s (auth=%s)", MONGO_HOST, "oui" if MONGO_USER else "non")
    MAX_RETRIES = 20
    retries = 0
    while retries < MAX_RETRIES:
        try:
            client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            logger.info("MongoDB connecte (Database: thumalien_db)")
            return client["thumalien_db"]["raw_posts"]
        except Exception as e:
            wait = 5 * (retries + 1)
            retries += 1
            logger.warning(
                "Base de donnees indisponible. Tentative %d/%d dans %ds... (%s)",
                retries,
                MAX_RETRIES,
                wait,
                e,
            )
            time.sleep(wait)
    raise ConnectionError(f"MongoDB inaccessible apres {MAX_RETRIES} tentatives")


def get_bluesky_client():
    try:
        client = Client()
        client.login(HANDLE, PASSWORD)
        logger.info("Authentification Bluesky reussie pour %s", HANDLE)
        return client
    except Exception as e:
        logger.critical("Echec authentification Bluesky : %s", e)
        return None


def extract_metadata(post: Any) -> tuple[bool, str | None, list[str]]:
    """
    Fonction d'ingénierie pour extraire proprement les métadonnées complexes.
    Indispensable pour l'IA (multimodalité).

    Returns
    -------
    (has_image, image_url, langs)
    """
    # 1. Détection des Images
    has_image = False
    image_url = None
    if hasattr(post, "embed") and post.embed and hasattr(post.embed, "images"):
        has_image = True
        # On prend la première image (fullsize)
        if len(post.embed.images) > 0:
            image_url = post.embed.images[0].fullsize

    # 2. Gestion des Langues déclarées
    langs = getattr(post.record, "langs", [])

    return has_image, image_url, langs


def _exponential_backoff(attempt, base=2, max_wait=120):
    """Calcule un delai d'attente exponentiel avec jitter."""
    wait = min(base**attempt + random.uniform(0, 1), max_wait)
    return wait


def run_collection_cycle(collection, client, monitor=None):
    reload_excluded_handles()
    total_new = 0
    consecutive_failures = 0
    start_time = datetime.datetime.now()
    if monitor:
        monitor.start_cycle()
    logger.info("Cycle de collecte Multi-Langues : %s", start_time.strftime("%H:%M:%S"))

    for lang, keywords in SEARCH_CONFIG.items():
        logger.info("Traitement de la langue : %s", lang.upper())

        for kw in keywords:
            # --- Circuit breaker : si trop d'erreurs consecutives, pause longue ---
            if consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
                logger.warning(
                    "Circuit breaker OUVERT : %d erreurs consecutives. Pause de %ds avant reprise.",
                    consecutive_failures,
                    CIRCUIT_BREAKER_TIMEOUT,
                )
                # Write alert file for external monitoring / webhook integration
                alert_path = (
                    Path(__file__).resolve().parent.parent.parent
                    / "logs"
                    / "circuit_breaker_alert.jsonl"
                )
                alert_path.parent.mkdir(parents=True, exist_ok=True)
                import json as _json

                alert_data = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "event": "circuit_breaker_open",
                    "consecutive_failures": consecutive_failures,
                    "timeout_seconds": CIRCUIT_BREAKER_TIMEOUT,
                    "lang": lang,
                    "last_keyword": kw,
                }
                with open(alert_path, "a", encoding="utf-8") as af:
                    af.write(_json.dumps(alert_data, ensure_ascii=False) + "\n")
                logger.critical("ALERT written to %s — circuit breaker opened", alert_path)
                time.sleep(CIRCUIT_BREAKER_TIMEOUT)
                consecutive_failures = 0

            # --- Tentatives avec exponential backoff ---
            success = False
            for attempt in range(MAX_RETRIES):
                try:
                    ops = []
                    skipped = 0
                    cursor = None

                    # Pagination : 2 pages × 100 posts = 200 posts/mot-clé max
                    for _page in range(2):
                        params = {"q": kw, "limit": 100, "sort": "latest", "lang": lang}
                        if cursor:
                            params["cursor"] = cursor
                        try:
                            data = client.app.bsky.feed.search_posts(params)
                        except Exception as page_err:
                            # Bluesky peut introduire de nouveaux types d'embed non supportés
                            # par la version actuelle d'atproto — on sauvegarde ce qu'on a
                            logger.debug('Page %d ignorée pour "%s": %s', _page + 1, kw, page_err)
                            break

                        for post in data.posts:
                            if getattr(post.author, "handle", None) in EXCLUDED_HANDLES:
                                skipped += 1
                                continue

                            is_valid, result = validate_text(post.record.text)
                            if not is_valid:
                                skipped += 1
                                continue

                            clean_text = result
                            has_image, image_url, detected_langs = extract_metadata(post)

                            doc = {
                                "uri": post.uri,
                                "cid": post.cid,
                                "text": clean_text,
                                "created_at": post.record.created_at,
                                "search_term": kw,
                                "search_lang": lang,
                                "collected_at": datetime.datetime.now(),
                                "author_did": pseudonymize(post.author.did),
                                "author_handle": pseudonymize(post.author.handle),
                                "author_display_name": pseudonymize(post.author.display_name or ""),
                                "has_image": has_image,
                                "image_url": image_url,
                                "reply_count": getattr(post, "reply_count", 0),
                                "repost_count": getattr(post, "repost_count", 0),
                                "like_count": getattr(post, "like_count", 0),
                                "declared_langs": detected_langs,
                                "text_word_count": compute_word_count(clean_text),
                                "text_language_hint": detect_language_hint(clean_text),
                                "ai_processed": False,
                            }

                            ops.append(UpdateOne({"uri": post.uri}, {"$set": doc}, upsert=True))

                        cursor = getattr(data, "cursor", None)
                        if not cursor:
                            break
                        time.sleep(2.0)  # pause prudente entre pages

                    if ops:
                        result = collection.bulk_write(ops)
                        added = result.upserted_count
                        duplicates = result.modified_count
                        total_new += added + duplicates
                        if monitor:
                            monitor.record_keyword(kw, lang, added=added, duplicates=duplicates)
                    else:
                        if monitor:
                            monitor.record_keyword(kw, lang, added=0, duplicates=0)

                    consecutive_failures = 0
                    success = True
                    break  # sortie de la boucle retry

                except Exception as e:
                    err_str = str(e).lower()
                    logger.warning(
                        'Erreur sur "%s" (%s), tentative %d/%d: %s',
                        kw,
                        lang,
                        attempt + 1,
                        MAX_RETRIES,
                        e,
                    )
                    if monitor:
                        monitor.record_keyword(kw, lang, errors=1, error_msg=e)

                    if "429" in err_str or "rate" in err_str or "too many" in err_str:
                        wait = random.uniform(30, 60)
                        logger.info("Rate limit detecte — pause %.0fs", wait)
                        time.sleep(wait)
                    else:
                        wait = _exponential_backoff(attempt)
                        logger.info("Backoff exponentiel: %.1fs", wait)
                        time.sleep(wait)

            if not success:
                consecutive_failures += 1
                logger.error(
                    'Echec definitif sur "%s" (%s) apres %d tentatives', kw, lang, MAX_RETRIES
                )

            # Delai inter-requetes (respect rate limit API)
            time.sleep(random.uniform(1.0, 2.5))

    logger.info("Cycle termine. %d documents traites/ajoutes.", total_new)
    if monitor:
        monitor.end_cycle()


# ---------------------------------------------------------------------------
#  Inference IA automatique (emotions + V5) apres chaque cycle de collecte
# ---------------------------------------------------------------------------

_emotion_model = None
_emotion_vocab = None
_emotion_le = None
_emotion_max_len = 100
_detector = None
_emo_extractor = None
_stage1_pipe = None
_stage1_threshold = 0.40


def _load_inference_models():
    """Charge les modeles d'inference (emotions + V5) une seule fois."""
    global _emotion_model, _emotion_vocab, _emotion_le, _emotion_max_len
    global _detector, _emo_extractor

    if _emotion_model is not None:
        return True

    import pickle as _pickle

    import torch

    # S'assurer que src/ est dans le path pour importer pipeline.*
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    model_dir = os.path.abspath(model_dir)

    try:
        # Emotions
        with open(os.path.join(model_dir, "emotion_vocab_bilingual.pickle"), "rb") as f:
            _emotion_vocab = _pickle.load(f)
        with open(os.path.join(model_dir, "emotion_label_encoder_bilingual.pickle"), "rb") as f:
            _emotion_le = _pickle.load(f)

        cp = torch.load(
            os.path.join(model_dir, "emotion_bilingual.pt"), map_location="cpu", weights_only=True
        )
        if isinstance(cp, dict) and "model_state_dict" in cp:
            sd = cp["model_state_dict"]
            _emotion_max_len = cp.get("max_len", 100)
        else:
            sd = cp
            _emotion_max_len = 100

        vs = sd["embedding.weight"].shape[0]
        ed = sd["embedding.weight"].shape[1]
        nc = sd["fc3.weight"].shape[0]

        from pipeline.expert_detector import _EmotionMLP as EmotionMLP

        _emotion_model = EmotionMLP(vs, ed, nc)
        _emotion_model.load_state_dict(sd)
        _emotion_model.eval()

        # V5 detector
        from pipeline.expert_detector import EmotionFeatureExtractor, ExpertFakeNewsDetector

        _detector = ExpertFakeNewsDetector(model_dir=model_dir, use_emotions=True)
        _detector.load(suffix="expert_v5")
        _emo_extractor = EmotionFeatureExtractor(model_dir=model_dir)
        _emo_extractor.load()

        # Stage 1 fait/opinion (V9)
        global _stage1_pipe, _stage1_threshold
        import joblib as _joblib

        s1_path = os.path.join(model_dir, "stage1_fact_opinion.joblib")
        if os.path.exists(s1_path):
            s1_data = _joblib.load(s1_path)
            _stage1_pipe = s1_data["pipeline"]
            _stage1_threshold = s1_data.get("threshold", 0.40)
            print("  Modeles d'inference charges (emotions + V5 + Stage1 V9)")
        else:
            print("  Modeles d'inference charges (emotions + V5, Stage1 non disponible)")
        return True
    except Exception as e:
        print(f"  Modeles d'inference non disponibles: {e}")
        return False


def run_inference_cycle(collection):
    """Applique l'inference IA sur les posts non encore traites."""
    import numpy as np
    import torch

    if not _load_inference_models():
        return

    query = {
        "text": {"$exists": True, "$ne": ""},
        "$or": [
            {"ai_emotion": {"$exists": False}},
            {"ai_emotion": None},
        ],
    }
    to_process = collection.count_documents(query)
    if to_process == 0:
        return

    print(f"  Inference sur {to_process} posts non analyses...")
    batch_size = 500
    processed = 0

    while processed < to_process:
        docs = list(collection.find(query, {"_id": 1, "text": 1}).limit(batch_size))
        if not docs:
            break

        texts = [d.get("text", "") for d in docs]
        ids = [d["_id"] for d in docs]

        # Emotions
        pad_idx = _emotion_vocab.get("<PAD>", 0)
        seqs = []
        for text in texts:
            tokens = str(text).lower().split()
            tok_ids = [
                _emotion_vocab.get(t, _emotion_vocab.get("<UNK>", 1))
                for t in tokens[:_emotion_max_len]
            ]
            if len(tok_ids) < _emotion_max_len:
                tok_ids += [pad_idx] * (_emotion_max_len - len(tok_ids))
            seqs.append(tok_ids)

        X = torch.tensor(seqs, dtype=torch.long)
        with torch.no_grad():
            logits = _emotion_model(X)
            probs = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        emo_labels = _emotion_le.inverse_transform(preds)

        # V5 fake news
        import pandas as pd

        v5_result = _detector.predict(pd.Series(texts))

        # Stage 1 fait/opinion (V9)
        post_types = [None] * len(texts)
        post_type_probas = [None] * len(texts)
        if _stage1_pipe is not None:
            try:
                s1_probas = _stage1_pipe.predict_proba(texts)
                for i in range(len(texts)):
                    pf = float(s1_probas[i, 1])
                    post_type_probas[i] = round(pf, 4)
                    post_types[i] = "factuel" if pf >= _stage1_threshold else "opinion"
            except Exception:
                logger.debug("Classification stage 1 indisponible", exc_info=True)

        ops = []
        for i, _id in enumerate(ids):
            v5_label = int(v5_result["prediction_label"].iloc[i])

            # V9 : opinions suspectes => reclassees fiables
            v9_label = v5_label
            if post_types[i] == "opinion" and v5_label == 1:
                v9_label = 0

            update_fields = {
                "ai_emotion": str(emo_labels[i]),
                "ai_score_credibility": float(v5_result["ai_score_credibility"].iloc[i]),
                "prediction_label": v5_label,
                "ai_v9_label": v9_label,
                "ai_language": str(v5_result["language"].iloc[i]),
                "ai_analysis_log": str(v5_result["ai_analysis_log"].iloc[i]),
                "ai_model_version": "expert_v5",
                "ai_model_name": "ExpertFakeNewsDetector_v5",
                "ai_processed_at": datetime.datetime.now(),
                "ai_processed": True,
            }
            if post_types[i] is not None:
                update_fields["ai_post_type"] = post_types[i]
                update_fields["ai_post_type_proba"] = post_type_probas[i]

            ops.append(UpdateOne({"_id": _id}, {"$set": update_fields}))

        if ops:
            collection.bulk_write(ops)

        processed += len(docs)
        print(f"    {processed}/{to_process} traites")

    print(f"  Inference terminee : {processed} posts")


if __name__ == "__main__":
    # Ensure logs/ directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "..", "logs"), exist_ok=True)

    logger.info("Demarrage du Collecteur Bluesky Intelligent (V3 — avec inference auto)")

    db_collection = connect_db()
    bsky_client = get_bluesky_client()
    monitor = PipelineMonitor()

    if bsky_client:
        while True:
            run_collection_cycle(db_collection, bsky_client, monitor=monitor)

            try:
                run_inference_cycle(db_collection)
            except Exception as e:
                logger.error("Erreur inference: %s", e, exc_info=True)

            logger.info("Mise en veille pour %ds...", SLEEP_TIME)
            time.sleep(SLEEP_TIME)
    else:
        logger.critical("Impossible de demarrer : Verifiez vos identifiants dans .env")
