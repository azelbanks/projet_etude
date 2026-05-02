import os
import sys
import re
import time
import datetime
import random
from atproto import Client
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
from src.collection.pipeline_monitor import PipelineMonitor

# Chargement de l'environnement
load_dotenv()

MONGO_HOST = os.getenv('MONGO_HOST', 'mongodb')
MONGO_USER = os.getenv('MONGO_USER')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')
HANDLE = os.getenv('BLUESKY_HANDLE')
PASSWORD = os.getenv('BLUESKY_PASSWORD')

# --- CONFIGURATION EXPERTE ---
# On structure les recherches par langue pour respecter le Cahier des Charges (FR/EN)
SEARCH_CONFIG = {
    "en": [
        # Thématiques à risque de désinformation
        "climate change", "vaccine", "conspiracy", "breaking news", "leaked",
        "censored", "urgent", "trump", "election",
        # Termes sensationnalistes / désinformation
        "exposed", "they lied", "cover up", "wake up",
        # Termes neutres (baseline contraste)
        "weekend", "art", "technology", "community",
    ],
    "fr": [
        # Thématiques à risque de désinformation
        "changement climatique", "vaccin", "complot", "alerte info", "scandale",
        "censuré", "urgent", "macron", "élection", "démission",
        # Termes sensationnalistes / désinformation
        "on nous cache", "révélation", "ils mentent", "manipulation",
        # Actualité / société (rééquilibrage volume FR)
        "politique", "santé", "éducation", "immigration", "retraite",
        "sécurité", "économie", "justice", "grève", "assemblée nationale",
        # Termes neutres (baseline contraste)
        "weekend", "art", "technologie", "communauté",
    ]
}

# Paramètres de résilience
SLEEP_TIME = 300  # 5 minutes
MAX_RETRIES = 3

# --- VALIDATION DU TEXTE ---
# Expression régulière pour détecter les textes composés uniquement d'URLs
_URL_PATTERN = re.compile(r'https?://\S+', re.IGNORECASE)

# Mots courants français pour la détection heuristique de langue
_FR_COMMON_WORDS = {"le", "la", "les", "de", "un", "une", "est", "et", "des",
                    "du", "en", "au", "aux", "ce", "qui", "que", "dans", "pour",
                    "pas", "sur", "il", "elle", "je", "tu", "nous", "vous", "sont"}


def validate_text(text):
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
    text_without_urls = _URL_PATTERN.sub('', stripped).strip()
    if len(text_without_urls) < 3:
        return False, "url_only"

    return True, stripped


def compute_word_count(text):
    """Retourne le nombre de mots dans le texte (hors URLs)."""
    text_without_urls = _URL_PATTERN.sub('', text).strip()
    words = text_without_urls.split()
    return len(words)


def detect_language_hint(text):
    """
    Heuristique simple de détection de langue.
    Si plus de 30% des mots sont des mots français courants, retourne 'fr', sinon 'en'.
    """
    text_without_urls = _URL_PATTERN.sub('', text).strip().lower()
    words = text_without_urls.split()
    if not words:
        return "en"
    fr_count = sum(1 for w in words if w in _FR_COMMON_WORDS)
    ratio = fr_count / len(words)
    return "fr" if ratio > 0.30 else "en"


def connect_db():
    if MONGO_USER and MONGO_PASSWORD:
        from urllib.parse import quote_plus
        uri = f"mongodb://{quote_plus(MONGO_USER)}:{quote_plus(MONGO_PASSWORD)}@{MONGO_HOST}:27017/?authSource=admin"
    else:
        uri = f"mongodb://{MONGO_HOST}:27017/"
    print(f"🔌 Connexion à MongoDB : {MONGO_HOST} (auth={'oui' if MONGO_USER else 'non'})")
    retries = 0
    while True:
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            print(f"✅ MongoDB connecté (Database: thumalien_db)!")
            # On retourne la collection
            return client['thumalien_db']['raw_posts']
        except Exception as e:
            wait = 5 * (retries + 1)
            print(f"⏳ Base de données indisponible. Nouvelle tentative dans {wait}s... ({e})")
            time.sleep(wait)
            retries += 1

def get_bluesky_client():
    try:
        client = Client()
        client.login(HANDLE, PASSWORD)
        print(f"✅ Authentification Bluesky réussie pour {HANDLE}")
        return client
    except Exception as e:
        print(f"❌ CRITIQUE : Échec authentification Bluesky : {e}")
        return None

def extract_metadata(post):
    """
    Fonction d'ingénierie pour extraire proprement les métadonnées complexes.
    Indispensable pour l'IA (multimodalité).
    """
    # 1. Détection des Images
    has_image = False
    image_url = None
    if hasattr(post, 'embed') and post.embed:
        if hasattr(post.embed, 'images'):
            has_image = True
            # On prend la première image (fullsize)
            if len(post.embed.images) > 0:
                image_url = post.embed.images[0].fullsize
    
    # 2. Gestion des Langues déclarées
    langs = getattr(post.record, 'langs', [])
    
    return has_image, image_url, langs

def run_collection_cycle(collection, client, monitor=None):
    total_new = 0
    start_time = datetime.datetime.now()
    if monitor:
        monitor.start_cycle()
    print(f"\n--- 🔄 Cycle de collecte Multi-Langues : {start_time.strftime('%H:%M:%S')} ---")

    # On itère sur chaque langue (FR, EN...)
    for lang, keywords in SEARCH_CONFIG.items():
        print(f"   🌍 Traitement de la langue : {lang.upper()}")

        for kw in keywords:
            try:
                # REQUÊTE API CIBLÉE
                # On ajoute le filtre 'lang' pour ne pas polluer la base avec du bruit
                data = client.app.bsky.feed.search_posts({
                    'q': kw,
                    'limit': 25, # On réduit légèrement par mot pour éviter le rate-limit
                    'sort': 'latest',
                    'lang': lang
                })

                ops = [] # Liste pour le Bulk Write (Optimisation Performance)
                skipped = 0

                for post in data.posts:
                    # --- Validation du texte ---
                    is_valid, result = validate_text(post.record.text)
                    if not is_valid:
                        skipped += 1
                        continue

                    clean_text = result

                    # Extraction avancée
                    has_image, image_url, detected_langs = extract_metadata(post)

                    doc = {
                        # Champs primaires
                        "uri": post.uri,
                        "cid": post.cid,
                        "text": clean_text,
                        "created_at": post.record.created_at,

                        # Contexte de collecte (Traçabilité)
                        "search_term": kw,
                        "search_lang": lang,
                        "collected_at": datetime.datetime.now(),

                        # Auteur
                        "author_did": post.author.did,
                        "author_handle": post.author.handle,
                        "author_display_name": post.author.display_name,

                        # Métadonnées IA (Feature Engineering)
                        "has_image": has_image,
                        "image_url": image_url,
                        "reply_count": getattr(post, 'reply_count', 0),
                        "repost_count": getattr(post, 'repost_count', 0),
                        "like_count": getattr(post, 'like_count', 0),
                        "declared_langs": detected_langs,

                        # Analyse textuelle (Feature Engineering)
                        "text_word_count": compute_word_count(clean_text),
                        "text_language_hint": detect_language_hint(clean_text),

                        # Placeholders pour l'IA (seront remplis plus tard)
                        "ai_processed": False
                    }

                    # On prépare l'opération d'insertion/mise à jour
                    ops.append(
                        UpdateOne({"uri": post.uri}, {"$set": doc}, upsert=True)
                    )

                # Exécution en masse (Beaucoup plus rapide que one-by-one)
                if ops:
                    result = collection.bulk_write(ops)
                    added = result.upserted_count
                    duplicates = result.modified_count
                    total_new += added + duplicates
                    # print(f"      -> '{kw}': {added} posts traités")
                    if monitor:
                        monitor.record_keyword(kw, lang, added=added, duplicates=duplicates)
                else:
                    if monitor:
                        monitor.record_keyword(kw, lang, added=0, duplicates=0)

                # Délai aléatoire entre requêtes (respect rate limit API)
                time.sleep(random.uniform(1.0, 2.5))

            except Exception as e:
                err_str = str(e).lower()
                print(f"⚠️ Erreur sur '{kw}' ({lang}): {e}")
                if monitor:
                    monitor.record_keyword(kw, lang, errors=1, error_msg=e)

                # Rate limit détecté : backoff progressif
                if '429' in err_str or 'rate' in err_str or 'too many' in err_str:
                    wait = random.uniform(30, 60)
                    print(f"   ⏳ Rate limit détecté — pause {wait:.0f}s")
                    time.sleep(wait)
                else:
                    # Erreur réseau ou autre : petite pause de sécurité
                    time.sleep(random.uniform(3, 5))

    print(f"📦 Cycle terminé. {total_new} documents traités/ajoutés.")
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
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    model_dir = os.path.abspath(model_dir)

    try:
        # Emotions
        with open(os.path.join(model_dir, 'emotion_vocab_bilingual.pickle'), 'rb') as f:
            _emotion_vocab = _pickle.load(f)
        with open(os.path.join(model_dir, 'emotion_label_encoder_bilingual.pickle'), 'rb') as f:
            _emotion_le = _pickle.load(f)

        cp = torch.load(os.path.join(model_dir, 'emotion_bilingual.pt'),
                        map_location='cpu', weights_only=False)
        if isinstance(cp, dict) and 'model_state_dict' in cp:
            sd = cp['model_state_dict']
            _emotion_max_len = cp.get('max_len', 100)
        else:
            sd = cp
            _emotion_max_len = 100

        vs = sd['embedding.weight'].shape[0]
        ed = sd['embedding.weight'].shape[1]
        nc = sd['fc3.weight'].shape[0]

        from pipeline.expert_detector import _EmotionMLP as EmotionMLP
        _emotion_model = EmotionMLP(vs, ed, nc)
        _emotion_model.load_state_dict(sd)
        _emotion_model.eval()

        # V5 detector
        from pipeline.expert_detector import ExpertFakeNewsDetector, EmotionFeatureExtractor
        _detector = ExpertFakeNewsDetector(model_dir=model_dir, use_emotions=True)
        _detector.load(suffix='expert_v5')
        _emo_extractor = EmotionFeatureExtractor(model_dir=model_dir)
        _emo_extractor.load()

        # Stage 1 fait/opinion (V9)
        global _stage1_pipe, _stage1_threshold
        import joblib as _joblib
        s1_path = os.path.join(model_dir, 'stage1_fact_opinion.joblib')
        if os.path.exists(s1_path):
            s1_data = _joblib.load(s1_path)
            _stage1_pipe = s1_data['pipeline']
            _stage1_threshold = s1_data.get('threshold', 0.40)
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
        'text': {'$exists': True, '$ne': ''},
        '$or': [
            {'ai_emotion': {'$exists': False}},
            {'ai_emotion': None},
        ]
    }
    to_process = collection.count_documents(query)
    if to_process == 0:
        return

    print(f"  Inference sur {to_process} posts non analyses...")
    batch_size = 500
    processed = 0

    while processed < to_process:
        docs = list(collection.find(query, {'_id': 1, 'text': 1}).limit(batch_size))
        if not docs:
            break

        texts = [d.get('text', '') for d in docs]
        ids = [d['_id'] for d in docs]

        # Emotions
        pad_idx = _emotion_vocab.get('<PAD>', 0)
        seqs = []
        for text in texts:
            tokens = str(text).lower().split()
            tok_ids = [_emotion_vocab.get(t, _emotion_vocab.get('<UNK>', 1))
                       for t in tokens[:_emotion_max_len]]
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
                    post_types[i] = 'factuel' if pf >= _stage1_threshold else 'opinion'
            except Exception:
                pass

        ops = []
        for i, _id in enumerate(ids):
            v5_label = int(v5_result['prediction_label'].iloc[i])

            # V9 : opinions suspectes => reclassees fiables
            v9_label = v5_label
            if post_types[i] == 'opinion' and v5_label == 1:
                v9_label = 0

            update_fields = {
                'ai_emotion': str(emo_labels[i]),
                'ai_score_credibility': float(v5_result['ai_score_credibility'].iloc[i]),
                'prediction_label': v5_label,
                'ai_v9_label': v9_label,
                'ai_language': str(v5_result['language'].iloc[i]),
                'ai_analysis_log': str(v5_result['ai_analysis_log'].iloc[i]),
                'ai_model_version': 'expert_v5',
                'ai_model_name': 'ExpertFakeNewsDetector_v5',
                'ai_processed_at': datetime.datetime.now(),
                'ai_processed': True,
            }
            if post_types[i] is not None:
                update_fields['ai_post_type'] = post_types[i]
                update_fields['ai_post_type_proba'] = post_type_probas[i]

            ops.append(UpdateOne(
                {'_id': _id},
                {'$set': update_fields}
            ))

        if ops:
            collection.bulk_write(ops)

        processed += len(docs)
        print(f"    {processed}/{to_process} traites")

    print(f"  Inference terminee : {processed} posts")


if __name__ == "__main__":
    print("🚀 Démarrage du Collecteur Bluesky Intelligent (V3 — avec inference auto)")

    db_collection = connect_db()
    bsky_client = get_bluesky_client()
    monitor = PipelineMonitor()

    if bsky_client:
        while True:
            run_collection_cycle(db_collection, bsky_client, monitor=monitor)

            # Inference IA automatique sur les nouveaux posts
            try:
                run_inference_cycle(db_collection)
            except Exception as e:
                print(f"⚠️ Erreur inference: {e}")

            # Gestion du temps d'attente
            print(f"💤 Mise en veille pour {SLEEP_TIME} secondes...")
            time.sleep(SLEEP_TIME)
    else:
        print("❌ Impossible de démarrer : Vérifiez vos identifiants dans .env")