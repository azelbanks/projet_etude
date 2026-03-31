import os
import time
import datetime
import random
from atproto import Client
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

# Chargement de l'environnement
load_dotenv()

MONGO_HOST = os.getenv('MONGO_HOST', 'mongodb')
HANDLE = os.getenv('BLUESKY_HANDLE')
PASSWORD = os.getenv('BLUESKY_PASSWORD')

# --- CONFIGURATION EXPERTE ---
# On structure les recherches par langue pour respecter le Cahier des Charges (FR/EN)
SEARCH_CONFIG = {
    "en": [
        "climate change", "vaccine", "conspiracy", "breaking news", "leaked", 
        "censored", "urgent", "happy", "weekend", "art", "technology", "trump"
    ],
    "fr": [
        "changement climatique", "vaccin", "complot", "alerte info", "scandale", 
        "censuré", "urgent", "macron", "joie", "weekend", "art", "technologie", "démission"
    ]
}

# Paramètres de résilience
SLEEP_TIME = 300  # 5 minutes
MAX_RETRIES = 3

def connect_db():
    uri = f"mongodb://{MONGO_HOST}:27017/"
    print(f"🔌 Connexion à MongoDB : {uri}")
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

def run_collection_cycle(collection, client):
    total_new = 0
    start_time = datetime.datetime.now()
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
                
                for post in data.posts:
                    # Extraction avancée
                    has_image, image_url, detected_langs = extract_metadata(post)
                    
                    doc = {
                        # Champs primaires
                        "uri": post.uri,
                        "cid": post.cid,
                        "text": post.record.text,
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
                    added = result.upserted_count + result.modified_count
                    total_new += added
                    # print(f"      -> '{kw}': {added} posts traités")
                
                # Petit délai aléatoire pour simuler un comportement humain (Anti-Bot)
                time.sleep(random.uniform(0.5, 1.5))
                    
            except Exception as e:
                print(f"⚠️ Erreur sur '{kw}' ({lang}): {e}")
            
    print(f"📦 Cycle terminé. {total_new} documents traités/ajoutés.")

if __name__ == "__main__":
    print("🚀 Démarrage du Collecteur Bluesky Intelligent (V2)")
    
    db_collection = connect_db()
    bsky_client = get_bluesky_client()

    if bsky_client:
        while True:
            run_collection_cycle(db_collection, bsky_client)
            
            # Gestion du temps d'attente
            print(f"💤 Mise en veille pour {SLEEP_TIME} secondes...")
            time.sleep(SLEEP_TIME)
    else:
        print("❌ Impossible de démarrer : Vérifiez vos identifiants dans .env")