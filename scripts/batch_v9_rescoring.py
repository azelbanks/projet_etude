"""
Batch V9 re-scoring — Applique le filtre fait/opinion (Stage 1) sur tous les posts.

Ajoute les champs :
  - ai_post_type : 'factuel' ou 'opinion'
  - ai_post_type_proba : P(factuel) entre 0 et 1
  - ai_v9_label : prediction_label final apres filtre (opinions forcees a 0=fiable)

Usage :
    python3 scripts/batch_v9_rescoring.py
"""

import os
import sys
import time
import joblib
import numpy as np

# Path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, '.env'))


def connect_db():
    host = os.getenv('MONGO_HOST', 'localhost')
    user = os.getenv('MONGO_USER', '')
    password = os.getenv('MONGO_PASSWORD', '')

    if user and password:
        from urllib.parse import quote_plus
        uri = f"mongodb://{quote_plus(user)}:{quote_plus(password)}@{host}:27017/?authSource=admin"
    else:
        uri = f"mongodb://{host}:27017/"

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.server_info()
    return client['thumalien_db']['raw_posts']


def load_stage1():
    path = os.path.join(PROJECT_ROOT, 'models', 'stage1_fact_opinion.joblib')
    if not os.path.exists(path):
        print(f"Stage 1 model not found: {path}")
        sys.exit(1)
    data = joblib.load(path)
    pipeline = data['pipeline']
    threshold = data.get('threshold', 0.40)
    print(f"Stage 1 loaded (threshold={threshold})")
    return pipeline, threshold


def main():
    print("=== Batch V9 Re-scoring ===")

    collection = connect_db()
    s1_pipe, s1_threshold = load_stage1()

    # Count posts to process (those without ai_post_type)
    query = {
        'text': {'$exists': True, '$ne': ''},
        'ai_post_type': {'$exists': False},
    }
    total = collection.count_documents(query)
    print(f"Posts a traiter : {total}")

    if total == 0:
        print("Rien a faire.")
        return

    batch_size = 1000
    processed = 0
    stats = {'factuel': 0, 'opinion': 0, 'reclassed': 0}
    start = time.time()

    while processed < total:
        docs = list(collection.find(query, {'_id': 1, 'text': 1, 'prediction_label': 1}).limit(batch_size))
        if not docs:
            break

        texts = [d.get('text', '') for d in docs]
        ids = [d['_id'] for d in docs]
        original_labels = [d.get('prediction_label', 0) for d in docs]

        # Stage 1 prediction
        try:
            probas = s1_pipe.predict_proba(texts)
            p_factuel = probas[:, 1]  # P(factuel)
        except Exception as e:
            print(f"  Erreur Stage 1: {e}")
            # Skip batch
            processed += len(docs)
            continue

        ops = []
        for i, _id in enumerate(ids):
            pf = float(p_factuel[i])
            post_type = 'factuel' if pf >= s1_threshold else 'opinion'

            # V9 logic: if opinion AND was suspect, reclassify as fiable
            v9_label = original_labels[i]
            if post_type == 'opinion' and v9_label == 1:
                v9_label = 0  # Opinion => not fake news
                stats['reclassed'] += 1

            stats[post_type] += 1

            ops.append(UpdateOne(
                {'_id': _id},
                {'$set': {
                    'ai_post_type': post_type,
                    'ai_post_type_proba': round(pf, 4),
                    'ai_v9_label': int(v9_label),
                }}
            ))

        if ops:
            collection.bulk_write(ops)

        processed += len(docs)
        elapsed = time.time() - start
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"  {processed}/{total} ({rate:.0f} posts/sec)")

    elapsed = time.time() - start
    print(f"\n=== Termine en {elapsed:.1f}s ===")
    print(f"  Factuel  : {stats['factuel']}")
    print(f"  Opinion  : {stats['opinion']}")
    print(f"  Reclasses (suspect->fiable) : {stats['reclassed']}")

    # Final stats
    total_posts = collection.count_documents({})
    v5_suspects = collection.count_documents({'prediction_label': 1})
    v9_suspects = collection.count_documents({'ai_v9_label': 1})
    print(f"\n  V5 suspects : {v5_suspects}/{total_posts} ({v5_suspects/total_posts*100:.1f}%)")
    print(f"  V9 suspects : {v9_suspects}/{total_posts} ({v9_suspects/total_posts*100:.1f}%)")
    print(f"  Reduction FP : {v5_suspects - v9_suspects} posts ({(v5_suspects - v9_suspects)/v5_suspects*100:.1f}%)")


if __name__ == '__main__':
    main()
