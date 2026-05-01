#!/usr/bin/env python3
"""
Inference emotionnelle batch sur tous les posts MongoDB non encore analyses.

Usage (depuis le conteneur Docker ou en local) :
    python scripts/batch_emotion_inference.py

Ce script :
1. Recupere tous les posts sans champ ai_emotion (ou ai_emotion=null)
2. Applique le modele MLP emotions bilingue (7 classes)
3. Met a jour chaque post avec ai_emotion dans MongoDB
4. Traite par batch de 1000 pour gerer la memoire

Prerequis : modeles dans models/ (emotion_bilingual.pt, vocab, label_encoder)
"""

import os
import sys
import time
import numpy as np
import pickle

# Ajouter src au path
_proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_proj, 'src'))

from pymongo import MongoClient, UpdateOne
from urllib.parse import quote_plus

BATCH_SIZE = 1000
MODEL_DIR = os.path.join(_proj, 'models')


def connect_db():
    mongo_host = os.getenv('MONGO_HOST', 'mongodb')
    mongo_user = os.getenv('MONGO_USER', '')
    mongo_pwd = os.getenv('MONGO_PASSWORD', '')

    if mongo_user and mongo_pwd:
        uri = f"mongodb://{quote_plus(mongo_user)}:{quote_plus(mongo_pwd)}@{mongo_host}:27017/?authSource=admin"
    else:
        uri = f"mongodb://{mongo_host}:27017/"

    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    client.admin.command('ping')
    return client['thumalien_db']['raw_posts']


def load_emotion_model():
    """Charge le modele MLP emotions bilingue."""
    import torch

    vocab_path = os.path.join(MODEL_DIR, 'emotion_vocab_bilingual.pickle')
    le_path = os.path.join(MODEL_DIR, 'emotion_label_encoder_bilingual.pickle')
    model_path = os.path.join(MODEL_DIR, 'emotion_bilingual.pt')

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(le_path, 'rb') as f:
        label_encoder = pickle.load(f)

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Le checkpoint peut etre un dict avec 'model_state_dict' ou directement un state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        max_len = checkpoint.get('max_len', 100)
    else:
        state_dict = checkpoint
        max_len = 100

    vocab_size = state_dict['embedding.weight'].shape[0]
    embed_dim = state_dict['embedding.weight'].shape[1]
    n_classes = state_dict['fc3.weight'].shape[0]

    from pipeline.expert_detector import _EmotionMLP as EmotionMLP
    model = EmotionMLP(vocab_size, embed_dim, n_classes)
    model.load_state_dict(state_dict)
    model.eval()

    return model, vocab, label_encoder, max_len


def predict_emotions_batch(texts, model, vocab, label_encoder, max_len):
    """Predit les emotions pour un batch de textes."""
    import torch

    pad_idx = vocab.get('<PAD>', 0)

    sequences = []
    for text in texts:
        tokens = str(text).lower().split()
        ids = [vocab.get(t, vocab.get('<UNK>', 1)) for t in tokens[:max_len]]
        if len(ids) < max_len:
            ids += [pad_idx] * (max_len - len(ids))
        sequences.append(ids)

    X = torch.tensor(sequences, dtype=torch.long)

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).numpy()

    preds = np.argmax(probs, axis=1)
    labels = label_encoder.inverse_transform(preds)

    return labels, probs


def main():
    print("=" * 60)
    print("BATCH EMOTION INFERENCE")
    print("=" * 60)
    t0 = time.time()

    # 1. Connexion MongoDB
    print("\n[1/3] Connexion MongoDB...")
    collection = connect_db()
    print("  OK")

    # 2. Charger le modele
    print("[2/3] Chargement du modele emotions...")
    model, vocab, label_encoder, max_len = load_emotion_model()
    print(f"  OK - {len(label_encoder.classes_)} classes: {list(label_encoder.classes_)}")

    # 3. Compter les posts a traiter
    query = {
        'text': {'$exists': True, '$ne': ''},
        '$or': [
            {'ai_emotion': {'$exists': False}},
            {'ai_emotion': None},
        ]
    }
    total = collection.count_documents(query)
    print(f"\n[3/3] Posts sans emotion : {total}")

    if total == 0:
        print("  Rien a faire.")
        return

    # 4. Traitement par batch
    processed = 0
    batch_num = 0

    while processed < total:
        batch_num += 1
        cursor = collection.find(
            query,
            {'_id': 1, 'text': 1}
        ).limit(BATCH_SIZE)

        docs = list(cursor)
        if not docs:
            break

        texts = [d.get('text', '') for d in docs]
        ids = [d['_id'] for d in docs]

        labels, probs = predict_emotions_batch(texts, model, vocab, label_encoder, max_len)

        ops = []
        for i, (_id, label) in enumerate(zip(ids, labels)):
            ops.append(UpdateOne(
                {'_id': _id},
                {'$set': {
                    'ai_emotion': str(label),
                    'ai_emotion_model': 'emotion_bilingual_v1',
                }}
            ))

        if ops:
            collection.bulk_write(ops)

        processed += len(docs)
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (total - processed) / rate if rate > 0 else 0
        print(f"  Batch {batch_num}: {processed}/{total} ({processed/total*100:.1f}%) "
              f"- {rate:.0f} posts/s - ETA {remaining/60:.1f}min")

    elapsed = time.time() - t0
    print(f"\nTermine : {processed} posts traites en {elapsed:.0f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
