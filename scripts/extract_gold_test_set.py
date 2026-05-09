"""
Script d'extraction de posts Bluesky pour annotation manuelle (Gold Test Set).

Extrait un echantillon stratifie de 500 posts depuis MongoDB :
- Equilibre FR / EN
- Equilibre suspect / fiable (selon prediction IA)
- Equilibre ultra-court (<15 mots) / court (15-30) / long (>30)
- Inclut les cas "frontiere" (score credibilite entre 0.3 et 0.7)

Usage:
    python scripts/extract_gold_test_set.py [--n_posts 500] [--output data/gold_test_set.csv]

Auteur: Azelie Bernard - Thumalien Team
"""

import os
import sys
import argparse
import random
from datetime import datetime

import pandas as pd
from pymongo import MongoClient
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()


def get_mongo_collection():
    """Connexion a MongoDB (meme pattern que collect_bluesky.py)."""
    host = os.getenv("MONGO_HOST", "localhost")
    user = os.getenv("MONGO_USER", "")
    password = os.getenv("MONGO_PASSWORD", "")

    if user and password:
        uri = f"mongodb://{quote_plus(user)}:{quote_plus(password)}@{host}:27017/?authSource=admin"
    else:
        uri = f"mongodb://{host}:27017/"

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    print(f"MongoDB connecte ({host})")
    db = client["thumalien_db"]
    return db["raw_posts"]


def extract_stratified_sample(collection, n_posts=500):
    """
    Extrait un echantillon stratifie de posts.

    Strates :
    1. Langue (FR / EN) : 50/50
    2. Prediction (suspect / fiable) : 50/50
    3. Longueur (ultra-court <15 / court 15-30 / long >30) : 40/30/30
    4. + 10% de cas "frontiere" (score entre 0.3 et 0.7)
    """
    per_lang = n_posts // 2  # 250 FR + 250 EN

    # Repartition par longueur au sein de chaque (langue, prediction)
    # 40% ultra-court, 30% court, 30% long
    strata = []
    for lang in ["fr", "en"]:
        for pred in [0, 1]:  # 0=fiable, 1=suspect
            n_strate = per_lang // 2  # 125 par (lang, pred)
            n_ultra = int(n_strate * 0.4)
            n_court = int(n_strate * 0.3)
            n_long = n_strate - n_ultra - n_court

            strata.append({
                "lang": lang, "pred": pred,
                "length": "ultra", "n": n_ultra,
                "query": {"ai_language": lang, "prediction_label": pred,
                          "text_word_count": {"$lt": 15}, "ai_processed": True}
            })
            strata.append({
                "lang": lang, "pred": pred,
                "length": "court", "n": n_court,
                "query": {"ai_language": lang, "prediction_label": pred,
                          "text_word_count": {"$gte": 15, "$lt": 30}, "ai_processed": True}
            })
            strata.append({
                "lang": lang, "pred": pred,
                "length": "long", "n": n_long,
                "query": {"ai_language": lang, "prediction_label": pred,
                          "text_word_count": {"$gte": 30}, "ai_processed": True}
            })

    all_posts = []
    for s in strata:
        cursor = collection.find(
            s["query"],
            {"_id": 0, "uri": 1, "text": 1, "ai_language": 1,
             "prediction_label": 1, "ai_score_credibility": 1,
             "text_word_count": 1, "created_at": 1, "author_handle": 1,
             "search_term": 1}
        )
        posts = list(cursor)

        if len(posts) > s["n"]:
            posts = random.sample(posts, s["n"])

        for p in posts:
            p["strate"] = f"{s['lang']}_{['fiable','suspect'][s['pred']]}_{s['length']}"

        all_posts.extend(posts)
        print(f"  Strate {s['lang']}/{['fiable','suspect'][s['pred']]}/{s['length']}: "
              f"{len(posts)}/{s['n']} posts extraits")

    # Ajouter des cas frontiere (score entre 0.3 et 0.7)
    n_frontiere = max(20, n_posts // 10)
    cursor_frontiere = collection.find(
        {"ai_processed": True,
         "ai_score_credibility": {"$gte": 0.3, "$lte": 0.7}},
        {"_id": 0, "uri": 1, "text": 1, "ai_language": 1,
         "prediction_label": 1, "ai_score_credibility": 1,
         "text_word_count": 1, "created_at": 1, "author_handle": 1,
         "search_term": 1}
    )
    frontiere_posts = list(cursor_frontiere)
    if len(frontiere_posts) > n_frontiere:
        frontiere_posts = random.sample(frontiere_posts, n_frontiere)
    for p in frontiere_posts:
        p["strate"] = "frontiere"
    all_posts.extend(frontiere_posts)
    print(f"  Cas frontiere (score 0.3-0.7): {len(frontiere_posts)}/{n_frontiere}")

    # Dedupliquer par URI
    seen = set()
    unique_posts = []
    for p in all_posts:
        if p["uri"] not in seen:
            seen.add(p["uri"])
            unique_posts.append(p)

    random.shuffle(unique_posts)
    print(f"\nTotal: {len(unique_posts)} posts uniques extraits")
    return unique_posts


def create_annotation_csv(posts, output_path):
    """
    Cree le CSV d'annotation avec colonnes pour 2 annotateurs.
    """
    rows = []
    for i, p in enumerate(posts, 1):
        rows.append({
            "id": i,
            "uri": p.get("uri", ""),
            "text": p.get("text", ""),
            "langue": p.get("ai_language", ""),
            "nb_mots": p.get("text_word_count", 0),
            "strate": p.get("strate", ""),
            "search_term": p.get("search_term", ""),
            # Colonnes d'annotation (a remplir manuellement)
            "annotateur_1": "",           # fiable / suspect
            "confiance_1": "",            # 1=pas sur, 2=assez sur, 3=certain
            "justification_1": "",        # texte libre court
            "annotateur_2": "",           # fiable / suspect
            "confiance_2": "",            # 1=pas sur, 2=assez sur, 3=certain
            "justification_2": "",        # texte libre court
            # Colonnes de resolution (apres annotation)
            "label_final": "",            # fiable / suspect (apres arbitrage)
            "desaccord": "",              # oui / non
            # Prediction IA (MASQUE pendant l'annotation, visible apres)
            "ia_prediction": p.get("prediction_label", ""),
            "ia_score": round(p.get("ai_score_credibility", 0), 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")

    # Version sans predictions IA (pour les annotateurs)
    blind_path = output_path.replace(".csv", "_blind.csv")
    df_blind = df.drop(columns=["ia_prediction", "ia_score"])
    df_blind.to_csv(blind_path, index=False, encoding="utf-8")

    print(f"\nFichiers crees :")
    print(f"  {output_path} (complet, avec predictions IA)")
    print(f"  {blind_path} (aveugle, pour les annotateurs)")

    return df


def print_stats(df):
    """Affiche les statistiques de l'echantillon."""
    print("\n--- Statistiques de l'echantillon ---")
    print(f"Total: {len(df)} posts")
    print(f"\nPar langue:")
    print(df["langue"].value_counts().to_string())
    print(f"\nPar strate:")
    print(df["strate"].value_counts().to_string())
    print(f"\nNombre de mots - stats:")
    print(f"  Moyenne: {df['nb_mots'].mean():.1f}")
    print(f"  Mediane: {df['nb_mots'].median():.0f}")
    print(f"  Min/Max: {df['nb_mots'].min()}/{df['nb_mots'].max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction gold test set")
    parser.add_argument("--n_posts", type=int, default=200,
                        help="Nombre total de posts (default: 200)")
    parser.add_argument("--output", type=str,
                        default="data/gold_test_set.csv",
                        help="Chemin de sortie")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed aleatoire")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=== Extraction Gold Test Set - Thumalien ===")
    print(f"Objectif: {args.n_posts} posts\n")

    collection = get_mongo_collection()
    total = collection.count_documents({"ai_processed": True})
    print(f"Posts disponibles dans MongoDB: {total}\n")

    posts = extract_stratified_sample(collection, args.n_posts)
    df = create_annotation_csv(posts, args.output)
    print_stats(df)

    print("\n=== Prochaines etapes ===")
    print("1. Distribuer le fichier *_blind.csv aux 2 annotateurs")
    print("2. Chaque annotateur remplit: annotateur_X, confiance_X, justification_X")
    print("3. Fusionner les annotations dans le fichier complet")
    print("4. Calculer le Cohen's Kappa (notebook 21)")
    print("5. Resoudre les desaccords par discussion")
