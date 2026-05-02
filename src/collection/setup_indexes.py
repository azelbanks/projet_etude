"""
setup_indexes.py — MongoDB index and schema validation setup for thumalien_db.raw_posts.

Usage:
    python3 src/collection/setup_indexes.py

Environment variables (optional, falls back to no-auth local connection):
    MONGO_USER, MONGO_PASSWORD, MONGO_HOST
"""

import os
from pymongo import MongoClient, DESCENDING, ASCENDING, TEXT
from dotenv import load_dotenv

load_dotenv()


def connect_db():
    """Connect to MongoDB with optional authentication."""
    user = os.getenv("MONGO_USER")
    password = os.getenv("MONGO_PASSWORD")
    host = os.getenv("MONGO_HOST", "localhost")

    if user and password:
        uri = f"mongodb://{user}:{password}@{host}:27017/"
    else:
        uri = f"mongodb://{host}:27017/"

    print(f"Connexion a MongoDB : {uri.replace(password, '***') if password else uri}")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    print("MongoDB connecte.")
    return client["thumalien_db"]


def setup_indexes(db):
    """Create all required indexes on the raw_posts collection."""
    col = db["raw_posts"]

    # 1. Unique index on uri (deduplication)
    name = col.create_index([("uri", ASCENDING)], unique=True, name="idx_uri_unique")
    print(f"  Index cree : {name}  (uri, unique)")

    # 2. Descending index on collected_at (temporal queries)
    name = col.create_index([("collected_at", DESCENDING)], name="idx_collected_at_desc")
    print(f"  Index cree : {name}  (collected_at descending)")

    # 3. Compound index for dashboard filters
    name = col.create_index(
        [("prediction_label", ASCENDING), ("ai_language", ASCENDING)],
        name="idx_prediction_label_ai_language",
    )
    print(f"  Index cree : {name}  (prediction_label + ai_language)")

    # 4. Index on ai_processed (batch processing filter)
    name = col.create_index([("ai_processed", ASCENDING)], name="idx_ai_processed")
    print(f"  Index cree : {name}  (ai_processed)")

    # 5. Index on search_term (keyword queries)
    name = col.create_index([("search_term", ASCENDING)], name="idx_search_term")
    print(f"  Index cree : {name}  (search_term)")

    # 6. Text index on text (full-text search)
    name = col.create_index([("text", TEXT)], name="idx_text_fulltext")
    print(f"  Index cree : {name}  (text, full-text)")

    # 7. TTL index on collected_at — automatic deletion after 12 months (RGPD data retention)
    name = col.create_index(
        [("collected_at", ASCENDING)],
        expireAfterSeconds=31536000,
        name="idx_collected_at_ttl_12months",
    )
    print(f"  Index cree : {name}  (collected_at TTL 365 jours)")

    print("Tous les index ont ete crees avec succes.")


def setup_schema_validation(db):
    """Apply JSON Schema validation to raw_posts (moderate level — new inserts only)."""
    validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["uri", "text", "collected_at", "author_did"],
            "properties": {
                "uri": {
                    "bsonType": "string",
                    "description": "Identifiant unique du post (AT URI)",
                },
                "text": {
                    "bsonType": "string",
                    "description": "Contenu textuel du post",
                },
                "collected_at": {
                    "bsonType": "date",
                    "description": "Date/heure de collecte",
                },
                "author_did": {
                    "bsonType": "string",
                    "description": "DID de l'auteur",
                },
            },
        }
    }

    db.command(
        "collMod",
        "raw_posts",
        validator=validator,
        validationLevel="moderate",
    )
    print("Schema validation appliquee (moderate) — champs requis : uri, text, collected_at, author_did")


if __name__ == "__main__":
    print("=== Setup des index et validation pour thumalien_db.raw_posts ===\n")
    db = connect_db()
    print()
    setup_indexes(db)
    print()
    setup_schema_validation(db)
    print("\nTermine.")
