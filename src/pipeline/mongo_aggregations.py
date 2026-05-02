"""
Thumalien -- MongoDB Aggregation Helpers
========================================

Reusable aggregation functions for the Thumalien dashboard and pipeline.
Designed to push computation into MongoDB rather than loading raw documents
into Python, which is critical for performance at scale (> 100K posts).

Usage from the dashboard::

    from pipeline.mongo_aggregations import get_mongo_collection, get_overview_stats, get_recent_posts

    collection = get_mongo_collection()
    if collection is not None:
        stats = get_overview_stats(collection)
        recent = get_recent_posts(collection, limit=50)
"""

import os
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  MongoDB connection helper
# ---------------------------------------------------------------------------

def get_mongo_collection(
    db_name: str = "thumalien_db",
    collection_name: str = "raw_posts",
    timeout_ms: int = 2000,
):
    """
    Return a pymongo Collection, or None if MongoDB is unreachable.

    Connection priority:
      1. ``MONGODB_URI`` environment variable (supports auth)
      2. ``mongodb://localhost:27017/``
      3. ``mongodb://mongodb:27017/`` (Docker service name)

    Authentication is handled transparently when the URI contains
    credentials (e.g. ``mongodb://user:pass@host:27017/``).  The env vars
    ``MONGO_USER`` and ``MONGO_PASSWORD`` are also supported as a
    convenience -- they are injected into the fallback URIs when present.
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        logger.warning("pymongo is not installed -- MongoDB unavailable.")
        return None

    mongo_uri_env = os.environ.get("MONGODB_URI")
    mongo_user = os.environ.get("MONGO_USER", "")
    mongo_password = os.environ.get("MONGO_PASSWORD", "")

    # Build candidate URIs
    candidates: List[str] = []
    if mongo_uri_env:
        candidates.append(mongo_uri_env)

    if mongo_user and mongo_password:
        from urllib.parse import quote_plus
        auth_prefix = f"{quote_plus(mongo_user)}:{quote_plus(mongo_password)}@"
        candidates.append(f"mongodb://{auth_prefix}localhost:27017/")
        candidates.append(f"mongodb://{auth_prefix}mongodb:27017/")
    else:
        candidates.append("mongodb://localhost:27017/")
        candidates.append("mongodb://mongodb:27017/")

    for uri in candidates:
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
            client.server_info()  # force connection check
            db = client[db_name]
            return db[collection_name]
        except Exception:
            continue

    logger.info("Could not connect to MongoDB on any candidate URI.")
    return None


# ---------------------------------------------------------------------------
#  get_overview_stats  -- single $facet aggregation
# ---------------------------------------------------------------------------

def get_overview_stats(collection) -> Dict:
    """
    Return a dict of overview statistics computed entirely inside MongoDB.

    Keys returned::

        {
            "total_posts": int,
            "by_label": {"FIABLE": int, "SUSPECT": int, ...},
            "by_emotion": {"neutre": int, "colere": int, ...},
            "by_language": {"fr": int, "en": int, ...},
            "avg_credibility": float | None,
        }

    Uses a single ``$facet`` aggregation so MongoDB performs one pass
    over the data regardless of how many facets we request.
    """
    pipeline = [
        {
            "$facet": {
                # Total document count
                "total": [{"$count": "count"}],

                # Count by prediction_label
                "by_label": [
                    {
                        "$group": {
                            "_id": "$prediction_label",
                            "count": {"$sum": 1},
                        }
                    }
                ],

                # Count by ai_emotion
                "by_emotion": [
                    {
                        "$group": {
                            "_id": "$ai_emotion",
                            "count": {"$sum": 1},
                        }
                    }
                ],

                # Count by ai_language
                "by_language": [
                    {
                        "$group": {
                            "_id": "$ai_language",
                            "count": {"$sum": 1},
                        }
                    }
                ],

                # Average credibility score (skip nulls)
                "avg_cred": [
                    {
                        "$match": {
                            "ai_score_credibility": {"$exists": True, "$ne": None}
                        }
                    },
                    {
                        "$group": {
                            "_id": None,
                            "avg": {"$avg": "$ai_score_credibility"},
                        }
                    },
                ],
            }
        }
    ]

    try:
        raw = list(collection.aggregate(pipeline))
    except Exception as exc:
        logger.error("MongoDB aggregation failed: %s", exc)
        return {
            "total_posts": 0,
            "by_label": {},
            "by_emotion": {},
            "by_language": {},
            "avg_credibility": None,
        }

    facets = raw[0] if raw else {}

    total_posts = facets.get("total", [{}])[0].get("count", 0) if facets.get("total") else 0

    by_label = {
        doc["_id"]: doc["count"]
        for doc in facets.get("by_label", [])
        if doc["_id"] is not None
    }

    by_emotion = {
        doc["_id"]: doc["count"]
        for doc in facets.get("by_emotion", [])
        if doc["_id"] is not None
    }

    by_language = {
        doc["_id"]: doc["count"]
        for doc in facets.get("by_language", [])
        if doc["_id"] is not None
    }

    avg_cred_docs = facets.get("avg_cred", [])
    avg_credibility = float(avg_cred_docs[0]["avg"]) if avg_cred_docs else None

    return {
        "total_posts": total_posts,
        "by_label": by_label,
        "by_emotion": by_emotion,
        "by_language": by_language,
        "avg_credibility": avg_credibility,
    }


# ---------------------------------------------------------------------------
#  get_recent_posts
# ---------------------------------------------------------------------------

def get_recent_posts(collection, limit: int = 50) -> List[Dict]:
    """
    Return the *limit* most recent posts as a list of dicts.

    Only the fields needed by the dashboard are projected to minimise
    data transfer from MongoDB.
    """
    projection = {
        "_id": 0,
        "uri": 1,
        "text": 1,
        "collected_at": 1,
        "ai_score_credibility": 1,
        "ai_emotion": 1,
        "prediction_label": 1,
        "ai_v9_label": 1,
        "ai_post_type": 1,
        "ai_language": 1,
        "ai_model_name": 1,
        "search_term": 1,
    }

    try:
        cursor = (
            collection
            .find({"text": {"$exists": True}}, projection)
            .sort("collected_at", -1)
            .limit(limit)
        )
        return list(cursor)
    except Exception as exc:
        logger.error("MongoDB query failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
#  get_score_distribution  -- histogram of credibility scores
# ---------------------------------------------------------------------------

def get_score_distribution(collection, bins: int = 20) -> List[Dict]:
    """
    Return histogram data of credibility scores computed inside MongoDB.

    Each element in the returned list is a dict::

        {"bin_start": float, "bin_end": float, "count": int}

    The scores are split into *bins* equal-width buckets between 0 and 1.
    Documents without ``ai_score_credibility`` are excluded.
    """
    bin_width = 1.0 / bins
    boundaries = [round(i * bin_width, 6) for i in range(bins + 1)]

    pipeline = [
        {
            "$match": {
                "ai_score_credibility": {"$exists": True, "$ne": None},
            }
        },
        {
            "$bucket": {
                "groupBy": "$ai_score_credibility",
                "boundaries": boundaries,
                "default": "_other",
                "output": {"count": {"$sum": 1}},
            }
        },
    ]

    try:
        raw = list(collection.aggregate(pipeline))
    except Exception as exc:
        logger.error("MongoDB score distribution aggregation failed: %s", exc)
        return []

    result: List[Dict] = []
    for doc in raw:
        bucket_id = doc["_id"]
        if bucket_id == "_other":
            continue
        idx = boundaries.index(bucket_id) if bucket_id in boundaries else -1
        if idx < 0 or idx >= len(boundaries) - 1:
            continue
        result.append({
            "bin_start": boundaries[idx],
            "bin_end": boundaries[idx + 1],
            "count": doc["count"],
        })
    return result
