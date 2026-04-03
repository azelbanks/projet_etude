"""
data_quality_check.py — Periodic data quality report for the Thumalien raw_posts collection.

Usage:
    python3 src/collection/data_quality_check.py
"""

import os
import sys
import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_HOST = os.getenv("MONGO_HOST", "mongodb")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
DB_NAME = "thumalien_db"
COLLECTION_NAME = "raw_posts"


def _connect():
    if MONGO_USER and MONGO_PASSWORD:
        from urllib.parse import quote_plus
        uri = f"mongodb://{quote_plus(MONGO_USER)}:{quote_plus(MONGO_PASSWORD)}@{MONGO_HOST}:27017/?authSource=admin"
    else:
        uri = f"mongodb://{MONGO_HOST}:27017/"
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[DB_NAME][COLLECTION_NAME]


def run_quality_check(collection):
    """Run all quality checks and return a report dict."""
    report = {}

    # 1. Total documents
    total = collection.count_documents({})
    report["total_documents"] = total

    if total == 0:
        return report

    # 2. Missing required fields
    required_fields = ["text", "uri", "collected_at"]
    missing = {}
    for field in required_fields:
        count = collection.count_documents({field: {"$exists": False}})
        if count > 0:
            missing[field] = count
    report["missing_required_fields"] = missing if missing else "none"

    # 3. Empty text
    empty_text = collection.count_documents({
        "$or": [
            {"text": ""},
            {"text": None},
        ]
    })
    report["empty_text_count"] = empty_text

    # 4. Documents by language (ai_language field)
    lang_pipeline = [
        {"$group": {"_id": "$ai_language", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    lang_counts = {doc["_id"]: doc["count"] for doc in collection.aggregate(lang_pipeline)}
    report["documents_by_ai_language"] = lang_counts

    # 5. Documents by prediction_label
    label_pipeline = [
        {"$group": {"_id": "$prediction_label", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    label_counts = {doc["_id"]: doc["count"] for doc in collection.aggregate(label_pipeline)}
    report["documents_by_prediction_label"] = label_counts

    # 6. Duplicate URIs
    dup_pipeline = [
        {"$group": {"_id": "$uri", "count": {"$sum": 1}}},
        {"$match": {"count": {"$gt": 1}}},
        {"$count": "duplicate_uri_groups"},
    ]
    dup_result = list(collection.aggregate(dup_pipeline))
    report["duplicate_uri_groups"] = dup_result[0]["duplicate_uri_groups"] if dup_result else 0

    return report


def print_report(report):
    """Pretty-print the quality report to stdout."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 60

    print(sep)
    print(f"  DATA QUALITY REPORT  --  {now}")
    print(sep)
    print(f"  Total documents:          {report.get('total_documents', 0)}")
    print()

    if report.get("total_documents", 0) == 0:
        print("  (collection is empty, nothing to check)")
        print(sep)
        return

    # Missing fields
    missing = report.get("missing_required_fields", "none")
    print("  Missing required fields:")
    if missing == "none":
        print("    All documents have text, uri, collected_at")
    else:
        for field, count in missing.items():
            print(f"    {field}: {count} documents")
    print()

    # Empty text
    print(f"  Empty text:               {report.get('empty_text_count', 0)}")
    print()

    # Language distribution
    print("  Documents by ai_language:")
    lang_counts = report.get("documents_by_ai_language", {})
    if lang_counts:
        for lang, count in lang_counts.items():
            label = lang if lang is not None else "(not set)"
            print(f"    {label:20s} {count}")
    else:
        print("    (no data)")
    print()

    # Prediction label distribution
    print("  Documents by prediction_label:")
    label_counts = report.get("documents_by_prediction_label", {})
    if label_counts:
        for label, count in label_counts.items():
            name = label if label is not None else "(not set)"
            print(f"    {name:20s} {count}")
    else:
        print("    (no data)")
    print()

    # Duplicates
    dup = report.get("duplicate_uri_groups", 0)
    print(f"  Duplicate URI groups:     {dup}")
    print(sep)


def main():
    try:
        collection = _connect()
        print(f"Connected to MongoDB ({MONGO_HOST}) - {DB_NAME}.{COLLECTION_NAME}")
    except Exception as e:
        print(f"ERROR: Could not connect to MongoDB: {e}", file=sys.stderr)
        sys.exit(1)

    report = run_quality_check(collection)
    print_report(report)


if __name__ == "__main__":
    main()
