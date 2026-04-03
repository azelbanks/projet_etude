"""
Thumalien — Weekly Score Monitoring
====================================

Connects to MongoDB, samples recent posts, runs the ExpertFakeNewsDetector V2,
computes key metrics, and writes a JSONL report to logs/weekly_scores.jsonl.

Usage:
    python -m src.monitoring.weekly_score_check

Environment variables (optional):
    MONGO_USER      MongoDB username (omit for no-auth)
    MONGO_PASSWORD  MongoDB password (omit for no-auth)
    MONGO_HOST      MongoDB host (default: localhost:27017)
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pymongo import MongoClient

# ---------------------------------------------------------------------------
#  Project root = 2 levels up from this file  (src/monitoring/ -> project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from pipeline.expert_detector import ExpertFakeNewsDetector

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
)

SAMPLE_SIZE = 1000
SUSPECT_RATE_ALERT_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
#  MongoDB connection
# ---------------------------------------------------------------------------

def _build_mongo_uri() -> str:
    """Build a MongoDB URI from env vars, supporting both auth and no-auth."""
    user = os.environ.get('MONGO_USER', '')
    password = os.environ.get('MONGO_PASSWORD', '')
    host = os.environ.get('MONGO_HOST', 'localhost:27017')

    if user and password:
        return f'mongodb://{user}:{password}@{host}/'
    return f'mongodb://{host}/'


def fetch_recent_posts(n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Fetch the *n* most recent posts from MongoDB."""
    uri = _build_mongo_uri()
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.server_info()  # fail fast if unreachable

    db = client['thumalien_db']
    docs = list(
        db['raw_posts']
        .find({'text': {'$exists': True}}, {'_id': 0, 'text': 1, 'collected_at': 1})
        .sort('collected_at', -1)
        .limit(n)
    )
    if not docs:
        raise RuntimeError('No posts found in MongoDB collection raw_posts.')
    return pd.DataFrame(docs)


# ---------------------------------------------------------------------------
#  Scoring
# ---------------------------------------------------------------------------

def run_scoring(df: pd.DataFrame, detector: ExpertFakeNewsDetector) -> dict:
    """Run predict() and compute aggregate metrics."""
    results = detector.predict(pd.Series(df['text'].values))

    labels = results['prediction_label'].values
    scores = results['ai_score_credibility'].values

    n_total = len(labels)
    n_suspect = int((labels == 1).sum())
    suspect_rate = round(float(n_suspect / n_total), 4)
    mean_credibility = round(float(np.mean(scores)), 4)
    std_credibility = round(float(np.std(scores)), 4)

    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'n_sampled': n_total,
        'n_suspect': n_suspect,
        'suspect_rate': suspect_rate,
        'mean_credibility': mean_credibility,
        'std_credibility': std_credibility,
    }


# ---------------------------------------------------------------------------
#  Report writing
# ---------------------------------------------------------------------------

def write_report(report: dict, path: str) -> None:
    """Append a single JSONL line to *path*, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(report, ensure_ascii=False) + '\n')
    logger.info('Report written to %s', path)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info('=== Weekly Score Check — start ===')

    # 1. Fetch posts
    logger.info('Connecting to MongoDB ...')
    df = fetch_recent_posts(SAMPLE_SIZE)
    logger.info('Fetched %d posts.', len(df))

    # 2. Load model
    model_dir = os.path.join(PROJECT_ROOT, 'models')
    detector = ExpertFakeNewsDetector(model_dir=model_dir, threshold=0.44)
    v2_path = os.path.join(model_dir, 'model_expert_v2.pkl')
    suffix = 'expert_v2' if os.path.exists(v2_path) else 'expert'
    detector.load(suffix=suffix)
    logger.info('Model loaded (suffix=%s).', suffix)

    # 3. Score
    report = run_scoring(df, detector)
    logger.info(
        'Results — suspect_rate=%.2f%%, mean_cred=%.4f, std_cred=%.4f',
        report['suspect_rate'] * 100,
        report['mean_credibility'],
        report['std_credibility'],
    )

    # 4. Write JSONL
    report_path = os.path.join(PROJECT_ROOT, 'logs', 'weekly_scores.jsonl')
    write_report(report, report_path)

    # 5. Alert
    if report['suspect_rate'] > SUSPECT_RATE_ALERT_THRESHOLD:
        logger.warning(
            'ALERT: suspect_rate %.2f%% exceeds threshold %.0f%%. '
            'Investigate potential model drift or data quality issue.',
            report['suspect_rate'] * 100,
            SUSPECT_RATE_ALERT_THRESHOLD * 100,
        )
        print(
            f"\n*** ALERT *** suspect_rate={report['suspect_rate']:.2%} "
            f"> {SUSPECT_RATE_ALERT_THRESHOLD:.0%} threshold\n"
        )
    else:
        logger.info('suspect_rate within normal range.')

    logger.info('=== Weekly Score Check — done ===')


if __name__ == '__main__':
    main()
