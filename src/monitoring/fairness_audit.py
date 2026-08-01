"""
ThumaCheck -- Audit d'equite algorithmique
==========================================
Calcule des metriques de fairness par sous-groupe (langue, longueur de texte).
Usage: python -m src.monitoring.fairness_audit
"""

import os, sys, json, logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from pipeline.expert_detector import ExpertFakeNewsDetector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')


def compute_fairness_metrics(texts, labels_true, languages, text_lengths, detector):
    """
    Compute fairness metrics across subgroups.

    Returns dict with:
    - demographic_parity: difference in positive prediction rates between groups
    - equalized_odds: difference in TPR and FPR between groups
    - per_group_metrics: detailed metrics per subgroup
    """
    results = detector.predict(pd.Series(texts))
    pred_labels = results['prediction_label'].values
    scores = results['ai_score_credibility'].values

    metrics = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'n_total': len(texts),
        'groups': {}
    }

    # By language
    for lang in ['fr', 'en']:
        mask = np.array(languages) == lang
        if mask.sum() < 10:
            continue

        group_preds = pred_labels[mask]
        group_true = np.array(labels_true)[mask]
        group_scores = scores[mask]

        tp = ((group_preds == 1) & (group_true == 1)).sum()
        fp = ((group_preds == 1) & (group_true == 0)).sum()
        tn = ((group_preds == 0) & (group_true == 0)).sum()
        fn = ((group_preds == 0) & (group_true == 1)).sum()

        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
        positive_rate = float(group_preds.mean())
        mean_score = float(group_scores.mean())

        metrics['groups'][f'lang_{lang}'] = {
            'n': int(mask.sum()),
            'positive_rate': round(positive_rate, 4),
            'tpr': round(tpr, 4),
            'fpr': round(fpr, 4),
            'mean_score': round(mean_score, 4),
        }

    # By text length
    lengths = np.array(text_lengths)
    for name, lo, hi in [('ultra_short', 0, 15), ('short', 15, 30), ('medium', 30, 100), ('long', 100, 99999)]:
        mask = (lengths >= lo) & (lengths < hi)
        if mask.sum() < 10:
            continue

        group_preds = pred_labels[mask]
        group_true = np.array(labels_true)[mask]
        group_scores = scores[mask]

        tp = ((group_preds == 1) & (group_true == 1)).sum()
        fp = ((group_preds == 1) & (group_true == 0)).sum()
        tn = ((group_preds == 0) & (group_true == 0)).sum()
        fn = ((group_preds == 0) & (group_true == 1)).sum()

        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
        positive_rate = float(group_preds.mean())

        metrics['groups'][f'length_{name}'] = {
            'n': int(mask.sum()),
            'positive_rate': round(positive_rate, 4),
            'tpr': round(tpr, 4),
            'fpr': round(fpr, 4),
            'mean_score': round(float(group_scores.mean()), 4),
        }

    # Demographic parity difference (max gap in positive rate between language groups)
    lang_groups = [v for k, v in metrics['groups'].items() if k.startswith('lang_')]
    if len(lang_groups) >= 2:
        rates = [g['positive_rate'] for g in lang_groups]
        metrics['demographic_parity_diff'] = round(max(rates) - min(rates), 4)

    # Equalized odds difference (max gap in TPR between groups)
    if len(lang_groups) >= 2:
        tprs = [g['tpr'] for g in lang_groups]
        fprs = [g['fpr'] for g in lang_groups]
        metrics['equalized_odds_tpr_diff'] = round(max(tprs) - min(tprs), 4)
        metrics['equalized_odds_fpr_diff'] = round(max(fprs) - min(fprs), 4)

    return metrics


def write_fairness_report(report: dict, path: str = None) -> None:
    if path is None:
        path = os.path.join(PROJECT_ROOT, 'logs', 'fairness_audit.jsonl')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(report, ensure_ascii=False) + '\n')
    logger.info('Fairness report written to %s', path)


if __name__ == '__main__':
    logger.info('=== Fairness Audit ===')
    logger.info('This module requires labeled test data. Use compute_fairness_metrics() programmatically.')
    logger.info('Example: compute_fairness_metrics(texts, true_labels, languages, word_counts, detector)')
