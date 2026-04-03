"""
pipeline_monitor.py — Lightweight monitoring for the Bluesky collection pipeline.

Writes one JSON line per collection cycle to logs/collection_metrics.jsonl.
"""

import json
import os
import datetime


class PipelineMonitor:
    """Tracks stats for a single collection cycle and persists them as JSONL."""

    LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
    METRICS_FILE = os.path.join(LOGS_DIR, "collection_metrics.jsonl")

    def __init__(self):
        self._start_time = None
        self._reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_cycle(self):
        """Call at the beginning of each collection cycle."""
        self._start_time = datetime.datetime.now()
        self._reset()

    def record_keyword(self, keyword, lang, added=0, duplicates=0, errors=0, error_msg=None):
        """Accumulate stats after processing one keyword."""
        self.posts_new += added
        self.duplicates_skipped += duplicates
        self.errors += errors
        self.keywords_processed += 1
        if error_msg:
            self.error_details.append({"keyword": keyword, "lang": lang, "error": str(error_msg)})

    def end_cycle(self):
        """Finalize the cycle: compute duration and write the JSONL report."""
        end_time = datetime.datetime.now()
        duration = (end_time - self._start_time).total_seconds() if self._start_time else 0.0

        report = {
            "cycle_start": self._start_time.isoformat() if self._start_time else None,
            "cycle_end": end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "posts_new": self.posts_new,
            "duplicates_skipped": self.duplicates_skipped,
            "errors": self.errors,
            "error_details": self.error_details,
            "keywords_processed": self.keywords_processed,
        }

        os.makedirs(self.LOGS_DIR, exist_ok=True)
        with open(self.METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")

        print(
            f"[Monitor] Cycle termine — {self.posts_new} nouveaux, "
            f"{self.duplicates_skipped} doublons, {self.errors} erreurs, "
            f"{self.keywords_processed} mots-cles, {round(duration, 1)}s"
        )
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self):
        self.posts_new = 0
        self.duplicates_skipped = 0
        self.errors = 0
        self.error_details = []
        self.keywords_processed = 0
