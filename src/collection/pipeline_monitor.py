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

    def get_health_status(self, num_cycles=5):
        """
        Read the last *num_cycles* cycle reports and return a health summary.

        Returns a dict with:
            - avg_posts_new: average new posts per cycle
            - error_rate: errors / keywords_processed (0.0 – 1.0)
            - status: "healthy", "degraded", or "unhealthy"
            - cycles_analysed: how many cycles were actually read
        """
        reports = self._read_last_reports(num_cycles)
        if not reports:
            return {
                "avg_posts_new": 0,
                "error_rate": 0.0,
                "status": "unhealthy",
                "cycles_analysed": 0,
            }

        total_posts = sum(r.get("posts_new", 0) for r in reports)
        total_errors = sum(r.get("errors", 0) for r in reports)
        total_keywords = sum(r.get("keywords_processed", 0) for r in reports)

        avg_posts = total_posts / len(reports)
        error_rate = (total_errors / total_keywords) if total_keywords > 0 else 0.0

        if error_rate < 0.10 and total_posts > 0:
            status = "healthy"
        elif error_rate <= 0.50:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "avg_posts_new": round(avg_posts, 2),
            "error_rate": round(error_rate, 4),
            "status": status,
            "cycles_analysed": len(reports),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _read_last_reports(cls, n=5):
        """Read the last *n* JSONL lines from the metrics file."""
        if not os.path.exists(cls.METRICS_FILE):
            return []
        lines = []
        try:
            with open(cls.METRICS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)
        except OSError:
            return []
        # Keep only the last n lines and parse them
        reports = []
        for raw in lines[-n:]:
            try:
                reports.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return reports

    def _reset(self):
        self.posts_new = 0
        self.duplicates_skipped = 0
        self.errors = 0
        self.error_details = []
        self.keywords_processed = 0
