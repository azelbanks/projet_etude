"""Tests pour PipelineMonitor — monitoring du pipeline de collecte."""

import sys
import os
import json
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from collection.pipeline_monitor import PipelineMonitor


@pytest.fixture
def monitor(tmp_path):
    """Create a monitor that writes to a temp directory."""
    m = PipelineMonitor()
    PipelineMonitor.LOGS_DIR = str(tmp_path)
    PipelineMonitor.METRICS_FILE = str(tmp_path / "collection_metrics.jsonl")
    return m


class TestPipelineMonitorCycle:
    def test_start_and_end_cycle(self, monitor):
        monitor.start_cycle()
        monitor.record_keyword("test", "en", added=5, duplicates=2)
        report = monitor.end_cycle()
        assert report['posts_new'] == 5
        assert report['duplicates_skipped'] == 2
        assert report['keywords_processed'] == 1
        assert report['errors'] == 0
        assert 'cycle_start' in report
        assert 'cycle_end' in report
        assert report['duration_seconds'] >= 0

    def test_multiple_keywords(self, monitor):
        monitor.start_cycle()
        monitor.record_keyword("climate", "en", added=10, duplicates=3)
        monitor.record_keyword("vaccin", "fr", added=7, duplicates=1)
        monitor.record_keyword("error_kw", "en", errors=1, error_msg="timeout")
        report = monitor.end_cycle()
        assert report['posts_new'] == 17
        assert report['duplicates_skipped'] == 4
        assert report['keywords_processed'] == 3
        assert report['errors'] == 1
        assert len(report['error_details']) == 1
        assert report['error_details'][0]['keyword'] == 'error_kw'

    def test_writes_jsonl_file(self, monitor):
        monitor.start_cycle()
        monitor.record_keyword("test", "en", added=1)
        monitor.end_cycle()
        assert os.path.exists(PipelineMonitor.METRICS_FILE)
        with open(PipelineMonitor.METRICS_FILE, 'r') as f:
            line = f.readline()
            data = json.loads(line)
            assert data['posts_new'] == 1

    def test_appends_multiple_cycles(self, monitor):
        for i in range(3):
            monitor.start_cycle()
            monitor.record_keyword(f"kw{i}", "en", added=i + 1)
            monitor.end_cycle()
        with open(PipelineMonitor.METRICS_FILE, 'r') as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 3

    def test_reset_between_cycles(self, monitor):
        monitor.start_cycle()
        monitor.record_keyword("a", "en", added=10)
        monitor.end_cycle()
        monitor.start_cycle()
        report = monitor.end_cycle()
        assert report['posts_new'] == 0
        assert report['keywords_processed'] == 0


class TestHealthStatus:
    def test_healthy_status(self, monitor):
        for _ in range(5):
            monitor.start_cycle()
            monitor.record_keyword("kw", "en", added=10)
            monitor.end_cycle()
        health = monitor.get_health_status(5)
        assert health['status'] == 'healthy'
        assert health['avg_posts_new'] == 10.0
        assert health['error_rate'] == 0.0
        assert health['cycles_analysed'] == 5

    def test_degraded_status(self, monitor):
        for _ in range(3):
            monitor.start_cycle()
            monitor.record_keyword("kw1", "en", added=5)
            monitor.record_keyword("kw2", "en", errors=1, error_msg="fail")
            monitor.end_cycle()
        health = monitor.get_health_status(3)
        assert health['status'] == 'degraded'

    def test_unhealthy_no_file(self, monitor):
        health = monitor.get_health_status(5)
        assert health['status'] == 'unhealthy'
        assert health['cycles_analysed'] == 0

    def test_all_errors_unhealthy(self, monitor):
        for _ in range(3):
            monitor.start_cycle()
            monitor.record_keyword("kw", "en", errors=1, error_msg="fail")
            monitor.end_cycle()
        health = monitor.get_health_status(3)
        assert health['status'] == 'unhealthy'
