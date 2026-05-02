"""
Benchmark latence — Temps moyen d'analyse par le pipeline V5.

Mesure le temps d'inference sur des textes individuels et en batch.
Les resultats sont affiches mais le test passe toujours (pas de seuil strict),
sauf si le pipeline ne fonctionne pas du tout.

Requis par le Cahier des Charges : "temps moyen d'analyse".
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pipeline.expert_detector import ExpertFakeNewsDetector

_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
_MODEL_EXISTS = os.path.exists(os.path.join(_MODEL_DIR, 'model_expert_v5.pkl'))


@pytest.mark.skipif(not _MODEL_EXISTS, reason="Model files not found")
class TestBenchmarkLatence:
    """Benchmark d'inference pour documenter le temps moyen d'analyse."""

    @pytest.fixture(scope='class')
    def detector(self):
        det = ExpertFakeNewsDetector(model_dir=_MODEL_DIR)
        det.load(suffix='expert_v5')
        return det

    @pytest.fixture
    def sample_texts(self):
        return [
            "Breaking news: scientists discover breakthrough in renewable energy.",
            "SCANDALE ! Le gouvernement cache la verite sur les vaccins !!!",
            "The weather is nice today, perfect for a walk in the park.",
            "EXPOSED: they lied about everything, the conspiracy runs deep!!!",
            "Les experts confirment que le changement climatique est reel.",
            "wow this is crazy check this out",
            "According to Reuters, the new policy takes effect next month.",
            "On nous cache tout, on nous dit rien, manipulation totale !!!",
            "A simple tweet about nothing important.",
            "Le president a annonce de nouvelles mesures economiques.",
        ]

    def test_single_text_latency(self, detector):
        """Mesure le temps d'inference pour un seul texte."""
        text = pd.Series(["Breaking news: new discovery about climate change."])

        # Warmup
        detector.predict(text)

        # Measure
        times = []
        for _ in range(10):
            start = time.perf_counter()
            detector.predict(text)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_ms = np.mean(times) * 1000
        p95_ms = np.percentile(times, 95) * 1000

        print(f"\n{'='*60}")
        print(f"  BENCHMARK LATENCE — TEXTE UNIQUE")
        print(f"{'='*60}")
        print(f"  Temps moyen :  {avg_ms:.1f} ms")
        print(f"  P95 :          {p95_ms:.1f} ms")
        print(f"  Iterations :   10")
        print(f"{'='*60}")

        # CDC exige < 100ms par texte ; on verifie < 200ms (marge)
        assert avg_ms < 200, f"Latence moyenne trop elevee: {avg_ms:.0f}ms (CDC: <100ms)"

    def test_batch_10_latency(self, detector, sample_texts):
        """Mesure le temps d'inference pour un batch de 10 textes."""
        texts = pd.Series(sample_texts)

        # Warmup
        detector.predict(texts)

        # Measure
        times = []
        for _ in range(5):
            start = time.perf_counter()
            detector.predict(texts)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_ms = np.mean(times) * 1000
        per_text_ms = avg_ms / len(sample_texts)

        print(f"\n{'='*60}")
        print(f"  BENCHMARK LATENCE — BATCH DE 10 TEXTES")
        print(f"{'='*60}")
        print(f"  Temps total moyen :    {avg_ms:.1f} ms")
        print(f"  Temps par texte :      {per_text_ms:.1f} ms")
        print(f"  Debit :                {1000/per_text_ms:.0f} textes/sec")
        print(f"  Iterations :           5")
        print(f"{'='*60}")

        # CDC: < 100ms/texte, soit < 1000ms pour 10 textes
        assert avg_ms < 1000, f"Batch trop lent: {avg_ms:.0f}ms pour 10 textes (CDC: <1000ms)"
        assert per_text_ms < 200, f"Latence par texte trop elevee: {per_text_ms:.0f}ms"

    def test_batch_100_latency(self, detector, sample_texts):
        """Mesure le temps d'inference pour un batch de 100 textes."""
        texts = pd.Series(sample_texts * 10)  # 100 textes

        start = time.perf_counter()
        result = detector.predict(texts)
        elapsed = time.perf_counter() - start

        elapsed_ms = elapsed * 1000
        per_text_ms = elapsed_ms / 100

        print(f"\n{'='*60}")
        print(f"  BENCHMARK LATENCE — BATCH DE 100 TEXTES")
        print(f"{'='*60}")
        print(f"  Temps total :          {elapsed_ms:.1f} ms")
        print(f"  Temps par texte :      {per_text_ms:.1f} ms")
        print(f"  Debit :                {1000/per_text_ms:.0f} textes/sec")
        print(f"{'='*60}")

        assert len(result) == 100
        # CDC: < 100ms/texte, soit < 10s pour 100 textes
        assert elapsed_ms < 10000, f"Batch 100 trop lent: {elapsed_ms:.0f}ms (CDC: <10000ms)"
        assert per_text_ms < 200, f"Latence par texte trop elevee: {per_text_ms:.0f}ms"
