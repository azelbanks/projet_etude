"""
ThumaCheck -- Prototype Kafka Consumer pour scalabilite
=======================================================

Consumer Kafka qui lit des messages (textes a analyser) depuis un topic
et les soumet au pipeline de detection de desinformation.

Ce module est un prototype demontrant la capacite de ThumaCheck a
s'integrer dans une architecture evenementielle (Kafka/Spark).

Usage :
    python -m src.scalability.kafka_consumer --topic thumacheck-texts --bootstrap-servers localhost:9092

Prerequis :
    pip install confluent-kafka

Auteur : Niamato Consulting (pour Thumalien)
"""

import json
import logging
import os
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    from confluent_kafka import Consumer, KafkaError

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.info("confluent-kafka non installe. Kafka consumer desactive.")


class ThumaCheckKafkaConsumer:
    """
    Consumer Kafka pour ThumaCheck.

    Lit des messages JSON depuis un topic Kafka et les analyse
    via le pipeline ExpertFakeNewsDetector.

    Format de message attendu :
        {"text": "Le texte a analyser", "id": "optionnel-uuid", "lang": "auto"}

    Resultats publies sur un topic de sortie (optionnel) :
        {"id": "...", "score": 0.72, "label": "FIABLE", "language": "fr"}
    """

    DEFAULT_CONFIG = {
        "bootstrap.servers": "localhost:9092",
        "group.id": "thumacheck-consumer-group",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "max.poll.interval.ms": 300000,
    }

    def __init__(
        self,
        topic: str = "thumacheck-texts",
        output_topic: str | None = "thumacheck-results",
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "thumacheck-consumer-group",
        batch_size: int = 10,
        model_dir: str | None = None,
    ):
        if not KAFKA_AVAILABLE:
            raise RuntimeError(
                "confluent-kafka non installe. Installez-le avec : pip install confluent-kafka"
            )

        self.topic = topic
        self.output_topic = output_topic
        self.batch_size = batch_size
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), "..", "..", "models")

        self._config = {
            **self.DEFAULT_CONFIG,
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
        }
        self._consumer = None
        self._producer = None
        self._detector = None
        self._running = False

        # Metrics
        # dict heterogene (compteurs int + horodatage float|None) : Any evite
        # de fausses erreurs de typage sur les operations arithmetiques.
        self.metrics: dict[str, Any] = {
            "messages_consumed": 0,
            "messages_processed": 0,
            "errors": 0,
            "start_time": None,  # float une fois start() appele
        }

    def _load_detector(self):
        """Load the ExpertFakeNewsDetector pipeline."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from pipeline.expert_detector import ExpertFakeNewsDetector

        detector = ExpertFakeNewsDetector(model_dir=self.model_dir)
        for suffix in ["expert_v5", "expert_v4", "expert_v3", "expert"]:
            model_path = os.path.join(self.model_dir, f"model_{suffix}.pkl")
            if os.path.exists(model_path):
                detector.load(suffix=suffix)
                logger.info("Kafka consumer: modele charge (%s)", suffix)
                return detector
        raise RuntimeError(f"Aucun modele trouve dans {self.model_dir}")

    def _init_producer(self):
        """Initialise un producer Kafka pour les resultats (optionnel)."""
        if self.output_topic is None:
            return
        try:
            from confluent_kafka import Producer

            self._producer = Producer(
                {
                    "bootstrap.servers": self._config["bootstrap.servers"],
                }
            )
        except Exception:
            logger.exception("Producer Kafka non disponible — resultats non publies")
            self._producer = None

    def start(self):
        """Demarre le consumer en boucle."""
        logger.info("Demarrage du consumer Kafka (topic=%s)", self.topic)

        self._detector = self._load_detector()
        self._consumer = Consumer(self._config)
        self._consumer.subscribe([self.topic])
        self._init_producer()

        self._running = True
        self.metrics["start_time"] = time.time()

        try:
            self._consume_loop()
        except KeyboardInterrupt:
            logger.info("Arret demande par l'utilisateur")
        finally:
            self.stop()

    def _consume_loop(self):
        """Boucle de consommation principale."""

        batch_texts = []
        batch_ids = []

        while self._running:
            msg = self._consumer.poll(timeout=1.0)
            if msg is None:
                # Timeout — process partial batch if any
                if batch_texts:
                    self._process_batch(batch_texts, batch_ids)
                    batch_texts, batch_ids = [], []
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("Kafka error: %s", msg.error())
                self.metrics["errors"] += 1
                continue

            try:
                payload = json.loads(msg.value().decode("utf-8"))
                text = payload.get("text", "").strip()
                msg_id = payload.get("id", f"msg-{self.metrics['messages_consumed']}")

                if text:
                    batch_texts.append(text)
                    batch_ids.append(msg_id)
                    self.metrics["messages_consumed"] += 1

                if len(batch_texts) >= self.batch_size:
                    self._process_batch(batch_texts, batch_ids)
                    batch_texts, batch_ids = [], []

            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning("Message mal forme: %s", e)
                self.metrics["errors"] += 1

    def _process_batch(self, texts, ids):
        """Analyse un batch de textes via le detector."""
        import pandas as pd

        try:
            results = self._detector.predict(pd.Series(texts))
            for i, msg_id in enumerate(ids):
                result = {
                    "id": msg_id,
                    "score": float(results["ai_score_credibility"].iloc[i]),
                    "label": "FIABLE"
                    if int(results["prediction_label"].iloc[i]) == 0
                    else "SUSPECT",
                    "language": str(results["language"].iloc[i]),
                }
                self._publish_result(result)
                self.metrics["messages_processed"] += 1

            logger.info(
                "Batch de %d messages traite (total: %d)",
                len(texts),
                self.metrics["messages_processed"],
            )
        except Exception:
            logger.exception("Erreur lors du traitement du batch")
            self.metrics["errors"] += len(texts)

    def _publish_result(self, result: dict):
        """Publie un resultat sur le topic de sortie."""
        if self._producer is None or self.output_topic is None:
            return
        try:
            self._producer.produce(
                self.output_topic,
                value=json.dumps(result).encode("utf-8"),
                key=result.get("id", "").encode("utf-8"),
            )
            self._producer.poll(0)
        except Exception:
            logger.exception("Publication du resultat echouee pour %s", result.get("id"))

    def stop(self):
        """Arrete le consumer proprement."""
        self._running = False
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None
        if self._producer is not None:
            self._producer.flush(timeout=5)
            self._producer = None

        elapsed = time.time() - (self.metrics["start_time"] or time.time())
        logger.info(
            "Consumer arrete — %d messages traites en %.1fs (%.1f msg/s)",
            self.metrics["messages_processed"],
            elapsed,
            self.metrics["messages_processed"] / max(elapsed, 1),
        )

    def get_metrics(self) -> dict:
        """Retourne les metriques du consumer."""
        elapsed = time.time() - (self.metrics["start_time"] or time.time())
        return {
            **self.metrics,
            "uptime_s": round(elapsed, 1),
            "throughput_msg_per_s": round(self.metrics["messages_processed"] / max(elapsed, 1), 2),
        }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="ThumaCheck Kafka Consumer")
    parser.add_argument("--topic", default="thumacheck-texts")
    parser.add_argument("--output-topic", default="thumacheck-results")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--group-id", default="thumacheck-consumer-group")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()

    consumer = ThumaCheckKafkaConsumer(
        topic=args.topic,
        output_topic=args.output_topic,
        bootstrap_servers=args.bootstrap_servers,
        group_id=args.group_id,
        batch_size=args.batch_size,
        model_dir=args.model_dir,
    )
    consumer.start()
