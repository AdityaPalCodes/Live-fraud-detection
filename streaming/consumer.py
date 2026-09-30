"""
Optional Kafka Fraud Detection Consumer.

Consumes transaction messages from Kafka topic 'financial_transactions',
scores each transaction in real time using FraudPredictor,
and produces high-risk alerts to 'fraud_alerts' topic.

Note: Kafka is strictly optional; the system operates independently via REST API.
"""

import os
import json
import logging
import argparse
from typing import Optional

from api.schemas import TransactionRequest
from src.inference.predictor import FraudPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("kafka_consumer")


def run_kafka_consumer(
    bootstrap_servers: str = "localhost:9092",
    input_topic: str = "financial_transactions",
    alerts_topic: str = "fraud_alerts",
    group_id: str = "fraud-detector-group"
) -> None:
    """
    Connect to Kafka, consume transaction events, and score in real time.
    """
    try:
        from kafka import KafkaConsumer, KafkaProducer
    except ImportError:
        logger.warning(
            "kafka-python package not installed. "
            "Kafka streaming is optional. To enable: pip install kafka-python-ng"
        )
        return

    logger.info(f"Connecting to Kafka at {bootstrap_servers}...")
    try:
        consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset="latest",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=10000
        )
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
    except Exception as e:
        logger.warning(f"Could not connect to Kafka broker ({e}). Exiting gracefully (Kafka is optional).")
        return

    predictor = FraudPredictor()
    logger.info(f"Listening for transactions on topic '{input_topic}'...")

    try:
        for message in consumer:
            tx_data = message.value
            try:
                req = TransactionRequest(**tx_data)
                result = predictor.predict(req)

                if result.decision in ["REVIEW", "DECLINE"]:
                    logger.warning(
                        f"ALERT [{result.decision}] {result.transaction_id}: "
                        f"Risk={result.risk_score:.4f}, Supervised={result.supervised_probability:.4f}"
                    )
                    producer.send(alerts_topic, value=result.model_dump())

            except Exception as ex:
                logger.error(f"Error scoring message offset {message.offset}: {ex}")
    except KeyboardInterrupt:
        logger.info("Kafka consumer stopped by user.")
    finally:
        consumer.close()
        producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka Streaming Fraud Consumer")
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    parser.add_argument("--input-topic", default="financial_transactions")
    parser.add_argument("--alerts-topic", default="fraud_alerts")
    args = parser.parse_args()

    run_kafka_consumer(
        bootstrap_servers=args.bootstrap,
        input_topic=args.input_topic,
        alerts_topic=args.alerts_topic
    )
