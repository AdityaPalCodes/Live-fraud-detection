"""
Real-time transaction generator and streaming simulator.

Generates realistic stream of financial transactions:
- ~98% legitimate transactions (normal amounts, standard latent profiles)
- ~2% suspicious & fraudulent transactions (burst amounts, nocturnal timing, high latent anomalies)

Supports sending directly to FastAPI (/predict) or publishing to a Kafka topic.
"""

import time
import uuid
import random
import logging
import argparse
from typing import Dict, Any, Generator, Optional
import httpx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("transaction_simulator")


def generate_transaction(is_fraud: Optional[bool] = None, fraud_rate: float = 0.05) -> Dict[str, Any]:
    """
    Generate a single realistic transaction.
    If is_fraud is None, samples based on fraud_rate.
    """
    if is_fraud is None:
        is_fraud = random.random() < fraud_rate

    tx_id = f"tx_{uuid.uuid4().hex[:10]}"
    t_seconds = time.time() % (48 * 3600)  # cyclical 48-hour window

    if is_fraud:
        # Fraudulent pattern: extreme or micro amounts, nocturnal, anomalous latent components
        amount = round(random.choice([
            random.uniform(0.5, 2.5),       # Card testing micro-auth
            random.uniform(450.0, 3200.0)   # High-ticket unauthorized drainage
        ]), 2)
        v_vector = np.random.normal(loc=0.0, scale=1.3, size=28)
        # Empirical fraud shifts
        v_vector[13] -= random.uniform(6.0, 9.5)  # V14
        v_vector[9] -= random.uniform(4.5, 7.5)   # V10
        v_vector[11] -= random.uniform(4.5, 6.5)  # V12
        v_vector[3] += random.uniform(3.5, 6.0)   # V4
        v_vector[10] += random.uniform(3.0, 5.0)  # V11
    else:
        # Legitimate consumer pattern: typical log-normal spend, standard PCA coordinates
        amount = round(float(np.exp(np.random.normal(loc=3.2, scale=1.1))), 2)
        v_vector = np.random.normal(loc=0.0, scale=1.0, size=28)

    tx_payload = {
        "transaction_id": tx_id,
        "amount": max(1.0, amount),
        "time": round(float(t_seconds), 2),
        "is_simulated_fraud": bool(is_fraud)
    }
    for i in range(1, 29):
        tx_payload[f"V{i}"] = round(float(v_vector[i - 1]), 4)

    return tx_payload


def stream_transactions_to_api(
    api_url: str = "http://localhost:8000/predict",
    rate_per_sec: float = 2.0,
    fraud_rate: float = 0.10,
    max_count: Optional[int] = None
) -> None:
    """
    Continuously send simulated transactions to the FastAPI endpoint.
    """
    logger.info(f"Starting transaction streaming simulator -> {api_url} at {rate_per_sec} tx/s...")
    count = 0
    delay = 1.0 / max(0.1, rate_per_sec)

    with httpx.Client(timeout=5.0) as client:
        while True:
            if max_count and count >= max_count:
                logger.info(f"Stream simulation reached max_count={max_count}. Done.")
                break

            tx = generate_transaction(fraud_rate=fraud_rate)
            is_ground_truth_fraud = tx.pop("is_simulated_fraud")

            try:
                t0 = time.perf_counter()
                resp = client.post(api_url, json=tx)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()
                    decision = data["decision"]
                    risk = data["risk_score"]
                    color = "🔴" if decision == "DECLINE" else ("🟡" if decision == "REVIEW" else "🟢")
                    logger.info(
                        f"{color} [{decision}] {tx['transaction_id']} | "
                        f"Amount=${tx['amount']:>7.2f} | Risk={risk:.4f} | "
                        f"Latency={elapsed_ms:.1f}ms | Truth={'FRAUD' if is_ground_truth_fraud else 'LEGIT'}"
                    )
                else:
                    logger.error(f"API Error {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.warning(f"Connection error reaching {api_url}: {e}")

            count += 1
            time.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Transaction Stream Simulator")
    parser.add_argument("--api-url", default="http://localhost:8000/predict", help="FastAPI predict endpoint URL")
    parser.add_argument("--rate", type=float, default=2.0, help="Transactions per second")
    parser.add_argument("--fraud-rate", type=float, default=0.10, help="Fraction of fraudulent transactions")
    parser.add_argument("--count", type=int, default=20, help="Maximum transactions to generate")
    args = parser.parse_args()

    stream_transactions_to_api(
        api_url=args.api_url,
        rate_per_sec=args.rate,
        fraud_rate=args.fraud_rate,
        max_count=args.count
    )
