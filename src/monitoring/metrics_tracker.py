"""
Operational and model performance tracking module.

Queries the live transaction audit store and computes:
- Real-time transaction volumes
- Approval, Review, and Decline rate distributions
- Mean and percentile risk score statistics
- System latency metrics
"""

import sqlite3
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.models.evaluator import evaluate_predictions

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks streaming operational KPIs and evaluates model performance
    when ground truth feedback is recorded.
    """

    def __init__(self, db_path: str = "fraud_detection.db"):
        self.db_path = db_path

    def get_operational_summary(self) -> Dict[str, Any]:
        """Fetch real-time inference volume and decision statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM transaction_logs", conn)
            conn.close()

            if df.empty:
                return {
                    "total_transactions": 0,
                    "approval_rate": 100.0,
                    "review_rate": 0.0,
                    "decline_rate": 0.0,
                    "average_risk_score": 0.0,
                    "latency_p50_ms": 0.0,
                    "latency_p95_ms": 0.0
                }

            total = len(df)
            decision_counts = df["decision"].value_counts().to_dict()

            return {
                "total_transactions": total,
                "approval_rate": round(decision_counts.get("APPROVE", 0) / total * 100.0, 2),
                "review_rate": round(decision_counts.get("REVIEW", 0) / total * 100.0, 2),
                "decline_rate": round(decision_counts.get("DECLINE", 0) / total * 100.0, 2),
                "average_risk_score": round(float(df["risk_score"].mean()), 4),
                "latency_p50_ms": round(float(df["latency_ms"].median()), 2),
                "latency_p95_ms": round(float(np.percentile(df["latency_ms"], 95)), 2),
                "latest_transactions": df.tail(10).to_dict(orient="records")
            }
        except Exception as e:
            logger.error(f"Error reading operational metrics: {e}")
            return {"error": str(e)}

    def evaluate_ground_truth_cohort(
        self,
        labeled_df: pd.DataFrame,
        risk_column: str = "risk_score",
        label_column: str = "ground_truth_label",
        threshold: float = 0.65
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on an audited cohort once chargeback ground-truth is available.
        """
        y_true = labeled_df[label_column].values
        y_prob = labeled_df[risk_column].values
        return evaluate_predictions(y_true, y_prob, threshold=threshold)
