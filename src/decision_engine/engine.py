"""
Unified Real-Time Decision Engine.

Coordinates:
1. Supervised Fraud Probability (calibrated)
2. Anomaly Signal (Isolation Forest outlier score)
3. Configurable weighted risk score fusion
4. Tri-State Decision Routing:
   - Risk < review_threshold: APPROVE
   - review_threshold <= Risk < decline_threshold: REVIEW
   - Risk >= decline_threshold: DECLINE
"""

import os
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DecisionResult(BaseModel):
    risk_score: float = Field(..., description="Combined unified risk score between 0.0 and 1.0")
    decision: str = Field(..., description="Operational decision: APPROVE, REVIEW, or DECLINE")
    supervised_probability: float = Field(..., description="Calibrated supervised fraud probability")
    anomaly_score: float = Field(..., description="Normalized unsupervised outlier score")
    review_threshold: float = Field(..., description="Active threshold for manual review queue")
    decline_threshold: float = Field(..., description="Active threshold for automatic decline")


class DecisionEngine:
    """
    Real-time fraud decision engine.
    """

    def __init__(
        self,
        review_threshold: Optional[float] = None,
        decline_threshold: Optional[float] = None,
        supervised_weight: Optional[float] = None,
        anomaly_weight: Optional[float] = None
    ):
        # Allow override or fallback to environment variables / defaults
        self.review_threshold = review_threshold if review_threshold is not None else float(
            os.getenv("REVIEW_THRESHOLD", "0.30")
        )
        self.decline_threshold = decline_threshold if decline_threshold is not None else float(
            os.getenv("FRAUD_THRESHOLD", "0.65")
        )
        self.supervised_weight = supervised_weight if supervised_weight is not None else float(
            os.getenv("SUPERVISED_WEIGHT", "0.80")
        )
        self.anomaly_weight = anomaly_weight if anomaly_weight is not None else float(
            os.getenv("ANOMALY_WEIGHT", "0.20")
        )

        # Normalize weights if needed
        total_w = self.supervised_weight + self.anomaly_weight
        if total_w > 0:
            self.supervised_weight /= total_w
            self.anomaly_weight /= total_w

    def compute_risk_score(self, supervised_prob: float, anomaly_score: float) -> float:
        """
        Synthesize supervised probability and unsupervised anomaly score
        into a unified, calibrated risk index in [0.0, 1.0].
        """
        combined = (self.supervised_weight * supervised_prob) + (self.anomaly_weight * anomaly_score)
        return float(min(1.0, max(0.0, combined)))

    def evaluate(self, supervised_prob: float, anomaly_score: float = 0.0) -> DecisionResult:
        """
        Evaluate a transaction risk score and determine operational disposition.
        """
        risk = self.compute_risk_score(supervised_prob, anomaly_score)

        if risk >= self.decline_threshold:
            decision = "DECLINE"
        elif risk >= self.review_threshold:
            decision = "REVIEW"
        else:
            decision = "APPROVE"

        return DecisionResult(
            risk_score=round(risk, 4),
            decision=decision,
            supervised_probability=round(float(supervised_prob), 4),
            anomaly_score=round(float(anomaly_score), 4),
            review_threshold=self.review_threshold,
            decline_threshold=self.decline_threshold
        )
