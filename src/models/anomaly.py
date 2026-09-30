"""
Anomaly detection module using Isolation Forest.

Key Distinction:
- Supervised Fraud Classifier answers: "Does this transaction match historical fraud patterns?"
- Unsupervised Anomaly Detector answers: "Does this transaction deviate significantly from normal baseline activity?"

Novel / zero-day fraud attacks often appear first as anomalies rather than established fraud signatures.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


class AnomalyRiskDetector:
    """
    Isolation Forest anomaly detector that produces normalized [0.0, 1.0] anomaly risk scores.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        contamination: float = 0.002,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.min_score: float = -0.5
        self.max_score: float = 0.5
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, train_on_normal_only: bool = True, y: np.ndarray = None) -> "AnomalyRiskDetector":
        """
        Fit Isolation Forest.
        Optionally fits on normal transactions only (semi-supervised anomaly detection).
        """
        if train_on_normal_only and y is not None:
            mask_normal = (y == 0)
            X_train = X[mask_normal]
            logger.info(f"Fitting Anomaly Detector on {len(X_train):,} verified normal transactions...")
        else:
            X_train = X
            logger.info(f"Fitting Anomaly Detector on all {len(X_train):,} training transactions...")

        self.model.fit(X_train)

        # Calibrate score range for [0, 1] normalization
        raw_scores = self.model.decision_function(X_train)
        self.min_score = float(np.percentile(raw_scores, 0.5))
        self.max_score = float(np.percentile(raw_scores, 99.5))
        self.is_fitted = True
        return self

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute normalized anomaly risk score in [0.0, 1.0].
        Higher score = more anomalous / unusual behavior.
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyRiskDetector must be fitted before predicting.")

        raw_scores = self.model.decision_function(X)
        # Raw decision_function: lower (more negative) = more abnormal
        # Invert so higher = more abnormal
        inverted_scores = -raw_scores
        norm_scores = (inverted_scores - (-self.max_score)) / ((-self.min_score) - (-self.max_score) + 1e-6)
        return np.clip(norm_scores, 0.0, 1.0)

    def evaluate_against_labels(self, X: pd.DataFrame, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate how well the unsupervised anomaly score correlates with actual fraud.
        """
        anomaly_scores = self.score_samples(X)
        roc_auc = roc_auc_score(y_true, anomaly_scores)
        pr_auc = average_precision_score(y_true, anomaly_scores)

        logger.info(f"Anomaly Detector Evaluation vs Fraud Labels: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
        return {
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4)
        }
