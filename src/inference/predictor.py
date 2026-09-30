"""
Real-time inference orchestrator for fraud detection.

Integrates:
- Artifact loading (Feature Engineer, Supervised Model, Anomaly Detector, Metadata)
- Low-latency single-transaction scoring (<10ms)
- Explanations generation
- Local SQLite logging for auditability and monitoring
"""

import os
import time
import json
import uuid
import sqlite3
import logging
from typing import Dict, Any, Optional
import joblib
import pandas as pd

from api.schemas import TransactionRequest, PredictionResponse, RiskReason
from src.decision_engine.engine import DecisionEngine
from src.explainability.explainer import FraudExplainer

logger = logging.getLogger(__name__)


class FraudPredictor:
    """
    High-performance real-time inference engine.
    """

    def __init__(self, artifacts_dir: str = "models/artifacts", db_path: str = "fraud_detection.db"):
        self.artifacts_dir = artifacts_dir
        self.db_path = db_path
        self.model = None
        self.feature_engineer = None
        self.anomaly_detector = None
        self.explainer = None
        self.decision_engine = None
        self.metadata = {}
        self.is_ready = False
        self._init_storage()
        self.load_artifacts()

    def _init_storage(self) -> None:
        """Initialize lightweight SQLite audit storage for predictions."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transaction_logs (
                    transaction_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    amount REAL,
                    risk_score REAL,
                    decision TEXT,
                    supervised_prob REAL,
                    anomaly_score REAL,
                    model_version TEXT,
                    latency_ms REAL,
                    top_reason TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize SQLite storage: {e}")

    def load_artifacts(self) -> None:
        """Load trained models and pipeline components."""
        model_path = os.path.join(self.artifacts_dir, "best_model.joblib")
        fe_path = os.path.join(self.artifacts_dir, "feature_engineer.joblib")
        anom_path = os.path.join(self.artifacts_dir, "isolation_forest.joblib")
        meta_path = os.path.join(self.artifacts_dir, "metadata.json")

        if not os.path.exists(model_path) or not os.path.exists(fe_path):
            logger.warning(f"Artifacts missing in '{self.artifacts_dir}'. Predictor not ready.")
            self.is_ready = False
            return

        logger.info("Loading serialized inference artifacts...")
        self.model = joblib.load(model_path)
        self.feature_engineer = joblib.load(fe_path)
        self.anomaly_detector = joblib.load(anom_path) if os.path.exists(anom_path) else None

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)

        self.decision_engine = DecisionEngine()
        self.explainer = FraudExplainer(self.model, self.metadata.get("feature_names", []))
        self.is_ready = True
        logger.info("FraudPredictor successfully initialized and ready for real-time inference.")

    def predict(self, request: TransactionRequest) -> PredictionResponse:
        """
        Execute end-to-end real-time scoring for a single transaction.
        """
        if not self.is_ready:
            raise RuntimeError("FraudPredictor artifacts are not loaded. Train the models first.")

        t_start = time.perf_counter()

        # 1. Prepare raw input dictionary
        tx_id = request.transaction_id or f"tx_{uuid.uuid4().hex[:12]}"
        tx_dict = {
            "Time": request.time if request.time is not None else 3600.0,
            "Amount": request.amount
        }
        # Populate V1 to V28
        req_dict = request.model_dump(by_alias=True)
        for i in range(1, 29):
            col = f"V{i}"
            tx_dict[col] = float(req_dict.get(col, 0.0))

        # 2. Feature transformation
        features_df = self.feature_engineer.transform_single(tx_dict)

        # 3. Supervised Calibrated Probability
        prob_arr = self.model.predict_proba(features_df)
        supervised_prob = float(prob_arr[0, 1])

        # 4. Anomaly score
        anomaly_score = 0.0
        if self.anomaly_detector is not None:
            anomaly_score = float(self.anomaly_detector.score_samples(features_df)[0])

        # 5. Decision Engine evaluation
        decision_res = self.decision_engine.evaluate(supervised_prob, anomaly_score)

        # 6. Top contributing explanation reasons
        raw_reasons = self.explainer.explain_transaction(features_df, top_k=3)
        reasons = [RiskReason(**r) for r in raw_reasons]

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # 7. Async audit logging
        self._log_transaction(
            tx_id=tx_id,
            amount=request.amount,
            risk_score=decision_res.risk_score,
            decision=decision_res.decision,
            supervised_prob=supervised_prob,
            anomaly_score=anomaly_score,
            model_version=self.metadata.get("model_name", "1.0.0"),
            latency_ms=latency_ms,
            top_reason=reasons[0].description if reasons else "Normal profile"
        )

        return PredictionResponse(
            transaction_id=tx_id,
            risk_score=decision_res.risk_score,
            decision=decision_res.decision,
            supervised_probability=supervised_prob,
            anomaly_score=anomaly_score,
            thresholds={
                "review_threshold": decision_res.review_threshold,
                "decline_threshold": decision_res.decline_threshold
            },
            top_reasons=reasons,
            model_version=self.metadata.get("model_name", "1.0.0"),
            processing_time_ms=round(latency_ms, 2)
        )

    def _log_transaction(
        self,
        tx_id: str,
        amount: float,
        risk_score: float,
        decision: str,
        supervised_prob: float,
        anomaly_score: float,
        model_version: str,
        latency_ms: float,
        top_reason: str
    ) -> None:
        """Store transaction record into audit SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO transaction_logs 
                (transaction_id, timestamp, amount, risk_score, decision, supervised_prob, anomaly_score, model_version, latency_ms, top_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (tx_id, time.time(), amount, risk_score, decision, supervised_prob, anomaly_score, model_version, latency_ms, top_reason))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to record transaction log: {e}")
