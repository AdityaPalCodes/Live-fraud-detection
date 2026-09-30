"""
FastAPI Real-Time Fraud Detection Inference Service.

Endpoints:
- POST /predict: Evaluates incoming financial transactions in real time (<10ms).
- GET  /health: System health and model artifact readiness check.
- GET  /metrics: Live streaming inference metrics and decision distributions.
"""

import time
import sqlite3
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import TransactionRequest, PredictionResponse, HealthResponse
from src.inference.predictor import FraudPredictor

# Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("fraud_api")

# State & Predictor
predictor: FraudPredictor = None
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for artifact pre-loading."""
    global predictor
    logger.info("Initializing Real-Time Fraud Detection Service...")
    predictor = FraudPredictor()
    yield
    logger.info("Shutting down Fraud Detection Service...")


app = FastAPI(
    title="Real-Time Fraud Detection Service",
    description="Production-grade ML inference API with calibrated risk scoring, anomaly detection, and SHAP explainability.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for dashboard and microservices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Secure error handler to prevent internal trace exposure."""
    logger.error(f"Unhandled error processing {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred while evaluating transaction risk."}
    )


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """Return health status and loaded model information."""
    is_loaded = predictor is not None and predictor.is_ready
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_version=predictor.metadata.get("model_name", "1.0.0") if predictor else "none",
        uptime_seconds=round(time.time() - START_TIME, 1)
    )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
def predict_transaction(transaction: TransactionRequest) -> PredictionResponse:
    """
    Score financial transaction in real-time.
    Calculates calibrated fraud probability, anomaly score, and returns explainable decision.
    """
    if predictor is None or not predictor.is_ready:
        logger.error("Predictor not ready to process transactions.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are loading or uninitialized."
        )

    try:
        response = predictor.predict(transaction)
        logger.info(
            f"Scored {response.transaction_id}: "
            f"Amount=${transaction.amount:.2f}, Risk={response.risk_score:.4f}, "
            f"Decision={response.decision} ({response.processing_time_ms:.1f}ms)"
        )
        return response
    except Exception as e:
        logger.error(f"Inference error for {transaction.transaction_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process transaction: {str(e)}"
        )


@app.get("/metrics", tags=["Monitoring"])
def get_inference_metrics() -> Dict[str, Any]:
    """Retrieve operational KPIs from the transaction log database."""
    try:
        conn = sqlite3.connect(predictor.db_path if predictor else "fraud_detection.db")
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*), AVG(risk_score), AVG(latency_ms) FROM transaction_logs")
        total_tx, avg_risk, avg_latency = cursor.fetchone()

        if not total_tx:
            conn.close()
            return {
                "total_transactions": 0,
                "approval_rate": 100.0,
                "review_rate": 0.0,
                "decline_rate": 0.0,
                "average_risk_score": 0.0,
                "average_latency_ms": 0.0
            }

        cursor.execute("SELECT decision, COUNT(*) FROM transaction_logs GROUP BY decision")
        decisions = dict(cursor.fetchall())
        conn.close()

        approve_ct = decisions.get("APPROVE", 0)
        review_ct = decisions.get("REVIEW", 0)
        decline_ct = decisions.get("DECLINE", 0)

        return {
            "total_transactions": total_tx,
            "approval_rate": round(approve_ct / total_tx * 100.0, 2),
            "review_rate": round(review_ct / total_tx * 100.0, 2),
            "decline_rate": round(decline_ct / total_tx * 100.0, 2),
            "counts": {
                "APPROVE": approve_ct,
                "REVIEW": review_ct,
                "DECLINE": decline_ct
            },
            "average_risk_score": round(avg_risk or 0.0, 4),
            "average_latency_ms": round(avg_latency or 0.0, 2)
        }
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
