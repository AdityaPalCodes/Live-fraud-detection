"""
Pydantic schemas for the Real-Time Fraud Detection API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class TransactionRequest(BaseModel):
    """Transaction payload sent for real-time risk scoring."""
    transaction_id: Optional[str] = Field(None, description="Unique transaction reference ID")
    amount: float = Field(..., gt=0.0, description="Transaction monetary amount (must be positive)")
    time: Optional[float] = Field(None, ge=0.0, description="Elapsed seconds since reference epoch")
    timestamp: Optional[str] = Field(None, description="Optional ISO timestamp string (e.g. 2026-09-30T14:30:00)")

    # Latent features V1 to V28
    v1: Optional[float] = Field(0.0, alias="V1")
    v2: Optional[float] = Field(0.0, alias="V2")
    v3: Optional[float] = Field(0.0, alias="V3")
    v4: Optional[float] = Field(0.0, alias="V4")
    v5: Optional[float] = Field(0.0, alias="V5")
    v6: Optional[float] = Field(0.0, alias="V6")
    v7: Optional[float] = Field(0.0, alias="V7")
    v8: Optional[float] = Field(0.0, alias="V8")
    v9: Optional[float] = Field(0.0, alias="V9")
    v10: Optional[float] = Field(0.0, alias="V10")
    v11: Optional[float] = Field(0.0, alias="V11")
    v12: Optional[float] = Field(0.0, alias="V12")
    v13: Optional[float] = Field(0.0, alias="V13")
    v14: Optional[float] = Field(0.0, alias="V14")
    v15: Optional[float] = Field(0.0, alias="V15")
    v16: Optional[float] = Field(0.0, alias="V16")
    v17: Optional[float] = Field(0.0, alias="V17")
    v18: Optional[float] = Field(0.0, alias="V18")
    v19: Optional[float] = Field(0.0, alias="V19")
    v20: Optional[float] = Field(0.0, alias="V20")
    v21: Optional[float] = Field(0.0, alias="V21")
    v22: Optional[float] = Field(0.0, alias="V22")
    v23: Optional[float] = Field(0.0, alias="V23")
    v24: Optional[float] = Field(0.0, alias="V24")
    v25: Optional[float] = Field(0.0, alias="V25")
    v26: Optional[float] = Field(0.0, alias="V26")
    v27: Optional[float] = Field(0.0, alias="V27")
    v28: Optional[float] = Field(0.0, alias="V28")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "transaction_id": "tx_99814421",
                "amount": 284.50,
                "time": 45120.0,
                "V4": 4.12,
                "V10": -4.89,
                "V12": -5.10,
                "V14": -6.75
            }
        }


class RiskReason(BaseModel):
    feature: str = Field(..., description="Feature driving the risk score")
    description: str = Field(..., description="Human-interpretable explanation of the anomaly")
    contribution: float = Field(..., description="Estimated risk contribution impact")


class PredictionResponse(BaseModel):
    """Structured response returned by real-time inference service."""
    transaction_id: str
    risk_score: float
    decision: str  # APPROVE, REVIEW, DECLINE
    supervised_probability: float
    anomaly_score: float
    thresholds: Dict[str, float]
    top_reasons: List[RiskReason]
    model_version: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check status response."""
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
