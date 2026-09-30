"""
Integration tests for FastAPI inference endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create test client with active lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test GET /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data


def test_predict_endpoint_valid_request(client):
    """Test POST /predict with valid payload returns structured response."""
    payload = {
        "transaction_id": "tx_test_001",
        "amount": 42.50,
        "time": 25000.0,
        "V1": -0.5,
        "V2": 0.2,
        "V14": -0.1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == "tx_test_001"
    assert data["decision"] in ["APPROVE", "REVIEW", "DECLINE"]
    assert 0.0 <= data["risk_score"] <= 1.0
    assert 0.0 <= data["supervised_probability"] <= 1.0
    assert "thresholds" in data
    assert "top_reasons" in data
    assert "processing_time_ms" in data


def test_predict_endpoint_invalid_amount(client):
    """Test POST /predict rejects negative or zero amount with 422 error."""
    invalid_payload = {
        "transaction_id": "tx_bad",
        "amount": -50.0  # Invalid negative amount
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_metrics_endpoint(client):
    """Test GET /metrics returns aggregated operational KPIs."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()

    assert "total_transactions" in data
    assert "approval_rate" in data
    assert "review_rate" in data
    assert "decline_rate" in data
