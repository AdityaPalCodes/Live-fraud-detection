"""
Unit tests for the Decision Engine, Cost Model, and Threshold Optimization.
"""

import pytest
import numpy as np
from src.decision_engine.engine import DecisionEngine
from src.decision_engine.cost_model import CostModel
from src.decision_engine.threshold import optimize_thresholds


def test_decision_routing_logic():
    """Verify tri-state APPROVE, REVIEW, DECLINE routing."""
    engine = DecisionEngine(review_threshold=0.30, decline_threshold=0.65)

    # Clean transaction
    res_approve = engine.evaluate(supervised_prob=0.10, anomaly_score=0.10)
    assert res_approve.decision == "APPROVE"
    assert res_approve.risk_score < 0.30

    # Suspicious transaction
    res_review = engine.evaluate(supervised_prob=0.45, anomaly_score=0.40)
    assert res_review.decision == "REVIEW"
    assert 0.30 <= res_review.risk_score < 0.65

    # Blatant fraud transaction
    res_decline = engine.evaluate(supervised_prob=0.90, anomaly_score=0.80)
    assert res_decline.decision == "DECLINE"
    assert res_decline.risk_score >= 0.65


def test_cost_model_calculation():
    """Verify financial cost calculation accuracy."""
    cost_model = CostModel(fp_cost=25.0, fn_cost=250.0)
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])  # 1 TN, 1 FP, 1 FN, 1 TP

    res = cost_model.calculate_cost(y_true, y_pred)
    assert res["fp_count"] == 1
    assert res["fn_count"] == 1
    assert res["fp_loss"] == 25.0
    assert res["fn_loss"] == 250.0
    assert res["total_loss"] == 275.0


def test_threshold_optimization_output():
    """Verify optimization yields valid thresholds."""
    y_true = np.array([0]*90 + [1]*10)
    y_prob = np.concatenate([np.random.uniform(0.0, 0.3, 90), np.random.uniform(0.7, 1.0, 10)])

    res = optimize_thresholds(y_true, y_prob, fp_cost=25.0, fn_cost=250.0)
    recs = res["recommendations"]

    assert 0.0 < recs["cost_optimal_threshold"] < 1.0
    assert 0.0 < recs["f1_optimal_threshold"] < 1.0
    assert recs["recommended_review_threshold"] < recs["recommended_decline_threshold"]
