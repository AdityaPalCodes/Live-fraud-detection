"""
Unit tests for model training, calibration, anomaly detection, and evaluation metrics.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.models.baseline import build_logistic_regression_baseline
from src.models.trees import build_random_forest_model, build_lightgbm_model
from src.models.anomaly import AnomalyRiskDetector
from src.models.calibration import calibrate_classifier, compare_calibration
from src.models.evaluator import evaluate_predictions


@pytest.fixture
def synthetic_data():
    """Generate minimal imbalanced dataset for testing."""
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=10,
        weights=[0.95, 0.05],
        random_state=42
    )
    df_X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(15)])
    return df_X, y


def test_baseline_logistic_regression(synthetic_data):
    """Test baseline model output probabilities."""
    X, y = synthetic_data
    model = build_logistic_regression_baseline()
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]

    assert len(probs) == len(y)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_anomaly_detector_scoring(synthetic_data):
    """Test Isolation Forest anomaly score normalization in [0, 1]."""
    X, y = synthetic_data
    detector = AnomalyRiskDetector(n_estimators=50)
    detector.fit(X, train_on_normal_only=True, y=y)
    scores = detector.score_samples(X)

    assert len(scores) == len(X)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_probability_calibration(synthetic_data):
    """Verify calibration preserves probability bounds and outputs valid metrics."""
    X, y = synthetic_data
    model = build_logistic_regression_baseline()
    model.fit(X, y)

    calibrated = calibrate_classifier(model, X.values, y, method="isotonic", cv=2)
    cal_probs = calibrated.predict_proba(X.values)[:, 1]

    assert len(cal_probs) == len(y)
    assert (cal_probs >= 0.0).all() and (cal_probs <= 1.0).all()

    comparison = compare_calibration(y, model.predict_proba(X.values)[:, 1], cal_probs)
    assert "brier_uncalibrated" in comparison
    assert "brier_calibrated" in comparison


def test_evaluator_metrics():
    """Verify evaluation metric calculations."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])

    metrics = evaluate_predictions(y_true, y_prob, threshold=0.5, fp_cost=10.0, fn_cost=100.0)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["pr_auc"] > 0.9
    assert metrics["total_cost"] == 0.0
