"""
Baseline model implementation: Logistic Regression with cost-sensitive class weights.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def build_logistic_regression_baseline(
    class_weight: Optional[str] = "balanced",
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42
) -> Pipeline:
    """
    Build a standard scaled Logistic Regression baseline pipeline.
    
    Args:
        class_weight: 'balanced' to penalize fraud misclassifications inversely
                      proportional to class frequencies, or None.
        C: Inverse regularization strength.
        max_iter: Maximum optimization solver iterations.
        random_state: Seed for reproducibility.

    Returns:
        Scikit-Learn Pipeline with StandardScaler and LogisticRegression.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight=class_weight,
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=random_state
        ))
    ])
    return pipeline
