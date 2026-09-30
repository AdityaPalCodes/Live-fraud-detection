"""
Unit tests for Feature Engineering and leakage prevention.
"""

import pytest
import numpy as np
import pandas as pd
from src.features.engineer import TransactionFeatureEngineer


@pytest.fixture
def sample_transactions():
    """Create a minimal synthetic sample DataFrame."""
    np.random.seed(42)
    df = pd.DataFrame({
        "Time": [0.0, 3600.0, 7200.0, 45000.0, 80000.0],
        "Amount": [0.0, 1.50, 25.0, 50.0, 1200.0],
        "Class": [0, 0, 0, 0, 1]
    })
    for i in range(1, 29):
        df[f"V{i}"] = np.random.normal(0, 1, len(df))
    return df


def test_feature_engineer_unfitted_raises():
    """Verify transform fails before fit."""
    fe = TransactionFeatureEngineer()
    df = pd.DataFrame({"Amount": [10.0], "Time": [100.0]})
    with pytest.raises(RuntimeError):
        fe.transform(df)


def test_feature_transform_shapes_and_columns(sample_transactions):
    """Test feature transformations and output schema."""
    fe = TransactionFeatureEngineer()
    X = sample_transactions.drop(columns=["Class"])
    fe.fit(X)
    X_trans = fe.transform(X)

    assert len(X_trans) == len(X)
    assert "log_amount" in X_trans.columns
    assert "hour_sin" in X_trans.columns
    assert "hour_cos" in X_trans.columns
    assert "is_night" in X_trans.columns
    assert "is_micro_amount" in X_trans.columns
    assert "v_vector_norm" in X_trans.columns

    # Verify no NaN values
    assert X_trans.isna().sum().sum() == 0


def test_cyclical_hour_bounds(sample_transactions):
    """Ensure sine and cosine values are strictly bounded in [-1.0, 1.0]."""
    fe = TransactionFeatureEngineer()
    fe.fit(sample_transactions.drop(columns=["Class"]))
    X_trans = fe.transform(sample_transactions.drop(columns=["Class"]))

    assert (X_trans["hour_sin"] >= -1.0).all() and (X_trans["hour_sin"] <= 1.0).all()
    assert (X_trans["hour_cos"] >= -1.0).all() and (X_trans["hour_cos"] <= 1.0).all()


def test_single_transaction_transform(sample_transactions):
    """Verify single-transaction inference mode produces exact matching schema."""
    fe = TransactionFeatureEngineer()
    fe.fit(sample_transactions.drop(columns=["Class"]))

    batch_res = fe.transform(sample_transactions.head(1).drop(columns=["Class"]))
    single_res = fe.transform_single(sample_transactions.iloc[0].drop("Class").to_dict())

    assert list(batch_res.columns) == list(single_res.columns)
    np.testing.assert_allclose(batch_res.values, single_res.values, rtol=1e-5)
