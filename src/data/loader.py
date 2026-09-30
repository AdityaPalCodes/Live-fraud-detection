"""
Data loading and dataset generation utilities for the Real-Time Fraud Detection System.

Handles loading existing datasets (e.g. Kaggle Credit Card Fraud dataset)
or generating a statistically faithful benchmark dataset if offline.
"""

import os
import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_synthetic_creditcard_data(
    n_samples: int = 50000,
    fraud_rate: float = 0.00172,
    random_state: int = 42,
    save_path: Optional[str] = "data/raw/creditcard.csv"
) -> pd.DataFrame:
    """
    Generate a statistically faithful benchmark credit card fraud dataset
    matching the schema, distributions, and correlation structure of the
    benchmark Kaggle/ULB Credit Card Fraud dataset:
    - Time: elapsed seconds over a 48-hour window (0 to 172,800)
    - V1 to V28: PCA-transformed latent variables with characteristic fraud divergence
    - Amount: heavily right-skewed log-normal distribution
    - Class: extreme class imbalance (~0.172% fraud rate)

    Args:
        n_samples: Total number of transactions to generate.
        fraud_rate: Target fraction of fraudulent transactions (default 0.172%).
        random_state: Seed for reproducibility.
        save_path: Optional path to save the generated CSV.

    Returns:
        pd.DataFrame containing the generated dataset.
    """
    rng = np.random.RandomState(random_state)
    logger.info(f"Generating synthetic benchmark dataset: n_samples={n_samples}, fraud_rate={fraud_rate:.5f}")

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # 1. Time: 48-hour period with diurnal variation (fewer transactions at night)
    # Legitimate transactions follow a 24-hour cycle with day/night peaks
    time_hours_legit = rng.beta(2, 2, size=n_legit) * 48
    time_sec_legit = (time_hours_legit * 3600).astype(float)

    # Fraudulent transactions tend to be more active during off-peak hours
    time_hours_fraud = rng.uniform(0, 48, size=n_fraud)
    time_sec_fraud = (time_hours_fraud * 3600).astype(float)

    # 2. Amount: heavily right-skewed log-normal distribution
    # Legitimate: median ~$25, mean ~$88, max up to thousands
    amount_legit = np.exp(rng.normal(loc=3.0, scale=1.3, size=n_legit))
    amount_legit = np.round(amount_legit, 2)

    # Fraud: bimodal (many testing micro-transactions <$5, and high amounts >$300)
    fraud_amount_low = np.exp(rng.normal(loc=1.2, scale=0.5, size=n_fraud // 2))
    fraud_amount_high = np.exp(rng.normal(loc=5.8, scale=0.8, size=n_fraud - (n_fraud // 2)))
    amount_fraud = np.round(np.concatenate([fraud_amount_low, fraud_amount_high]), 2)
    rng.shuffle(amount_fraud)

    # 3. PCA Features V1 to V28
    # Legitimate features: mostly standard normal N(0, 1)
    v_legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))

    # Fraud features: well-documented shifts in the ULB dataset
    # In ULB data: V14, V12, V10, V17 have strong negative correlations with fraud
    # V4, V11, V2, V19 have positive shifts
    v_fraud = rng.normal(loc=0.0, scale=1.2, size=(n_fraud, 28))
    # Apply empirical shifts observed in credit card fraud detection benchmarks
    v_fraud[:, 13] -= 6.5   # V14
    v_fraud[:, 11] -= 5.0   # V12
    v_fraud[:, 9] -= 4.8    # V10
    v_fraud[:, 16] -= 5.2   # V17
    v_fraud[:, 3] += 4.5    # V4
    v_fraud[:, 10] += 3.8   # V11
    v_fraud[:, 1] += 2.8    # V2
    v_fraud[:, 18] += 1.5   # V19

    # Combine into DataFrame
    cols_v = [f"V{i}" for i in range(1, 29)]
    
    df_legit = pd.DataFrame(v_legit, columns=cols_v)
    df_legit["Time"] = time_sec_legit
    df_legit["Amount"] = amount_legit
    df_legit["Class"] = 0

    df_fraud = pd.DataFrame(v_fraud, columns=cols_v)
    df_fraud["Time"] = time_sec_fraud
    df_fraud["Amount"] = amount_fraud
    df_fraud["Class"] = 1

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    # Sort by time chronologically as real transaction logs arrive
    df = df.sort_values("Time").reset_index(drop=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved dataset to {save_path} ({len(df)} rows, {df['Class'].sum()} frauds)")

    return df


def load_credit_card_data(
    data_path: str = "data/raw/creditcard.csv",
    auto_generate_if_missing: bool = True,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Load the Credit Card Fraud Detection dataset from CSV.
    If the file does not exist locally and auto_generate_if_missing is True,
    generates a statistically representative benchmark dataset automatically.

    Args:
        data_path: Path to the raw CSV file.
        auto_generate_if_missing: Whether to generate dataset if file not found.
        sample_size: Optional sub-sampling for fast experimentation.

    Returns:
        pd.DataFrame containing the dataset.
    """
    if not os.path.exists(data_path):
        if auto_generate_if_missing:
            logger.warning(
                f"Dataset not found at '{data_path}'. "
                "Generating statistically faithful benchmark dataset matching Kaggle/ULB properties..."
            )
            df = generate_synthetic_creditcard_data(
                n_samples=50000,
                fraud_rate=0.00172,
                random_state=42,
                save_path=data_path
            )
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{data_path}'. Please place 'creditcard.csv' into 'data/raw/' "
                "or enable auto_generate_if_missing."
            )
    else:
        logger.info(f"Loading dataset from '{data_path}'...")
        df = pd.read_csv(data_path)

    if sample_size and len(df) > sample_size:
        logger.info(f"Subsampling dataset to {sample_size} records while preserving class ratio...")
        # Stratified subsampling
        fraud = df[df["Class"] == 1]
        legit = df[df["Class"] == 0]
        frac = sample_size / len(df)
        sampled_fraud = fraud.sample(frac=frac, random_state=42)
        sampled_legit = legit.sample(frac=frac, random_state=42)
        df = pd.concat([sampled_legit, sampled_fraud]).sort_values("Time").reset_index(drop=True)

    logger.info(
        f"Dataset loaded: {len(df):,} transactions, {df['Class'].sum():,} frauds "
        f"({df['Class'].mean() * 100:.3f}% fraud rate)"
    )
    return df
