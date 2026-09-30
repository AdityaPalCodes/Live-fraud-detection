"""
Unified model training, comparison, calibration, and serialization pipeline.

Executes:
1. Leakage-free split (Train / Validation / Test)
2. Feature engineering fitted ONLY on Train
3. Model comparison:
   - Baseline: Logistic Regression
   - Random Forest
   - LightGBM / XGBoost
   - Isolation Forest (Unsupervised Anomaly)
4. Probability Calibration (Isotonic / Sigmoid)
5. Metric evaluation & reproducible comparison table
6. Artifact serialization for production inference
7. Optional MLflow experiment tracking
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
import mlflow

from src.data.loader import load_credit_card_data
from src.data.split import time_based_split, stratified_split
from src.features.engineer import TransactionFeatureEngineer
from src.models.baseline import build_logistic_regression_baseline
from src.models.trees import build_random_forest_model, build_lightgbm_model, build_xgboost_model
from src.models.anomaly import AnomalyRiskDetector
from src.models.calibration import calibrate_classifier, compare_calibration
from src.models.evaluator import evaluate_predictions, print_evaluation_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline(
    data_path: str = "data/raw/creditcard.csv",
    artifacts_dir: str = "models/artifacts",
    split_method: str = "time",
    fp_cost: float = 25.0,
    fn_cost: float = 250.0,
    use_mlflow: bool = True
) -> Dict[str, Any]:
    """
    Execute end-to-end model training, comparison, calibration, and artifact generation.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Load Data
    logger.info("Loading dataset...")
    df = load_credit_card_data(data_path=data_path, auto_generate_if_missing=True)

    # 2. Strict Train / Validation / Test Split
    if split_method == "time":
        logger.info("Performing chronological time-based split (Out-of-Time)...")
        train_df, val_df, test_df = time_based_split(df, val_size=0.15, test_size=0.15)
    else:
        logger.info("Performing stratified split...")
        train_df, val_df, test_df = stratified_split(df, val_size=0.15, test_size=0.15)

    # 3. Feature Engineering fitted strictly on training data
    logger.info("Fitting Feature Engineer on training split only...")
    fe = TransactionFeatureEngineer()
    fe.fit(train_df.drop(columns=["Class"]))

    X_train = fe.transform(train_df.drop(columns=["Class"]))
    y_train = train_df["Class"].values

    X_val = fe.transform(val_df.drop(columns=["Class"]))
    y_val = val_df["Class"].values

    X_test = fe.transform(test_df.drop(columns=["Class"]))
    y_test = test_df["Class"].values

    feature_names = list(X_train.columns)
    logger.info(f"Dataset split shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # Track MLflow experiment
    if use_mlflow:
        mlflow.set_experiment("fraud-detection-model-comparison")

    # 4. Model Training & Comparison
    models = {
        "Logistic Regression (Baseline)": build_logistic_regression_baseline(class_weight="balanced"),
        "Random Forest": build_random_forest_model(n_estimators=100, max_depth=10),
        "LightGBM": build_lightgbm_model(n_estimators=120, max_depth=5, scale_pos_weight=50.0),
        "XGBoost": build_xgboost_model(n_estimators=120, max_depth=5, scale_pos_weight=50.0)
    }

    experiment_results = []
    trained_models = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Validation prediction
        val_probs = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(y_val, val_probs, threshold=0.5, fp_cost=fp_cost, fn_cost=fn_cost)
        metrics["model_name"] = name
        experiment_results.append(metrics)
        print_evaluation_summary(metrics, model_name=name)

        if use_mlflow:
            with mlflow.start_run(run_name=f"train_{name.replace(' ', '_')}", nested=True):
                mlflow.log_params({"model_type": name, "train_records": len(X_train)})
                mlflow.log_metrics({
                    "val_pr_auc": metrics["pr_auc"],
                    "val_roc_auc": metrics["roc_auc"],
                    "val_f1": metrics["f1"],
                    "val_recall": metrics["recall"],
                    "val_precision": metrics["precision"],
                    "val_total_cost": metrics["total_cost"]
                })

    # 5. Train Anomaly Detector (Isolation Forest)
    logger.info("Training Isolation Forest Anomaly Detector...")
    anomaly_detector = AnomalyRiskDetector(n_estimators=100, contamination=0.002)
    anomaly_detector.fit(X_train, train_on_normal_only=True, y=y_train)
    anomaly_metrics = anomaly_detector.evaluate_against_labels(X_val, y_val)

    # 6. Select Best Model by PR-AUC & Perform Probability Calibration
    df_exp = pd.DataFrame(experiment_results).sort_values("pr_auc", ascending=False)
    best_model_name = df_exp.iloc[0]["model_name"]
    best_uncalibrated_model = trained_models[best_model_name]
    logger.info(f"Selected Best Supervised Model: {best_model_name} (Val PR-AUC: {df_exp.iloc[0]['pr_auc']})")

    logger.info("Performing Isotonic Probability Calibration on best model...")
    calibrated_best_model = calibrate_classifier(
        base_model=best_uncalibrated_model,
        X_train=X_train,
        y_train=y_train,
        method="isotonic",
        cv=3
    )

    # Compare calibration
    val_uncal_probs = best_uncalibrated_model.predict_proba(X_val)[:, 1]
    val_cal_probs = calibrated_best_model.predict_proba(X_val)[:, 1]
    cal_comparison = compare_calibration(y_val, val_uncal_probs, val_cal_probs)

    # 7. Final Untouched Out-of-Time Test Evaluation
    logger.info("Performing final evaluation on untouched Test Set...")
    test_cal_probs = calibrated_best_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_cal_probs, threshold=0.5, fp_cost=fp_cost, fn_cost=fn_cost)
    test_metrics["model_name"] = f"{best_model_name} (Calibrated)"
    print_evaluation_summary(test_metrics, model_name=f"FINAL TEST SET - {best_model_name}")

    # 8. Save Artifacts for Production Inference
    logger.info(f"Saving production artifacts to '{artifacts_dir}'...")
    joblib.dump(calibrated_best_model, os.path.join(artifacts_dir, "best_model.joblib"))
    joblib.dump(best_uncalibrated_model, os.path.join(artifacts_dir, "uncalibrated_best_model.joblib"))
    joblib.dump(fe, os.path.join(artifacts_dir, "feature_engineer.joblib"))
    joblib.dump(anomaly_detector, os.path.join(artifacts_dir, "isolation_forest.joblib"))

    # Save feature metadata
    metadata = {
        "model_name": best_model_name,
        "is_calibrated": True,
        "calibration_method": "isotonic",
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "validation_pr_auc": float(df_exp.iloc[0]["pr_auc"]),
        "test_pr_auc": float(test_metrics["pr_auc"]),
        "test_recall": float(test_metrics["recall"]),
        "test_precision": float(test_metrics["precision"]),
        "test_f1": float(test_metrics["f1"]),
        "brier_score": float(test_metrics["brier_score"])
    }
    with open(os.path.join(artifacts_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training and serialization completed successfully.")
    return {
        "comparison_table": df_exp,
        "anomaly_metrics": anomaly_metrics,
        "calibration_comparison": cal_comparison,
        "test_metrics": test_metrics,
        "metadata": metadata
    }


if __name__ == "__main__":
    run_training_pipeline()
