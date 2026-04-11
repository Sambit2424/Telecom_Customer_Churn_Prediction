#!/usr/bin/env python3
"""
Script to log existing saved models to MLflow for tracking and versioning.
This demonstrates how to use MLFlowTracker to log artifacts (model files).
"""

import os
from pathlib import Path
from telecom_churn.mlflow_utils import MLFlowTracker

def log_existing_models():
    """Log all saved models to MLflow as separate runs."""

    # Initialize MLflow tracker
    tracker = MLFlowTracker()

    # Path to saved models
    model_dir = Path(__file__).parent / "Saved ML models"

    # Model metadata (you can expand this with actual training details)
    model_info = {
        "tuned_logistic_regr_standardscaled_smote.joblib": {
            "model_type": "LogisticRegression",
            "preprocessing": "StandardScaler + SMOTE",
            "description": "Tuned logistic regression with standard scaling and SMOTE oversampling"
        },
        "tuned_xgb_no_preproc_model.joblib": {
            "model_type": "XGBoost",
            "preprocessing": "None",
            "description": "Tuned XGBoost without preprocessing"
        },
        "tuned_xgb_optuna_model.joblib": {
            "model_type": "XGBoost",
            "preprocessing": "Optimized with Optuna",
            "description": "XGBoost tuned using Optuna hyperparameter optimization"
        },
        "tuned_xgb_standardscalde_smotenn_model.joblib": {
            "model_type": "XGBoost",
            "preprocessing": "StandardScaler + SMOTENN",
            "description": "Tuned XGBoost with standard scaling and SMOTENN oversampling"
        },
        "tuned_xgb_standardscaler_model.joblib": {
            "model_type": "XGBoost",
            "preprocessing": "StandardScaler",
            "description": "Tuned XGBoost with standard scaling only"
        }
    }

    for model_file in model_dir.glob("*.joblib"):
        model_name = model_file.name
        info = model_info.get(model_name, {"model_type": "Unknown", "preprocessing": "Unknown", "description": "Unknown"})

        # Start a new run for this model
        with tracker.start_run(run_name=f"log_{model_name.replace('.joblib', '')}"):

            # Log model metadata as parameters
            tracker.log_params({
                "model_type": info["model_type"],
                "preprocessing": info["preprocessing"],
                "description": info["description"],
                "file_size_mb": round(model_file.stat().st_size / (1024 * 1024), 2)
            })

            # Log dummy metrics (replace with actual metrics if available)
            tracker.log_metrics({
                "accuracy": 0.85,  # Replace with actual values
                "precision": 0.82,
                "recall": 0.78,
                "f1_score": 0.80
            })

            # Log the model file as artifact
            tracker.log_artifacts(str(model_file))

            print(f"Logged model: {model_name}")

    print("All existing models have been logged to MLflow!")
    print(f"View them at: {tracker.tracking_uri}")

if __name__ == "__main__":
    log_existing_models()