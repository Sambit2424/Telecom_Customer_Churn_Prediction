#!/usr/bin/env python3
"""
Comprehensive MLflow demonstration script.
This shows how to use MLFlowTracker for logging experiments during model training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path

from telecom_churn.mlflow_utils import MLFlowTracker
from telecom_churn.preprocessing import preprocess_data

def load_and_preprocess_data():
    """Load and preprocess the telecom churn data."""
    # Load data
    data_path = Path(__file__).parent / "Customer_Churn.csv"
    df = pd.read_csv(data_path)

    # Basic preprocessing (simplified for demo)
    # In real scenario, use your full preprocessing pipeline
    df = df.dropna()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    # Convert categorical to numeric (simplified)
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                       'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract',
                       'PaperlessBilling', 'PaymentMethod']

    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    # Features and target
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    return X, y

def train_and_log_model(model_name, model_params, preprocessing_steps=None):
    """
    Train a model and log the experiment to MLflow.

    Args:
        model_name: Name for the MLflow run
        model_params: Dictionary of model hyperparameters
        preprocessing_steps: List of preprocessing steps applied
    """

    # Initialize MLflow tracker
    tracker = MLFlowTracker()

    with tracker.start_run(run_name=model_name):

        # Log preprocessing information
        if preprocessing_steps:
            tracker.log_params({"preprocessing": ", ".join(preprocessing_steps)})
        else:
            tracker.log_params({"preprocessing": "None"})

        # Log model parameters
        tracker.log_params(model_params)

        # Load and preprocess data
        X, y = load_and_preprocess_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        model = RandomForestClassifier(**model_params, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        # Log metrics
        tracker.log_metrics(metrics)

        # Save model locally (for demo purposes)
        model_path = Path(__file__).parent / "demo_models" / f"{model_name}.joblib"
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(model, model_path)

        # Log the model file as artifact
        tracker.log_artifacts(str(model_path))

        print(f"Model {model_name} trained and logged!")
        print(f"Metrics: {metrics}")

        return metrics

def run_multiple_experiments():
    """Run multiple experiments with different hyperparameters."""

    experiments = [
        {
            "name": "rf_baseline",
            "params": {"n_estimators": 100, "max_depth": None},
            "preprocessing": []
        },
        {
            "name": "rf_tuned_shallow",
            "params": {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5},
            "preprocessing": ["standard_scaler"]
        },
        {
            "name": "rf_tuned_deep",
            "params": {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 2},
            "preprocessing": ["standard_scaler"]
        }
    ]

    results = []

    for exp in experiments:
        print(f"\n--- Running experiment: {exp['name']} ---")
        metrics = train_and_log_model(
            exp['name'],
            exp['params'],
            exp['preprocessing']
        )
        results.append({
            "experiment": exp['name'],
            "metrics": metrics,
            "params": exp['params']
        })

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for result in results:
        print(f"\n{result['experiment']}:")
        print(f"  Parameters: {result['params']}")
        print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"  F1-Score: {result['metrics']['f1_score']:.4f}")

def demonstrate_mlflow_features():
    """Demonstrate various MLflow features."""

    print("\n" + "="*60)
    print("MLFLOW FEATURES DEMONSTRATION")
    print("="*60)

    tracker = MLFlowTracker()

    print("1. Current experiment:", tracker.experiment_name)
    print("2. Tracking URI:", tracker.tracking_uri)

    # Get experiment details
    import mlflow
    experiment = mlflow.get_experiment_by_name(tracker.experiment_name)
    if experiment:
        print("3. Experiment ID:", experiment.experiment_id)
        print("4. Artifact location:", experiment.artifact_location)

        # Get recent runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        print(f"5. Total runs in experiment: {len(runs)}")

        if not runs.empty:
            print("6. Recent runs:")
            for _, run in runs.tail(3).iterrows():
                run_name = run.get('tags.mlflow.runName', 'Unnamed')
                accuracy = run.get('metrics.accuracy', 'N/A')
                print(f"   - {run_name}: accuracy = {accuracy}")

    print("\nNext steps:")
    print("- View experiments in MLflow UI at:", tracker.tracking_uri)
    print("- Compare runs in the UI")
    print("- Register best models using register_models.py")

if __name__ == "__main__":
    print("Starting MLflow demonstration...")

    # Run multiple experiments
    run_multiple_experiments()

    # Demonstrate MLflow features
    demonstrate_mlflow_features()

    print("\nDemo complete! Check your MLflow UI to see the logged experiments.")