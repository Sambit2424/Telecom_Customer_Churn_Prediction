#!/usr/bin/env python3
"""
Script to demonstrate model registration in MLflow.
This shows how to register logged models for version control and deployment.
"""

import mlflow
from telecom_churn.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

def register_models():
    """Register models from completed runs to the MLflow Model Registry."""

    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Get all runs from the experiment
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if not experiment:
        print(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found!")
        return

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if runs.empty:
        print("No runs found in the experiment. Please log some models first.")
        return

    print(f"Found {len(runs)} runs in experiment '{MLFLOW_EXPERIMENT_NAME}'")

    # Register each run's model (assuming each run has a model artifact)
    for _, run in runs.iterrows():
        run_id = run.run_id
        run_name = run.get('tags.mlflow.runName', f'run_{run_id[:8]}')

        try:
            # Register the model from this run
            # Note: This assumes the model was logged with mlflow.log_model() or similar
            # For joblib models, you might need to log them properly first
            model_uri = f"runs:/{run_id}/model"

            # Check if model exists at this URI
            try:
                mlflow.get_run(run_id)
                model_name = f"churn_predictor_{run_name}"
                mlflow.register_model(model_uri, model_name)
                print(f"Registered model '{model_name}' from run '{run_name}'")
            except Exception as e:
                print(f"Could not register model from run '{run_name}': {e}")
                print("Make sure the model was logged with mlflow.log_model()")

        except Exception as e:
            print(f"Error registering model from run {run_id}: {e}")

    print("\nModel registration complete!")
    print("View registered models in the MLflow UI under the 'Models' tab.")

def demonstrate_model_lifecycle():
    """Demonstrate the complete model lifecycle: logging -> registration -> staging."""

    print("=== MLflow Model Lifecycle Demo ===")

    # This would be done after training a model
    print("1. During training, you would log the model like this:")
    print("""
    import mlflow.sklearn
    # After training your model
    mlflow.sklearn.log_model(model, "model")
    """)

    print("2. Then register the model:")
    print("""
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, "my_model_name")
    """)

    print("3. Transition model stages:")
    print("""
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="my_model_name",
        version=1,
        stage="Production"
    )
    """)

    print("4. Load model for inference:")
    print("""
    model = mlflow.sklearn.load_model("models:/my_model_name/Production")
    """)

if __name__ == "__main__":
    print("Registering existing models...")
    register_models()

    print("\n" + "="*50)
    demonstrate_model_lifecycle()