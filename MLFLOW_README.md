# MLflow Setup and Usage Guide

## Overview
This guide explains how to use MLflow with your deployed telecom churn prediction services on AWS EC2.

## Prerequisites
- Your three services (API, Streamlit, MLflow) are running on EC2
- You know your EC2 instance's public IP address (e.g., `54.123.45.67`)

## Configuration
1. Update the tracking URI in `telecom_churn/config.py`:
   ```python
   # Replace <EC2_PUBLIC_IP> with your actual EC2 public IP
   MLFLOW_TRACKING_URI = "http://54.123.45.67:5000"
   ```

## Running the Scripts

### 1. Log Existing Models
To log your saved models to MLflow:
```bash
# On your EC2 instance (SSH into it first)
cd /path/to/your/project
python log_existing_models.py
```

### 2. Register Models
To register logged models for version control:
```bash
python register_models.py
```

### 3. Run Demo Experiments
To see MLflow logging in action with sample experiments:
```bash
python mlflow_demo.py
```

## Accessing MLflow UI
- Open your browser and go to: `http://<EC2_PUBLIC_IP>:5000`
- You'll see:
  - **Experiments**: Your logged runs and metrics
  - **Models**: Registered models (after running register_models.py)
  - **Artifacts**: Downloaded model files and other outputs

## Key MLflow Concepts

### Experiments
- Group related runs together
- Your experiment: "telecom_churn_prediction"

### Runs
- Individual training sessions
- Each run logs parameters, metrics, and artifacts

### Model Registry
- Version control for models
- Stages: Development → Staging → Production

## Using MLflow in Your Code

### Basic Logging
```python
from telecom_churn.mlflow_utils import MLFlowTracker

tracker = MLFlowTracker()

with tracker.start_run(run_name="my_experiment"):
    tracker.log_params({"learning_rate": 0.01, "epochs": 100})
    tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})
    tracker.log_artifacts("./my_model_files/")
```

### Model Registration
```python
import mlflow

# Register a model from a run
model_uri = "runs:/<run_id>/model"
mlflow.register_model(model_uri, "my_churn_model")

# Load a registered model
model = mlflow.sklearn.load_model("models:/my_churn_model/Production")
```

## Troubleshooting

### Connection Issues
- Ensure your EC2 security group allows inbound traffic on port 5000
- Verify the MLFLOW_TRACKING_URI is correct
- Check that the MLflow service is running: `docker ps`

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Run scripts from the project root directory

### Model Registration Fails
- Models need to be logged with `mlflow.log_model()` for registration
- Use the artifact logging approach shown in the scripts

## Next Steps
1. Run the demo script to see MLflow in action
2. Log your existing models
3. Register your best performing models
4. Use the MLflow UI to compare experiments and manage models
5. Integrate MLflow logging into your training pipelines</content>
<parameter name="filePath">c:\ai_projects\telecom_customer_churn_prediction\MLFLOW_README.md