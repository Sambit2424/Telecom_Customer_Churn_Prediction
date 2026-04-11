from pathlib import Path

# Application configuration and default filesystem paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "Customer_Churn.csv"
MODEL_DIR = ROOT_DIR / "Saved ML models"
MODEL_PATH = MODEL_DIR / "tuned_xgb_optuna_model.joblib"
LOG_DIR = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "telecom_churn_app.log"

# MLflow configuration
# For running mlflow_demo.py on EC2 instance (outside Docker container, localhost access)
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# For Docker internal networking (use inside containers)
# MLFLOW_TRACKING_URI = "http://mlflow:5000"
# For external access from outside EC2 (replace <EC2_PUBLIC_IP> with your actual EC2 public IP)
# MLFLOW_TRACKING_URI = "http://YOUR_EC2_PUBLIC_IP:5000"
MLFLOW_EXPERIMENT_NAME = "telecom_churn_prediction"

# ABSOLUTE PATH for artifacts (Fixes the Permission Denied /app error)
# This will evaluate to /home/ubuntu/telecom_customer_churn_prediction/mlruns
MLFLOW_ARTIFACT_ROOT = str(ROOT_DIR / "mlruns")

RAW_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
