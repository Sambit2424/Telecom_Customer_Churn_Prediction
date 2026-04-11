from pathlib import Path

# Application configuration and default filesystem paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "Customer_Churn.csv"
MODEL_DIR = ROOT_DIR / "Saved ML models"
MODEL_PATH = MODEL_DIR / "tuned_xgb_optuna_model.joblib"
LOG_DIR = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "telecom_churn_app.log"

# MLflow configuration
# For Docker internal networking (use this when running services together)
# MLFLOW_TRACKING_URI = "http://mlflow:5000"
# For external access from outside Docker (replace <EC2_PUBLIC_IP> with your actual EC2 public IP)
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "telecom_churn_prediction"

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
