import pandas as pd

from telecom_churn.batch_processor import BatchProcessor
from telecom_churn.model_manager import ModelManager
from telecom_churn.preprocessing import DataPreprocessor


class DummyModelManager:
    def predict(self, X):
        return [0] * len(X)

    def predict_proba(self, X):
        return [0.1] * len(X)


def _build_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customerID": "0001",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 84.80,
                "TotalCharges": 1020.50,
                "Churn": "No",
            }
        ]
    )


def test_batch_processor_predict_batch():
    df = _build_sample_dataframe()
    preprocessor = DataPreprocessor()
    preprocessor.fit(df)

    model_manager = DummyModelManager()
    batch_processor = BatchProcessor(preprocessor, model_manager, max_workers=2)

    records = [
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 10,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 72.45,
            "TotalCharges": 760.25,
        }
    ]

    result = batch_processor.predict_batch(records)
    assert len(result) == 1
    assert result[0]["prediction"] == 0
    assert 0.0 <= result[0]["probability"] <= 1.0
