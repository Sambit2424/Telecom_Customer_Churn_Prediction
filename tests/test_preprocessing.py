import pandas as pd

from telecom_churn.preprocessing import DataPreprocessor


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
            },
            {
                "customerID": "0002",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 65.30,
                "TotalCharges": 1540.20,
                "Churn": "Yes",
            },
        ]
    )


def test_preprocessor_fit_transform():
    df = _build_sample_dataframe()
    preprocessor = DataPreprocessor()
    preprocessor.fit(df)

    transformed = preprocessor.transform(df)
    assert transformed.shape[0] == 2
    assert any(col.startswith("tenure_bin_") for col in transformed.columns)
    assert transformed.isnull().sum().sum() == 0


def test_preprocessor_transform_single():
    df = _build_sample_dataframe()
    preprocessor = DataPreprocessor()
    preprocessor.fit(df)

    record = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "Yes",
        "tenure": 8,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.50,
        "TotalCharges": 620.00,
    }

    transformed = preprocessor.transform_single(record)
    assert transformed.shape == (1, len(preprocessor.template_columns))
