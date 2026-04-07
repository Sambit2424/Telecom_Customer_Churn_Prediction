import pytest

from telecom_churn.api import PredictionPayload


def test_prediction_payload_valid():
    record = {
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

    payload = PredictionPayload(input_data=record)
    assert payload.input_data["gender"] == "Female"
    assert payload.input_data["tenure"] == 10


def test_prediction_payload_missing_fields():
    incomplete_record = {"gender": "Female", "SeniorCitizen": 0}
    with pytest.raises(ValueError):
        PredictionPayload(input_data=incomplete_record)
