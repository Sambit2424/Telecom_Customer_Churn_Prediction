from locust import HttpUser, between, task

class ChurnPredictionUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        payload = {
            "input_data": {
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
            }
        }
        self.client.post('/predict', json=payload)
