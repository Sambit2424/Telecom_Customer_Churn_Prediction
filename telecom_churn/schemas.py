from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from typing import Any

from .config import RAW_FEATURES


# Pydantic schemas validate single and batch prediction inputs
class PredictionPayload(BaseModel):
    input_data: dict[str, Any] = Field(..., description="Feature vector for a single churn prediction")

    @model_validator(mode="after")
    def validate_payload(self):
        if not isinstance(self.input_data, dict):
            raise ValueError("input_data must be a dictionary of features")
        missing = [feature for feature in RAW_FEATURES if feature not in self.input_data]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return self


class BatchPredictionPayload(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1, description="List of feature payloads for batch prediction")

    @model_validator(mode="after")
    def validate_records(self):
        if not isinstance(self.records, list) or not self.records:
            raise ValueError("records must be a non-empty list")
        for record in self.records:
            missing = [feature for feature in RAW_FEATURES if feature not in record]
            if missing:
                raise ValueError(f"One or more records are missing required features: {missing}")
        return self
