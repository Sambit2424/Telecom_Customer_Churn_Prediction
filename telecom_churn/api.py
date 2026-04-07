from __future__ import annotations

from typing import Any

from .batch_processor import BatchProcessor
from .exceptions import PredictionError, ValidationError
from .logger import LoggerFactory
from .model_manager import ModelManager
from .preprocessing import DataPreprocessor
from .schemas import BatchPredictionPayload, PredictionPayload


def create_app() -> 'FastAPI':
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    logger = LoggerFactory.get_logger("TelecomChurnAPI")
    app = FastAPI(
        title="Telecom Customer Churn Prediction API",
        description="A production-grade API for telecom churn prediction with validation, batch processing, and MLflow readiness.",
        version="1.0.0",
    )

    preprocessor = DataPreprocessor()
    preprocessor.fit()
    model_manager = ModelManager()
    model_manager.load_model()
    batch_processor = BatchProcessor(preprocessor, model_manager)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(PredictionError)
    async def prediction_exception_handler(request: Request, exc: PredictionError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def validation_exception_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "healthy", "service": "telecom_churn_prediction_api"}

    @app.post("/predict")
    async def predict(payload: PredictionPayload) -> dict[str, Any]:
        try:
            X = preprocessor.transform_single(payload.input_data)
            prediction = model_manager.predict(X)[0]
            probability = model_manager.predict_proba(X)[0]
            return {
                "prediction": int(prediction),
                "probability": float(probability),
            }
        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            raise PredictionError(str(exc)) from exc

    @app.post("/batch_predict")
    async def batch_predict(payload: BatchPredictionPayload) -> dict[str, Any]:
        try:
            records = payload.records
            predictions = batch_processor.predict_batch(records)
            return {"predictions": predictions, "count": len(predictions)}
        except Exception as exc:
            logger.error("Batch prediction failed: %s", exc)
            raise PredictionError(str(exc)) from exc

    return app
