from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from .model_manager import ModelManager
from .preprocessing import DataPreprocessor
from .exceptions import PredictionError
from .logger import LoggerFactory


class BatchProcessor:
    def __init__(self, preprocessor: DataPreprocessor, model_manager: ModelManager, max_workers: int = 4):
        self.preprocessor = preprocessor
        self.model_manager = model_manager
        self.max_workers = max_workers
        self.logger = LoggerFactory.get_logger("BatchProcessor")

    def _predict_record(self, record: dict) -> dict:
        try:
            X = self.preprocessor.transform_single(record)
            prediction = self.model_manager.predict(X)
            probability = self.model_manager.predict_proba(X)

            if isinstance(probability[0], (list, tuple)):
                probability = [item[1] if len(item) > 1 else item[0] for item in probability]

            return {
                "input": record,
                "prediction": int(prediction[0]),
                "probability": float(probability[0]),
            }
        except Exception as error:
            self.logger.error("Batch prediction failed for record: %s, error: %s", record, error)
            raise PredictionError(str(error)) from error

    def predict_batch(self, records: Iterable[dict]) -> list[dict]:
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_record = {executor.submit(self._predict_record, record): record for record in records}
            for future in as_completed(future_to_record):
                results.append(future.result())
        return results
