from __future__ import annotations

import joblib
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import MODEL_PATH
from .exceptions import ModelLoadError
from .logger import LoggerFactory
from .mlflow_utils import MLFlowTracker


class ModelManager:
    def __init__(self, model_path: Path | str = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model: BaseEstimator | None = None
        self.logger = LoggerFactory.get_logger("ModelManager")

    def load_model(self) -> BaseEstimator:
        if not self.model_path.exists():
            raise ModelLoadError(f"Model file does not exist at {self.model_path}")
        self.logger.info("Loading model from %s", self.model_path)
        self.model = joblib.load(self.model_path)
        return self.model

    def save_model(self, model: BaseEstimator, path: Path | str | None = None) -> Path:
        target_path = Path(path or self.model_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, target_path)
        self.logger.info("Saved model to %s", target_path)
        return target_path

    def predict(self, X) -> list:
        if self.model is None:
            self.load_model()
        return self.model.predict(X).tolist()

    def predict_proba(self, X) -> list:
        if self.model is None:
            self.load_model()
        probabilities = self.model.predict_proba(X)
        return probabilities[:, 1].tolist()

    def train_xgboost(self, X, y, params: dict | None = None, tracking: bool = False) -> BaseEstimator:
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "XGBoost is required for train_xgboost. Install it with `pip install xgboost`."
            ) from exc

        params = params or {
            "n_estimators": 200,
            "learning_rate": 0.08,
            "max_depth": 5,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 24,
        }
        self.logger.info("Training XGBoost model with params: %s", params)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y)
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = {
            "f1_score": float(f1_score(y_val, y_pred, pos_label=1)),
            "roc_auc": float(roc_auc_score(y_val, y_proba)),
        }
        self.logger.info("Training metrics: %s", metrics)

        if tracking:
            tracker = MLFlowTracker()
            with tracker.start_run(run_name="xgboost_training"):
                tracker.log_params(params)
                tracker.log_metrics(metrics)
                tracker.log_artifacts(str(self.model_path.parent))

        self.save_model(model)
        self.model = model
        return model
