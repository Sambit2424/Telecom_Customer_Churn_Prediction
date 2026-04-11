from __future__ import annotations

from .config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_ROOT
from .logger import LoggerFactory


# MLFlowTracker wraps mlflow client setup and logging for training runs
class MLFlowTracker:
    def __init__(self, tracking_uri: str = MLFLOW_TRACKING_URI, experiment_name: str = MLFLOW_EXPERIMENT_NAME):
        try:
            import mlflow
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "mlflow is required for MLFlowTracker. Install it with `pip install mlflow`."
            ) from exc

        self.mlflow = mlflow
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.logger = LoggerFactory.get_logger("MLFlowTracker")
        self.mlflow.set_tracking_uri(self.tracking_uri)
        self.mlflow.set_experiment(self.experiment_name)
        self.logger.info("MLflow configured at %s for experiment %s", self.tracking_uri, self.experiment_name)

    def start_run(self, run_name: str | None = None):
        return self.mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict) -> None:
        self.mlflow.log_params(params)
        self.logger.debug("Logged params: %s", params)

    def log_metrics(self, metrics: dict) -> None:
        self.mlflow.log_metrics(metrics)
        self.logger.debug("Logged metrics: %s", metrics)

    def log_artifacts(self, artifact_path: str) -> None:
        self.mlflow.log_artifacts(artifact_path)
        self.logger.debug("Logged artifacts from: %s", artifact_path)
