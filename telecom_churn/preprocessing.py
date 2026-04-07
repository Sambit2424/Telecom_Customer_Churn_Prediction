from __future__ import annotations

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from .config import DATA_PATH, RAW_FEATURES
from .logger import LoggerFactory


# DataPreprocessor handles dataset loading, cleaning, encoding, and scaling
class DataPreprocessor:
    TENURE_BINS = [0, 12, 24, 36, 48, 60, 72]
    TENURE_LABELS = ["1-12", "13-24", "25-36", "37-48", "49-60", "61-72"]

    def __init__(self, data_path: Path | str = DATA_PATH):
        self.data_path = Path(data_path)
        self.template_columns: list[str] = []
        self.scaler = StandardScaler()
        self.logger = LoggerFactory.get_logger("DataPreprocessor")
        self.fitted = False

    def load_data(self) -> pd.DataFrame:
        self.logger.info("Loading dataset from %s", self.data_path)
        return pd.read_csv(self.data_path)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Cleaning raw dataset")
        df = df.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna(subset=["TotalCharges"])
        df = df.drop(columns=["customerID"], errors="ignore")
        return df

    def _build_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "tenure" in df.columns:
            df["tenure_bin"] = pd.cut(
                df["tenure"],
                bins=self.TENURE_BINS,
                labels=self.TENURE_LABELS,
                include_lowest=True,
            )
        df = df.drop(columns=["tenure"], errors="ignore")
        df = df.drop(columns=["Churn"], errors="ignore")
        df = pd.get_dummies(df, drop_first=True)
        return df

    def fit(self, df: pd.DataFrame | None = None) -> DataPreprocessor:
        if df is None:
            df = self.load_data()
        df = self.clean_data(df)
        X = self._build_feature_frame(df)
        self.scaler.fit(X)
        self.template_columns = X.columns.tolist()
        self.fitted = True
        self.logger.info("Preprocessor fit complete with %d feature columns", len(self.template_columns))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            self.fit(df)

        df = self.clean_data(df)
        X = self._build_feature_frame(df)
        X = X.reindex(columns=self.template_columns, fill_value=0)
        transformed = pd.DataFrame(self.scaler.transform(X), columns=self.template_columns)
        return transformed

    def transform_single(self, record: dict) -> pd.DataFrame:
        self.logger.info("Transforming single record for prediction")
        self._validate_record(record)
        df = pd.DataFrame([record])
        transformed = self.transform(df)
        return transformed

    def _validate_record(self, record: dict) -> None:
        if not isinstance(record, dict):
            raise ValueError("Input record must be a dictionary of feature values")

        missing = [feature for feature in RAW_FEATURES if feature not in record]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        extra = [key for key in record if key not in RAW_FEATURES]
        if extra:
            self.logger.warning("Dropping extra input keys: %s", extra)
            for key in extra:
                record.pop(key)

    def get_raw_feature_names(self) -> list[str]:
        return RAW_FEATURES.copy()
