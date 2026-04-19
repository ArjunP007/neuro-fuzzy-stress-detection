"""Comprehensive preprocessing pipeline for the stress detection project."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


LOGGER = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Container for preprocessed train, validation, and test splits."""

    X_train: pd.DataFrame
    X_val: Optional[pd.DataFrame]
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: Optional[pd.Series]
    y_test: pd.Series


class DataPreprocessor:
    """End-to-end preprocessing pipeline with fit/transform semantics."""

    FORBIDDEN_FEATURE_COLUMNS = {"stress_score", "stress_level"}

    DEFAULT_FEATURE_COLUMNS = [
        "sleep_hours",
        "work_hours",
        "screen_time",
        "physical_activity_hours",
        "mental_fatigue_score",
        "heart_rate",
        "caffeine_intake",
        "social_interaction_hours",
        "work_pressure_score",
    ]

    def __init__(
        self,
        *,
        feature_columns: Optional[Sequence[str]] = None,
        target_column: str = "stress_level",
        missing_strategy: str = "median",
        outlier_method: str = "zscore",
        outlier_threshold: float = 3.0,
        outlier_iqr_multiplier: float = 1.5,
        scaling_method: str = "standard",
        normalization_method: Optional[str] = None,
        variance_threshold: float = 0.0,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True,
    ) -> None:
        """Initialize preprocessing components and configuration."""

        self.feature_columns = list(feature_columns or self.DEFAULT_FEATURE_COLUMNS)
        self.target_column = target_column
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.outlier_threshold = float(outlier_threshold)
        self.outlier_iqr_multiplier = float(outlier_iqr_multiplier)
        self.scaling_method = scaling_method
        self.normalization_method = normalization_method
        self.variance_threshold = float(variance_threshold)
        self.test_size = float(test_size)
        self.validation_size = float(validation_size)
        self.random_state = int(random_state)
        self.stratify = bool(stratify)

        self._validate_configuration()

        self.imputer = SimpleImputer(strategy=self.missing_strategy)
        self.scaler = self._build_scaler(self.scaling_method)
        self.normalizer = self._build_scaler(self.normalization_method) if self.normalization_method else None
        self.label_encoder = LabelEncoder()
        self.feature_selector = VarianceThreshold(threshold=self.variance_threshold)

        self.selected_features_: list[str] = []
        self.numeric_feature_columns_: list[str] = []
        self.statistics_: Dict[str, Any] = {}
        self.is_fitted_: bool = False

    def load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load a dataset from CSV."""

        source_path = Path(filepath)
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {source_path}")

        dataset = pd.read_csv(source_path)
        LOGGER.info("Loaded dataset from %s with shape %s", source_path, dataset.shape)
        return dataset

    def fit(self, dataset: pd.DataFrame) -> "DataPreprocessor":
        """Fit preprocessing steps on a dataset."""

        df = self._validate_input_dataframe(dataset)
        df = self._filter_outliers(df)

        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()

        self.numeric_feature_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(self.numeric_feature_columns_) != len(self.feature_columns):
            missing_numeric = sorted(set(self.feature_columns) - set(self.numeric_feature_columns_))
            raise ValueError(f"All feature columns must be numeric. Non-numeric columns: {missing_numeric}")

        imputed = self.imputer.fit_transform(X[self.numeric_feature_columns_])
        scaled = self.scaler.fit_transform(imputed)

        if self.normalizer is not None:
            scaled = self.normalizer.fit_transform(scaled)

        selected = self.feature_selector.fit_transform(scaled)
        selected_mask = self.feature_selector.get_support()
        self.selected_features_ = [
            column for column, keep in zip(self.numeric_feature_columns_, selected_mask) if keep
        ]

        if not self.selected_features_:
            raise ValueError("Feature selection removed all columns. Lower the variance_threshold.")

        self.label_encoder.fit(y.astype(str))
        self.statistics_ = self._build_statistics(df, selected.shape[0])
        self.is_fitted_ = True
        LOGGER.info("Preprocessor fitted with %d selected features.", len(self.selected_features_))
        return self

    def transform(
        self,
        dataset: pd.DataFrame,
        *,
        encode_target: bool = True,
        drop_outliers: bool = False,
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Transform a dataset using previously fitted preprocessing steps."""

        self._ensure_fitted()
        df = self._validate_input_dataframe(dataset)

        if drop_outliers:
            df = self._filter_outliers(df)

        X = df[self.feature_columns].copy()
        imputed = self.imputer.transform(X[self.numeric_feature_columns_])
        scaled = self.scaler.transform(imputed)

        if self.normalizer is not None:
            scaled = self.normalizer.transform(scaled)

        selected = self.feature_selector.transform(scaled)
        X_transformed = pd.DataFrame(selected, columns=self.selected_features_, index=df.index)

        y_transformed: Optional[pd.Series] = None
        if encode_target and self.target_column in df.columns:
            encoded_target = self.label_encoder.transform(df[self.target_column].astype(str))
            y_transformed = pd.Series(encoded_target, index=df.index, name=self.target_column)

        return X_transformed, y_transformed

    def fit_transform(
        self,
        dataset: pd.DataFrame,
        *,
        encode_target: bool = True,
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Fit the preprocessing pipeline and transform the same dataset."""

        self.fit(dataset)
        return self.transform(dataset, encode_target=encode_target, drop_outliers=True)

    def split_data(self, dataset: pd.DataFrame) -> SplitData:
        """Split the dataset into stratified train, validation, and test sets."""

        df = self._validate_input_dataframe(dataset)
        stratify_values = df[self.target_column] if self.stratify else None

        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_values,
        )

        val_df: Optional[pd.DataFrame] = None
        if self.validation_size > 0:
            adjusted_validation_size = self.validation_size / (1.0 - self.test_size)
            train_stratify = train_df[self.target_column] if self.stratify else None
            train_df, val_df = train_test_split(
                train_df,
                test_size=adjusted_validation_size,
                random_state=self.random_state,
                stratify=train_stratify,
            )

        X_train, y_train = self.fit_transform(train_df)
        X_test, y_test = self.transform(test_df, drop_outliers=False)

        X_val: Optional[pd.DataFrame] = None
        y_val: Optional[pd.Series] = None
        if val_df is not None:
            X_val, y_val = self.transform(val_df, drop_outliers=False)

        if y_train is None or y_test is None:
            raise ValueError("Encoded target values could not be generated during splitting.")

        return SplitData(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )

    def preprocess_pipeline(
        self,
        filepath: Union[str, Path],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Compatibility helper returning train and test splits."""

        dataset = self.load_data(filepath)
        split_data = self.split_data(dataset)
        return split_data.X_train, split_data.X_test, split_data.y_train, split_data.y_test

    def inverse_transform_labels(self, labels: Sequence[int]) -> np.ndarray:
        """Convert encoded labels back to original class names."""

        self._ensure_fitted()
        return self.label_encoder.inverse_transform(np.asarray(labels))

    def get_selected_features(self) -> list[str]:
        """Return the selected feature names after fitting."""

        self._ensure_fitted()
        return list(self.selected_features_)

    def get_statistics(self) -> Dict[str, Any]:
        """Return preprocessing summary statistics."""

        self._ensure_fitted()
        return dict(self.statistics_)

    def _validate_configuration(self) -> None:
        """Validate preprocessor configuration values."""

        valid_missing = {"mean", "median", "most_frequent", "constant"}
        valid_outlier_methods = {"zscore", "iqr", "none"}
        valid_scalers = {"standard", "minmax", None}

        if not self.feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        forbidden_features = sorted(set(self.feature_columns) & self.FORBIDDEN_FEATURE_COLUMNS)
        if forbidden_features:
            raise ValueError(
                "Target leakage detected. The following columns cannot be used as input features: "
                f"{forbidden_features}"
            )
        if self.target_column in self.feature_columns:
            raise ValueError("target_column cannot also appear in feature_columns.")
        if self.missing_strategy not in valid_missing:
            raise ValueError(f"missing_strategy must be one of {sorted(valid_missing)}.")
        if self.outlier_method not in valid_outlier_methods:
            raise ValueError(f"outlier_method must be one of {sorted(valid_outlier_methods)}.")
        if self.scaling_method not in valid_scalers:
            raise ValueError("scaling_method must be 'standard', 'minmax', or None.")
        if self.normalization_method not in valid_scalers:
            raise ValueError("normalization_method must be 'standard', 'minmax', or None.")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must satisfy 0.0 < test_size < 1.0.")
        if not 0.0 <= self.validation_size < 1.0:
            raise ValueError("validation_size must satisfy 0.0 <= validation_size < 1.0.")
        if self.test_size + self.validation_size >= 1.0:
            raise ValueError("The sum of test_size and validation_size must be less than 1.0.")
        if self.outlier_threshold <= 0:
            raise ValueError("outlier_threshold must be strictly positive.")
        if self.outlier_iqr_multiplier <= 0:
            raise ValueError("outlier_iqr_multiplier must be strictly positive.")
        if self.variance_threshold < 0:
            raise ValueError("variance_threshold must be non-negative.")

    def _validate_input_dataframe(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Validate input dataset structure."""

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(
                f"Expected dataset to be a pandas DataFrame, received {type(dataset).__name__}."
            )

        required_columns = set(self.feature_columns + [self.target_column])
        missing_columns = sorted(required_columns - set(dataset.columns))
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")

        if dataset.empty:
            raise ValueError("Dataset cannot be empty.")

        forbidden_features = sorted(set(self.feature_columns) & self.FORBIDDEN_FEATURE_COLUMNS)
        if forbidden_features:
            raise ValueError(
                "Target leakage detected during preprocessing. Forbidden feature columns present: "
                f"{forbidden_features}"
            )

        return dataset.copy()

    def _filter_outliers(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using z-score or IQR logic."""

        if self.outlier_method == "none":
            return dataset.copy()

        numeric_data = dataset[self.feature_columns].astype(float)

        if self.outlier_method == "zscore":
            means = numeric_data.mean()
            stds = numeric_data.std(ddof=0).replace(0, np.nan)
            zscores = ((numeric_data - means) / stds).abs()
            mask = (zscores <= self.outlier_threshold) | zscores.isna()
            filtered = dataset.loc[mask.all(axis=1)].copy()
        elif self.outlier_method == "iqr":
            q1 = numeric_data.quantile(0.25)
            q3 = numeric_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.outlier_iqr_multiplier * iqr
            upper_bound = q3 + self.outlier_iqr_multiplier * iqr
            mask = ((numeric_data >= lower_bound) & (numeric_data <= upper_bound)) | iqr.eq(0)
            filtered = dataset.loc[mask.all(axis=1)].copy()
        else:
            raise ValueError(f"Unsupported outlier_method: {self.outlier_method}")

        if filtered.empty:
            raise ValueError("Outlier filtering removed all rows. Relax the threshold or review the data.")

        LOGGER.info("Outlier filtering reduced rows from %d to %d.", len(dataset), len(filtered))
        return filtered

    @staticmethod
    def _build_scaler(method: Optional[str]) -> Any:
        """Create a scaler object for the requested method."""

        if method is None:
            return _IdentityTransformer()
        if method == "standard":
            return StandardScaler()
        if method == "minmax":
            return MinMaxScaler()
        raise ValueError(f"Unsupported scaling method: {method}")

    def _ensure_fitted(self) -> None:
        """Ensure the preprocessor has been fitted before transform usage."""

        if not self.is_fitted_:
            raise RuntimeError("DataPreprocessor must be fitted before calling transform-related methods.")

    def _build_statistics(self, dataset: pd.DataFrame, fitted_rows: int) -> Dict[str, Any]:
        """Build summary statistics for fitted preprocessing state."""

        return {
            "original_rows": int(len(dataset)),
            "fitted_rows": int(fitted_rows),
            "selected_feature_count": len(self.selected_features_),
            "selected_features": list(self.selected_features_),
            "class_mapping": {
                class_name: int(index) for index, class_name in enumerate(self.label_encoder.classes_)
            },
            "missing_strategy": self.missing_strategy,
            "outlier_method": self.outlier_method,
            "scaling_method": self.scaling_method,
            "normalization_method": self.normalization_method,
        }


class _IdentityTransformer:
    """Minimal transformer implementing fit/transform as pass-through."""

    def fit(self, values: np.ndarray) -> "_IdentityTransformer":
        """Fit no-op transformer."""

        _ = values
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Return input unchanged."""

        return np.asarray(values)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        """Fit and transform in one pass."""

        return self.fit(values).transform(values)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline("stress_dataset.csv")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(preprocessor.get_statistics())
