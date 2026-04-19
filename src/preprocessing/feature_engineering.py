"""Feature engineering utilities for the stress detection project."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureRankingResult:
    """Container for feature ranking outputs."""

    feature_scores: pd.DataFrame
    top_features: list[str]


class FeatureEngineer:
    """Create a compact, domain-driven feature set for stress prediction."""

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

    ENGINEERED_FEATURE_COLUMNS = [
        "fatigue_sleep_ratio",
        "work_screen_load",
        "recovery_balance",
        "stress_physiology",
        "lifestyle_balance",
    ]

    def __init__(
        self,
        *,
        feature_columns: Optional[Sequence[str]] = None,
        target_column: Optional[str] = None,
        epsilon: float = 1e-6,
        top_k_features: Optional[int] = None,
    ) -> None:
        """Initialize the feature engineering configuration."""

        self.feature_columns = list(feature_columns or self.DEFAULT_FEATURE_COLUMNS)
        self.target_column = target_column
        self.epsilon = float(epsilon)
        self.top_k_features = top_k_features

        self._validate_configuration()

        self.is_fitted_: bool = False
        self.input_feature_columns_: list[str] = []
        self.engineered_feature_columns_: list[str] = []
        self.feature_scores_: Optional[pd.DataFrame] = None

    def fit(self, dataset: pd.DataFrame) -> "FeatureEngineer":
        """Fit the feature engineer by validating the input feature set."""

        df = self._validate_dataframe(dataset)
        numeric_frame = self._extract_numeric_features(df)
        self.input_feature_columns_ = list(numeric_frame.columns)
        self.engineered_feature_columns_ = self.input_feature_columns_ + self.ENGINEERED_FEATURE_COLUMNS
        self.is_fitted_ = True
        LOGGER.info("FeatureEngineer fitted on %d base features.", len(self.input_feature_columns_))
        return self

    def transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Transform a dataset into a compact engineered feature representation."""

        self._ensure_fitted()
        df = self._validate_dataframe(dataset)
        numeric_frame = self._extract_numeric_features(df)
        engineered = self._generate_meaningful_features(numeric_frame)
        self.engineered_feature_columns_ = list(engineered.columns)
        return engineered

    def fit_transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the provided dataset."""

        self.fit(dataset)
        return self.transform(dataset)

    def rank_features(
        self,
        dataset: pd.DataFrame,
        target: Optional[Sequence[Any]] = None,
        *,
        task_type: str = "classification",
    ) -> FeatureRankingResult:
        """Rank engineered features using mutual information scoring."""

        engineered = self.fit_transform(dataset) if not self.is_fitted_ else self.transform(dataset)
        target_values = self._resolve_target(dataset, target)

        if task_type == "classification":
            scores = mutual_info_classif(engineered, target_values, random_state=0)
        elif task_type == "regression":
            scores = mutual_info_regression(engineered, target_values, random_state=0)
        else:
            raise ValueError("task_type must be either 'classification' or 'regression'.")

        ranking_frame = pd.DataFrame(
            {
                "feature": engineered.columns,
                "mutual_information_score": scores,
            }
        ).sort_values(by="mutual_information_score", ascending=False, ignore_index=True)

        if self.top_k_features is not None:
            top_features = ranking_frame.head(self.top_k_features)["feature"].tolist()
        else:
            top_features = ranking_frame["feature"].tolist()

        self.feature_scores_ = ranking_frame
        return FeatureRankingResult(feature_scores=ranking_frame, top_features=top_features)

    def get_top_features(self, n: int = 10) -> list[str]:
        """Return the top-ranked engineered feature names."""

        if self.feature_scores_ is None:
            raise RuntimeError("Feature scores are not available. Call rank_features first.")
        return self.feature_scores_.head(n)["feature"].tolist()

    def _generate_meaningful_features(self, numeric_frame: pd.DataFrame) -> pd.DataFrame:
        """Generate a compact, signal-focused feature set."""

        engineered = numeric_frame.copy()
        engineered["fatigue_sleep_ratio"] = (
            numeric_frame["mental_fatigue_score"] / (numeric_frame["sleep_hours"] + 0.5)
        )
        engineered["work_screen_load"] = (
            numeric_frame["work_hours"] + numeric_frame["screen_time"]
        )
        engineered["recovery_balance"] = (
            numeric_frame["sleep_hours"] + numeric_frame["physical_activity_hours"]
        )
        engineered["stress_physiology"] = (
            numeric_frame["heart_rate"] * numeric_frame["mental_fatigue_score"] / 100.0
        )
        engineered["lifestyle_balance"] = (
            numeric_frame["sleep_hours"]
            + numeric_frame["physical_activity_hours"]
            + numeric_frame["social_interaction_hours"]
        ) / 3.0

        engineered = engineered.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        LOGGER.info("Generated %d engineered features.", engineered.shape[1])
        return engineered

    def _resolve_target(
        self,
        dataset: pd.DataFrame,
        target: Optional[Sequence[Any]],
    ) -> np.ndarray:
        """Resolve the target array for feature ranking."""

        if target is not None:
            target_array = np.asarray(target)
        elif self.target_column is not None and self.target_column in dataset.columns:
            target_array = dataset[self.target_column].to_numpy()
        else:
            raise ValueError("A target sequence or valid target_column is required for feature ranking.")

        if target_array.ndim != 1:
            raise ValueError("Target values must be one-dimensional.")
        return target_array

    def _extract_numeric_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Extract and validate the configured numeric feature columns."""

        missing_columns = [column for column in self.feature_columns if column not in dataset.columns]
        if missing_columns:
            raise ValueError(f"Dataset is missing required feature columns: {missing_columns}")

        numeric_frame = dataset[self.feature_columns].copy()
        for column in numeric_frame.columns:
            if not pd.api.types.is_numeric_dtype(numeric_frame[column]):
                raise ValueError(f"Feature column '{column}' must be numeric for feature engineering.")

        return numeric_frame.astype(float)

    @staticmethod
    def _validate_dataframe(dataset: pd.DataFrame) -> pd.DataFrame:
        """Validate dataframe inputs."""

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, received {type(dataset).__name__}.")
        if dataset.empty:
            raise ValueError("Dataset cannot be empty.")
        return dataset.copy()

    def _ensure_fitted(self) -> None:
        """Ensure the feature engineer is fitted before transform usage."""

        if not self.is_fitted_:
            raise RuntimeError("FeatureEngineer must be fitted before calling transform.")

    def _validate_configuration(self) -> None:
        """Validate constructor configuration."""

        if not self.feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be strictly positive.")
        if self.top_k_features is not None and self.top_k_features <= 0:
            raise ValueError("top_k_features must be strictly positive when provided.")


__all__ = [
    "FeatureEngineer",
    "FeatureRankingResult",
]
