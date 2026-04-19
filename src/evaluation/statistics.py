"""Shared statistical analysis utilities for model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass
class StatisticalSummary:
    """Structured statistical summary for metric distributions."""

    mean: float
    std: float
    variance: float
    minimum: float
    maximum: float
    median: float
    confidence_interval_95: tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize summary values to a dictionary."""

        return {
            "mean": self.mean,
            "std": self.std,
            "variance": self.variance,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "median": self.median,
            "confidence_interval_95": self.confidence_interval_95,
        }


class StatisticalAnalysis:
    """Utility methods for statistical analysis of evaluation results."""

    @staticmethod
    def validate_inputs(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Validate and normalize classification targets."""

        true_values = np.asarray(y_true)
        predicted_values = np.asarray(y_pred)

        if true_values.ndim != 1 or predicted_values.ndim != 1:
            raise ValueError("y_true and y_pred must be one-dimensional arrays.")
        if true_values.shape[0] != predicted_values.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")
        if true_values.shape[0] == 0:
            raise ValueError("y_true and y_pred cannot be empty.")

        return true_values.astype(int), predicted_values.astype(int)

    @staticmethod
    def infer_labels(
        y_true: Sequence[int],
        y_pred: Sequence[int],
        labels: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Infer or validate class labels."""

        true_values, predicted_values = StatisticalAnalysis.validate_inputs(y_true, y_pred)
        if labels is not None:
            label_values = np.asarray(labels, dtype=int)
            if label_values.ndim != 1 or label_values.size == 0:
                raise ValueError("labels must be a non-empty one-dimensional sequence.")
            return label_values
        return np.unique(np.concatenate([true_values, predicted_values]))

    @staticmethod
    def summarize(values: Sequence[float]) -> StatisticalSummary:
        """Compute descriptive statistics and a 95% confidence interval."""

        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("values must be a non-empty one-dimensional sequence.")

        mean = float(np.mean(array))
        std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        variance = float(np.var(array, ddof=1)) if array.size > 1 else 0.0
        minimum = float(np.min(array))
        maximum = float(np.max(array))
        median = float(np.median(array))
        margin = 1.96 * std / np.sqrt(array.size) if array.size > 1 else 0.0
        confidence_interval = (mean - margin, mean + margin)

        return StatisticalSummary(
            mean=mean,
            std=std,
            variance=variance,
            minimum=minimum,
            maximum=maximum,
            median=median,
            confidence_interval_95=confidence_interval,
        )

    @staticmethod
    def bootstrap_metric(
        metric_function: Any,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        *,
        n_bootstrap: int = 200,
        random_state: int = 42,
    ) -> StatisticalSummary:
        """Estimate metric variability using bootstrap resampling."""

        true_values, predicted_values = StatisticalAnalysis.validate_inputs(y_true, y_pred)
        if n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be strictly positive.")

        rng = np.random.default_rng(random_state)
        sample_size = true_values.shape[0]
        estimates = []

        for _ in range(n_bootstrap):
            indices = rng.integers(0, sample_size, size=sample_size)
            estimates.append(metric_function(true_values[indices], predicted_values[indices]))

        return StatisticalAnalysis.summarize(estimates)


__all__ = [
    "StatisticalAnalysis",
    "StatisticalSummary",
]
