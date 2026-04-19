"""Accuracy metric utilities."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from src.evaluation.statistics import StatisticalAnalysis


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute classification accuracy."""

    true_values, predicted_values = StatisticalAnalysis.validate_inputs(y_true, y_pred)
    return float(np.mean(true_values == predicted_values))


def accuracy_analysis(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    """Return accuracy and bootstrap-based statistical summary."""

    score = accuracy_score(y_true, y_pred)
    summary = StatisticalAnalysis.bootstrap_metric(accuracy_score, y_true, y_pred)
    return {
        "accuracy": score,
        "statistics": summary.to_dict(),
    }


__all__ = [
    "accuracy_analysis",
    "accuracy_score",
]
