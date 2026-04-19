"""Precision metric utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.evaluation.confusion_matrix import compute_confusion_matrix
from src.evaluation.statistics import StatisticalAnalysis


def precision_score(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    average: str = "macro",
    labels: Optional[Sequence[int]] = None,
) -> float:
    """Compute multiclass precision."""

    matrix = compute_confusion_matrix(y_true, y_pred, labels)
    true_positives = np.diag(matrix).astype(float)
    predicted_positives = matrix.sum(axis=0).astype(float)
    per_class_precision = np.divide(
        true_positives,
        predicted_positives,
        out=np.zeros_like(true_positives),
        where=predicted_positives != 0,
    )

    if average == "macro":
        return float(np.mean(per_class_precision))
    if average == "weighted":
        supports = matrix.sum(axis=1).astype(float)
        total_support = np.sum(supports)
        if total_support == 0:
            return 0.0
        return float(np.sum(per_class_precision * supports) / total_support)
    raise ValueError("average must be either 'macro' or 'weighted'.")


def precision_analysis(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    average: str = "macro",
    labels: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Return precision and bootstrap statistics."""

    score = precision_score(y_true, y_pred, average=average, labels=labels)
    summary = StatisticalAnalysis.bootstrap_metric(
        lambda yt, yp: precision_score(yt, yp, average=average, labels=labels),
        y_true,
        y_pred,
    )
    return {
        "precision": score,
        "average": average,
        "statistics": summary.to_dict(),
    }


__all__ = [
    "precision_analysis",
    "precision_score",
]
