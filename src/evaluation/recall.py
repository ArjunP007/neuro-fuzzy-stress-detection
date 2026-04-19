"""Recall metric utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.evaluation.confusion_matrix import compute_confusion_matrix
from src.evaluation.statistics import StatisticalAnalysis


def recall_score(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    average: str = "macro",
    labels: Optional[Sequence[int]] = None,
) -> float:
    """Compute multiclass recall."""

    matrix = compute_confusion_matrix(y_true, y_pred, labels)
    true_positives = np.diag(matrix).astype(float)
    actual_positives = matrix.sum(axis=1).astype(float)
    per_class_recall = np.divide(
        true_positives,
        actual_positives,
        out=np.zeros_like(true_positives),
        where=actual_positives != 0,
    )

    if average == "macro":
        return float(np.mean(per_class_recall))
    if average == "weighted":
        supports = actual_positives
        total_support = np.sum(supports)
        if total_support == 0:
            return 0.0
        return float(np.sum(per_class_recall * supports) / total_support)
    raise ValueError("average must be either 'macro' or 'weighted'.")


def recall_analysis(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    average: str = "macro",
    labels: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Return recall and bootstrap statistics."""

    score = recall_score(y_true, y_pred, average=average, labels=labels)
    summary = StatisticalAnalysis.bootstrap_metric(
        lambda yt, yp: recall_score(yt, yp, average=average, labels=labels),
        y_true,
        y_pred,
    )
    return {
        "recall": score,
        "average": average,
        "statistics": summary.to_dict(),
    }


__all__ = [
    "recall_analysis",
    "recall_score",
]
