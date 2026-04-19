"""Confusion matrix computation utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.evaluation.statistics import StatisticalAnalysis


def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Compute the confusion matrix for multiclass classification."""

    true_values, predicted_values = StatisticalAnalysis.validate_inputs(y_true, y_pred)
    label_values = StatisticalAnalysis.infer_labels(true_values, predicted_values, labels)
    label_to_index = {label: index for index, label in enumerate(label_values)}

    matrix = np.zeros((label_values.size, label_values.size), dtype=int)
    for true_label, predicted_label in zip(true_values, predicted_values):
        matrix[label_to_index[int(true_label)], label_to_index[int(predicted_label)]] += 1
    return matrix


def confusion_matrix_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Return confusion matrix plus useful derived statistics."""

    matrix = compute_confusion_matrix(y_true, y_pred, labels)
    label_values = StatisticalAnalysis.infer_labels(y_true, y_pred, labels)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    return {
        "labels": label_values.tolist(),
        "matrix": matrix,
        "normalized_matrix": normalized,
        "true_support": matrix.sum(axis=1).tolist(),
        "predicted_support": matrix.sum(axis=0).tolist(),
    }


__all__ = [
    "compute_confusion_matrix",
    "confusion_matrix_report",
]
