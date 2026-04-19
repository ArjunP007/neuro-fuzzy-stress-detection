"""Visualization utilities for training, evaluation, and data exploration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class VisualizationError(Exception):
    """Raised when visualization inputs are invalid."""


class PlotManager:
    """Reusable plotting manager for the stress detection project."""

    def __init__(
        self,
        *,
        style: str = "whitegrid",
        palette: str = "deep",
        figure_size: tuple[int, int] = (10, 6),
        dpi: int = 120,
    ) -> None:
        """Initialize plotting defaults."""

        self.style = style
        self.palette = palette
        self.figure_size = figure_size
        self.dpi = dpi
        sns.set_theme(style=self.style, palette=self.palette)

    def plot_training_curves(
        self,
        train_loss: Sequence[float],
        val_loss: Optional[Sequence[float]] = None,
        *,
        title: str = "Training Curves",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot training and validation loss curves."""

        train_values = np.asarray(train_loss, dtype=float)
        if train_values.ndim != 1 or train_values.size == 0:
            raise VisualizationError("train_loss must be a non-empty one-dimensional sequence.")

        figure, axis = self._create_figure()
        epochs = np.arange(1, train_values.size + 1)
        axis.plot(epochs, train_values, label="Train Loss", linewidth=2.0)

        if val_loss is not None:
            validation_values = np.asarray(val_loss, dtype=float)
            if validation_values.shape != train_values.shape:
                raise VisualizationError("val_loss must have the same length as train_loss.")
            axis.plot(epochs, validation_values, label="Validation Loss", linewidth=2.0)

        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.legend()
        axis.grid(True, linestyle="--", alpha=0.5)
        return self._finalize_figure(figure, save_path=save_path, show=show)

    def plot_feature_importance(
        self,
        feature_names: Sequence[str],
        importance_scores: Sequence[float],
        *,
        top_k: Optional[int] = None,
        title: str = "Feature Importance",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot feature importance as a ranked horizontal bar chart."""

        names = list(feature_names)
        scores = np.asarray(importance_scores, dtype=float)
        if len(names) == 0 or scores.size == 0:
            raise VisualizationError("feature_names and importance_scores cannot be empty.")
        if len(names) != scores.size:
            raise VisualizationError("feature_names and importance_scores must have the same length.")

        ranking = pd.DataFrame({"feature": names, "importance": scores}).sort_values(
            by="importance",
            ascending=False,
        )
        if top_k is not None:
            if top_k <= 0:
                raise VisualizationError("top_k must be strictly positive when provided.")
            ranking = ranking.head(top_k)

        figure, axis = self._create_figure(height=max(6, int(0.4 * len(ranking) + 2)))
        sns.barplot(data=ranking, x="importance", y="feature", ax=axis, orient="h")
        axis.set_title(title)
        axis.set_xlabel("Importance Score")
        axis.set_ylabel("Feature")
        return self._finalize_figure(figure, save_path=save_path, show=show)

    def plot_confusion_matrix_heatmap(
        self,
        confusion_matrix: np.ndarray,
        *,
        labels: Optional[Sequence[str]] = None,
        normalize: bool = False,
        title: str = "Confusion Matrix",
        cmap: str = "Blues",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot a confusion matrix heatmap."""

        matrix = np.asarray(confusion_matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise VisualizationError("confusion_matrix must be a square 2D array.")

        display_matrix = matrix.astype(float)
        annotation_format = ".2f"
        if normalize:
            row_sums = display_matrix.sum(axis=1, keepdims=True)
            display_matrix = np.divide(
                display_matrix,
                row_sums,
                out=np.zeros_like(display_matrix),
                where=row_sums != 0,
            )
        else:
            display_matrix = matrix.astype(int)
            annotation_format = "d"

        tick_labels = list(labels) if labels is not None else [str(index) for index in range(matrix.shape[0])]
        if len(tick_labels) != matrix.shape[0]:
            raise VisualizationError("labels length must match confusion_matrix dimensions.")

        figure, axis = self._create_figure()
        sns.heatmap(
            display_matrix,
            annot=True,
            fmt=annotation_format,
            cmap=cmap,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("Predicted Label")
        axis.set_ylabel("True Label")
        return self._finalize_figure(figure, save_path=save_path, show=show)

    def plot_roc_curve(
        self,
        y_true_binary: Sequence[int],
        y_score: Sequence[float],
        *,
        title: str = "ROC Curve",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot ROC curve for binary classification scores."""

        true_values, score_values = self._validate_binary_curve_inputs(y_true_binary, y_score)
        thresholds = np.unique(score_values)[::-1]
        thresholds = np.concatenate(([np.inf], thresholds, [-np.inf]))

        tpr_values = []
        fpr_values = []
        for threshold in thresholds:
            predictions = (score_values >= threshold).astype(int)
            tp = np.sum((predictions == 1) & (true_values == 1))
            fp = np.sum((predictions == 1) & (true_values == 0))
            tn = np.sum((predictions == 0) & (true_values == 0))
            fn = np.sum((predictions == 0) & (true_values == 1))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr_values.append(tpr)
            fpr_values.append(fpr)

        order = np.argsort(fpr_values)
        fpr_sorted = np.asarray(fpr_values)[order]
        tpr_sorted = np.asarray(tpr_values)[order]
        auc_score = float(np.trapezoid(tpr_sorted, fpr_sorted))

        figure, axis = self._create_figure()
        axis.plot(fpr_sorted, tpr_sorted, label=f"ROC (AUC = {auc_score:.3f})", linewidth=2.0)
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")
        axis.set_title(title)
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.legend(loc="lower right")
        axis.grid(True, linestyle="--", alpha=0.5)
        return self._finalize_figure(figure, save_path=save_path, show=show)

    def plot_precision_recall_curve(
        self,
        y_true_binary: Sequence[int],
        y_score: Sequence[float],
        *,
        title: str = "Precision-Recall Curve",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot precision-recall curve for binary classification scores."""

        true_values, score_values = self._validate_binary_curve_inputs(y_true_binary, y_score)
        thresholds = np.unique(score_values)[::-1]
        thresholds = np.concatenate((thresholds, [-np.inf]))

        precision_values = []
        recall_values = []
        for threshold in thresholds:
            predictions = (score_values >= threshold).astype(int)
            tp = np.sum((predictions == 1) & (true_values == 1))
            fp = np.sum((predictions == 1) & (true_values == 0))
            fn = np.sum((predictions == 0) & (true_values == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision_values.append(precision)
            recall_values.append(recall)

        order = np.argsort(recall_values)
        recall_sorted = np.asarray(recall_values)[order]
        precision_sorted = np.asarray(precision_values)[order]
        pr_auc = float(np.trapezoid(precision_sorted, recall_sorted))

        figure, axis = self._create_figure()
        axis.plot(recall_sorted, precision_sorted, label=f"PR Curve (AUC = {pr_auc:.3f})", linewidth=2.0)
        axis.set_title(title)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.legend(loc="lower left")
        axis.grid(True, linestyle="--", alpha=0.5)
        return self._finalize_figure(figure, save_path=save_path, show=show)

    def plot_feature_distributions(
        self,
        dataset: pd.DataFrame,
        *,
        columns: Optional[Sequence[str]] = None,
        bins: int = 25,
        kde: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot distributions for selected numeric features."""

        frame = self._validate_dataframe(dataset)
        selected_columns = list(columns) if columns is not None else frame.select_dtypes(include=[np.number]).columns.tolist()
        if not selected_columns:
            raise VisualizationError("No numeric columns available for distribution plotting.")

        missing_columns = [column for column in selected_columns if column not in frame.columns]
        if missing_columns:
            raise VisualizationError(f"Dataset is missing requested columns: {missing_columns}")

        numeric_frame = frame[selected_columns]
        num_features = len(selected_columns)
        rows = int(np.ceil(num_features / 2))
        figure, axes = plt.subplots(rows, 2, figsize=(14, max(4 * rows, 4)), dpi=self.dpi)
        axes_array = np.atleast_1d(axes).ravel()

        for axis, column in zip(axes_array, selected_columns):
            sns.histplot(numeric_frame[column], bins=bins, kde=kde, ax=axis)
            axis.set_title(f"Distribution: {column}")
            axis.set_xlabel(column)
            axis.set_ylabel("Frequency")

        for axis in axes_array[num_features:]:
            axis.axis("off")

        figure.tight_layout()
        return self._finalize_figure(figure, save_path=save_path, show=show)

    def plot_correlation_heatmap(
        self,
        dataset: pd.DataFrame,
        *,
        columns: Optional[Sequence[str]] = None,
        method: str = "pearson",
        title: str = "Correlation Heatmap",
        cmap: str = "coolwarm",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot a correlation heatmap for numeric features."""

        frame = self._validate_dataframe(dataset)
        selected_columns = list(columns) if columns is not None else frame.select_dtypes(include=[np.number]).columns.tolist()
        if not selected_columns:
            raise VisualizationError("No numeric columns available for correlation plotting.")

        correlation_matrix = frame[selected_columns].corr(method=method, numeric_only=True)
        figure, axis = self._create_figure(width=max(8, len(selected_columns)), height=max(6, len(selected_columns) * 0.7))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap=cmap, square=True, ax=axis)
        axis.set_title(title)
        return self._finalize_figure(figure, save_path=save_path, show=show)

    def _create_figure(
        self,
        *,
        width: Optional[float] = None,
        height: Optional[float] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a single-axis figure."""

        final_width = width if width is not None else self.figure_size[0]
        final_height = height if height is not None else self.figure_size[1]
        figure, axis = plt.subplots(figsize=(final_width, final_height), dpi=self.dpi)
        return figure, axis

    def _finalize_figure(
        self,
        figure: plt.Figure,
        *,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Finalize, save, and optionally show a figure."""

        figure.tight_layout()
        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(figure)
        return figure

    @staticmethod
    def _validate_dataframe(dataset: pd.DataFrame) -> pd.DataFrame:
        """Validate dataframe plotting input."""

        if not isinstance(dataset, pd.DataFrame):
            raise VisualizationError(
                f"Expected dataset to be a pandas DataFrame, received {type(dataset).__name__}."
            )
        if dataset.empty:
            raise VisualizationError("Dataset cannot be empty.")
        return dataset.copy()

    @staticmethod
    def _validate_binary_curve_inputs(
        y_true_binary: Sequence[int],
        y_score: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate binary labels and decision scores for curve plotting."""

        true_values = np.asarray(y_true_binary, dtype=int)
        score_values = np.asarray(y_score, dtype=float)
        if true_values.ndim != 1 or score_values.ndim != 1:
            raise VisualizationError("y_true_binary and y_score must be one-dimensional sequences.")
        if true_values.size == 0 or score_values.size == 0:
            raise VisualizationError("y_true_binary and y_score cannot be empty.")
        if true_values.size != score_values.size:
            raise VisualizationError("y_true_binary and y_score must have the same length.")
        if not np.all(np.isin(true_values, [0, 1])):
            raise VisualizationError("y_true_binary must contain only binary values 0 and 1.")
        return true_values, score_values


__all__ = [
    "PlotManager",
    "VisualizationError",
]
