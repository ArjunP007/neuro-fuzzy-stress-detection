"""Hybrid neuro-fuzzy system combining neural predictions with fuzzy inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.fuzzy_logic.inference import FuzzyInferenceEngine, InferenceResult
from src.fuzzy_logic.rules import FuzzyRuleBase
from src.neural_network.network import NeuralNetwork, TrainingHistory


LOGGER = logging.getLogger(__name__)


@dataclass
class NeuroFuzzyResult:
    """Container for hybrid neuro-fuzzy inference outputs."""

    neural_probabilities: np.ndarray
    fuzzy_probabilities: np.ndarray
    ensemble_probabilities: np.ndarray
    ensemble_predictions: np.ndarray
    neural_confidence: np.ndarray
    fuzzy_confidence: np.ndarray
    combined_confidence: np.ndarray
    fuzzy_outputs: list[InferenceResult] = field(default_factory=list)


class NeuroFuzzySystem:
    """Hybrid architecture that fuses neural-network and fuzzy-inference predictions."""

    def __init__(
        self,
        neural_network: NeuralNetwork,
        fuzzy_inference_engine: FuzzyInferenceEngine,
        *,
        feature_names: Sequence[str],
        class_labels: Sequence[str] = ("Low", "Medium", "High"),
        fuzzy_output_variable: str = "stress_level",
        fuzzy_score_variable: str = "stress_score",
        neural_weight: float = 0.6,
        fuzzy_weight: float = 0.4,
        adaptive_tuning_rate: float = 0.05,
    ) -> None:
        """Initialize the neuro-fuzzy system."""

        if not feature_names:
            raise ValueError("feature_names cannot be empty.")
        if len(class_labels) < 2:
            raise ValueError("class_labels must contain at least two classes.")
        if not 0.0 <= neural_weight <= 1.0:
            raise ValueError("neural_weight must satisfy 0.0 <= weight <= 1.0.")
        if not 0.0 <= fuzzy_weight <= 1.0:
            raise ValueError("fuzzy_weight must satisfy 0.0 <= weight <= 1.0.")
        if abs((neural_weight + fuzzy_weight) - 1.0) > 1e-8:
            raise ValueError("neural_weight and fuzzy_weight must sum to 1.0.")
        if adaptive_tuning_rate <= 0.0:
            raise ValueError("adaptive_tuning_rate must be strictly positive.")

        self.neural_network = neural_network
        self.fuzzy_inference_engine = fuzzy_inference_engine
        self.feature_names = list(feature_names)
        self.class_labels = [label.strip() for label in class_labels]
        self.class_to_index = {label.lower(): index for index, label in enumerate(self.class_labels)}
        self.fuzzy_output_variable = fuzzy_output_variable.strip().lower()
        self.fuzzy_score_variable = fuzzy_score_variable.strip().lower()
        self.neural_weight = float(neural_weight)
        self.fuzzy_weight = float(fuzzy_weight)
        self.adaptive_tuning_rate = float(adaptive_tuning_rate)
        self.rule_performance_: Dict[str, float] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> TrainingHistory:
        """Train the neural component and adapt fuzzy rules from validation feedback."""

        history = self.neural_network.fit(
            X,
            y,
            X_val=X_val,
            y_val=y_val,
            verbose=verbose,
        )

        reference_X = X_val if X_val is not None else X
        reference_y = y_val if y_val is not None else y
        self.hybrid_learn(reference_X, reference_y)
        return history

    def hybrid_learn(self, X: np.ndarray, y: np.ndarray) -> None:
        """Adapt fuzzy rule weights using observed prediction performance."""

        features = self._to_feature_frame(X)
        labels = self._prepare_labels(y)
        rule_feedback: Dict[str, list[float]] = {}

        for row_index in range(len(features)):
            crisp_inputs = self._row_to_input_dict(features.iloc[row_index])
            rule_evaluations = self.fuzzy_inference_engine.rule_base.evaluate_rules(crisp_inputs)
            fuzzy_result = self.fuzzy_inference_engine.infer(crisp_inputs)
            predicted_label_index = self._map_fuzzy_result_to_label_index(fuzzy_result)
            target_label_index = int(labels[row_index])

            reward = 1.0 if predicted_label_index == target_label_index else -1.0
            for evaluation in rule_evaluations:
                weighted_feedback = reward * float(evaluation["firing_strength"])
                rule_feedback.setdefault(evaluation["rule_id"], []).append(weighted_feedback)

        averaged_feedback = {
            rule_id: float(np.mean(feedback_values))
            for rule_id, feedback_values in rule_feedback.items()
        }
        self.rule_performance_ = averaged_feedback
        self.adaptive_rule_tuning(averaged_feedback)

    def adaptive_rule_tuning(self, performance_feedback: Dict[str, float]) -> None:
        """Adapt fuzzy rule weights using averaged performance signals."""

        scaled_feedback = {
            rule_id: self.adaptive_tuning_rate * score
            for rule_id, score in performance_feedback.items()
        }
        self.fuzzy_inference_engine.rule_base.optimize_rules(scaled_feedback)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ensemble class labels."""

        result = self.ensemble_predict(X)
        return result.ensemble_predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict ensemble class probabilities."""

        result = self.ensemble_predict(X)
        return result.ensemble_probabilities

    def ensemble_predict(self, X: np.ndarray) -> NeuroFuzzyResult:
        """Combine neural and fuzzy predictions into a confidence-weighted ensemble."""

        feature_frame = self._to_feature_frame(X)
        neural_probabilities = self.neural_network.predict_proba(feature_frame.to_numpy())
        fuzzy_results = [self.fuzzy_inference_engine.infer(self._row_to_input_dict(row)) for _, row in feature_frame.iterrows()]
        fuzzy_probabilities = np.vstack([self._fuzzy_result_to_probabilities(result) for result in fuzzy_results])

        neural_confidence = self._compute_neural_confidence(neural_probabilities)
        fuzzy_confidence = np.asarray(
            [
                result.confidence_scores.get(self.fuzzy_output_variable, 0.0)
                for result in fuzzy_results
            ],
            dtype=float,
        )
        combined_confidence = self._combine_confidences(neural_confidence, fuzzy_confidence)

        ensemble_probabilities = self._combine_predictions(
            neural_probabilities=neural_probabilities,
            fuzzy_probabilities=fuzzy_probabilities,
            neural_confidence=neural_confidence,
            fuzzy_confidence=fuzzy_confidence,
        )
        ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)

        return NeuroFuzzyResult(
            neural_probabilities=neural_probabilities,
            fuzzy_probabilities=fuzzy_probabilities,
            ensemble_probabilities=ensemble_probabilities,
            ensemble_predictions=ensemble_predictions,
            neural_confidence=neural_confidence,
            fuzzy_confidence=fuzzy_confidence,
            combined_confidence=combined_confidence,
            fuzzy_outputs=fuzzy_results,
        )

    def confidence_weighting(
        self,
        neural_probabilities: np.ndarray,
        fuzzy_probabilities: np.ndarray,
        neural_confidence: np.ndarray,
        fuzzy_confidence: np.ndarray,
    ) -> np.ndarray:
        """Apply confidence-aware weighting to neural and fuzzy probabilities."""

        neural_probabilities = np.asarray(neural_probabilities, dtype=float)
        fuzzy_probabilities = np.asarray(fuzzy_probabilities, dtype=float)
        neural_confidence = np.asarray(neural_confidence, dtype=float).reshape(-1, 1)
        fuzzy_confidence = np.asarray(fuzzy_confidence, dtype=float).reshape(-1, 1)

        neural_weights = self.neural_weight * np.clip(neural_confidence, 0.0, 1.0)
        fuzzy_weights = self.fuzzy_weight * np.clip(fuzzy_confidence, 0.0, 1.0)
        total_weights = neural_weights + fuzzy_weights
        total_weights = np.where(total_weights == 0.0, 1.0, total_weights)

        combined = (neural_probabilities * neural_weights + fuzzy_probabilities * fuzzy_weights) / total_weights
        combined_sum = np.sum(combined, axis=1, keepdims=True)
        combined_sum = np.where(combined_sum == 0.0, 1.0, combined_sum)
        return combined / combined_sum

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the hybrid model on labeled data."""

        labels = self._prepare_labels(y)
        result = self.ensemble_predict(X)
        accuracy = float(np.mean(result.ensemble_predictions == labels))
        return {
            "accuracy": accuracy,
            "average_neural_confidence": float(np.mean(result.neural_confidence)),
            "average_fuzzy_confidence": float(np.mean(result.fuzzy_confidence)),
            "average_combined_confidence": float(np.mean(result.combined_confidence)),
        }

    def _combine_predictions(
        self,
        *,
        neural_probabilities: np.ndarray,
        fuzzy_probabilities: np.ndarray,
        neural_confidence: np.ndarray,
        fuzzy_confidence: np.ndarray,
    ) -> np.ndarray:
        """Combine neural and fuzzy outputs into ensemble probabilities."""

        return self.confidence_weighting(
            neural_probabilities=neural_probabilities,
            fuzzy_probabilities=fuzzy_probabilities,
            neural_confidence=neural_confidence,
            fuzzy_confidence=fuzzy_confidence,
        )

    def _compute_neural_confidence(self, probabilities: np.ndarray) -> np.ndarray:
        """Estimate neural confidence from class-probability separation."""

        sorted_probabilities = np.sort(probabilities, axis=1)[:, ::-1]
        top_scores = sorted_probabilities[:, 0]
        second_scores = sorted_probabilities[:, 1] if sorted_probabilities.shape[1] > 1 else np.zeros_like(top_scores)
        confidence = 0.7 * top_scores + 0.3 * (top_scores - second_scores)
        return np.clip(confidence, 0.0, 1.0)

    def _combine_confidences(
        self,
        neural_confidence: np.ndarray,
        fuzzy_confidence: np.ndarray,
    ) -> np.ndarray:
        """Combine per-sample neural and fuzzy confidence scores."""

        combined = self.neural_weight * neural_confidence + self.fuzzy_weight * fuzzy_confidence
        return np.clip(combined, 0.0, 1.0)

    def _fuzzy_result_to_probabilities(self, result: InferenceResult) -> np.ndarray:
        """Convert fuzzy output activations into class probabilities."""

        if self.fuzzy_output_variable not in result.aggregated_outputs:
            if self.fuzzy_score_variable in result.crisp_outputs:
                return self._score_to_probabilities(result.crisp_outputs[self.fuzzy_score_variable])
            return np.full(len(self.class_labels), 1.0 / len(self.class_labels), dtype=float)

        term_strengths = result.aggregated_outputs[self.fuzzy_output_variable]
        probabilities = np.zeros(len(self.class_labels), dtype=float)

        for term_name, strength in term_strengths.items():
            label_index = self.class_to_index.get(term_name.lower())
            if label_index is not None:
                probabilities[label_index] = max(probabilities[label_index], float(strength))

        total = float(np.sum(probabilities))
        if total <= 0.0:
            if self.fuzzy_score_variable in result.crisp_outputs:
                return self._score_to_probabilities(result.crisp_outputs[self.fuzzy_score_variable])
            return np.full(len(self.class_labels), 1.0 / len(self.class_labels), dtype=float)
        return probabilities / total

    def _score_to_probabilities(self, score: float) -> np.ndarray:
        """Map a crisp fuzzy stress score to class probabilities."""

        centers = np.linspace(2.0, 8.0, len(self.class_labels))
        scale = 1.5
        distances = np.exp(-0.5 * np.square((float(score) - centers) / scale))
        total = float(np.sum(distances))
        if total <= 0.0:
            return np.full(len(self.class_labels), 1.0 / len(self.class_labels), dtype=float)
        return distances / total

    def _map_fuzzy_result_to_label_index(self, result: InferenceResult) -> int:
        """Convert fuzzy inference output to a predicted class index."""

        probabilities = self._fuzzy_result_to_probabilities(result)
        return int(np.argmax(probabilities))

    def _to_feature_frame(self, X: np.ndarray) -> pd.DataFrame:
        """Convert feature inputs to a validated DataFrame."""

        if isinstance(X, pd.DataFrame):
            missing_columns = [name for name in self.feature_names if name not in X.columns]
            if missing_columns:
                raise ValueError(f"Feature DataFrame is missing required columns: {missing_columns}")
            return X[self.feature_names].copy()

        array = np.asarray(X, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("Input features must be a 2D array or DataFrame.")
        if array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} feature columns, received {array.shape[1]}."
            )
        return pd.DataFrame(array, columns=self.feature_names)

    def _row_to_input_dict(self, row: pd.Series) -> Dict[str, float]:
        """Convert a feature row into fuzzy-engine input mapping."""

        return {
            feature_name.strip().lower(): float(row[feature_name])
            for feature_name in self.feature_names
        }

    def _prepare_labels(self, y: np.ndarray) -> np.ndarray:
        """Prepare integer class labels."""

        labels = np.asarray(y)
        if labels.ndim > 1:
            labels = labels.reshape(-1)
        return labels.astype(int)


__all__ = [
    "NeuroFuzzyResult",
    "NeuroFuzzySystem",
]
