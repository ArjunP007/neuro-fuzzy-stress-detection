"""Fuzzy inference engine with fuzzification, aggregation, defuzzification, and confidence scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np

from src.fuzzy_logic.rules import FuzzyRuleBase, LinguisticVariable


class FuzzyInferenceError(Exception):
    """Raised when fuzzy inference cannot be completed."""


@dataclass
class InferenceResult:
    """Container for fuzzy inference outputs."""

    crisp_outputs: Dict[str, float]
    aggregated_outputs: Dict[str, Dict[str, float]]
    fuzzified_inputs: Dict[str, Dict[str, float]]
    confidence_scores: Dict[str, float]
    method: str


class FuzzyInferenceEngine:
    """End-to-end fuzzy inference engine for Mamdani-style reasoning."""

    def __init__(
        self,
        rule_base: FuzzyRuleBase,
        *,
        output_universes: Optional[Dict[str, np.ndarray]] = None,
        default_resolution: int = 500,
    ) -> None:
        """Initialize the fuzzy inference engine."""

        if default_resolution < 50:
            raise FuzzyInferenceError("default_resolution must be at least 50.")

        self.rule_base = rule_base
        self.output_universes = output_universes or {}
        self.default_resolution = int(default_resolution)

    def fuzzify(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Convert crisp inputs into fuzzy membership degrees."""

        fuzzified_inputs: Dict[str, Dict[str, float]] = {}
        for variable_name, crisp_value in inputs.items():
            normalized_name = variable_name.strip().lower()
            if normalized_name not in self.rule_base.variables:
                continue

            variable = self.rule_base.variables[normalized_name]
            fuzzified_inputs[normalized_name] = {}
            for term_name in variable.membership_functions:
                fuzzified_inputs[normalized_name][term_name] = variable.evaluate(term_name, crisp_value)

        return fuzzified_inputs

    def aggregate_rules(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Evaluate and aggregate rule outputs."""

        return self.rule_base.evaluate(inputs)

    def defuzzify(
        self,
        aggregated_outputs: Dict[str, Dict[str, float]],
        *,
        method: str = "centroid",
    ) -> Dict[str, float]:
        """Defuzzify aggregated fuzzy outputs into crisp values."""

        normalized_method = method.strip().lower()
        if normalized_method not in {"centroid", "bisector"}:
            raise FuzzyInferenceError("Defuzzification method must be 'centroid' or 'bisector'.")

        crisp_outputs: Dict[str, float] = {}
        for variable_name, activated_terms in aggregated_outputs.items():
            if variable_name not in self.rule_base.variables:
                raise FuzzyInferenceError(
                    f"Unknown output variable '{variable_name}' during defuzzification."
                )

            universe = self._get_universe(variable_name)
            aggregated_membership = self._build_aggregated_membership(
                variable_name=variable_name,
                activated_terms=activated_terms,
                universe=universe,
            )

            if normalized_method == "centroid":
                crisp_outputs[variable_name] = self._centroid(universe, aggregated_membership)
            else:
                crisp_outputs[variable_name] = self._bisector(universe, aggregated_membership)

        return crisp_outputs

    def compute_confidence(
        self,
        aggregated_outputs: Dict[str, Dict[str, float]],
        crisp_outputs: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute confidence scores for inferred outputs."""

        confidence_scores: Dict[str, float] = {}
        for variable_name, term_strengths in aggregated_outputs.items():
            strengths = np.asarray(list(term_strengths.values()), dtype=float)
            if strengths.size == 0:
                confidence_scores[variable_name] = 0.0
                continue

            sorted_strengths = np.sort(strengths)[::-1]
            peak_strength = float(sorted_strengths[0])
            separation = float(sorted_strengths[0] - sorted_strengths[1]) if sorted_strengths.size > 1 else peak_strength
            consistency = float(np.mean(strengths))
            crisp_bonus = 1.0 if variable_name in crisp_outputs else 0.0
            confidence = 0.5 * peak_strength + 0.3 * separation + 0.15 * consistency + 0.05 * crisp_bonus
            confidence_scores[variable_name] = float(np.clip(confidence, 0.0, 1.0))

        return confidence_scores

    def infer(
        self,
        inputs: Dict[str, float],
        *,
        defuzz_method: str = "centroid",
    ) -> InferenceResult:
        """Run the full fuzzy inference pipeline."""

        fuzzified_inputs = self.fuzzify(inputs)
        aggregated_outputs = self.aggregate_rules(inputs)
        crisp_outputs = self.defuzzify(aggregated_outputs, method=defuzz_method)
        confidence_scores = self.compute_confidence(aggregated_outputs, crisp_outputs)

        return InferenceResult(
            crisp_outputs=crisp_outputs,
            aggregated_outputs=aggregated_outputs,
            fuzzified_inputs=fuzzified_inputs,
            confidence_scores=confidence_scores,
            method=defuzz_method,
        )

    def _get_universe(self, variable_name: str) -> np.ndarray:
        """Return the output universe for a linguistic variable."""

        if variable_name in self.output_universes:
            return np.asarray(self.output_universes[variable_name], dtype=float)

        variable = self.rule_base.variables[variable_name]
        lower_bound, upper_bound = self._infer_universe_bounds(variable)
        return np.linspace(lower_bound, upper_bound, self.default_resolution)

    def _build_aggregated_membership(
        self,
        *,
        variable_name: str,
        activated_terms: Dict[str, float],
        universe: np.ndarray,
    ) -> np.ndarray:
        """Build the aggregated membership function for an output variable."""

        variable = self.rule_base.variables[variable_name]
        aggregated = np.zeros_like(universe, dtype=float)

        for term_name, strength in activated_terms.items():
            if term_name not in variable.membership_functions:
                raise FuzzyInferenceError(
                    f"Unknown term '{term_name}' for output variable '{variable_name}'."
                )
            term_membership = variable.membership_functions[term_name].compute(universe)
            clipped_membership = np.minimum(term_membership, float(strength))
            aggregated = np.maximum(aggregated, clipped_membership)

        return aggregated

    @staticmethod
    def _centroid(universe: np.ndarray, membership: np.ndarray) -> float:
        """Compute centroid defuzzification."""

        denominator = np.sum(membership)
        if denominator <= 0.0:
            return float(np.mean(universe))
        return float(np.sum(universe * membership) / denominator)

    @staticmethod
    def _bisector(universe: np.ndarray, membership: np.ndarray) -> float:
        """Compute bisector defuzzification."""

        area = np.trapezoid(membership, universe)
        if area <= 0.0:
            return float(np.mean(universe))

        cumulative = np.cumsum((membership[:-1] + membership[1:]) * np.diff(universe) / 2.0)
        half_area = area / 2.0
        index = int(np.searchsorted(cumulative, half_area, side="left"))
        if index >= universe.size:
            index = universe.size - 1
        return float(universe[index])

    @staticmethod
    def _infer_universe_bounds(variable: LinguisticVariable) -> tuple[float, float]:
        """Infer a reasonable universe range from membership parameters."""

        lower_candidates: list[float] = []
        upper_candidates: list[float] = []

        for membership_function in variable.membership_functions.values():
            parameters = membership_function.get_parameters()
            lower_candidates.extend(parameters.values())
            upper_candidates.extend(parameters.values())

        if not lower_candidates or not upper_candidates:
            raise FuzzyInferenceError(
                f"Cannot infer universe bounds for variable '{variable.name}'."
            )

        lower_bound = min(lower_candidates)
        upper_bound = max(upper_candidates)
        if lower_bound == upper_bound:
            lower_bound -= 1.0
            upper_bound += 1.0
        return float(lower_bound), float(upper_bound)


__all__ = [
    "FuzzyInferenceEngine",
    "FuzzyInferenceError",
    "InferenceResult",
]
