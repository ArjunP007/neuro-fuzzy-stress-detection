"""Fuzzy rule base system with parsing, evaluation, and optimization support."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np

from src.fuzzy_logic.membership_functions import BaseMembershipFunction


class FuzzyRuleError(Exception):
    """Raised when fuzzy rule parsing or evaluation fails."""


@dataclass
class LinguisticVariable:
    """Represent a fuzzy linguistic variable with named membership functions."""

    name: str
    membership_functions: Dict[str, BaseMembershipFunction] = field(default_factory=dict)

    def add_term(self, term: str, membership_function: BaseMembershipFunction) -> None:
        """Register a new linguistic term."""

        normalized_term = term.strip().lower()
        if normalized_term in self.membership_functions:
            raise FuzzyRuleError(
                f"Term '{normalized_term}' already exists for variable '{self.name}'."
            )
        self.membership_functions[normalized_term] = membership_function

    def evaluate(self, term: str, value: float) -> float:
        """Evaluate the membership degree of a term for a crisp value."""

        normalized_term = term.strip().lower()
        if normalized_term not in self.membership_functions:
            raise FuzzyRuleError(
                f"Unknown term '{normalized_term}' for variable '{self.name}'."
            )
        membership = self.membership_functions[normalized_term].compute([value])[0]
        return float(membership)

    def tune_term(
        self,
        term: str,
        parameter_updates: Dict[str, float],
    ) -> None:
        """Tune the parameters of an existing linguistic term."""

        normalized_term = term.strip().lower()
        if normalized_term not in self.membership_functions:
            raise FuzzyRuleError(
                f"Unknown term '{normalized_term}' for variable '{self.name}'."
            )
        self.membership_functions[normalized_term].tune(parameter_updates)


@dataclass
class RuleCondition:
    """Represent a single antecedent clause."""

    variable: str
    term: str
    operator_to_next: Optional[str] = None


@dataclass
class RuleConsequent:
    """Represent the consequent part of a fuzzy rule."""

    variable: str
    term: str


@dataclass
class FuzzyRule:
    """Represent a weighted fuzzy IF-THEN rule."""

    rule_id: str
    conditions: list[RuleCondition]
    consequent: RuleConsequent
    weight: float = 1.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate a fuzzy rule definition."""

        if not self.rule_id.strip():
            raise FuzzyRuleError("Rule identifier cannot be empty.")
        if not self.conditions:
            raise FuzzyRuleError(f"Rule '{self.rule_id}' must contain at least one condition.")
        if not 0.0 <= self.weight <= 1.0:
            raise FuzzyRuleError(f"Rule '{self.rule_id}' weight must satisfy 0.0 <= weight <= 1.0.")


class RuleParser:
    """Parse text-based fuzzy IF-THEN rules into structured rule objects."""

    RULE_PATTERN = re.compile(
        r"^\s*IF\s+(?P<antecedent>.+?)\s+THEN\s+(?P<consequent>.+?)\s*$",
        flags=re.IGNORECASE,
    )
    CLAUSE_PATTERN = re.compile(
        r"(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\s+IS\s+(?P<term>[A-Za-z_][A-Za-z0-9_]*)",
        flags=re.IGNORECASE,
    )

    def parse(self, rule_id: str, rule_expression: str, weight: float = 1.0) -> FuzzyRule:
        """Parse a rule expression into a structured fuzzy rule."""

        matched = self.RULE_PATTERN.match(rule_expression)
        if matched is None:
            raise FuzzyRuleError(
                f"Rule '{rule_id}' could not be parsed. Expected format: IF ... THEN ..."
            )

        antecedent_text = matched.group("antecedent")
        consequent_text = matched.group("consequent")
        conditions = self._parse_antecedent(antecedent_text)
        consequent = self._parse_consequent(consequent_text)

        rule = FuzzyRule(
            rule_id=rule_id,
            conditions=conditions,
            consequent=consequent,
            weight=float(weight),
        )
        rule.validate()
        return rule

    def _parse_antecedent(self, antecedent_text: str) -> list[RuleCondition]:
        """Parse the antecedent clauses and logical connectors."""

        tokens = re.split(r"\s+(AND|OR)\s+", antecedent_text, flags=re.IGNORECASE)
        if not tokens:
            raise FuzzyRuleError("Antecedent cannot be empty.")

        conditions: list[RuleCondition] = []
        for index in range(0, len(tokens), 2):
            clause = tokens[index].strip()
            matched_clause = self.CLAUSE_PATTERN.fullmatch(clause)
            if matched_clause is None:
                raise FuzzyRuleError(f"Invalid antecedent clause: '{clause}'.")

            operator_to_next = None
            if index + 1 < len(tokens):
                operator_to_next = tokens[index + 1].strip().upper()

            conditions.append(
                RuleCondition(
                    variable=matched_clause.group("variable").strip().lower(),
                    term=matched_clause.group("term").strip().lower(),
                    operator_to_next=operator_to_next,
                )
            )
        return conditions

    def _parse_consequent(self, consequent_text: str) -> RuleConsequent:
        """Parse the consequent clause."""

        matched_clause = self.CLAUSE_PATTERN.fullmatch(consequent_text.strip())
        if matched_clause is None:
            raise FuzzyRuleError(f"Invalid consequent clause: '{consequent_text}'.")

        return RuleConsequent(
            variable=matched_clause.group("variable").strip().lower(),
            term=matched_clause.group("term").strip().lower(),
        )


class RuleEvaluator:
    """Evaluate fuzzy rules against crisp input values."""

    def __init__(
        self,
        *,
        conjunction_operator: str = "min",
        disjunction_operator: str = "max",
    ) -> None:
        """Initialize rule evaluation operators."""

        if conjunction_operator not in {"min", "product"}:
            raise FuzzyRuleError("conjunction_operator must be 'min' or 'product'.")
        if disjunction_operator not in {"max", "probabilistic_or"}:
            raise FuzzyRuleError("disjunction_operator must be 'max' or 'probabilistic_or'.")

        self.conjunction_operator = conjunction_operator
        self.disjunction_operator = disjunction_operator

    def evaluate_rule(
        self,
        rule: FuzzyRule,
        inputs: Dict[str, float],
        variables: Dict[str, LinguisticVariable],
    ) -> Dict[str, Any]:
        """Evaluate a single fuzzy rule and return firing metadata."""

        rule.validate()
        if not rule.enabled:
            return {
                "rule_id": rule.rule_id,
                "firing_strength": 0.0,
                "consequent_variable": rule.consequent.variable,
                "consequent_term": rule.consequent.term,
                "weight": rule.weight,
            }

        membership_values: list[float] = []
        operators: list[str] = []

        for condition in rule.conditions:
            variable_name = condition.variable
            if variable_name not in inputs:
                raise FuzzyRuleError(
                    f"Missing crisp input for variable '{variable_name}' while evaluating rule '{rule.rule_id}'."
                )
            if variable_name not in variables:
                raise FuzzyRuleError(
                    f"Unknown linguistic variable '{variable_name}' in rule '{rule.rule_id}'."
                )

            membership_degree = variables[variable_name].evaluate(condition.term, inputs[variable_name])
            membership_values.append(membership_degree)
            if condition.operator_to_next is not None:
                operators.append(condition.operator_to_next)

        aggregated_strength = self._aggregate_memberships(membership_values, operators)
        weighted_strength = float(np.clip(aggregated_strength * rule.weight, 0.0, 1.0))

        return {
            "rule_id": rule.rule_id,
            "firing_strength": weighted_strength,
            "raw_strength": aggregated_strength,
            "consequent_variable": rule.consequent.variable,
            "consequent_term": rule.consequent.term,
            "weight": rule.weight,
        }

    def _aggregate_memberships(
        self,
        memberships: list[float],
        operators: list[str],
    ) -> float:
        """Aggregate antecedent memberships according to logical operators."""

        if not memberships:
            raise FuzzyRuleError("At least one membership value is required for aggregation.")

        strength = memberships[0]
        for index, operator in enumerate(operators, start=1):
            current_value = memberships[index]
            if operator == "AND":
                if self.conjunction_operator == "min":
                    strength = min(strength, current_value)
                else:
                    strength = strength * current_value
            elif operator == "OR":
                if self.disjunction_operator == "max":
                    strength = max(strength, current_value)
                else:
                    strength = strength + current_value - (strength * current_value)
            else:
                raise FuzzyRuleError(f"Unsupported logical operator '{operator}'.")

        return float(np.clip(strength, 0.0, 1.0))


class ConflictResolver:
    """Resolve conflicts among multiple activated rules."""

    def __init__(self, strategy: str = "max_activation") -> None:
        """Initialize conflict resolution strategy."""

        if strategy not in {"max_activation", "weighted_average", "sum"}:
            raise FuzzyRuleError(
                "Conflict resolution strategy must be one of "
                "{'max_activation', 'weighted_average', 'sum'}."
            )
        self.strategy = strategy

    def resolve(self, evaluations: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Resolve conflicting consequents into aggregated output activations."""

        grouped: Dict[str, Dict[str, list[float]]] = {}
        for evaluation in evaluations:
            variable = evaluation["consequent_variable"]
            term = evaluation["consequent_term"]
            grouped.setdefault(variable, {}).setdefault(term, []).append(evaluation["firing_strength"])

        resolved: Dict[str, Dict[str, float]] = {}
        for variable, term_mapping in grouped.items():
            resolved[variable] = {}
            for term, strengths in term_mapping.items():
                if self.strategy == "max_activation":
                    resolved_value = float(np.max(strengths))
                elif self.strategy == "weighted_average":
                    weights = np.arange(1, len(strengths) + 1, dtype=float)
                    resolved_value = float(np.average(np.asarray(strengths, dtype=float), weights=weights))
                else:
                    resolved_value = float(np.clip(np.sum(strengths), 0.0, 1.0))
                resolved[variable][term] = float(np.clip(resolved_value, 0.0, 1.0))

        return resolved


class RuleOptimizer:
    """Optimize rule weights using simple performance feedback."""

    def __init__(self, learning_rate: float = 0.05) -> None:
        """Initialize optimizer hyperparameters."""

        if learning_rate <= 0.0:
            raise FuzzyRuleError("learning_rate must be strictly positive.")
        self.learning_rate = float(learning_rate)

    def optimize(
        self,
        rules: Dict[str, FuzzyRule],
        rule_performance: Dict[str, float],
    ) -> Dict[str, FuzzyRule]:
        """Adjust rule weights based on observed performance feedback."""

        optimized_rules: Dict[str, FuzzyRule] = {}
        for rule_id, rule in rules.items():
            performance = float(rule_performance.get(rule_id, 0.0))
            updated_rule = FuzzyRule(
                rule_id=rule.rule_id,
                conditions=list(rule.conditions),
                consequent=rule.consequent,
                weight=float(np.clip(rule.weight + self.learning_rate * performance, 0.0, 1.0)),
                enabled=rule.enabled,
                metadata=dict(rule.metadata),
            )
            optimized_rules[rule_id] = updated_rule
        return optimized_rules


class FuzzyRuleBase:
    """Manage linguistic variables, fuzzy rules, parsing, and evaluation."""

    def __init__(
        self,
        *,
        conjunction_operator: str = "min",
        disjunction_operator: str = "max",
        conflict_resolution: str = "max_activation",
    ) -> None:
        """Initialize the fuzzy rule base system."""

        self.variables: Dict[str, LinguisticVariable] = {}
        self.rules: Dict[str, FuzzyRule] = {}
        self.parser = RuleParser()
        self.evaluator = RuleEvaluator(
            conjunction_operator=conjunction_operator,
            disjunction_operator=disjunction_operator,
        )
        self.conflict_resolver = ConflictResolver(strategy=conflict_resolution)
        self.optimizer = RuleOptimizer()

    def add_linguistic_variable(self, variable: LinguisticVariable) -> None:
        """Register a linguistic variable in the rule base."""

        normalized_name = variable.name.strip().lower()
        if normalized_name in self.variables:
            raise FuzzyRuleError(f"Linguistic variable '{normalized_name}' already exists.")
        self.variables[normalized_name] = variable

    def add_rule(
        self,
        rule_id: str,
        rule_expression: str,
        *,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FuzzyRule:
        """Parse and add a new fuzzy rule dynamically."""

        normalized_rule_id = rule_id.strip()
        if not normalized_rule_id:
            raise FuzzyRuleError("rule_id cannot be empty.")
        if normalized_rule_id in self.rules:
            raise FuzzyRuleError(f"Rule '{normalized_rule_id}' already exists.")

        rule = self.parser.parse(normalized_rule_id, rule_expression, weight=weight)
        rule.metadata = dict(metadata or {})
        self._validate_rule_variables(rule)
        self.rules[normalized_rule_id] = rule
        return rule

    def remove_rule(self, rule_id: str) -> None:
        """Remove a fuzzy rule from the rule base."""

        if rule_id not in self.rules:
            raise FuzzyRuleError(f"Rule '{rule_id}' does not exist.")
        del self.rules[rule_id]

    def evaluate(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Evaluate all rules for crisp inputs and resolve consequent conflicts."""

        evaluations = [
            self.evaluator.evaluate_rule(rule, inputs, self.variables)
            for rule in self.rules.values()
        ]
        return self.conflict_resolver.resolve(evaluations)

    def evaluate_rules(self, inputs: Dict[str, float]) -> list[Dict[str, Any]]:
        """Return raw per-rule evaluation outputs."""

        return [
            self.evaluator.evaluate_rule(rule, inputs, self.variables)
            for rule in self.rules.values()
        ]

    def optimize_rules(self, rule_performance: Dict[str, float]) -> None:
        """Optimize rule weights using external performance feedback."""

        self.rules = self.optimizer.optimize(self.rules, rule_performance)

    def set_rule_weight(self, rule_id: str, weight: float) -> None:
        """Update the weight of an existing rule."""

        if rule_id not in self.rules:
            raise FuzzyRuleError(f"Rule '{rule_id}' does not exist.")
        if not 0.0 <= weight <= 1.0:
            raise FuzzyRuleError("Rule weight must satisfy 0.0 <= weight <= 1.0.")
        self.rules[rule_id].weight = float(weight)

    def enable_rule(self, rule_id: str, enabled: bool = True) -> None:
        """Enable or disable a rule."""

        if rule_id not in self.rules:
            raise FuzzyRuleError(f"Rule '{rule_id}' does not exist.")
        self.rules[rule_id].enabled = bool(enabled)

    def _validate_rule_variables(self, rule: FuzzyRule) -> None:
        """Ensure referenced variables and terms exist in the linguistic model."""

        for condition in rule.conditions:
            if condition.variable not in self.variables:
                raise FuzzyRuleError(
                    f"Rule '{rule.rule_id}' references unknown variable '{condition.variable}'."
                )
            if condition.term not in self.variables[condition.variable].membership_functions:
                raise FuzzyRuleError(
                    f"Rule '{rule.rule_id}' references unknown term '{condition.term}' "
                    f"for variable '{condition.variable}'."
                )

        consequent_variable = rule.consequent.variable
        consequent_term = rule.consequent.term
        if consequent_variable not in self.variables:
            raise FuzzyRuleError(
                f"Rule '{rule.rule_id}' references unknown consequent variable '{consequent_variable}'."
            )
        if consequent_term not in self.variables[consequent_variable].membership_functions:
            raise FuzzyRuleError(
                f"Rule '{rule.rule_id}' references unknown consequent term '{consequent_term}' "
                f"for variable '{consequent_variable}'."
            )


__all__ = [
    "ConflictResolver",
    "FuzzyRule",
    "FuzzyRuleBase",
    "FuzzyRuleError",
    "LinguisticVariable",
    "RuleCondition",
    "RuleConsequent",
    "RuleEvaluator",
    "RuleOptimizer",
    "RuleParser",
]
