"""Reusable data validation utilities for the stress detection project."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails in strict mode."""


@dataclass(frozen=True)
class RangeRule:
    """Define numeric range constraints for a single feature."""

    minimum: Optional[float] = None
    maximum: Optional[float] = None
    inclusive_min: bool = True
    inclusive_max: bool = True


@dataclass(frozen=True)
class DistributionRule:
    """Define expected statistical properties for a feature."""

    min_mean: Optional[float] = None
    max_mean: Optional[float] = None
    min_std: Optional[float] = None
    max_std: Optional[float] = None
    allowed_skew_abs: Optional[float] = None
    quantile_bounds: Optional[Dict[float, tuple[float, float]]] = None


@dataclass(frozen=True)
class AnomalyRule:
    """Define anomaly detection sensitivity for a feature."""

    zscore_threshold: float = 3.0
    max_anomaly_ratio: float = 0.05


@dataclass
class ValidationIssue:
    """Represent a single validation problem."""

    rule: str
    message: str
    column: Optional[str] = None
    severity: str = "error"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Aggregate validation outcome details."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Append a validation issue and mark the report invalid."""

        self.issues.append(issue)
        self.is_valid = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the report into a dictionary."""

        return {
            "is_valid": self.is_valid,
            "issues": [
                {
                    "rule": issue.rule,
                    "message": issue.message,
                    "column": issue.column,
                    "severity": issue.severity,
                    "details": issue.details,
                }
                for issue in self.issues
            ],
            "summary": self.summary,
        }

    def raise_if_invalid(self) -> None:
        """Raise a single error if the report contains failures."""

        if self.is_valid:
            return

        joined_messages = "; ".join(
            f"{issue.rule}: {issue.message}" if issue.column is None else f"{issue.rule}[{issue.column}]: {issue.message}"
            for issue in self.issues
        )
        raise ValidationError(joined_messages)


class DataValidator:
    """Reusable framework for validating tabular datasets with configurable rules."""

    DEFAULT_SCHEMA: Dict[str, str] = {
        "sleep_hours": "float",
        "work_hours": "float",
        "screen_time": "float",
        "physical_activity_hours": "float",
        "mental_fatigue_score": "float",
        "heart_rate": "float",
        "caffeine_intake": "float",
        "social_interaction_hours": "float",
        "work_pressure_score": "float",
        "stress_score": "float",
        "stress_level": "object",
    }

    DEFAULT_RANGE_RULES: Dict[str, RangeRule] = {
        "sleep_hours": RangeRule(0.0, 12.0),
        "work_hours": RangeRule(0.0, 16.0),
        "screen_time": RangeRule(0.0, 18.0),
        "physical_activity_hours": RangeRule(0.0, 8.0),
        "mental_fatigue_score": RangeRule(0.0, 10.0),
        "heart_rate": RangeRule(40.0, 160.0),
        "caffeine_intake": RangeRule(0.0, 12.0),
        "social_interaction_hours": RangeRule(0.0, 12.0),
        "work_pressure_score": RangeRule(0.0, 10.0),
        "stress_score": RangeRule(0.0, 10.0),
    }

    DEFAULT_DISTRIBUTION_RULES: Dict[str, DistributionRule] = {
        "sleep_hours": DistributionRule(min_mean=4.0, max_mean=9.0, min_std=0.4, max_std=2.5),
        "work_hours": DistributionRule(min_mean=4.0, max_mean=11.5, min_std=0.5, max_std=3.5),
        "screen_time": DistributionRule(min_mean=1.0, max_mean=12.0, min_std=0.5, max_std=4.0),
        "physical_activity_hours": DistributionRule(min_mean=0.2, max_mean=4.0, min_std=0.2, max_std=2.0),
        "mental_fatigue_score": DistributionRule(min_mean=1.0, max_mean=9.0, min_std=0.5, max_std=3.5),
        "heart_rate": DistributionRule(min_mean=55.0, max_mean=105.0, min_std=2.0, max_std=25.0),
        "caffeine_intake": DistributionRule(min_mean=0.0, max_mean=6.0, min_std=0.2, max_std=3.0),
        "social_interaction_hours": DistributionRule(min_mean=0.2, max_mean=6.0, min_std=0.2, max_std=3.5),
        "work_pressure_score": DistributionRule(min_mean=1.0, max_mean=9.0, min_std=0.5, max_std=3.5),
        "stress_score": DistributionRule(min_mean=1.0, max_mean=9.5, min_std=0.5, max_std=4.0),
    }

    DEFAULT_ANOMALY_RULES: Dict[str, AnomalyRule] = {
        "sleep_hours": AnomalyRule(zscore_threshold=3.0, max_anomaly_ratio=0.03),
        "work_hours": AnomalyRule(zscore_threshold=3.0, max_anomaly_ratio=0.03),
        "screen_time": AnomalyRule(zscore_threshold=3.2, max_anomaly_ratio=0.04),
        "physical_activity_hours": AnomalyRule(zscore_threshold=3.0, max_anomaly_ratio=0.04),
        "mental_fatigue_score": AnomalyRule(zscore_threshold=3.0, max_anomaly_ratio=0.04),
        "heart_rate": AnomalyRule(zscore_threshold=3.2, max_anomaly_ratio=0.03),
        "caffeine_intake": AnomalyRule(zscore_threshold=3.0, max_anomaly_ratio=0.05),
        "social_interaction_hours": AnomalyRule(zscore_threshold=3.0, max_anomaly_ratio=0.05),
        "work_pressure_score": AnomalyRule(zscore_threshold=3.0, max_anomaly_ratio=0.04),
        "stress_score": AnomalyRule(zscore_threshold=3.0, max_anomaly_ratio=0.04),
    }

    def __init__(
        self,
        *,
        schema: Optional[Mapping[str, str]] = None,
        required_columns: Optional[Sequence[str]] = None,
        dtype_rules: Optional[Mapping[str, str]] = None,
        range_rules: Optional[Mapping[str, RangeRule]] = None,
        distribution_rules: Optional[Mapping[str, DistributionRule]] = None,
        anomaly_rules: Optional[Mapping[str, AnomalyRule]] = None,
        allowed_null_ratio: float = 0.0,
        strict: bool = False,
    ) -> None:
        """Initialize the validator with configurable rule sets."""

        if not 0.0 <= allowed_null_ratio <= 1.0:
            raise ValueError("allowed_null_ratio must be within [0.0, 1.0].")

        self.schema = dict(schema or self.DEFAULT_SCHEMA)
        self.required_columns = list(required_columns or self.schema.keys())
        self.dtype_rules = dict(dtype_rules or self.schema)
        self.range_rules = dict(range_rules or self.DEFAULT_RANGE_RULES)
        self.distribution_rules = dict(distribution_rules or self.DEFAULT_DISTRIBUTION_RULES)
        self.anomaly_rules = dict(anomaly_rules or self.DEFAULT_ANOMALY_RULES)
        self.allowed_null_ratio = float(allowed_null_ratio)
        self.strict = bool(strict)

    def validate(
        self,
        dataset: pd.DataFrame,
        *,
        run_distribution_checks: bool = True,
        run_anomaly_detection: bool = True,
    ) -> ValidationReport:
        """Run the full validation pipeline on a dataset."""

        self._ensure_dataframe(dataset)
        report = ValidationReport(is_valid=True)

        self._merge_report(report, self.validate_schema(dataset))
        self._merge_report(report, self.validate_data_types(dataset))
        self._merge_report(report, self.check_nulls(dataset))
        self._merge_report(report, self.validate_ranges(dataset))

        if run_distribution_checks:
            self._merge_report(report, self.validate_feature_distribution(dataset))
        if run_anomaly_detection:
            self._merge_report(report, self.detect_anomalies(dataset))

        report.summary = self._build_summary(dataset, report)
        if self.strict:
            report.raise_if_invalid()

        LOGGER.info("Validation completed. valid=%s issues=%d", report.is_valid, len(report.issues))
        return report

    def validate_schema(self, dataset: pd.DataFrame) -> ValidationReport:
        """Validate required columns and schema completeness."""

        self._ensure_dataframe(dataset)
        report = ValidationReport(is_valid=True)

        dataset_columns = set(dataset.columns)
        required_columns = set(self.required_columns)
        missing_columns = sorted(required_columns - dataset_columns)
        unexpected_columns = sorted(dataset_columns - set(self.schema.keys()))

        if missing_columns:
            report.add_issue(
                ValidationIssue(
                    rule="schema_validation",
                    message=f"Missing required columns: {missing_columns}",
                    details={"missing_columns": missing_columns},
                )
            )

        if unexpected_columns:
            report.add_issue(
                ValidationIssue(
                    rule="schema_validation",
                    message=f"Unexpected columns detected: {unexpected_columns}",
                    details={"unexpected_columns": unexpected_columns},
                    severity="warning",
                )
            )

        return report

    def validate_data_types(self, dataset: pd.DataFrame) -> ValidationReport:
        """Validate column data types against configured expectations."""

        self._ensure_dataframe(dataset)
        report = ValidationReport(is_valid=True)

        for column, expected_type in self.dtype_rules.items():
            if column not in dataset.columns:
                continue

            actual_dtype = dataset[column].dtype
            if not self._dtype_matches(actual_dtype, expected_type):
                report.add_issue(
                    ValidationIssue(
                        rule="data_type_validation",
                        column=column,
                        message=(
                            f"Column '{column}' has dtype '{actual_dtype}' but expected type category '{expected_type}'."
                        ),
                        details={
                            "expected_type": expected_type,
                            "actual_dtype": str(actual_dtype),
                        },
                    )
                )

        return report

    def check_nulls(self, dataset: pd.DataFrame) -> ValidationReport:
        """Validate null ratios for configured columns."""

        self._ensure_dataframe(dataset)
        report = ValidationReport(is_valid=True)

        for column in self.required_columns:
            if column not in dataset.columns:
                continue

            null_count = int(dataset[column].isnull().sum())
            null_ratio = float(null_count / len(dataset)) if len(dataset) > 0 else 0.0
            if null_ratio > self.allowed_null_ratio:
                report.add_issue(
                    ValidationIssue(
                        rule="null_check",
                        column=column,
                        message=(
                            f"Column '{column}' contains {null_count} null values "
                            f"({null_ratio:.2%}), exceeding allowed ratio {self.allowed_null_ratio:.2%}."
                        ),
                        details={
                            "null_count": null_count,
                            "null_ratio": null_ratio,
                            "allowed_null_ratio": self.allowed_null_ratio,
                        },
                    )
                )

        return report

    def validate_ranges(self, dataset: pd.DataFrame) -> ValidationReport:
        """Validate numeric columns against configured range rules."""

        self._ensure_dataframe(dataset)
        report = ValidationReport(is_valid=True)

        for column, rule in self.range_rules.items():
            if column not in dataset.columns:
                continue
            if not pd.api.types.is_numeric_dtype(dataset[column]):
                continue

            series = dataset[column].dropna()
            if series.empty:
                continue

            if rule.minimum is not None:
                invalid_min = series < rule.minimum if rule.inclusive_min else series <= rule.minimum
                if bool(invalid_min.any()):
                    report.add_issue(
                        ValidationIssue(
                            rule="range_validation",
                            column=column,
                            message=(
                                f"Column '{column}' contains values below the minimum allowed bound "
                                f"{rule.minimum}."
                            ),
                            details={
                                "minimum": rule.minimum,
                                "observed_min": float(series.min()),
                                "inclusive_min": rule.inclusive_min,
                            },
                        )
                    )

            if rule.maximum is not None:
                invalid_max = series > rule.maximum if rule.inclusive_max else series >= rule.maximum
                if bool(invalid_max.any()):
                    report.add_issue(
                        ValidationIssue(
                            rule="range_validation",
                            column=column,
                            message=(
                                f"Column '{column}' contains values above the maximum allowed bound "
                                f"{rule.maximum}."
                            ),
                            details={
                                "maximum": rule.maximum,
                                "observed_max": float(series.max()),
                                "inclusive_max": rule.inclusive_max,
                            },
                        )
                    )

        return report

    def validate_feature_distribution(self, dataset: pd.DataFrame) -> ValidationReport:
        """Validate feature distributions against expected statistical properties."""

        self._ensure_dataframe(dataset)
        report = ValidationReport(is_valid=True)

        for column, rule in self.distribution_rules.items():
            if column not in dataset.columns:
                continue
            if not pd.api.types.is_numeric_dtype(dataset[column]):
                continue

            series = dataset[column].dropna().astype(float)
            if len(series) < 3:
                report.add_issue(
                    ValidationIssue(
                        rule="feature_distribution_validation",
                        column=column,
                        message=f"Column '{column}' has insufficient non-null values for distribution validation.",
                        severity="warning",
                    )
                )
                continue

            mean_value = float(series.mean())
            std_value = float(series.std(ddof=1))
            skew_value = float(series.skew())

            if rule.min_mean is not None and mean_value < rule.min_mean:
                report.add_issue(
                    ValidationIssue(
                        rule="feature_distribution_validation",
                        column=column,
                        message=f"Mean of '{column}' is {mean_value:.4f}, below expected minimum {rule.min_mean}.",
                        details={"mean": mean_value, "min_mean": rule.min_mean},
                    )
                )

            if rule.max_mean is not None and mean_value > rule.max_mean:
                report.add_issue(
                    ValidationIssue(
                        rule="feature_distribution_validation",
                        column=column,
                        message=f"Mean of '{column}' is {mean_value:.4f}, above expected maximum {rule.max_mean}.",
                        details={"mean": mean_value, "max_mean": rule.max_mean},
                    )
                )

            if rule.min_std is not None and std_value < rule.min_std:
                report.add_issue(
                    ValidationIssue(
                        rule="feature_distribution_validation",
                        column=column,
                        message=f"Standard deviation of '{column}' is {std_value:.4f}, below expected minimum {rule.min_std}.",
                        details={"std": std_value, "min_std": rule.min_std},
                    )
                )

            if rule.max_std is not None and std_value > rule.max_std:
                report.add_issue(
                    ValidationIssue(
                        rule="feature_distribution_validation",
                        column=column,
                        message=f"Standard deviation of '{column}' is {std_value:.4f}, above expected maximum {rule.max_std}.",
                        details={"std": std_value, "max_std": rule.max_std},
                    )
                )

            if rule.allowed_skew_abs is not None and abs(skew_value) > rule.allowed_skew_abs:
                report.add_issue(
                    ValidationIssue(
                        rule="feature_distribution_validation",
                        column=column,
                        message=(
                            f"Absolute skewness of '{column}' is {abs(skew_value):.4f}, "
                            f"exceeding allowed limit {rule.allowed_skew_abs}."
                        ),
                        details={"skew": skew_value, "allowed_skew_abs": rule.allowed_skew_abs},
                        severity="warning",
                    )
                )

            if rule.quantile_bounds:
                for quantile, bounds in rule.quantile_bounds.items():
                    observed = float(series.quantile(quantile))
                    lower, upper = bounds
                    if observed < lower or observed > upper:
                        report.add_issue(
                            ValidationIssue(
                                rule="feature_distribution_validation",
                                column=column,
                                message=(
                                    f"Quantile {quantile:.2f} of '{column}' is {observed:.4f}, "
                                    f"outside expected range [{lower}, {upper}]."
                                ),
                                details={
                                    "quantile": quantile,
                                    "observed": observed,
                                    "lower": lower,
                                    "upper": upper,
                                },
                            )
                        )

        return report

    def detect_anomalies(self, dataset: pd.DataFrame) -> ValidationReport:
        """Detect anomalous values using configurable z-score thresholds."""

        self._ensure_dataframe(dataset)
        report = ValidationReport(is_valid=True)

        for column, rule in self.anomaly_rules.items():
            if column not in dataset.columns:
                continue
            if not pd.api.types.is_numeric_dtype(dataset[column]):
                continue

            series = dataset[column].dropna().astype(float)
            if len(series) < 3:
                continue

            std_value = float(series.std(ddof=1))
            if std_value == 0.0:
                continue

            zscores = np.abs((series.to_numpy() - float(series.mean())) / std_value)
            anomaly_mask = zscores > rule.zscore_threshold
            anomaly_count = int(np.sum(anomaly_mask))
            anomaly_ratio = float(anomaly_count / len(series))

            if anomaly_ratio > rule.max_anomaly_ratio:
                report.add_issue(
                    ValidationIssue(
                        rule="anomaly_detection",
                        column=column,
                        message=(
                            f"Column '{column}' has anomaly ratio {anomaly_ratio:.2%}, "
                            f"exceeding allowed limit {rule.max_anomaly_ratio:.2%}."
                        ),
                        details={
                            "anomaly_count": anomaly_count,
                            "sample_size": int(len(series)),
                            "anomaly_ratio": anomaly_ratio,
                            "zscore_threshold": rule.zscore_threshold,
                            "max_anomaly_ratio": rule.max_anomaly_ratio,
                        },
                    )
                )

        return report

    @staticmethod
    def _merge_report(target: ValidationReport, source: ValidationReport) -> None:
        """Merge one report into another."""

        for issue in source.issues:
            target.add_issue(issue)

    @staticmethod
    def _ensure_dataframe(dataset: pd.DataFrame) -> None:
        """Validate dataframe input type."""

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(
                f"Expected dataset to be a pandas DataFrame, received {type(dataset).__name__}."
            )

    @staticmethod
    def _dtype_matches(actual_dtype: Any, expected_type: str) -> bool:
        """Check whether a pandas dtype matches an abstract expected type."""

        normalized = expected_type.strip().lower()

        if normalized in {"float", "numeric"}:
            return pd.api.types.is_numeric_dtype(actual_dtype)
        if normalized == "int":
            return pd.api.types.is_integer_dtype(actual_dtype)
        if normalized in {"str", "string", "object"}:
            return pd.api.types.is_object_dtype(actual_dtype) or pd.api.types.is_string_dtype(actual_dtype)
        if normalized == "bool":
            return pd.api.types.is_bool_dtype(actual_dtype)
        if normalized == "datetime":
            return pd.api.types.is_datetime64_any_dtype(actual_dtype)

        raise ValueError(f"Unsupported expected dtype category: {expected_type}")

    def _build_summary(self, dataset: pd.DataFrame, report: ValidationReport) -> Dict[str, Any]:
        """Build a compact summary after validation."""

        numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        return {
            "row_count": int(len(dataset)),
            "column_count": int(dataset.shape[1]),
            "numeric_columns": numeric_columns,
            "issue_count": len(report.issues),
            "null_counts": dataset.isnull().sum().to_dict(),
        }


__all__ = [
    "AnomalyRule",
    "DataValidator",
    "DistributionRule",
    "RangeRule",
    "ValidationError",
    "ValidationIssue",
    "ValidationReport",
]
