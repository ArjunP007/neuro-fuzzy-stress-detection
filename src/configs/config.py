"""Enterprise configuration system for the neuro-fuzzy stress project."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Type, TypeVar, Union, get_args, get_origin


LOGGER = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseConfig")


class ConfigError(Exception):
    """Base exception raised for configuration failures."""


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""


class ConfigSerializationError(ConfigError):
    """Raised when configuration cannot be serialized or deserialized."""


def _is_optional(annotation: Any) -> bool:
    """Return ``True`` when a type annotation is Optional."""

    origin = get_origin(annotation)
    if origin is Union:
        return type(None) in get_args(annotation)
    return False


def _strip_optional(annotation: Any) -> Any:
    """Return the non-None type from an Optional annotation."""

    if not _is_optional(annotation):
        return annotation
    return next(arg for arg in get_args(annotation) if arg is not type(None))


def _is_config_instance(value: Any) -> bool:
    """Return ``True`` when the value is an instance of ``BaseConfig``."""

    return isinstance(value, BaseConfig)


def _is_config_class(annotation: Any) -> bool:
    """Return ``True`` when the annotation refers to a config subclass."""

    annotation = _strip_optional(annotation)
    return isinstance(annotation, type) and issubclass(annotation, BaseConfig)


def _normalize_key(key: str) -> str:
    """Normalize environment and payload keys for matching."""

    return key.strip().lower()


def _coerce_bool(value: Any) -> bool:
    """Coerce a value into a boolean with explicit accepted literals."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        raise ConfigValidationError(f"Cannot coerce numeric value {value!r} to bool.")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    raise ConfigValidationError(f"Cannot coerce value {value!r} to bool.")


def _coerce_scalar(expected_type: Any, value: Any) -> Any:
    """Coerce a scalar value to the expected runtime type."""

    if value is None:
        return None

    expected_type = _strip_optional(expected_type)

    if expected_type is Any:
        return value
    if expected_type is bool:
        return _coerce_bool(value)
    if expected_type is int:
        if isinstance(value, bool):
            raise ConfigValidationError("Boolean values are not valid integers in configuration.")
        return int(value)
    if expected_type is float:
        if isinstance(value, bool):
            raise ConfigValidationError("Boolean values are not valid floats in configuration.")
        return float(value)
    if expected_type is str:
        return str(value)
    if expected_type is Path:
        return Path(value)
    if isinstance(expected_type, type) and isinstance(value, expected_type):
        return value
    return value


def _coerce_collection(annotation: Any, value: Any) -> Any:
    """Coerce common collection annotations recursively."""

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigValidationError(f"Expected sequence for {annotation!r}, got {type(value).__name__}.")
        item_type = args[0] if args else Any
        items = [_coerce_value(item_type, item) for item in value]
        return items if origin is list else tuple(items)

    if origin in (dict, Dict):
        if not isinstance(value, Mapping):
            raise ConfigValidationError(f"Expected mapping for {annotation!r}, got {type(value).__name__}.")
        key_type = args[0] if len(args) > 0 else Any
        value_type = args[1] if len(args) > 1 else Any
        return {
            _coerce_value(key_type, item_key): _coerce_value(value_type, item_value)
            for item_key, item_value in value.items()
        }

    if origin is Union:
        last_error: Optional[Exception] = None
        for union_type in args:
            if union_type is type(None) and value is None:
                return None
            try:
                return _coerce_value(union_type, value)
            except Exception as exc:
                last_error = exc
        raise ConfigValidationError(
            f"Unable to coerce value {value!r} into any supported union type for {annotation!r}."
        ) from last_error

    return _coerce_scalar(annotation, value)


def _coerce_value(annotation: Any, value: Any) -> Any:
    """Coerce an arbitrary value to match a field annotation."""

    if _is_config_class(annotation):
        config_cls = _strip_optional(annotation)
        if isinstance(value, config_cls):
            return value
        if isinstance(value, Mapping):
            return config_cls.from_dict(dict(value))
        raise ConfigValidationError(
            f"Cannot coerce value of type {type(value).__name__} into {config_cls.__name__}."
        )

    origin = get_origin(annotation)
    if origin is not None:
        return _coerce_collection(annotation, value)
    return _coerce_scalar(annotation, value)


def _deep_merge(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge two mapping structures."""

    for key, value in updates.items():
        if key in base and isinstance(base[key], MutableMapping) and isinstance(value, Mapping):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _parse_json_value(value: str) -> Any:
    """Attempt JSON parsing and gracefully fall back to raw string values."""

    stripped = value.strip()
    if not stripped:
        return stripped
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


@dataclass(kw_only=True)
class BaseConfig:
    """Base class offering validation, serialization, and environment overrides."""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the configuration object recursively."""

        self._validate_required_fields()
        self._validate_field_types()
        self._validate_nested_configs()
        self._custom_validate()

    def _validate_required_fields(self) -> None:
        """Ensure required dataclass fields are populated."""

        for config_field in fields(self):
            value = getattr(self, config_field.name)
            if value is None and not _is_optional(config_field.type):
                raise ConfigValidationError(
                    f"Field '{config_field.name}' in {self.__class__.__name__} cannot be None."
                )

    def _validate_field_types(self) -> None:
        """Validate and coerce field values according to annotations."""

        for config_field in fields(self):
            value = getattr(self, config_field.name)
            if value is None and _is_optional(config_field.type):
                continue
            setattr(self, config_field.name, _coerce_value(config_field.type, value))

    def _validate_nested_configs(self) -> None:
        """Validate nested configuration objects."""

        for config_field in fields(self):
            value = getattr(self, config_field.name)
            if _is_config_instance(value):
                value.validate()

    def _custom_validate(self) -> None:
        """Hook for child classes to implement business validation rules."""

    def to_dict(self) -> Dict[str, Any]:
        """Return a recursively serialized dictionary."""

        payload: Dict[str, Any] = {}
        for config_field in fields(self):
            payload[config_field.name] = self._serialize_value(getattr(self, config_field.name))
        return payload

    def to_json(self, *, indent: int = 4) -> str:
        """Serialize the configuration to JSON text."""

        try:
            return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ConfigSerializationError("Failed to serialize configuration to JSON.") from exc

    def export(self, filepath: Union[str, Path], *, indent: int = 4) -> Path:
        """Export the configuration to a JSON file."""

        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            target_path.write_text(self.to_json(indent=indent), encoding="utf-8")
        except OSError as exc:
            raise ConfigSerializationError(f"Failed to write configuration to {target_path}.") from exc
        LOGGER.info("Configuration exported to %s", target_path)
        return target_path

    @classmethod
    def from_dict(cls: Type[T], payload: Mapping[str, Any]) -> T:
        """Construct a configuration object from a dictionary."""

        if not isinstance(payload, Mapping):
            raise ConfigSerializationError(
                f"Expected mapping to build {cls.__name__}, got {type(payload).__name__}."
            )

        init_kwargs: Dict[str, Any] = {}
        for config_field in fields(cls):
            if config_field.name in payload:
                init_kwargs[config_field.name] = _coerce_value(config_field.type, payload[config_field.name])
        return cls(**init_kwargs)

    @classmethod
    def from_json(cls: Type[T], json_text: str) -> T:
        """Construct a configuration object from a JSON string."""

        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ConfigSerializationError(f"Invalid JSON supplied for {cls.__name__}.") from exc
        return cls.from_dict(payload)

    @classmethod
    def load(cls: Type[T], filepath: Union[str, Path]) -> T:
        """Load a configuration object from a JSON file."""

        source_path = Path(filepath)
        if not source_path.exists():
            raise ConfigSerializationError(f"Configuration file does not exist: {source_path}")

        try:
            content = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigSerializationError(f"Failed to read configuration file: {source_path}") from exc

        LOGGER.info("Loading configuration from %s", source_path)
        return cls.from_json(content)

    def update(self: T, **kwargs: Any) -> T:
        """Update fields in place and revalidate the configuration."""

        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ConfigValidationError(
                    f"Unknown configuration field '{key}' for {self.__class__.__name__}."
                )
            setattr(self, key, value)
        self.validate()
        return self

    def get(self, dotted_path: str, default: Any = None) -> Any:
        """Return a nested configuration value using dot notation."""

        if not dotted_path:
            raise ConfigValidationError("A dotted path is required for configuration lookup.")

        current: Any = self
        for part in dotted_path.split("."):
            if _is_config_instance(current):
                if not hasattr(current, part):
                    return default
                current = getattr(current, part)
            elif isinstance(current, Mapping):
                current = current.get(part, default)
            else:
                return default
        return current

    def apply_environment_overrides(self: T, prefix: str = "STRESS_APP") -> T:
        """Apply environment variable overrides using a nested key convention."""

        normalized_prefix = prefix.strip().upper()
        matched_items = {
            key: value
            for key, value in os.environ.items()
            if key.upper().startswith(f"{normalized_prefix}_")
        }

        if not matched_items:
            LOGGER.debug("No environment overrides found for prefix %s", normalized_prefix)
            return self

        overrides: Dict[str, Any] = {}
        for env_key, raw_value in matched_items.items():
            relative_key = env_key[len(normalized_prefix) + 1 :]
            path_parts = [_normalize_key(part) for part in relative_key.split("__") if part.strip()]
            if path_parts:
                self._assign_nested_override(overrides, path_parts, _parse_json_value(raw_value))

        merged = _deep_merge(self.to_dict(), overrides)
        refreshed = self.from_dict(merged)
        for config_field in fields(self):
            setattr(self, config_field.name, getattr(refreshed, config_field.name))
        self.validate()
        LOGGER.info("Applied %d environment overrides using prefix %s", len(matched_items), normalized_prefix)
        return self

    @staticmethod
    def _assign_nested_override(target: MutableMapping[str, Any], path_parts: Iterable[str], value: Any) -> None:
        """Assign a nested override path into a mutable mapping."""

        parts = list(path_parts)
        current = target
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize nested values to JSON-compatible structures."""

        if _is_config_instance(value):
            return value.to_dict()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, list):
            return [BaseConfig._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): BaseConfig._serialize_value(item) for key, item in value.items()}
        return value


@dataclass(kw_only=True)
class DatasetConfig(BaseConfig):
    """Configuration controlling synthetic dataset generation and loading."""

    dataset_path: Path = Path("stress_dataset.csv")
    export_directory: Path = Path("artifacts/datasets")
    num_samples: int = 5000
    random_seed: int = 42
    train_split_ratio: float = 0.8
    validation_split_ratio: float = 0.1
    shuffle: bool = True
    sleep_range: tuple[float, float] = (0.0, 10.0)
    work_hours_range: tuple[float, float] = (0.0, 14.0)
    screen_time_range: tuple[float, float] = (0.0, 16.0)
    activity_range: tuple[float, float] = (0.0, 6.0)
    fatigue_range: tuple[float, float] = (0.0, 10.0)
    heart_rate_range: tuple[int, int] = (50, 150)
    allowed_labels: list[str] = field(default_factory=lambda: ["Low", "Medium", "High"])
    missing_value_strategy: str = "raise"
    outlier_zscore_threshold: float = 3.0
    persist_generated_data: bool = True
    enable_feature_clipping: bool = True

    def _custom_validate(self) -> None:
        self._validate_positive_int("num_samples", self.num_samples, minimum=100)
        self._validate_positive_int("random_seed", self.random_seed, minimum=0)
        self._validate_ratio("train_split_ratio", self.train_split_ratio)
        self._validate_ratio("validation_split_ratio", self.validation_split_ratio, allow_zero=True)

        if self.train_split_ratio + self.validation_split_ratio >= 1.0:
            raise ConfigValidationError(
                "The sum of train_split_ratio and validation_split_ratio must be less than 1.0."
            )

        self._validate_range("sleep_range", self.sleep_range)
        self._validate_range("work_hours_range", self.work_hours_range)
        self._validate_range("screen_time_range", self.screen_time_range)
        self._validate_range("activity_range", self.activity_range)
        self._validate_range("fatigue_range", self.fatigue_range)
        self._validate_range("heart_rate_range", self.heart_rate_range)

        if len(self.allowed_labels) < 2:
            raise ConfigValidationError("DatasetConfig.allowed_labels must contain at least two classes.")

        valid_missing_strategies = {"raise", "drop", "mean_impute", "median_impute"}
        if self.missing_value_strategy not in valid_missing_strategies:
            raise ConfigValidationError(
                f"missing_value_strategy must be one of {sorted(valid_missing_strategies)}."
            )

        if self.outlier_zscore_threshold <= 0:
            raise ConfigValidationError("outlier_zscore_threshold must be strictly positive.")

    @staticmethod
    def _validate_positive_int(name: str, value: int, minimum: int = 1) -> None:
        if value < minimum:
            raise ConfigValidationError(f"{name} must be greater than or equal to {minimum}.")

    @staticmethod
    def _validate_ratio(name: str, value: float, allow_zero: bool = False) -> None:
        if allow_zero:
            if not 0.0 <= value < 1.0:
                raise ConfigValidationError(f"{name} must satisfy 0.0 <= value < 1.0.")
            return
        if not 0.0 < value < 1.0:
            raise ConfigValidationError(f"{name} must satisfy 0.0 < value < 1.0.")

    @staticmethod
    def _validate_range(name: str, value: tuple[float, float]) -> None:
        if len(value) != 2:
            raise ConfigValidationError(f"{name} must contain exactly two values.")
        lower, upper = value
        if lower >= upper:
            raise ConfigValidationError(f"{name} must define a strictly increasing range.")

    def get_dataset_path(self) -> Path:
        """Return the configured dataset path."""

        return self.dataset_path

    def get_export_directory(self) -> Path:
        """Return the configured export directory."""

        return self.export_directory


@dataclass(kw_only=True)
class TrainingConfig(BaseConfig):
    """Configuration controlling optimization and experiment execution."""

    learning_rate: float = 0.001
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 1.0
    epochs: int = 150
    batch_size: int = 32
    optimizer: str = "adam"
    loss_function: str = "categorical_crossentropy"
    early_stopping: bool = True
    early_stopping_patience: int = 15
    gradient_clip_norm: Optional[float] = 5.0
    l2_regularization: float = 0.0001
    dropout_rate: float = 0.1
    validation_frequency: int = 1
    shuffle_each_epoch: bool = True
    random_seed: int = 42
    num_workers: int = 1
    checkpoint_frequency: int = 10
    checkpoint_directory: Path = Path("artifacts/checkpoints")
    enable_checkpointing: bool = True
    enable_mixed_precision: bool = False
    metrics: list[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"])

    def _custom_validate(self) -> None:
        self._validate_open_interval("learning_rate", self.learning_rate, 0.0, self.max_learning_rate)
        self._validate_open_interval("min_learning_rate", self.min_learning_rate, 0.0, self.max_learning_rate)
        self._validate_open_interval("max_learning_rate", self.max_learning_rate, 0.0, 10.0)

        if self.min_learning_rate > self.learning_rate:
            raise ConfigValidationError("min_learning_rate cannot exceed learning_rate.")
        if self.learning_rate > self.max_learning_rate:
            raise ConfigValidationError("learning_rate cannot exceed max_learning_rate.")

        self._validate_positive_int("epochs", self.epochs)
        self._validate_positive_int("batch_size", self.batch_size)
        self._validate_positive_int("early_stopping_patience", self.early_stopping_patience)
        self._validate_non_negative_float("l2_regularization", self.l2_regularization)
        self._validate_rate("dropout_rate", self.dropout_rate, max_inclusive=False)
        self._validate_positive_int("validation_frequency", self.validation_frequency)
        self._validate_positive_int("num_workers", self.num_workers)
        self._validate_positive_int("checkpoint_frequency", self.checkpoint_frequency)

        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ConfigValidationError("gradient_clip_norm must be strictly positive when provided.")

        valid_optimizers = {"sgd", "momentum", "rmsprop", "adam"}
        if self.optimizer not in valid_optimizers:
            raise ConfigValidationError(f"optimizer must be one of {sorted(valid_optimizers)}.")

        valid_losses = {"categorical_crossentropy", "cross_entropy", "mse"}
        if self.loss_function not in valid_losses:
            raise ConfigValidationError(f"loss_function must be one of {sorted(valid_losses)}.")

        if not self.metrics:
            raise ConfigValidationError("At least one evaluation metric must be configured.")

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        if value <= 0:
            raise ConfigValidationError(f"{name} must be strictly positive.")

    @staticmethod
    def _validate_non_negative_float(name: str, value: float) -> None:
        if value < 0:
            raise ConfigValidationError(f"{name} must be non-negative.")

    @staticmethod
    def _validate_open_interval(name: str, value: float, lower: float, upper: float) -> None:
        if not lower < value <= upper:
            raise ConfigValidationError(f"{name} must satisfy {lower} < value <= {upper}.")

    @staticmethod
    def _validate_rate(name: str, value: float, max_inclusive: bool = True) -> None:
        if max_inclusive:
            if not 0.0 <= value <= 1.0:
                raise ConfigValidationError(f"{name} must satisfy 0.0 <= value <= 1.0.")
            return
        if not 0.0 <= value < 1.0:
            raise ConfigValidationError(f"{name} must satisfy 0.0 <= value < 1.0.")

    def get_checkpoint_directory(self) -> Path:
        """Return the checkpoint directory."""

        return self.checkpoint_directory

    def get_metrics(self) -> list[str]:
        """Return a defensive copy of configured metrics."""

        return list(self.metrics)


@dataclass(kw_only=True)
class FuzzyConfig(BaseConfig):
    """Configuration for fuzzy membership functions and inference rules."""

    enabled: bool = True
    inference_type: str = "mamdani"
    defuzzification_method: str = "centroid"
    aggregation_method: str = "max"
    conjunction_operator: str = "product"
    disjunction_operator: str = "max"
    implication_operator: str = "min"
    membership_function_type: str = "gaussian"
    input_universe_resolution: int = 1000
    output_universe_range: tuple[float, float] = (0.0, 10.0)
    low_range: tuple[float, float] = (0.0, 3.5)
    medium_range: tuple[float, float] = (3.0, 7.0)
    high_range: tuple[float, float] = (6.5, 10.0)
    rule_weight_default: float = 1.0
    enable_rule_learning: bool = True
    max_rules: int = 64
    certainty_threshold: float = 0.55
    smooth_membership_functions: bool = True
    linguistic_labels: list[str] = field(default_factory=lambda: ["low", "medium", "high"])

    def _custom_validate(self) -> None:
        valid_inference = {"mamdani", "sugeno", "tsukamoto"}
        valid_defuzzification = {"centroid", "bisector", "mom", "som", "lom"}
        valid_aggregation = {"max", "sum", "probabilistic_or"}
        valid_conjunction = {"min", "product"}
        valid_membership = {"gaussian", "triangular", "trapezoidal", "gbell"}

        if self.inference_type not in valid_inference:
            raise ConfigValidationError(f"inference_type must be one of {sorted(valid_inference)}.")
        if self.defuzzification_method not in valid_defuzzification:
            raise ConfigValidationError(
                f"defuzzification_method must be one of {sorted(valid_defuzzification)}."
            )
        if self.aggregation_method not in valid_aggregation:
            raise ConfigValidationError(f"aggregation_method must be one of {sorted(valid_aggregation)}.")
        if self.conjunction_operator not in valid_conjunction:
            raise ConfigValidationError(f"conjunction_operator must be one of {sorted(valid_conjunction)}.")
        if self.implication_operator not in {"min", "product"}:
            raise ConfigValidationError("implication_operator must be either 'min' or 'product'.")
        if self.disjunction_operator not in {"max", "probabilistic_or"}:
            raise ConfigValidationError("disjunction_operator must be either 'max' or 'probabilistic_or'.")
        if self.membership_function_type not in valid_membership:
            raise ConfigValidationError(
                f"membership_function_type must be one of {sorted(valid_membership)}."
            )

        if self.input_universe_resolution < 100:
            raise ConfigValidationError("input_universe_resolution must be at least 100.")

        self._validate_range("output_universe_range", self.output_universe_range)
        self._validate_range("low_range", self.low_range)
        self._validate_range("medium_range", self.medium_range)
        self._validate_range("high_range", self.high_range)

        if not 0.0 < self.rule_weight_default <= 1.0:
            raise ConfigValidationError("rule_weight_default must satisfy 0.0 < value <= 1.0.")
        if self.max_rules <= 0:
            raise ConfigValidationError("max_rules must be strictly positive.")
        if not 0.0 <= self.certainty_threshold <= 1.0:
            raise ConfigValidationError("certainty_threshold must satisfy 0.0 <= value <= 1.0.")
        if len(self.linguistic_labels) < 3:
            raise ConfigValidationError("linguistic_labels must contain at least three fuzzy sets.")

    @staticmethod
    def _validate_range(name: str, value: tuple[float, float]) -> None:
        if len(value) != 2:
            raise ConfigValidationError(f"{name} must contain exactly two values.")
        if value[0] >= value[1]:
            raise ConfigValidationError(f"{name} must define an increasing range.")

    def get_output_universe_range(self) -> tuple[float, float]:
        """Return the fuzzy output universe."""

        return self.output_universe_range


@dataclass(kw_only=True)
class ModelConfig(BaseConfig):
    """Configuration for ANN and neuro-fuzzy model topology."""

    model_name: str = "stress_neuro_fuzzy_model"
    version: str = "1.0.0"
    input_size: int = 6
    output_size: int = 3
    hidden_layers: list[int] = field(default_factory=lambda: [32, 16, 8])
    activation_hidden: str = "relu"
    activation_output: str = "softmax"
    weight_init: str = "he_normal"
    bias_init: str = "zeros"
    enable_bias: bool = True
    neuro_fuzzy_fusion_strategy: str = "weighted_average"
    neural_weight: float = 0.65
    fuzzy_weight: float = 0.35
    use_batch_norm: bool = False
    model_directory: Path = Path("artifacts/models")
    export_onnx: bool = False
    save_trained_weights: bool = True
    class_labels: list[str] = field(default_factory=lambda: ["Low", "Medium", "High"])

    def _custom_validate(self) -> None:
        if self.input_size <= 0:
            raise ConfigValidationError("input_size must be strictly positive.")
        if self.output_size <= 1:
            raise ConfigValidationError("output_size must be greater than 1.")
        if not self.hidden_layers:
            raise ConfigValidationError("hidden_layers must contain at least one layer.")
        if any(units <= 0 for units in self.hidden_layers):
            raise ConfigValidationError("All hidden layer sizes must be strictly positive.")

        valid_hidden_activations = {"relu", "tanh", "sigmoid", "leaky_relu"}
        valid_output_activations = {"softmax", "sigmoid", "linear"}
        valid_weight_init = {"xavier", "xavier_uniform", "he_normal", "he_uniform", "normal"}
        valid_bias_init = {"zeros", "ones", "normal"}
        valid_fusion = {"weighted_average", "rule_guided", "confidence_gate", "stacked"}

        if self.activation_hidden not in valid_hidden_activations:
            raise ConfigValidationError(
                f"activation_hidden must be one of {sorted(valid_hidden_activations)}."
            )
        if self.activation_output not in valid_output_activations:
            raise ConfigValidationError(
                f"activation_output must be one of {sorted(valid_output_activations)}."
            )
        if self.weight_init not in valid_weight_init:
            raise ConfigValidationError(f"weight_init must be one of {sorted(valid_weight_init)}.")
        if self.bias_init not in valid_bias_init:
            raise ConfigValidationError(f"bias_init must be one of {sorted(valid_bias_init)}.")
        if self.neuro_fuzzy_fusion_strategy not in valid_fusion:
            raise ConfigValidationError(
                f"neuro_fuzzy_fusion_strategy must be one of {sorted(valid_fusion)}."
            )

        if not 0.0 <= self.neural_weight <= 1.0:
            raise ConfigValidationError("neural_weight must satisfy 0.0 <= value <= 1.0.")
        if not 0.0 <= self.fuzzy_weight <= 1.0:
            raise ConfigValidationError("fuzzy_weight must satisfy 0.0 <= value <= 1.0.")
        if abs((self.neural_weight + self.fuzzy_weight) - 1.0) > 1e-8:
            raise ConfigValidationError("neural_weight and fuzzy_weight must sum to 1.0.")
        if len(self.class_labels) != self.output_size:
            raise ConfigValidationError("class_labels length must equal output_size.")

    def get_hidden_layers(self) -> list[int]:
        """Return a defensive copy of hidden layer sizes."""

        return list(self.hidden_layers)

    def get_model_directory(self) -> Path:
        """Return the model artifact directory."""

        return self.model_directory


@dataclass(kw_only=True)
class LoggingConfig(BaseConfig):
    """Configuration controlling log formatting and output destinations."""

    logger_name: str = "stress_detection"
    level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    log_file: Path = Path("logs/system.log")
    max_file_size_mb: int = 10
    backup_count: int = 5
    log_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    enable_json_logging: bool = False
    capture_warnings: bool = True
    propagate: bool = False

    def _custom_validate(self) -> None:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise ConfigValidationError(f"level must be one of {sorted(valid_levels)}.")
        self.level = self.level.upper()

        if not self.log_to_file and not self.log_to_console:
            raise ConfigValidationError("At least one logging destination must be enabled.")
        if self.max_file_size_mb <= 0:
            raise ConfigValidationError("max_file_size_mb must be strictly positive.")
        if self.backup_count < 0:
            raise ConfigValidationError("backup_count must be non-negative.")
        if not self.log_format.strip():
            raise ConfigValidationError("log_format cannot be empty.")
        if not self.date_format.strip():
            raise ConfigValidationError("date_format cannot be empty.")

    def get_log_file(self) -> Path:
        """Return the configured log file path."""

        return self.log_file


@dataclass(kw_only=True)
class SystemConfig(BaseConfig):
    """Top-level nested application configuration container."""

    project_name: str = "Stress Level Detection using Neuro-Fuzzy Approach"
    project_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fuzzy: FuzzyConfig = field(default_factory=FuzzyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def _custom_validate(self) -> None:
        valid_environments = {"development", "testing", "staging", "production"}
        if self.environment not in valid_environments:
            raise ConfigValidationError(
                f"environment must be one of {sorted(valid_environments)}."
            )

        if self.dataset.allowed_labels != self.model.class_labels:
            raise ConfigValidationError(
                "Dataset allowed_labels must match Model class_labels to keep the pipeline consistent."
            )

        if self.model.output_size != len(self.dataset.allowed_labels):
            raise ConfigValidationError(
                "Model output_size must equal the number of dataset class labels."
            )

    @classmethod
    def from_sources(
        cls,
        *,
        json_path: Optional[Union[str, Path]] = None,
        overrides: Optional[Mapping[str, Any]] = None,
        env_prefix: str = "STRESS_APP",
        apply_env: bool = True,
    ) -> "SystemConfig":
        """Build configuration from defaults, JSON file, direct overrides, and environment."""

        config = cls()

        if json_path is not None:
            loaded = cls.load(json_path)
            config = cls.from_dict(loaded.to_dict())

        if overrides:
            merged = _deep_merge(config.to_dict(), dict(overrides))
            config = cls.from_dict(merged)

        if apply_env:
            config.apply_environment_overrides(prefix=env_prefix)

        config.validate()
        return config

    def export_config(self, filepath: Union[str, Path], *, indent: int = 4) -> Path:
        """Export the complete system configuration to disk."""

        return self.export(filepath, indent=indent)

    def get_dataset_config(self) -> DatasetConfig:
        """Return dataset configuration."""

        return self.dataset

    def get_training_config(self) -> TrainingConfig:
        """Return training configuration."""

        return self.training

    def get_fuzzy_config(self) -> FuzzyConfig:
        """Return fuzzy inference configuration."""

        return self.fuzzy

    def get_model_config(self) -> ModelConfig:
        """Return model configuration."""

        return self.model

    def get_logging_config(self) -> LoggingConfig:
        """Return logging configuration."""

        return self.logging

    def get_project_name(self) -> str:
        """Return the configured project name."""

        return self.project_name

    def get_project_version(self) -> str:
        """Return the configured project version."""

        return self.project_version

    def get_environment(self) -> str:
        """Return the current runtime environment."""

        return self.environment

    def with_metadata(self, **metadata: Any) -> "SystemConfig":
        """Attach metadata values and revalidate the configuration."""

        merged_metadata = dict(self.metadata)
        merged_metadata.update(metadata)
        self.metadata = merged_metadata
        self.validate()
        return self


def export_default_config(filepath: Union[str, Path]) -> Path:
    """Export a default system configuration to the given file path."""

    config = SystemConfig()
    return config.export_config(filepath)


def load_system_config(
    filepath: Optional[Union[str, Path]] = None,
    *,
    env_prefix: str = "STRESS_APP",
    overrides: Optional[Mapping[str, Any]] = None,
) -> SystemConfig:
    """Load the application configuration using supported layered sources."""

    return SystemConfig.from_sources(
        json_path=filepath,
        overrides=overrides,
        env_prefix=env_prefix,
        apply_env=True,
    )


__all__ = [
    "BaseConfig",
    "ConfigError",
    "ConfigSerializationError",
    "ConfigValidationError",
    "DatasetConfig",
    "FuzzyConfig",
    "LoggingConfig",
    "ModelConfig",
    "SystemConfig",
    "TrainingConfig",
    "export_default_config",
    "load_system_config",
]
