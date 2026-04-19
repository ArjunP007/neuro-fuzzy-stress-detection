"""Train the neuro-fuzzy model once and save reusable artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.configs.config import load_system_config
from src.evaluation.accuracy import accuracy_analysis
from src.evaluation.confusion_matrix import compute_confusion_matrix
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.preprocess import DataPreprocessor
from src.fuzzy_logic.inference import FuzzyInferenceEngine
from src.fuzzy_logic.membership_functions import build_membership_function
from src.fuzzy_logic.rules import FuzzyRuleBase, LinguisticVariable
from src.neural_network.network import NeuralNetwork
from src.neuro_fuzzy.neuro_fuzzy_system import NeuroFuzzySystem


DATA_PATH = Path("artifacts/stress_dataset.csv")
MODEL_PATH = Path("artifacts/model.pkl")
PREPROCESSOR_PATH = Path("artifacts/preprocessor.pkl")
FEATURE_ENGINEER_PATH = Path("artifacts/feature_engineer.pkl")
METADATA_PATH = Path("artifacts/model_metadata.json")
CONFUSION_MATRIX_PATH = Path("artifacts/confusion_matrix.png")

BASE_FEATURE_COLUMNS = [
    "sleep_hours",
    "work_hours",
    "screen_time",
    "physical_activity_hours",
    "mental_fatigue_score",
    "heart_rate",
    "caffeine_intake",
    "social_interaction_hours",
    "work_pressure_score",
]


def create_membership_triplet(function_type: str, low: float, center: float, high: float) -> tuple[Any, Any, Any]:
    """Create low, medium, and high membership functions."""

    spread = max((high - low) / 4.0, 1e-3)
    normalized_type = function_type.strip().lower()

    if normalized_type == "gaussian":
        return (
            build_membership_function("gaussian", mean=low, sigma=spread),
            build_membership_function("gaussian", mean=center, sigma=spread),
            build_membership_function("gaussian", mean=high, sigma=spread),
        )

    return (
        build_membership_function("triangular", a=low - spread, b=low, c=center),
        build_membership_function("triangular", a=low, b=center, c=high),
        build_membership_function("triangular", a=center, b=high, c=high + spread),
    )


def build_fuzzy_system(feature_frame: pd.DataFrame, class_labels: list[str], config: Any) -> FuzzyInferenceEngine:
    """Build a fuzzy inference engine over the selected processed features."""

    fuzzy_config = config.fuzzy
    rule_base = FuzzyRuleBase(
        conjunction_operator=fuzzy_config.conjunction_operator,
        disjunction_operator=fuzzy_config.disjunction_operator,
        conflict_resolution="max_activation",
    )

    for column in feature_frame.columns:
        series = feature_frame[column].astype(float)
        variable = LinguisticVariable(name=column)
        low_mf, medium_mf, high_mf = create_membership_triplet(
            fuzzy_config.membership_function_type,
            float(series.quantile(0.2)),
            float(series.quantile(0.5)),
            float(series.quantile(0.8)),
        )
        variable.add_term("low", low_mf)
        variable.add_term("medium", medium_mf)
        variable.add_term("high", high_mf)
        rule_base.add_linguistic_variable(variable)

    output_variable = LinguisticVariable(name="stress_level")
    low_mf, medium_mf, high_mf = create_membership_triplet(
        fuzzy_config.membership_function_type,
        config.fuzzy.low_range[1],
        float(np.mean(config.fuzzy.medium_range)),
        config.fuzzy.high_range[0],
    )
    output_variable.add_term("low", low_mf)
    output_variable.add_term("medium", medium_mf)
    output_variable.add_term("high", high_mf)
    rule_base.add_linguistic_variable(output_variable)

    selected_columns = list(feature_frame.columns[:3])
    if len(selected_columns) >= 2:
        rule_base.add_rule(
            "rule_high",
            f"IF {selected_columns[0]} IS high AND {selected_columns[1]} IS high THEN stress_level IS high",
            weight=config.fuzzy.rule_weight_default,
        )
        rule_base.add_rule(
            "rule_low",
            f"IF {selected_columns[0]} IS low AND {selected_columns[1]} IS low THEN stress_level IS low",
            weight=config.fuzzy.rule_weight_default,
        )
    if len(selected_columns) >= 3:
        rule_base.add_rule(
            "rule_medium",
            f"IF {selected_columns[2]} IS medium THEN stress_level IS medium",
            weight=config.fuzzy.rule_weight_default,
        )

    if not rule_base.rules:
        fallback_feature = feature_frame.columns[0]
        rule_base.add_rule(
            "rule_fallback",
            f"IF {fallback_feature} IS medium THEN stress_level IS medium",
            weight=config.fuzzy.rule_weight_default,
        )

    output_universe = np.linspace(
        config.fuzzy.output_universe_range[0],
        config.fuzzy.output_universe_range[1],
        config.fuzzy.input_universe_resolution,
    )
    return FuzzyInferenceEngine(
        rule_base=rule_base,
        output_universes={"stress_level": output_universe},
        default_resolution=config.fuzzy.input_universe_resolution,
    )


def save_confusion_matrix(matrix: np.ndarray, labels: list[str]) -> None:
    """Save the confusion matrix as an image."""

    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_title("Neuro-Fuzzy Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Train the neuro-fuzzy model and save artifacts."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    config = load_system_config()
    dataset = pd.read_csv(DATA_PATH)

    preprocessor = DataPreprocessor(
        feature_columns=BASE_FEATURE_COLUMNS,
        target_column="stress_level",
        missing_strategy="median",
        outlier_method="none",
        scaling_method="standard",
        normalization_method=None,
        variance_threshold=0.0,
        test_size=0.2,
        validation_size=0.1,
        random_state=config.training.random_seed,
        stratify=True,
    )
    split_data = preprocessor.split_data(dataset)
    X_train_base = split_data.X_train
    X_test_base = split_data.X_test
    y_train = split_data.y_train.to_numpy()
    y_test = split_data.y_test.to_numpy()

    feature_engineer = FeatureEngineer(
        feature_columns=X_train_base.columns.tolist(),
        target_column="stress_level",
        top_k_features=12,
    )
    X_train = feature_engineer.fit_transform(X_train_base)
    X_test = feature_engineer.transform(X_test_base)

    neural_network = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_layers=[16, 12],
        output_size=3,
        activation="relu",
        output_activation="softmax",
        weight_init="he",
        optimizer="adam",
        learning_rate=0.01,
        batch_size=32,
        epochs=60,
        dropout_rate=0.05,
        early_stopping=True,
        patience=8,
        validation_split=0.0,
        random_state=42,
    )
    neural_network.fit(X_train.to_numpy(), y_train, verbose=False)

    fuzzy_engine = build_fuzzy_system(X_train, config.model.class_labels, config)
    model = NeuroFuzzySystem(
        neural_network=neural_network,
        fuzzy_inference_engine=fuzzy_engine,
        feature_names=X_train.columns.tolist(),
        class_labels=config.model.class_labels,
        neural_weight=config.model.neural_weight,
        fuzzy_weight=config.model.fuzzy_weight,
    )
    model.hybrid_learn(X_train.to_numpy(), y_train)

    predictions = model.predict(X_test.to_numpy())
    accuracy_value = accuracy_analysis(y_test, predictions)["accuracy"]
    confusion_matrix = compute_confusion_matrix(
        y_test,
        predictions,
        labels=list(range(len(config.model.class_labels))),
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(feature_engineer, FEATURE_ENGINEER_PATH)

    save_confusion_matrix(confusion_matrix, config.model.class_labels)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "accuracy": accuracy_value,
                "dataset_size": int(len(dataset)),
                "base_feature_count": int(X_train_base.shape[1]),
                "engineered_feature_count": int(X_train.shape[1]),
                "class_labels": config.model.class_labels,
                "model_path": str(MODEL_PATH),
                "preprocessor_path": str(PREPROCESSOR_PATH),
                "feature_engineer_path": str(FEATURE_ENGINEER_PATH),
                "confusion_matrix_path": str(CONFUSION_MATRIX_PATH),
            },
            indent=4,
        ),
        encoding="utf-8",
    )

    print(f"Final accuracy: {accuracy_value:.4f}")


if __name__ == "__main__":
    main()
