"""Advanced synthetic dataset generation for neuro-fuzzy stress detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


class StressDatasetGenerator:
    """Generate realistic synthetic lifestyle data for stress prediction research."""

    FEATURE_COLUMNS = [
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

    TARGET_COLUMNS = ["stress_score", "stress_level"]
    LABELS = ("Low", "Medium", "High")

    def __init__(
        self,
        num_samples: int = 5000,
        random_state: int = 42,
        noise_level: float = 0.08,
        enable_balancing: bool = True,
        validate_after_generation: bool = True,
    ) -> None:
        """Initialize the generator with reproducible simulation settings."""

        if num_samples < 300:
            raise ValueError("num_samples must be at least 300 for stable class balancing.")
        if noise_level < 0:
            raise ValueError("noise_level must be non-negative.")

        self.num_samples = int(num_samples)
        self.random_state = int(random_state)
        self.noise_level = float(noise_level)
        self.enable_balancing = bool(enable_balancing)
        self.validate_after_generation = bool(validate_after_generation)
        self.rng = np.random.default_rng(self.random_state)

    def generate_dataset(self) -> pd.DataFrame:
        """Generate a complete synthetic dataset with balancing and validation."""

        generation_size = self._determine_generation_size()
        latent_factors = self._generate_latent_factors(generation_size)
        dataset = self.simulate_correlations(latent_factors)
        dataset = self.add_noise(dataset)
        dataset["stress_score"] = self.compute_stress_score(dataset)
        dataset["stress_level"] = self.assign_labels(dataset["stress_score"].to_numpy())

        if self.enable_balancing:
            dataset = self._balance_classes(dataset)

        dataset = dataset.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)

        if self.validate_after_generation:
            self.validate_distribution(dataset)

        LOGGER.info("Generated dataset with shape %s", dataset.shape)
        return dataset

    def simulate_correlations(self, latent_factors: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
        """Simulate feature correlations driven by latent behavioral and physiological factors."""

        latent = latent_factors or self._generate_latent_factors(self.num_samples)
        size = len(next(iter(latent.values())))

        chronic_stress = latent["chronic_stress"]
        workload_pressure = latent["workload_pressure"]
        digital_exposure = latent["digital_exposure"]
        health_discipline = latent["health_discipline"]
        social_support = latent["social_support"]
        physiological_load = latent["physiological_load"]

        work_hours = self._clip(
            7.6
            + 1.9 * workload_pressure
            + 0.6 * chronic_stress
            - 0.35 * health_discipline
            + self.rng.normal(0.0, 0.7, size),
            3.0,
            14.0,
        )

        work_pressure_score = self._clip(
            5.0
            + 1.9 * workload_pressure
            + 1.3 * chronic_stress
            - 0.45 * social_support
            + self.rng.normal(0.0, 0.9, size),
            0.0,
            10.0,
        )

        screen_time = self._clip(
            5.8
            + 1.3 * digital_exposure
            + 0.45 * workload_pressure
            + 0.35 * chronic_stress
            - 0.2 * health_discipline
            + self.rng.normal(0.0, 1.0, size),
            0.5,
            15.0,
        )

        physical_activity_hours = self._clip(
            1.7
            + 1.15 * health_discipline
            - 0.55 * workload_pressure
            - 0.45 * chronic_stress
            + 0.15 * social_support
            + self.rng.normal(0.0, 0.7, size),
            0.0,
            4.5,
        )

        social_interaction_hours = self._clip(
            2.4
            + 1.35 * social_support
            - 0.4 * chronic_stress
            - 0.35 * workload_pressure
            + self.rng.normal(0.0, 0.8, size),
            0.0,
            8.0,
        )

        caffeine_intake = self._clip(
            1.3
            + 0.8 * workload_pressure
            + 0.7 * chronic_stress
            + 0.35 * digital_exposure
            - 0.3 * health_discipline
            + self.rng.normal(0.0, 0.65, size),
            0.0,
            8.0,
        )

        sleep_hours = self._clip(
            7.4
            - 1.0 * chronic_stress
            - 0.55 * workload_pressure
            - 0.3 * digital_exposure
            + 0.45 * health_discipline
            + 0.2 * social_support
            + self.rng.normal(0.0, 0.75, size),
            3.0,
            10.0,
        )

        mental_fatigue_score = self._clip(
            4.6
            + 1.35 * chronic_stress
            + 0.95 * workload_pressure
            + 0.65 * digital_exposure
            + 0.35 * physiological_load
            - 0.95 * self._normalize(sleep_hours, 3.0, 10.0)
            - 0.55 * physical_activity_hours
            - 0.25 * social_interaction_hours
            + self.rng.normal(0.0, 0.8, size),
            0.0,
            10.0,
        )

        heart_rate = self._clip(
            73.0
            + 4.5 * chronic_stress
            + 3.8 * workload_pressure
            + 2.2 * physiological_load
            + 1.8 * caffeine_intake
            - 2.8 * physical_activity_hours
            - 1.3 * sleep_hours
            + self.rng.normal(0.0, 4.0, size),
            55.0,
            125.0,
        )

        dataset = pd.DataFrame(
            {
                "sleep_hours": sleep_hours,
                "work_hours": work_hours,
                "screen_time": screen_time,
                "physical_activity_hours": physical_activity_hours,
                "mental_fatigue_score": mental_fatigue_score,
                "heart_rate": heart_rate,
                "caffeine_intake": caffeine_intake,
                "social_interaction_hours": social_interaction_hours,
                "work_pressure_score": work_pressure_score,
            }
        )

        return dataset

    def add_noise(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Inject realistic noise while preserving physiological plausibility."""

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset must be a pandas DataFrame.")

        noisy_dataset = dataset.copy()
        if self.noise_level == 0:
            return noisy_dataset

        scale_map = {
            "sleep_hours": 0.30,
            "work_hours": 0.35,
            "screen_time": 0.45,
            "physical_activity_hours": 0.25,
            "mental_fatigue_score": 0.40,
            "heart_rate": 2.20,
            "caffeine_intake": 0.25,
            "social_interaction_hours": 0.30,
            "work_pressure_score": 0.35,
        }

        bounds = {
            "sleep_hours": (0.0, 10.0),
            "work_hours": (0.0, 16.0),
            "screen_time": (0.0, 18.0),
            "physical_activity_hours": (0.0, 6.0),
            "mental_fatigue_score": (0.0, 10.0),
            "heart_rate": (50.0, 140.0),
            "caffeine_intake": (0.0, 10.0),
            "social_interaction_hours": (0.0, 10.0),
            "work_pressure_score": (0.0, 10.0),
        }

        for column in self.FEATURE_COLUMNS:
            column_noise = self.rng.normal(
                loc=0.0,
                scale=scale_map[column] * self.noise_level,
                size=len(noisy_dataset),
            )
            noisy_dataset[column] = self._clip(
                noisy_dataset[column].to_numpy() + column_noise,
                bounds[column][0],
                bounds[column][1],
            )

        return noisy_dataset

    def compute_stress_score(self, dataset: pd.DataFrame) -> np.ndarray:
        """Compute a continuous stress score using nonlinear lifestyle interactions."""

        self._validate_required_columns(dataset, self.FEATURE_COLUMNS)

        sleep = dataset["sleep_hours"].to_numpy()
        work = dataset["work_hours"].to_numpy()
        screen = dataset["screen_time"].to_numpy()
        activity = dataset["physical_activity_hours"].to_numpy()
        fatigue = dataset["mental_fatigue_score"].to_numpy()
        heart_rate = dataset["heart_rate"].to_numpy()
        caffeine = dataset["caffeine_intake"].to_numpy()
        social = dataset["social_interaction_hours"].to_numpy()
        pressure = dataset["work_pressure_score"].to_numpy()

        sleep_penalty = self._scaled_inverse(sleep, 4.0, 9.0)
        work_load = self._normalize(work, 4.0, 14.0)
        screen_load = self._normalize(screen, 1.0, 14.0)
        activity_buffer = self._scaled_inverse(activity, 0.0, 3.5)
        fatigue_load = self._normalize(fatigue, 0.0, 10.0)
        heart_load = self._normalize(heart_rate, 55.0, 120.0)
        caffeine_load = self._normalize(caffeine, 0.0, 7.0)
        social_buffer = self._scaled_inverse(social, 0.0, 6.0)
        pressure_load = self._normalize(pressure, 0.0, 10.0)

        interaction_1 = np.clip(work_load * pressure_load, 0.0, 1.0)
        interaction_2 = np.clip(sleep_penalty * fatigue_load, 0.0, 1.0)
        interaction_3 = np.clip(screen_load * sleep_penalty, 0.0, 1.0)
        recovery_factor = np.clip((1.0 - activity_buffer) * (1.0 - social_buffer), 0.0, 1.0)

        score = (
            0.16 * sleep_penalty
            + 0.12 * work_load
            + 0.08 * screen_load
            + 0.09 * activity_buffer
            + 0.18 * fatigue_load
            + 0.10 * heart_load
            + 0.07 * caffeine_load
            + 0.07 * social_buffer
            + 0.13 * pressure_load
            + 0.08 * interaction_1
            + 0.07 * interaction_2
            + 0.05 * interaction_3
            - 0.08 * recovery_factor
        )

        score = score + self.rng.normal(0.0, 0.05, len(dataset))
        return np.clip(score * 10.0, 0.0, 10.0)

    def assign_labels(self, stress_scores: np.ndarray) -> np.ndarray:
        """Assign categorical stress labels using percentile-aware thresholds."""

        scores = np.asarray(stress_scores, dtype=float)
        if scores.ndim != 1:
            raise ValueError("stress_scores must be a one-dimensional array.")

        lower_threshold = max(3.6, float(np.quantile(scores, 0.34)))
        upper_threshold = min(6.8, float(np.quantile(scores, 0.67)))
        upper_threshold = max(upper_threshold, lower_threshold + 0.75)

        labels = np.full(scores.shape, "Medium", dtype=object)
        labels[scores < lower_threshold] = "Low"
        labels[scores >= upper_threshold] = "High"
        return labels.astype(str)

    def validate_distribution(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataset quality, label balance, and expected statistical correlations."""

        self._validate_required_columns(dataset, self.FEATURE_COLUMNS + self.TARGET_COLUMNS)

        if dataset.isnull().any().any():
            raise ValueError("Generated dataset contains missing values.")

        class_counts = dataset["stress_level"].value_counts().reindex(self.LABELS, fill_value=0)
        class_proportions = class_counts / len(dataset)

        if (class_counts == 0).any():
            raise ValueError("Generated dataset is missing one or more stress classes.")

        max_imbalance = float(class_proportions.max() - class_proportions.min())
        if self.enable_balancing and max_imbalance > 0.12:
            raise ValueError(f"Balanced generation failed. Observed imbalance delta: {max_imbalance:.3f}")

        correlations = dataset[self.FEATURE_COLUMNS + ["stress_score"]].corr(numeric_only=True)
        sleep_fatigue_corr = float(correlations.loc["sleep_hours", "mental_fatigue_score"])
        work_pressure_corr = float(correlations.loc["work_hours", "work_pressure_score"])
        activity_stress_corr = float(correlations.loc["physical_activity_hours", "stress_score"])

        if sleep_fatigue_corr > -0.20:
            raise ValueError(
                "Expected negative sleep-fatigue correlation was not observed strongly enough."
            )
        if work_pressure_corr < 0.30:
            raise ValueError(
                "Expected positive work-hours and work-pressure correlation was not observed strongly enough."
            )
        if activity_stress_corr > -0.10:
            raise ValueError(
                "Expected stress-reduction relationship with physical activity was not observed strongly enough."
            )

        summary = {
            "num_samples": int(len(dataset)),
            "class_counts": class_counts.to_dict(),
            "class_proportions": class_proportions.round(4).to_dict(),
            "mean_stress_score": float(dataset["stress_score"].mean()),
            "std_stress_score": float(dataset["stress_score"].std(ddof=1)),
            "sleep_fatigue_corr": sleep_fatigue_corr,
            "work_pressure_corr": work_pressure_corr,
            "activity_stress_corr": activity_stress_corr,
        }
        LOGGER.info("Dataset validation successful: %s", summary)
        return summary

    def save_dataset(self, dataset: pd.DataFrame, filepath: Union[str, Path] = "stress_dataset.csv") -> Path:
        """Persist the generated dataset to CSV."""

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset must be a pandas DataFrame.")

        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(target_path, index=False)
        LOGGER.info("Dataset saved to %s", target_path)
        return target_path

    def load_dataset(self, filepath: Union[str, Path] = "stress_dataset.csv") -> pd.DataFrame:
        """Load a dataset from disk and validate its expected structure."""

        source_path = Path(filepath)
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {source_path}")

        dataset = pd.read_csv(source_path)
        self._validate_required_columns(dataset, self.FEATURE_COLUMNS + self.TARGET_COLUMNS)
        LOGGER.info("Loaded dataset from %s with shape %s", source_path, dataset.shape)
        return dataset

    def dataset_summary(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Return a structured summary of the dataset for analysis or reporting."""

        self._validate_required_columns(dataset, self.FEATURE_COLUMNS + self.TARGET_COLUMNS)

        summary = {
            "shape": tuple(dataset.shape),
            "class_distribution": dataset["stress_level"].value_counts().to_dict(),
            "feature_means": dataset[self.FEATURE_COLUMNS].mean().round(3).to_dict(),
            "feature_std": dataset[self.FEATURE_COLUMNS].std().round(3).to_dict(),
            "stress_score_summary": dataset["stress_score"].describe().round(3).to_dict(),
            "correlation_with_stress": dataset[self.FEATURE_COLUMNS + ["stress_score"]]
            .corr(numeric_only=True)["stress_score"]
            .drop("stress_score")
            .round(3)
            .to_dict(),
            "missing_values": dataset.isnull().sum().to_dict(),
        }
        return summary

    def _generate_latent_factors(self, size: int) -> Dict[str, np.ndarray]:
        """Generate latent variables that drive realistic human behavior patterns."""

        if size <= 0:
            raise ValueError("size must be strictly positive.")

        baseline = self.rng.normal(0.0, 1.0, size)
        chronic_stress = 0.75 * baseline + self.rng.normal(0.0, 0.65, size)
        workload_pressure = 0.55 * chronic_stress + self.rng.normal(0.0, 0.75, size)
        digital_exposure = 0.35 * workload_pressure + self.rng.normal(0.0, 0.9, size)
        health_discipline = -0.30 * chronic_stress + self.rng.normal(0.0, 0.85, size)
        social_support = -0.25 * chronic_stress + self.rng.normal(0.0, 0.8, size)
        physiological_load = 0.45 * chronic_stress + 0.20 * workload_pressure + self.rng.normal(0.0, 0.7, size)

        return {
            "chronic_stress": chronic_stress,
            "workload_pressure": workload_pressure,
            "digital_exposure": digital_exposure,
            "health_discipline": health_discipline,
            "social_support": social_support,
            "physiological_load": physiological_load,
        }

    def _balance_classes(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Balance the stress classes using controlled stratified resampling."""

        if "stress_level" not in dataset.columns:
            raise ValueError("stress_level column is required for class balancing.")

        target_per_class = self.num_samples // len(self.LABELS)
        remainder = self.num_samples % len(self.LABELS)
        class_targets = {
            label: target_per_class + (1 if index < remainder else 0)
            for index, label in enumerate(self.LABELS)
        }

        balanced_parts = []
        for label in self.LABELS:
            label_frame = dataset[dataset["stress_level"] == label]
            if label_frame.empty:
                raise ValueError(f"Cannot balance dataset because class '{label}' has no samples.")

            replace = len(label_frame) < class_targets[label]
            sampled = label_frame.sample(
                n=class_targets[label],
                replace=replace,
                random_state=self.random_state,
            )
            balanced_parts.append(sampled)

        return pd.concat(balanced_parts, axis=0, ignore_index=True)

    def _determine_generation_size(self) -> int:
        """Determine a larger raw simulation pool before class balancing."""

        if not self.enable_balancing:
            return self.num_samples
        return max(self.num_samples * 3, 3000)

    @staticmethod
    def _validate_required_columns(dataset: pd.DataFrame, required_columns: list[str]) -> None:
        """Ensure that a dataset contains all required columns."""

        missing_columns = [column for column in required_columns if column not in dataset.columns]
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    @staticmethod
    def _clip(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
        """Clip values to a valid range."""

        return np.clip(values, lower, upper)

    @staticmethod
    def _normalize(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
        """Normalize values to the range [0, 1]."""

        values = np.asarray(values, dtype=float)
        return np.clip((values - lower) / (upper - lower), 0.0, 1.0)

    @staticmethod
    def _scaled_inverse(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
        """Return the inverted normalized scale so larger values reduce penalty."""

        return 1.0 - StressDatasetGenerator._normalize(values, lower, upper)


if __name__ == "__main__":

    generator = StressDatasetGenerator(
        num_samples=5000,
        random_state=42
    )

    dataset = generator.generate_dataset()

    print(dataset.head())

    print(generator.dataset_summary(dataset))

    # SAVE DATASET
    generator.save_dataset(
        dataset,
        "data/stress_dataset.csv"
    )

    print("\nDataset saved at: data/stress_dataset.csv")
