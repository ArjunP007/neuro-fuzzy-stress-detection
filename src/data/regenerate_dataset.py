"""Regenerate a better-distributed stress dataset for model training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    """Generate and save a balanced synthetic stress dataset."""

    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "artifacts" / "stress_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n = 4000

    sleep = np.clip(rng.normal(7.0, 1.0, n), 4.0, 10.0)
    work = np.clip(rng.normal(8.0, 2.0, n), 2.0, 14.0)
    screen = np.clip(rng.normal(5.0, 2.0, n), 0.0, 12.0)
    activity = np.clip(rng.normal(1.5, 1.0, n), 0.0, 5.0)
    fatigue = np.clip(rng.normal(5.0, 2.0, n), 0.0, 10.0)
    hr = np.clip(rng.normal(75.0, 10.0, n), 55.0, 110.0)

    caffeine = np.clip(0.3 * screen + rng.normal(0.5, 0.5, n), 0.0, 6.0)
    social = np.clip(4.0 - 0.2 * work + 0.3 * activity + rng.normal(0.0, 0.5, n), 0.0, 6.0)
    pressure = np.clip(0.5 * work + 0.4 * fatigue + rng.normal(0.0, 1.0, n), 0.0, 10.0)

    sleep_n = (sleep - 6.5) / 2.0
    work_n = (work - 8.0) / 3.0
    fatigue_n = fatigue / 10.0
    screen_n = screen / 10.0
    activity_n = activity / 3.0
    hr_n = (hr - 70.0) / 25.0
    social_n = social / 5.0
    caffeine_n = caffeine / 5.0

    noise = rng.normal(0.0, 0.15, n)

    raw_stress_score = (
        1.2 * fatigue_n
        + 1.0 * work_n
        + 0.6 * screen_n
        + 0.5 * hr_n
        + 0.3 * caffeine_n
        - 1.1 * sleep_n
        - 0.8 * activity_n
        - 0.4 * social_n
    ) + noise
    raw_min = float(np.min(raw_stress_score))
    raw_max = float(np.max(raw_stress_score))
    stress_score = (raw_stress_score - raw_min) / max(raw_max - raw_min, 1e-8)
    stress_score = np.clip(stress_score, 0.0, 1.0) * 10.0

    q_low = float(np.quantile(stress_score, 0.35))
    q_high = float(np.quantile(stress_score, 0.70))
    stress_level = pd.cut(
        stress_score,
        bins=[-np.inf, q_low, q_high, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    df = pd.DataFrame(
        {
            "sleep_hours": sleep,
            "work_hours": work,
            "screen_time": screen,
            "physical_activity_hours": activity,
            "mental_fatigue_score": fatigue,
            "heart_rate": hr,
            "caffeine_intake": caffeine,
            "social_interaction_hours": social,
            "work_pressure_score": pressure,
            "stress_score": stress_score,
            "stress_level": stress_level.astype(str),
        }
    )

    df.to_csv(output_path, index=False)

    print(df["stress_level"].value_counts())


if __name__ == "__main__":
    main()
