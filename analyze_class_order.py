import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('artifacts/model.pkl')
preprocessor = joblib.load('artifacts/preprocessor.pkl')
feature_engineer = joblib.load('artifacts/feature_engineer.pkl')

# The data
dataset = pd.read_csv('artifacts/stress_dataset.csv')

ALL_FEATURE_COLUMNS = [
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

def derive_missing_features(input_frame):
    """Derive the additional features expected by the saved pipeline."""
    enriched = input_frame.copy()
    enriched["caffeine_intake"] = np.clip(
        0.22 * enriched["screen_time"] + 0.15 * enriched["work_hours"] + 0.08 * enriched["mental_fatigue_score"],
        0.0,
        8.0,
    )
    enriched["social_interaction_hours"] = np.clip(
        4.5 - 0.18 * enriched["work_hours"] - 0.10 * enriched["mental_fatigue_score"] + 0.25 * enriched["physical_activity_hours"],
        0.0,
        8.0,
    )
    enriched["work_pressure_score"] = np.clip(
        0.45 * enriched["work_hours"] + 0.35 * enriched["mental_fatigue_score"] + 0.03 * (enriched["heart_rate"] - 60.0),
        0.0,
        10.0,
    )
    return enriched[ALL_FEATURE_COLUMNS]

print("=" * 70)
print("ANALYZING TRUE CLASS ORDER FROM PROBABILITIES")
print("=" * 70)

# Test with a few samples
test_samples = [
    ('Extreme stress (expected High)', dataset[dataset['stress_level'] == 'High'].iloc[0]),
    ('Healthy (expected Low)', dataset[dataset['stress_level'] == 'Low'].iloc[0]),
    ('Balanced (expected Medium)', dataset[dataset['stress_level'] == 'Medium'].iloc[0]),
]

all_probs = []
for desc, sample in test_samples:
    test_df = pd.DataFrame([sample])
    
    # Preprocess
    test_df_prep = test_df.copy()
    test_df_prep['stress_level'] = 'Low'
    X_processed, _ = preprocessor.transform(test_df_prep, encode_target=False, drop_outliers=False)
    
    # Engineer features
    X_processed = X_processed.reindex(columns=getattr(feature_engineer, 'feature_columns', X_processed.columns))
    X_engineered = feature_engineer.transform(X_processed)
    X_engineered = X_engineered.reindex(columns=getattr(model, 'feature_names', X_engineered.columns.tolist()))
    
    # Predict
    probs = model.predict_proba(X_engineered.to_numpy())[0]
    actual_label = sample['stress_level']
    max_idx = np.argmax(probs)
    
    all_probs.append((actual_label, probs, max_idx))
    
    print(f"\n{desc}")
    print(f"  Actual: {actual_label}")
    print(f"  Probabilities at indices [0, 1, 2]: {probs}")
    print(f"  Max probability {probs[max_idx]:.4f} at index {max_idx}")

print("\n" + "=" * 70)
print("DEDUCED CLASS ORDER")
print("=" * 70)

deduced_order = [None, None, None]

# High sample: max at index 0 → index 0 = High
high_label, high_probs, high_idx = all_probs[0]
deduced_order[high_idx] = "High"
print(f"From High sample: index {high_idx} = High")

# Low sample: max at index 1 → index 1 = Low
low_label, low_probs, low_idx = all_probs[1]
deduced_order[low_idx] = "Low"
print(f"From Low sample: index {low_idx} = Low")

# Medium sample: max at index 2 → index 2 = Medium
medium_label, medium_probs, medium_idx = all_probs[2]
deduced_order[medium_idx] = "Medium"
print(f"From Medium sample: index {medium_idx} = Medium")

print(f"\nDeduced class order: {deduced_order}")

print("\n" + "=" * 70)
print("VERIFICATION WITH DEDUCED ORDER")
print("=" * 70)

for desc, (actual_label, probs, max_idx) in zip(
    ['Extreme stress', 'Healthy', 'Balanced'],
    all_probs
):
    predicted_label = deduced_order[max_idx]
    match = "✓" if predicted_label == actual_label else "✗"
    print(f"{desc}: actual={actual_label}, predicted={predicted_label} {match}")
