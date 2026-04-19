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
print("FIXED PREDICTION TEST - Using Reversed Class Labels")
print("=" * 70)

# Test with a few samples
test_samples = [
    ('Extreme stress (expected High)', dataset[dataset['stress_level'] == 'High'].iloc[0]),
    ('Healthy (expected Low)', dataset[dataset['stress_level'] == 'Low'].iloc[0]),
    ('Balanced (expected Medium)', dataset[dataset['stress_level'] == 'Medium'].iloc[0]),
]

for desc, sample in test_samples:
    test_df = pd.DataFrame([sample])
    
    # Original class labels from model
    original_labels = getattr(model, "class_labels", ["Low", "Medium", "High"])
    
    # FIXED: Reverse the class labels to correct the inversion
    fixed_labels = list(reversed(original_labels))
    
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
    predicted_idx = np.argmax(probs)
    predicted_label = fixed_labels[predicted_idx]
    actual_label = sample['stress_level']
    
    # Probability map with fixed labels
    probability_map = {label: float(probs[i]) for i, label in enumerate(fixed_labels)}
    
    # Stress score with correct mapping
    stress_score = probability_map.get("Medium", 0.0) * 0.5 + probability_map.get("High", 0.0)
    
    match = "✓ CORRECT" if predicted_label == actual_label else "✗ WRONG"
    
    print(f"\n{desc}")
    print(f"  Actual: {actual_label}")
    print(f"  Original class order: {original_labels}")
    print(f"  Fixed class order: {fixed_labels}")
    print(f"  Raw probabilities: {probs}")
    print(f"  Probability map (FIXED): {probability_map}")
    print(f"  Predicted: {predicted_label} (prob {probs[predicted_idx]:.3f})")
    print(f"  Stress score: {stress_score:.2f}")
    print(f"  Match: {match}")
