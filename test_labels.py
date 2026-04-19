import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('artifacts/model.pkl')
preprocessor = joblib.load('artifacts/preprocessor.pkl')
feature_engineer = joblib.load('artifacts/feature_engineer.pkl')

# The data
dataset = pd.read_csv('artifacts/stress_dataset.csv')

print("=" * 60)
print("Model class_labels:", model.class_labels)
print("=" * 60)

# Test with a few samples
test_samples = [
    ('Extreme stress (expected High)', dataset[dataset['stress_level'] == 'High'].iloc[0]),
    ('Healthy (expected Low)', dataset[dataset['stress_level'] == 'Low'].iloc[0]),
    ('Balanced (expected Medium)', dataset[dataset['stress_level'] == 'Medium'].iloc[0]),
]

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
    predicted_idx = np.argmax(probs)
    predicted_label = model.class_labels[predicted_idx]
    actual_label = sample['stress_level']
    
    print(f"\n{desc}")
    print(f"  Actual: {actual_label}")
    print(f"  Probabilities (order {model.class_labels}): {probs}")
    print(f"  Predicted: {predicted_label} (index {predicted_idx}, prob {probs[predicted_idx]:.3f})")
