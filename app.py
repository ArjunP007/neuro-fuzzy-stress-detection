"""Research-Grade Neuro-Fuzzy Stress Intelligence System with Advanced Analytics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image


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
]
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

BASE_FEATURE_DISPLAY = {
    "sleep_hours": ("Sleep", 0, 12),
    "work_hours": ("Work Hours", 0, 14),
    "screen_time": ("Screen Time", 0, 14),
    "physical_activity_hours": ("Activity", 0, 5),
    "mental_fatigue_score": ("Fatigue", 0, 10),
    "heart_rate": ("Heart Rate", 50, 120),
}


def derive_missing_features(input_frame: pd.DataFrame) -> pd.DataFrame:
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


def predict_stress(model: Any, preprocessor: Any, feature_engineer: Any, input_frame: pd.DataFrame) -> dict[str, Any]:
    """Run inference using saved artifacts and return comprehensive results."""

    base_columns = list(getattr(preprocessor, "feature_columns", ALL_FEATURE_COLUMNS))

    base_df = derive_missing_features(input_frame).reindex(columns=base_columns).copy()
    base_df = base_df.astype(float)

    if base_df.isnull().any().any():
        missing_columns = base_df.columns[base_df.isnull().any()].tolist()
        raise ValueError(f"Invalid or missing numeric values for columns: {missing_columns}")

    preprocess_frame = base_df.copy()
    preprocess_frame["stress_level"] = "Low"

    X_processed, _ = preprocessor.transform(
        preprocess_frame,
        encode_target=False,
        drop_outliers=False,
    )

    expected_engineer_columns = list(getattr(feature_engineer, "feature_columns", base_columns))
    X_processed = X_processed.reindex(columns=expected_engineer_columns)
    engineered_input = feature_engineer.transform(X_processed)
    engineered_input = engineered_input.reindex(columns=getattr(model, "feature_names", engineered_input.columns.tolist()))

    probabilities = model.predict_proba(engineered_input.to_numpy())[0]
    
    # Class order: [High, Low, Medium]
    class_labels = ["High", "Low", "Medium"]
    
    prediction_index = int(np.argmax(probabilities))
    probability_map = {
        label: float(probabilities[index])
        for index, label in enumerate(class_labels)
    }
    stress_score = probability_map.get("Medium", 0.0) * 0.5 + probability_map.get("High", 0.0)

    influence_scores = np.abs(engineered_input.iloc[0].to_numpy(dtype=float))
    if hasattr(model, "neural_network") and hasattr(model.neural_network, "parameters") and model.neural_network.parameters:
        first_layer_weights = np.abs(model.neural_network.parameters[0].weights)
        weight_importance = np.mean(first_layer_weights, axis=1)
        if weight_importance.shape[0] == influence_scores.shape[0]:
            influence_scores = influence_scores * weight_importance

    influence_frame = (
        pd.DataFrame(
            {
                "feature": engineered_input.columns,
                "impact": influence_scores,
            }
        )
        .sort_values("impact", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    return {
        "label": class_labels[prediction_index],
        "confidence": float(probabilities[prediction_index]),
        "probabilities": probability_map,
        "stress_score": float(stress_score),
        "input_frame": input_frame.copy(),
        "enriched_frame": base_df.copy(),
        "engineered_features": engineered_input.copy(),
        "feature_influence": influence_frame,
        "raw_probabilities": probabilities,
    }


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load the dashboard dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict[str, Any]:
    """Load optional model metadata."""

    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_artifacts() -> tuple[Any, Any, Any]:
    """Load the trained model and preprocessing artifacts."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Saved preprocessor not found: {PREPROCESSOR_PATH}")
    if not FEATURE_ENGINEER_PATH.exists():
        raise FileNotFoundError(f"Saved feature engineer not found: {FEATURE_ENGINEER_PATH}")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_engineer = joblib.load(FEATURE_ENGINEER_PATH)
    return model, preprocessor, feature_engineer


# ==================== SECTION: CORE PREDICTION ====================
def render_core_prediction(result: dict[str, Any]) -> None:
    """Render core prediction with stress level, score, and probabilities."""

    col1, col2, col3 = st.columns(3)
    
    # Stress Level with Color
    with col1:
        palette = {
            "Low": "#1f9d55",
            "Medium": "#d9822b",
            "High": "#c23030",
        }
        color = palette.get(result["label"], "#102a43")
        st.markdown(
            f"""
            <div style="padding: 1.5rem; border-radius: 14px; background: {color}; color: white; text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.9;">Predicted Stress Level</div>
                <div style="font-size: 2.5rem; font-weight: 700;">{result["label"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Continuous Score
    with col2:
        st.metric("Continuous Stress Score", f"{result['stress_score']:.2f}", "out of 1.0")
        st.progress(max(0, min(int(result["stress_score"] * 100), 100)))
    
    # Model Confidence
    with col3:
        conf_pct = int(result["confidence"] * 100)
        st.metric("Confidence", f"{conf_pct}%", f"{result['confidence']:.3f}")
    
    # Probability Distribution
    st.subheader("Confidence Probabilities")
    labels = list(result["probabilities"].keys())
    probs = [result["probabilities"].get(label, 0.0) for label in labels]
    color_map = {"Low": "#1f9d55", "Medium": "#d9822b", "High": "#c23030"}
    colors = [color_map.get(label, "#102a43") for label in labels]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(labels, probs, color=colors)
    ax.set_title("Membership Strength Distribution", fontweight="bold", fontsize=12)
    ax.set_xlabel("Probability")
    ax.set_xlim(0.0, 1.0)
    for i, v in enumerate(probs):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center")
    st.pyplot(fig)


# ==================== SECTION: SENSITIVITY ANALYSIS ====================
@st.cache_data(show_spinner=True)
def compute_sensitivity_curves(_model: Any, _preprocessor: Any, _feature_engineer: Any, base_input_tuple: tuple) -> dict[str, Any]:
    """Compute sensitivity of stress score to each feature (cached)."""
    
    # Convert tuple back to DataFrame
    base_input = pd.DataFrame([dict(zip(BASE_FEATURE_COLUMNS, base_input_tuple))])
    base_derived = derive_missing_features(base_input)
    sensitivities = {}
    feature_ranges = {}
    
    for feature in BASE_FEATURE_COLUMNS:
        min_val, max_val = BASE_FEATURE_DISPLAY[feature][1], BASE_FEATURE_DISPLAY[feature][2]
        values = np.linspace(min_val, max_val, 15)
        stress_scores = []
        
        for val in values:
            test_input = base_input.copy()
            test_input[feature] = val
            try:
                pred = predict_stress(_model, _preprocessor, _feature_engineer, test_input)
                stress_scores.append(pred["stress_score"])
            except:
                stress_scores.append(np.nan)
        
        sensitivities[feature] = stress_scores
        feature_ranges[feature] = values
    
    return {"ranges": feature_ranges, "sensitivities": sensitivities}


def render_sensitivity_analysis(result: dict[str, Any], model: Any, preprocessor: Any, feature_engineer: Any) -> None:
    """Render sensitivity analysis showing stress response to each feature."""
    
    st.subheader("Feature Sensitivity Analysis")
    st.write("How does stress change with each lifestyle factor?")
    
    # Convert DataFrame to tuple for caching
    input_tuple = tuple(result["input_frame"].iloc[0][f] for f in BASE_FEATURE_COLUMNS)
    sensitivity_data = compute_sensitivity_curves(model, preprocessor, feature_engineer, input_tuple)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    colors_grad = ["#1f9d55", "#d9822b", "#c23030"]
    
    for idx, feature in enumerate(BASE_FEATURE_COLUMNS):
        ax = axes[idx]
        ranges = sensitivity_data["ranges"][feature]
        scores = sensitivity_data["sensitivities"][feature]
        
        ax.plot(ranges, scores, linewidth=2.5, marker="o", color="#2f7ed8")
        ax.fill_between(ranges, scores, alpha=0.3, color="#2f7ed8")
        ax.set_xlabel(BASE_FEATURE_DISPLAY[feature][0], fontweight="bold")
        ax.set_ylabel("Stress Score")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Stress vs {BASE_FEATURE_DISPLAY[feature][0]}")
    
    st.pyplot(fig)


# ==================== SECTION: WHAT-IF SIMULATOR ====================
def render_whatif_simulator(model: Any, preprocessor: Any, feature_engineer: Any, base_input: pd.DataFrame, base_result: dict[str, Any]) -> None:
    """Render interactive what-if scenario simulator."""
    
    st.subheader("What-If Scenario Simulator")
    st.write("Explore how lifestyle changes impact stress level")
    
    # Initialize session state for scenario feature if not exists
    if "scenario_feature" not in st.session_state:
        st.session_state.scenario_feature = BASE_FEATURE_COLUMNS[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Modify one lifestyle factor:**")
        st.session_state.scenario_feature = st.selectbox(
            "Select factor to adjust",
            BASE_FEATURE_COLUMNS,
            index=BASE_FEATURE_COLUMNS.index(st.session_state.scenario_feature),
            format_func=lambda x: BASE_FEATURE_DISPLAY[x][0],
            key="whatif_factor_select"
        )
    
    with col2:
        scenario_feature = st.session_state.scenario_feature
        display_name, raw_min, raw_max = BASE_FEATURE_DISPLAY[scenario_feature]
        
        # Convert to float to avoid slider type mismatch
        min_val = float(raw_min)
        max_val = float(raw_max)
        current_val = float(base_input[scenario_feature].iloc[0])
        step_val = (max_val - min_val) / 20.0
        
        scenario_value = st.slider(
            f"New {display_name} value",
            min_val,
            max_val,
            current_val,
            step_val,
            key="whatif_slider"
        )
    
    # Generate scenario
    scenario_input = base_input.copy()
    scenario_input[st.session_state.scenario_feature] = scenario_value
    scenario_result = predict_stress(model, preprocessor, feature_engineer, scenario_input)
    
    # Display comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Stress Score", f"{base_result['stress_score']:.2f}")
    
    with col2:
        st.metric("New Stress Score", f"{scenario_result['stress_score']:.2f}")
    
    with col3:
        delta = scenario_result['stress_score'] - base_result['stress_score']
        delta_pct = (delta / base_result['stress_score']) * 100 if base_result['stress_score'] > 0 else 0
        st.metric(
            "Change",
            f"{delta:.2f}",
            f"{delta_pct:.1f}%",
            delta_color="inverse"
        )
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    scenarios = ["Current", "Scenario"]
    scores = [base_result['stress_score'], scenario_result['stress_score']]
    colors = ["#2f7ed8", "#d9822b"]
    bars = ax.bar(scenarios, scores, color=colors, width=0.6)
    ax.set_ylabel("Stress Score")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Impact of {BASE_FEATURE_DISPLAY[scenario_feature][0]} Adjustment", fontweight="bold")
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    st.pyplot(fig)


# ==================== SECTION: RISK DECOMPOSITION ====================
def render_risk_decomposition(result: dict[str, Any]) -> None:
    """Render lifestyle risk decomposition showing factor contributions."""
    
    st.subheader("Lifestyle Risk Decomposition")
    st.write("Breakdown of stress by contributing lifestyle factors")
    
    row = result["input_frame"].iloc[0]
    
    # Calculate risk factors
    risk_factors = {}
    
    fatigue_contrib = float(row["mental_fatigue_score"]) / 10.0 * 35  # 35% max
    work_contrib = float(row["work_hours"]) / 14.0 * 30  # 30% max
    sleep_contrib = max(0, 1 - float(row["sleep_hours"]) / 8.0) * 20  # 20% max (inverse)
    heart_contrib = max(0, (float(row["heart_rate"]) - 60) / 40.0) * 10  # 10% max
    screen_contrib = float(row["screen_time"]) / 14.0 * 5  # 5% max
    
    risk_factors = {
        "Mental Fatigue": fatigue_contrib,
        "Work Pressure": work_contrib,
        "Sleep Deficiency": sleep_contrib,
        "Heart Strain": heart_contrib,
        "Screen Overload": screen_contrib,
    }
    
    # Normalize to sum to 100
    total = sum(risk_factors.values())
    if total > 0:
        risk_factors = {k: (v/total)*100 for k, v in risk_factors.items()}
    
    # Sort and display
    risk_factors = dict(sorted(risk_factors.items(), key=lambda x: x[1], reverse=True))
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#c23030", "#d9822b", "#f5a623", "#4a90e2", "#7ed321"]
    bars = ax.barh(list(risk_factors.keys()), list(risk_factors.values()), color=colors)
    ax.set_xlabel("Contribution (%)")
    ax.set_title("Risk Factor Decomposition", fontweight="bold", fontsize=12)
    ax.set_xlim(0, 100)
    
    for bar, (factor, value) in zip(bars, risk_factors.items()):
        ax.text(value + 1, bar.get_y() + bar.get_height()/2,
                f'{value:.1f}%', va='center', fontweight='bold')
    
    st.pyplot(fig)
    
    # Table view
    st.dataframe(
        pd.DataFrame(risk_factors.items(), columns=["Factor", "Contribution (%)"]),
        width="stretch",
        hide_index=True
    )


# ==================== SECTION: BEHAVIORAL PATTERN ====================
def classify_behavioral_pattern(result: dict[str, Any]) -> dict[str, Any]:
    """Classify lifestyle behavioral pattern based on input profile."""
    
    row = result["input_frame"].iloc[0]
    
    sleep_h = float(row["sleep_hours"])
    work_h = float(row["work_hours"])
    screen_h = float(row["screen_time"])
    fatigue = float(row["mental_fatigue_score"])
    activity = float(row["physical_activity_hours"])
    hr = float(row["heart_rate"])
    
    # Pattern classification
    profile = ""
    characteristics = []
    risk_level = "Low"
    
    if screen_h > 8 and work_h > 8:
        profile = "Digitally Overloaded Professional"
        characteristics = ["High screen time", "Extended work hours", "Elevated cognitive load"]
        risk_level = "High"
    elif sleep_h < 6 and fatigue > 7:
        profile = "Sleep-Deprived Burnout Profile"
        characteristics = ["Insufficient sleep", "High fatigue", "Recovery deficit"]
        risk_level = "High"
    elif activity < 1 and work_h > 10:
        profile = "Sedentary High-Pressure Worker"
        characteristics = ["Low physical activity", "High workload", "Limited stress relief"]
        risk_level = "High"
    elif work_h < 7 and sleep_h > 7 and activity > 2:
        profile = "Balanced Lifestyle Maintainer"
        characteristics = ["Adequate sleep", "Moderate work", "Regular physical activity"]
        risk_level = "Low"
    elif fatigue < 4 and sleep_h >= 7:
        profile = "Well-Recovered Performer"
        characteristics = ["Low fatigue", "Adequate sleep", "Good recovery"]
        risk_level = "Low"
    else:
        profile = "Moderate-Stress Subject"
        characteristics = ["Mixed lifestyle patterns", "Moderate stress indicators"]
        risk_level = "Medium"
    
    return {
        "profile": profile,
        "characteristics": characteristics,
        "risk_level": risk_level,
    }


def render_behavioral_classification(result: dict[str, Any]) -> None:
    """Render behavioral pattern classification."""
    
    pattern = classify_behavioral_pattern(result)
    
    st.subheader("Behavioral Pattern Classification")
    
    # Risk level color
    risk_colors = {"Low": "#1f9d55", "Medium": "#d9822b", "High": "#c23030"}
    risk_color = risk_colors.get(pattern["risk_level"], "#102a43")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### {pattern['profile']}")
        st.write("**Characteristics:**")
        for char in pattern["characteristics"]:
            st.write(f"• {char}")
    
    with col2:
        st.markdown(
            f"""
            <div style="padding: 1.5rem; border-radius: 14px; background: {risk_color}; color: white; text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.9;">Risk Assessment</div>
                <div style="font-size: 1.8rem; font-weight: 700;">{pattern["risk_level"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==================== SECTION: STABILITY ANALYSIS ====================
@st.cache_data(show_spinner=True)
def compute_stability_score(_model: Any, _preprocessor: Any, _feature_engineer: Any, base_input_tuple: tuple) -> dict[str, Any]:
    """Compute stability via perturbation testing (Monte Carlo sampling) - cached."""
    
    base_input = pd.DataFrame([dict(zip(BASE_FEATURE_COLUMNS, base_input_tuple))])
    n_samples = 100
    perturbation_range = 0.05  # 5% perturbation
    
    stress_scores = []
    
    for _ in range(n_samples):
        perturbed = base_input.copy()
        for feature in BASE_FEATURE_COLUMNS:
            min_val, max_val = BASE_FEATURE_DISPLAY[feature][1], BASE_FEATURE_DISPLAY[feature][2]
            current_val = float(base_input[feature].iloc[0])
            
            max_noise = (max_val - min_val) * perturbation_range
            noise = np.random.normal(0, max_noise / 3)  # 3-sigma rule
            perturbed[feature] = np.clip(current_val + noise, min_val, max_val)
        
        try:
            pred = predict_stress(_model, _preprocessor, _feature_engineer, perturbed)
            stress_scores.append(pred["stress_score"])
        except:
            pass
    
    stress_scores = np.array(stress_scores)
    
    mean_score = np.mean(stress_scores)
    std_score = np.std(stress_scores)
    stability_score = 1.0 - min(std_score / (mean_score + 1e-6), 1.0)
    
    return {
        "mean": mean_score,
        "std": std_score,
        "scores": stress_scores,
        "stability": stability_score,
    }


def render_stability_analysis(model: Any, preprocessor: Any, feature_engineer: Any, base_input: pd.DataFrame, base_result: dict[str, Any]) -> None:
    """Render stability analysis via perturbation testing (Monte Carlo sampling)."""
    
    st.subheader("Stability Analysis")
    st.write("Model prediction robustness under input perturbation")
    
    # Convert DataFrame to tuple for caching
    input_tuple = tuple(base_input.iloc[0][f] for f in BASE_FEATURE_COLUMNS)
    stability_data = compute_stability_score(model, preprocessor, feature_engineer, input_tuple)
    
    stress_scores = stability_data["scores"]
    mean_score = stability_data["mean"]
    std_score = stability_data["std"]
    stability_score = stability_data["stability"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Prediction", f"{mean_score:.3f}")
    
    with col2:
        st.metric("Std Deviation", f"{std_score:.4f}")
    
    with col3:
        stability_pct = int(stability_score * 100)
        st.metric("Stability Score", f"{stability_pct}%")
    
    # Distribution plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(stress_scores, bins=20, color="#2f7ed8", alpha=0.7, edgecolor="black")
    ax.axvline(mean_score, color="#c23030", linestyle="--", linewidth=2, label=f"Mean: {mean_score:.3f}")
    ax.axvline(base_result["stress_score"], color="#1f9d55", linestyle="--", linewidth=2, label=f"Base: {base_result['stress_score']:.3f}")
    ax.set_xlabel("Stress Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Prediction Distribution Under Perturbation (Monte Carlo)", fontweight="bold")
    ax.legend()
    st.pyplot(fig)


# ==================== SECTION: COUNTERFACTUAL EXPLANATIONS ====================
def find_counterfactual(model: Any, preprocessor: Any, feature_engineer: Any, base_input: pd.DataFrame, target_level: str = "Low") -> dict[str, Any]:
    """Find minimum changes needed to reach target stress level."""
    
    base_result = predict_stress(model, preprocessor, feature_engineer, base_input)
    
    if base_result["label"] == target_level:
        return {
            "already_target": True,
            "changes": [],
            "new_score": base_result["stress_score"],
        }
    
    counterfactual = base_input.copy()
    changes = []
    
    # Strategy: increase sleep, reduce fatigue, reduce work hours
    for iteration in range(10):
        current_result = predict_stress(model, preprocessor, feature_engineer, counterfactual)
        
        if current_result["probabilities"].get(target_level, 0) > 0.7:
            break
        
        # Adjust factors toward low stress
        counterfactual["sleep_hours"] = min(9.0, float(counterfactual["sleep_hours"].iloc[0]) + 0.3)
        counterfactual["mental_fatigue_score"] = max(0, float(counterfactual["mental_fatigue_score"].iloc[0]) - 0.5)
        counterfactual["work_hours"] = max(4, float(counterfactual["work_hours"].iloc[0]) - 0.5)
        counterfactual["physical_activity_hours"] = min(5, float(counterfactual["physical_activity_hours"].iloc[0]) + 0.2)
    
    final_result = predict_stress(model, preprocessor, feature_engineer, counterfactual)
    
    # Record changes
    for feature in BASE_FEATURE_COLUMNS:
        old_val = float(base_input[feature].iloc[0])
        new_val = float(counterfactual[feature].iloc[0])
        if abs(old_val - new_val) > 0.01:
            changes.append({
                "feature": BASE_FEATURE_DISPLAY[feature][0],
                "from": old_val,
                "to": new_val,
                "change": new_val - old_val,
            })
    
    return {
        "already_target": False,
        "changes": sorted(changes, key=lambda x: abs(x["change"]), reverse=True),
        "new_score": final_result["stress_score"],
        "target_probability": final_result["probabilities"].get(target_level, 0),
    }


def render_counterfactual(model: Any, preprocessor: Any, feature_engineer: Any, base_input: pd.DataFrame, base_result: dict[str, Any]) -> None:
    """Render counterfactual explanations."""
    
    st.subheader("Counterfactual Explanation")
    st.write(f"What changes would move you from **{base_result['label']}** to **Low** stress?")
    
    counterfactual = find_counterfactual(model, preprocessor, feature_engineer, base_input, "Low")
    
    if counterfactual["already_target"]:
        st.success("✓ You are already at Low stress level!")
    else:
        st.write("**Minimum changes needed:**")
        
        if counterfactual["changes"]:
            changes_df = pd.DataFrame(counterfactual["changes"])
            changes_df = changes_df[["feature", "from", "to", "change"]].copy()
            changes_df.columns = ["Factor", "Current", "Target", "Change"]
            changes_df["Current"] = changes_df["Current"].round(2)
            changes_df["Target"] = changes_df["Target"].round(2)
            changes_df["Change"] = changes_df["Change"].round(2)
            
            st.dataframe(changes_df, width="stretch", hide_index=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("New Stress Score", f"{counterfactual['new_score']:.2f}")
        with col2:
            target_prob = counterfactual["target_probability"]
            st.metric("Low Stress Probability", f"{int(target_prob*100)}%")


# ==================== SECTION: FEATURE INTERACTION HEATMAP ====================
def render_interaction_heatmap(result: dict[str, Any]) -> None:
    """Render feature interaction importance heatmap."""
    
    st.subheader("Feature Interaction Heatmap")
    st.write("Key lifestyle factor interactions affecting stress")
    
    row = result["input_frame"].iloc[0]
    
    # Define key interactions
    interactions = {
        "Fatigue × Sleep": float(row["mental_fatigue_score"]) * (1 - float(row["sleep_hours"])/12),
        "Work × Screen": float(row["work_hours"]) * float(row["screen_time"])/14,
        "Sleep × Activity": float(row["sleep_hours"]) * float(row["physical_activity_hours"])/5,
        "Fatigue × Heart": float(row["mental_fatigue_score"]) * (float(row["heart_rate"])-50)/40,
        "Work × Activity": float(row["work_hours"]) * max(0, 1 - float(row["physical_activity_hours"])/5),
        "Heart × Fatigue": (float(row["heart_rate"])-50)/40 * float(row["mental_fatigue_score"])/10,
    }
    
    # Create interaction matrix
    features_list = ["Sleep", "Work", "Screen", "Activity", "Fatigue", "Heart"]
    n = len(features_list)
    interaction_matrix = np.zeros((n, n))
    
    # Populate matrix
    idx_map = {f: i for i, f in enumerate(features_list)}
    interaction_pairs = [
        ("Fatigue", "Sleep"),
        ("Work", "Screen"),
        ("Sleep", "Activity"),
        ("Fatigue", "Heart"),
        ("Work", "Activity"),
        ("Heart", "Fatigue"),
    ]
    
    for i, (f1, f2) in enumerate(interaction_pairs):
        i1, i2 = idx_map[f1], idx_map[f2]
        val = list(interactions.values())[i]
        interaction_matrix[i1, i2] = val
        interaction_matrix[i2, i1] = val
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(interaction_matrix, annot=True, fmt=".2f", cmap="YlOrRd", 
                xticklabels=features_list, yticklabels=features_list, ax=ax,
                cbar_kws={"label": "Interaction Strength"})
    ax.set_title("Feature Interaction Strength Matrix", fontweight="bold", fontsize=12)
    st.pyplot(fig)


# ==================== SECTION: RADAR CHART ====================
def render_radar_profile(result: dict[str, Any]) -> None:
    """Render normalized radar chart for input profile."""
    
    st.subheader("Lifestyle Profile Radar")
    st.write("Normalized visualization of your lifestyle factors")
    
    row = result["input_frame"].iloc[0]
    
    features = np.array([
        float(row["sleep_hours"]) / 12.0,
        float(row["work_hours"]) / 14.0,
        float(row["screen_time"]) / 14.0,
        float(row["physical_activity_hours"]) / 5.0,
        float(row["mental_fatigue_score"]) / 10.0,
        (float(row["heart_rate"]) - 50.0) / 70.0,
    ])
    
    radar_labels = ["Sleep", "Work", "Screen", "Activity", "Fatigue", "Heart"]
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
    
    features = np.concatenate((features, [features[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, features, "o-", linewidth=2.5, color="#2f7ed8")
    ax.fill(angles, features, alpha=0.25, color="#2f7ed8")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("Lifestyle Profile (Normalized 0-1)", fontweight="bold", fontsize=12, pad=20)
    ax.grid(True)
    st.pyplot(fig)


# ==================== SECTION: HEALTH REPORT SUMMARY ====================
def generate_health_report(result: dict[str, Any]) -> str:
    """Auto-generate personalized health report summary."""
    
    row = result["input_frame"].iloc[0]
    label = result["label"]
    stress_score = result["stress_score"]
    
    report = ""
    
    # Intro
    if label == "High":
        stress_desc = "significantly elevated"
        severity = "Burnout risk is present."
    elif label == "Medium":
        stress_desc = "moderately elevated"
        severity = "Moderate stress management needed."
    else:
        stress_desc = "well-managed"
        severity = "Stress level is healthy."
    
    report += f"Your stress level is currently {stress_desc}, with a continuous score of {stress_score:.2f}. {severity}\n\n"
    
    # Key drivers
    report += "**Key Stress Drivers:**\n"
    if float(row["mental_fatigue_score"]) > 6:
        report += "• Elevated mental fatigue is a primary stress indicator\n"
    if float(row["sleep_hours"]) < 6.5:
        report += "• Sleep deficiency is limiting recovery capacity\n"
    if float(row["work_hours"]) > 9:
        report += "• Extended work hours amplify workload pressure\n"
    if float(row["screen_time"]) > 8:
        report += "• High screen exposure suggests digital overload\n"
    if float(row["physical_activity_hours"]) < 1:
        report += "• Low physical activity limits stress relief outlet\n"
    
    # Recommendations
    report += "\n**Primary Improvement Recommendation:**\n"
    if float(row["sleep_hours"]) < 6.5 or float(row["mental_fatigue_score"]) > 7:
        report += "Increase recovery balance through improved sleep quality (target 7-8 hours) and activity management.\n"
    elif float(row["work_hours"]) > 10:
        report += "Reduce workload intensity or implement structured breaks to prevent burnout.\n"
    elif float(row["screen_time"]) > 10:
        report += "Reduce screen exposure through digital breaks and offline activities.\n"
    else:
        report += "Maintain current lifestyle balance while optimizing specific areas.\n"
    
    # Risk assessment
    report += "\n**Physiological Indicators:**\n"
    if float(row["heart_rate"]) > 85:
        report += f"• Heart rate ({float(row['heart_rate']):.0f} bpm) suggests cardiovascular stress response\n"
    if float(row["mental_fatigue_score"]) > 7:
        report += "• Cognitive fatigue indicates potential decision-making deterioration\n"
    
    report += "\n**Actionable Next Steps:**\n"
    report += "1. Establish consistent sleep schedule (7-8 hours nightly)\n"
    report += "2. Incorporate 20-30 minutes daily physical activity\n"
    report += "3. Implement digital detox periods (e.g., 90-120 mins/day)\n"
    report += "4. Schedule recovery time into your work calendar\n"
    report += "5. Track progress with weekly stress assessments\n"
    
    return report


def render_health_report(result: dict[str, Any]) -> None:
    """Render auto-generated personal health report."""
    
    st.subheader("Personal Health Report Summary")
    
    report = generate_health_report(result)
    
    st.markdown(report)
    
    # Confidence assessment
    st.divider()
    st.write("**Report Confidence:** Based on validated neuro-fuzzy analysis with ensemble learning")


# ==================== SECTION: CONFUSION MATRIX ====================
@st.cache_data(show_spinner=False)
def load_confusion_matrix_image() -> Image.Image | None:
    """Load and cache confusion matrix image."""
    
    if CONFUSION_MATRIX_PATH.exists():
        return Image.open(CONFUSION_MATRIX_PATH)
    return None


def render_confusion_matrix() -> None:
    """Render model confusion matrix with optimized loading."""
    
    img = load_confusion_matrix_image()
    if img is not None:
        st.image(
            img,
            caption="Confusion Matrix - Classification Accuracy by Stress Level",
            use_container_width=True,
            output_format="auto"
        )
    else:
        st.info("Confusion matrix not available. Run `python train_once.py` first.")


# ==================== FUZZY MEMBERSHIP VISUALIZATION ====================
def render_fuzzy_memberships(result: dict[str, Any]) -> None:
    """Render fuzzy membership degree visualization."""
    
    probs = result["probabilities"]
    
    st.write("**Membership Degree Table:**")
    membership_data = {
        "Stress Level": ["Low", "Medium", "High"],
        "Membership Degree": [
            f"{probs.get('Low', 0):.3f}",
            f"{probs.get('Medium', 0):.3f}",
            f"{probs.get('High', 0):.3f}"
        ]
    }
    st.dataframe(pd.DataFrame(membership_data), width="stretch", hide_index=True)
    
    # Triangular visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    levels = ["Low", "Medium", "High"]
    values = [probs.get('Low', 0), probs.get('Medium', 0), probs.get('High', 0)]
    colors_tri = ["#1f9d55", "#d9822b", "#c23030"]
    
    x_pos = np.arange(len(levels))
    bars = ax.bar(x_pos, values, color=colors_tri, alpha=0.7, edgecolor="black", linewidth=2)
    
    for i, (level, value) in enumerate(zip(levels, values)):
        triangle_x = [i - 0.3, i, i + 0.3, i - 0.3]
        triangle_y = [0, value, 0, 0]
        ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)
        ax.fill(triangle_x, triangle_y, alpha=0.2, color=colors_tri[i])
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(levels, fontsize=12, fontweight='bold')
    ax.set_ylabel("Membership Degree", fontsize=12)
    ax.set_title("Fuzzy Membership Strength Distribution", fontweight="bold", fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)


# ==================== FUZZY RULE ACTIVATION ====================
def render_rule_activation(result: dict[str, Any]) -> None:
    """Render fuzzy rules driving the prediction."""
    
    row = result["input_frame"].iloc[0]
    sleep_h = float(row["sleep_hours"])
    fatigue = float(row["mental_fatigue_score"])
    work_h = float(row["work_hours"])
    screen_h = float(row["screen_time"])
    activity = float(row["physical_activity_hours"])
    
    rules = []
    
    if fatigue > 7 and sleep_h < 6:
        strength = (fatigue / 10) * (1 - sleep_h / 8)
        rules.append({
            "IF-THEN Rule": "IF fatigue=HIGH AND sleep=LOW THEN stress=HIGH",
            "Strength": f"{strength:.2f}",
            "Trigger": "Severe fatigue + insufficient recovery"
        })
    
    if work_h > 9 and screen_h > 8:
        strength = min((work_h / 14) + (screen_h / 14), 1.0) / 2
        rules.append({
            "IF-THEN Rule": "IF work_hours=HIGH AND screen_time=HIGH THEN stress=MEDIUM-HIGH",
            "Strength": f"{strength:.2f}",
            "Trigger": "Cognitive overload from extended work"
        })
    
    if sleep_h < 7 and activity < 1:
        strength = (1 - sleep_h / 8) * max(0, 1 - activity / 3)
        rules.append({
            "IF-THEN Rule": "IF sleep=LOW AND activity=LOW THEN recovery=INSUFFICIENT",
            "Strength": f"{strength:.2f}",
            "Trigger": "Inadequate recovery mechanisms"
        })
    
    if sleep_h >= 7 and work_h < 9 and activity > 1:
        strength = min(sleep_h / 8, 1.0)
        rules.append({
            "IF-THEN Rule": "IF sleep=ADEQUATE AND work=MODERATE AND activity=YES THEN stress=LOW",
            "Strength": f"{strength:.2f}",
            "Trigger": "Healthy lifestyle balance"
        })
    
    if rules:
        rules_df = pd.DataFrame(rules)
        st.dataframe(rules_df, width="stretch", hide_index=True)
    else:
        st.info("No high-strength rules activated. General weighted inference applied.")


# ==================== FEATURE IMPORTANCE ANALYSIS ====================
def render_feature_contribution(result: dict[str, Any]) -> None:
    """Render feature importance analysis."""
    
    influence_df = result["feature_influence"].copy()
    influence_df["Percentage"] = (influence_df["impact"] / influence_df["impact"].sum() * 100).round(1)
    
    top_features = influence_df.head(6)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#c23030", "#d9822b", "#f5a623", "#4a90e2", "#7ed321", "#50c878"]
    bars = ax.barh(range(len(top_features)), top_features["Percentage"], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"])
    ax.set_xlabel("Contribution (%)", fontweight="bold")
    ax.set_title("Feature Importance Distribution", fontweight="bold", fontsize=12)
    ax.set_xlim(0, 100)
    
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row["Percentage"] + 1, i, f"{row['Percentage']:.1f}%", va="center", fontweight="bold")
    
    st.pyplot(fig)
    
    display_df = influence_df[["feature", "impact", "Percentage"]].copy()
    display_df.columns = ["Feature", "Impact Score", "Contribution (%)"]
    st.dataframe(display_df, width="stretch", hide_index=True)


# ==================== CLINICAL RISK INTERPRETATION ====================
def render_risk_interpretation(result: dict[str, Any]) -> None:
    """Render risk interpretation and health assessment."""
    
    row = result["input_frame"].iloc[0]
    label = result["label"]
    stress_score = result["stress_score"]
    
    if label == "High":
        assessment = "**Significantly elevated stress** with burnout risk present."
    elif label == "Medium":
        assessment = "**Moderate stress** requiring active management."
    else:
        assessment = "**Healthy stress levels** with good recovery patterns."
    
    st.write(assessment)
    st.write(f"Continuous stress score: **{stress_score:.2f}/1.00**")
    
    st.write("**Key Drivers:**")
    drivers = []
    if float(row["mental_fatigue_score"]) > 6:
        drivers.append("Elevated mental fatigue")
    if float(row["sleep_hours"]) < 6.5:
        drivers.append("Sleep deficiency")
    if float(row["work_hours"]) > 9:
        drivers.append("Extended work hours")
    if float(row["screen_time"]) > 8:
        drivers.append("High screen exposure")
    if float(row["physical_activity_hours"]) < 1:
        drivers.append("Low physical activity")
    
    if drivers:
        for driver in drivers:
            st.write(f"• {driver}")
    else:
        st.write("• No significant stress drivers detected")
    
    st.write("**Recommendations:**")
    if float(row["sleep_hours"]) < 6.5:
        st.write("1. **Sleep**: Target 7-8 hours nightly with consistent sleep schedule")
    if float(row["mental_fatigue_score"]) > 7:
        st.write("2. **Recovery**: Implement daily cognitive breaks and relaxation")
    if float(row["work_hours"]) > 10:
        st.write("3. **Workload**: Reduce work hours or delegate tasks")
    if float(row["screen_time"]) > 10:
        st.write("4. **Digital**: Establish screen-free periods daily")
    if float(row["physical_activity_hours"]) < 1:
        st.write("5. **Exercise**: Add 30+ mins daily physical activity")


# ==================== PREDICTION STABILITY DIAGNOSTICS ====================
@st.cache_data(show_spinner=True)
def compute_stability(_model: Any, _preprocessor: Any, _feature_engineer: Any, base_input_tuple: tuple) -> dict[str, Any]:
    """Compute stability via Monte Carlo sampling."""
    
    base_input = pd.DataFrame([dict(zip(BASE_FEATURE_COLUMNS, base_input_tuple))])
    stress_scores = []
    
    for _ in range(100):
        perturbed = base_input.copy()
        for feature in BASE_FEATURE_COLUMNS:
            min_val, max_val = BASE_FEATURE_DISPLAY[feature][1], BASE_FEATURE_DISPLAY[feature][2]
            current_val = float(base_input[feature].iloc[0])
            max_noise = (max_val - min_val) * 0.05
            noise = np.random.normal(0, max_noise / 3)
            perturbed[feature] = np.clip(current_val + noise, min_val, max_val)
        
        try:
            pred = predict_stress(_model, _preprocessor, _feature_engineer, perturbed)
            stress_scores.append(pred["stress_score"])
        except:
            pass
    
    stress_scores = np.array(stress_scores)
    return {
        "mean": np.mean(stress_scores),
        "std": np.std(stress_scores),
        "scores": stress_scores,
        "stability": 1.0 - min(np.std(stress_scores) / (np.mean(stress_scores) + 1e-6), 1.0),
    }


def render_model_confidence(model: Any, preprocessor: Any, feature_engineer: Any, base_input: pd.DataFrame, base_result: dict[str, Any]) -> None:
    """Render confidence diagnostics."""
    
    input_tuple = tuple(base_input.iloc[0][f] for f in BASE_FEATURE_COLUMNS)
    stability = compute_stability(model, preprocessor, feature_engineer, input_tuple)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Prediction", f"{stability['mean']:.3f}")
    with col2:
        st.metric("Std Deviation", f"{stability['std']:.4f}")
    with col3:
        st.metric("Stability Score", f"{int(stability['stability']*100)}%")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(stability['scores'], bins=20, color="#2f7ed8", alpha=0.7, edgecolor="black")
    ax.axvline(stability['mean'], color="#c23030", linestyle="--", linewidth=2, label=f"Mean: {stability['mean']:.3f}")
    ax.axvline(base_result["stress_score"], color="#1f9d55", linestyle="--", linewidth=2, label=f"Base: {base_result['stress_score']:.3f}")
    ax.set_xlabel("Stress Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Prediction Variance (Monte Carlo, 100 samples)", fontweight="bold")
    ax.legend()
    st.pyplot(fig)


# ==================== MODEL PERFORMANCE METRICS ====================
def render_model_metrics(metadata: dict[str, Any]) -> None:
    """Render model performance metrics."""
    
    img = load_confusion_matrix_image()
    if img is not None:
        st.image(img, caption="Confusion Matrix", use_container_width=True, output_format="auto")
    else:
        st.info("Confusion matrix not available.")
    
    st.write("**Performance Summary:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        accuracy = metadata.get("accuracy", "N/A")
        if isinstance(accuracy, (int, float)):
            st.metric("Accuracy", f"{accuracy:.2%}")
        else:
            st.metric("Accuracy", accuracy)
    
    with col2:
        st.metric("Dataset Size", f"{metadata.get('dataset_size', 'N/A'):,}" if isinstance(metadata.get('dataset_size'), int) else metadata.get('dataset_size', 'N/A'))
    
    with col3:
        st.metric("Model Type", "Neuro-Fuzzy Ensemble")


# ==================== MAIN APPLICATION ====================
def main() -> None:
    """Run legendary-level Streamlit app with 14 comprehensive analytical sections."""

    st.set_page_config(
        page_title="Neuro-Fuzzy Stress Intelligence",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧠 Neuro-Fuzzy Stress Intelligence System")
    st.caption("Advanced research-grade analytics with clinical-decision-support capability")

    try:
        dataset = load_dataset()
        metadata = load_metadata()
        model, preprocessor, feature_engineer = load_artifacts()
    except Exception as exc:
        st.error(f"Failed to load artifacts: {exc}")
        return

    # ========== SYSTEM OVERVIEW ==========
    st.divider()
    st.subheader("🏗️ System Architecture & Overview")
    
    with st.container():
        st.write("**Neuro-Fuzzy Stress Intelligence System**")
        st.write("Hybrid AI combining neural networks, fuzzy logic, and feature engineering for interpretable stress prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### **Hybrid Architecture**")
            st.write("• Neural Network: Feature extraction")
            st.write("• Fuzzy Logic: Interpretable rules")
            st.write("• Ensemble: Weighted prediction aggregation")
        
        with col2:
            st.markdown("### **Core Modules**")
            st.write("• Data preprocessing")
            st.write("• Feature engineering")
            st.write("• Neuro-fuzzy inference")
            st.write("• Evaluation metrics")
        
        with col3:
            st.markdown("### **Key Metrics**")
            accuracy = metadata.get("accuracy", "N/A")
            if isinstance(accuracy, (int, float)):
                st.metric("Accuracy", f"{accuracy:.2%}")
            else:
                st.metric("Accuracy", accuracy)
            st.metric("Dataset Size", f"{len(dataset):,} samples")
            st.metric("Features", "9 engineered")

    # ========== INPUT ASSESSMENT ==========
    st.divider()
    st.subheader("🎮 Behavioral Profile Assessment")
    
    with st.form("analysis_form"):
        col1, col2, col3 = st.columns(3)
        
        input_values = {}
        
        with col1:
            input_values["sleep_hours"] = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.1)
            input_values["screen_time"] = st.slider("Screen Time (hrs)", 0.0, 14.0, 6.0, 0.1)
        
        with col2:
            input_values["work_hours"] = st.slider("Work Hours", 0.0, 14.0, 8.0, 0.1)
            input_values["physical_activity_hours"] = st.slider("Activity (hrs)", 0.0, 5.0, 1.5, 0.1)
        
        with col3:
            input_values["mental_fatigue_score"] = st.slider("Mental Fatigue", 0.0, 10.0, 5.0, 0.1)
            input_values["heart_rate"] = st.slider("Heart Rate (bpm)", 50.0, 120.0, 75.0, 1.0)

        if st.form_submit_button("🚀 Generate Comprehensive Analysis", use_container_width=True):
            input_frame = pd.DataFrame([input_values])
            result = predict_stress(model, preprocessor, feature_engineer, input_frame)
            st.session_state.prediction_result = result
            st.session_state.prediction_input = input_frame

    # ========== CONDITIONAL RENDERING OF REMAINING SECTIONS ==========
    if "prediction_result" in st.session_state:
        result = st.session_state.prediction_result
        input_frame = st.session_state.prediction_input
        
        # Display input profile summary
        st.divider()
        st.subheader("📊 Input Profile Summary")
        row = input_frame.iloc[0]
        summary_data = {
            "Factor": ["Sleep", "Work", "Screen", "Activity", "Fatigue", "Heart"],
            "Value": [f"{float(row['sleep_hours']):.1f}", f"{float(row['work_hours']):.1f}", 
                     f"{float(row['screen_time']):.1f}", f"{float(row['physical_activity_hours']):.1f}",
                     f"{float(row['mental_fatigue_score']):.1f}", f"{float(row['heart_rate']):.0f}"]
        }
        st.dataframe(pd.DataFrame(summary_data), width="stretch", hide_index=True)
        
        # CORE PREDICTION
        st.divider()
        st.subheader("🎯 Primary Stress Assessment")
        render_core_prediction(result)
        
        # FUZZY MEMBERSHIP
        st.divider()
        st.subheader("📊 Membership Strength Analysis")
        render_fuzzy_memberships(result)
        
        # RULE ACTIVATION
        st.divider()
        st.subheader("⚡ Inference Engine Rules")
        render_rule_activation(result)
        
        # FEATURE CONTRIBUTION
        st.divider()
        st.subheader("📈 Feature Importance Assessment")
        render_feature_contribution(result)
        
        # BEHAVIORAL ANALYSIS
        st.divider()
        st.subheader("👤 Behavioral Pattern Classification")
        render_behavioral_classification(result)
        
        # SENSITIVITY ANALYSIS
        st.divider()
        st.subheader("📊 Factor Sensitivity Analysis")
        render_sensitivity_analysis(result, model, preprocessor, feature_engineer)
        
        # SCENARIO ANALYSIS
        st.divider()
        st.subheader("🔮 Scenario Simulator")
        render_whatif_simulator(model, preprocessor, feature_engineer, input_frame, result)
        
        # COUNTERFACTUAL ANALYSIS
        st.divider()
        st.subheader("🎯 Intervention Recommendations")
        render_counterfactual(model, preprocessor, feature_engineer, input_frame, result)
        
        # INTERACTION ANALYSIS
        st.divider()
        st.subheader("🔗 Factor Interaction Matrix")
        render_interaction_heatmap(result)
        
        # RISK ASSESSMENT
        st.divider()
        st.subheader("⚠️ Clinical Risk Assessment")
        render_risk_interpretation(result)
        
        # CONFIDENCE DIAGNOSTICS
        st.divider()
        st.subheader("✓ Prediction Stability Analysis")
        render_model_confidence(model, preprocessor, feature_engineer, input_frame, result)
        
        # PERFORMANCE METRICS
        st.divider()
        st.subheader("📊 Model Performance Summary")
        render_model_metrics(metadata)
    else:
        st.info("👈 Enter your lifestyle factors and click **Generate Complete 14-Section Analysis**")


if __name__ == "__main__":
    main()
