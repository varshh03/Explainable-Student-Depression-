import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np

# ---------------- 1. LOAD FILES ----------------
@st.cache_resource
def load_assets():
    model = pickle.load(open("model.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    feature_columns = pickle.load(open("features.pkl", "rb"))
    return model, encoders, feature_columns

model, encoders, feature_columns = load_assets()

# ---------------- 2. UI HEADER ----------------
st.set_page_config(page_title="Student Mental Health AI", layout="centered")
st.title(" Explainable Student Depression Detection")
st.write("Fill in the details below to check the risk status and see the AI's reasoning.")

# ---------------- 3. USER INPUT ----------------
st.sidebar.header("Student Input Features")
user_input = {}

# Fields that must be restricted to 0–5 scale
scale_0_5_fields = [
    "Academic Pressure",
    "Work Pressure",
    "Study Satisfaction",
    "Job Satisfaction"
]

# Create two columns for cleaner UI
col1, col2 = st.columns(2)

for i, col in enumerate(feature_columns):
    target_col = col1 if i % 2 == 0 else col2

    # -------- CATEGORICAL FEATURES --------
    if col in encoders:
        options = list(encoders[col].classes_)
        val = target_col.selectbox(f"Select {col}", options)
        user_input[col] = encoders[col].transform([val])[0]

    # -------- SCALE 0–5 FEATURES --------
    elif col in scale_0_5_fields:
        val = target_col.slider(
            f"{col} (0 = Very Low, 5 = Very High)",
            min_value=0,
            max_value=5,
            value=2
        )
        user_input[col] = val

    # -------- OTHER NUMERICAL FEATURES (INTEGER ONLY) --------
    else:
        val = target_col.number_input(
            f"Enter {col}",
            value=0,
            step=1,        # prevents decimals
            format="%d"    # display as integer
        )
        user_input[col] = val

# Convert input → dataframe with correct column order
input_df = pd.DataFrame([user_input])[feature_columns]

# ---------------- 4. PREDICTION & XAI ----------------
if st.button("Predict + Explain Risk"):

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error("### High Risk Detected")
        st.write(f"Confidence Score: **{probability:.2%}**")
    else:
        st.success("###  Low Risk Detected")
        st.write(f"Confidence Score: **{(1 - probability):.2%}**")

    # ---------------- 5. SHAP EXPLANATION ----------------
    st.subheader(" Why did the AI say this?")
    st.info("Red factors increase risk • Blue factors reduce risk")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        # Handle different SHAP formats
        if isinstance(shap_values, list):
            exp_val = explainer.expected_value[1]
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            exp_val = explainer.expected_value[1]
            sv = shap_values[0][:, 1]
        else:
            exp_val = explainer.expected_value
            sv = shap_values[0]

        # Ensure scalar expected value
        if isinstance(exp_val, (list, pd.Series, np.ndarray)):
            exp_val = exp_val[0]

        # Plot SHAP Waterfall
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots._waterfall.waterfall_legacy(
            exp_val,
            sv,
            feature_names=feature_columns,
            show=False
        )
        st.pyplot(fig)

    except Exception as e:
        st.error("Could not generate the explanation plot.")
        st.write(f"Error detail: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Disclaimer: This AI tool is for educational purposes and not a medical diagnosis.")