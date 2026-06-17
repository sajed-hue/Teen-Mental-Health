import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf


MODEL_PATH = "teen_mental_health_model.keras"
SCALER_PATH = "scaler.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"


@st.cache_resource
def load_model_and_tools():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, scaler, feature_columns


model, scaler, feature_columns = load_model_and_tools()


def predict_depression(
    age,
    gender,
    daily_social_media_hours,
    sleep_hours,
    screen_time_before_sleep,
    academic_performance,
    physical_activity,
    stress_level,
    anxiety_level,
    addiction_level,
    platform_usage,
    social_interaction_level,
    threshold
):
    row = {col: 0.0 for col in feature_columns}

    numeric_values = {
        "age": age,
        "daily_social_media_hours": daily_social_media_hours,
        "sleep_hours": sleep_hours,
        "screen_time_before_sleep": screen_time_before_sleep,
        "academic_performance": academic_performance,
        "physical_activity": physical_activity,
        "stress_level": stress_level,
        "anxiety_level": anxiety_level,
        "addiction_level": addiction_level,
    }

    for col, value in numeric_values.items():
        if col in row:
            row[col] = float(value)

    gender = str(gender).strip().lower()

    if "gender" in row:
        if gender == "male":
            row["gender"] = 0.0
        elif gender == "female":
            row["gender"] = 1.0

    platform_usage = str(platform_usage).strip().lower()

    for col in feature_columns:
        if col.lower() in ["platform_usage_instagram", "platform_usage_tiktok"]:
            row[col] = 0.0

    if platform_usage == "instagram":
        for col in feature_columns:
            if col.lower() == "platform_usage_instagram":
                row[col] = 1.0

    elif platform_usage == "tiktok":
        for col in feature_columns:
            if col.lower() == "platform_usage_tiktok":
                row[col] = 1.0

    social_interaction_level = str(social_interaction_level).strip().lower()

    for col in feature_columns:
        if col.lower() in [
            "social_interaction_level_low",
            "social_interaction_level_medium"
        ]:
            row[col] = 0.0

    if social_interaction_level == "low":
        for col in feature_columns:
            if col.lower() == "social_interaction_level_low":
                row[col] = 1.0

    elif social_interaction_level == "medium":
        for col in feature_columns:
            if col.lower() == "social_interaction_level_medium":
                row[col] = 1.0

    input_df = pd.DataFrame([row])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    input_df = input_df.astype(float)

    input_scaled = scaler.transform(input_df).astype(np.float32)

    probability = float(model.predict(input_scaled, verbose=0)[0][0])
    prediction = int(probability >= threshold)

    if prediction == 1:
        result = "High Depression Risk"
    else:
        result = "Low Depression Risk"

    return result, probability, input_df


st.set_page_config(
    page_title="Teen Mental Health Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("Teen Mental Health Prediction System")

st.write(
    "This app uses trained Keras neural network model to predict depression risk. "
    "Educational use only — not a medical diagnosis."
)

st.warning(
    "This tool is for educational purposes only and should not replace professional mental health advice."
)


with st.form("prediction_form"):
    st.subheader("Enter Teen Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=13,
            max_value=19,
            value=16
        )

        gender = st.selectbox(
            "Gender",
            ["male", "female"]
        )

        daily_social_media_hours = st.number_input(
            "Daily Social Media Hours",
            min_value=0.0,
            max_value=24.0,
            value=10.0,
            step=0.5
        )

        sleep_hours = st.number_input(
            "Sleep Hours",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
            step=0.5
        )

        screen_time_before_sleep = st.number_input(
            "Screen Time Before Sleep",
            min_value=0.0,
            max_value=12.0,
            value=5.0,
            step=0.5
        )

        academic_performance = st.slider(
            "Academic Performance",
            min_value=0.0,
            max_value=4.0,
            value=3.18,
            step=0.1
        )

    with col2:
        physical_activity = st.number_input(
            "Physical Activity",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.5
        )

        stress_level = st.slider(
            "Stress Level",
            min_value=1,
            max_value=10,
            value=10
        )

        anxiety_level = st.slider(
            "Anxiety Level",
            min_value=1,
            max_value=10,
            value=10
        )

        addiction_level = st.slider(
            "Addiction Level",
            min_value=1,
            max_value=10,
            value=10
        )

        platform_usage = st.selectbox(
            "Platform Usage",
            ["both", "instagram", "tiktok"]
        )

        social_interaction_level = st.selectbox(
            "Social Interaction Level",
            ["high", "medium", "low"]
        )

    # Fixed threshold value
    # This is not shown in the Streamlit interface
    threshold = 0.5

    submitted = st.form_submit_button("Predict")


if submitted:
    result, probability, encoded_input = predict_depression(
        age,
        gender,
        daily_social_media_hours,
        sleep_hours,
        screen_time_before_sleep,
        academic_performance,
        physical_activity,
        stress_level,
        anxiety_level,
        addiction_level,
        platform_usage,
        social_interaction_level,
        threshold
    )

    st.subheader("Prediction Result")

    if result == "High Depression Risk":
        st.error(result)
    else:
        st.success(result)

    st.metric("Predicted Probability", f"{probability:.4f}")

    st.progress(min(max(probability, 0.0), 1.0))

    with st.expander("Show encoded model input"):
        st.dataframe(encoded_input)

    with st.expander("Model information"):
        st.write("Number of features:", len(feature_columns))
        st.write("Feature columns:")
        st.write(feature_columns)
st.markdown(
    """
<style>
.custom-footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: rgba(14, 17, 23, 0.97);
    color: #FAFAFA;
    padding: 12px 16px;
    z-index: 9999;
    border-top: 1px solid rgba(10, 102, 194, 0.45);
    box-shadow: 0 -4px 18px rgba(0, 0, 0, 0.30);
}

.footer-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 7px;
    font-size: 14px;
}

.footer-label {
    color: #B8C1CC;
    font-weight: 700;
    font-size: 15px;
}

.footer-author {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 230px;
    padding: 7px 13px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #FFFFFF !important;
    text-decoration: none !important;
    font-weight: 700;
}

.footer-author:hover {
    background: rgba(10, 102, 194, 0.20);
    border-color: rgba(10, 102, 194, 0.75);
}

.linkedin-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 6px;
    background: #0A66C2;
    color: white;
    font-size: 14px;
    font-weight: 800;
    font-family: Arial, sans-serif;
}

.block-container {
    padding-bottom: 150px;
}
</style>
<div class="custom-footer"><div class="footer-content"><div class="footer-label">Developed by</div><a class="footer-author" href="https://www.linkedin.com/in/sajed-kittanh/" target="_blank"><span>Sajed Kittanh</span><span class="linkedin-badge">in</span></a><a class="footer-author" href="https://www.linkedin.com/in/thimar-arda-02bb87347/" target="_blank"><span>Thimar Arda</span><span class="linkedin-badge">in</span></a></div></div>
""",
    unsafe_allow_html=True
)
