import streamlit as st
import numpy as np
import joblib

# Page Config
st.set_page_config(
    page_title="Parkinson Detection App",
    page_icon="🧠",
    layout="wide"
)

# Load Model
model = joblib.load("parkinson_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.markdown("<h1 style='text-align: center; color: #6C63FF;'>Parkinson Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("### Voice-based Machine Learning Detection System")

st.divider()

# Sidebar
st.sidebar.header("Enter Voice Features")

features = []

for i in range(22):
    val = st.sidebar.number_input(f"Feature {i+1}", value=0.0)
    features.append(val)

# Prediction Button
if st.sidebar.button("Predict Now"):
    input_data = np.array([features])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    st.divider()

    if prediction[0] == 1:
        st.error("⚠ Parkinson Detected")
    else:
        st.success("✅ Healthy")

    st.write(f"Confidence Score: {np.max(probability)*100:.2f}%")

st.divider()

# Footer
st.markdown(
    "<p style='text-align: center;'>Developed by Farhat 💛 | ML Healthcare App</p>",
    unsafe_allow_html=True
)