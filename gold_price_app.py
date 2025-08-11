import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load('gold_rf_model.pkl')

st.set_page_config(page_title="Gold Price Predictor", layout="centered")

st.title("🌟 Gold Price Prediction App")
st.markdown("Enter market values to predict the Gold Price (GLD)")

# Input fields
spx = st.number_input("S&P 500 Index (SPX)", min_value=0.0, format="%.2f")
uso = st.number_input("Oil Fund Price (USO)", min_value=0.0, format="%.2f")
slv = st.number_input("Silver Price (SLV)", min_value=0.0, format="%.2f")
eur = st.number_input("EUR/USD Exchange Rate", min_value=0.0, format="%.4f")

# Predict button
if st.button("Predict Gold Price"):
    input_data = np.array([[spx, uso, slv, eur]])
    prediction = model.predict(input_data)
    st.success(f"🌟 Predicted Gold Price (GLD): ${prediction[0]:.2f}")
