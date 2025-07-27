# app.py
import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("🛡️ Hate Speech Detection")

# User input
text = st.text_area("Enter a sentence to analyze:")

if st.button("Detect"):
    result = model.predict([text])[0]
    label = "Hate Speech" if result == 0 else "Offensive" if result == 1 else "Neutral"
    st.success(f"Prediction: {label}")

