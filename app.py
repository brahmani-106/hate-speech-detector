import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Hate Speech Detector", layout="centered")
st.title(" Hate Speech Detection Web App")
st.write("This app detects whether the input text contains hate or offensive language.")

try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Make sure 'model.pkl' and 'vectorizer.pkl' are in the same folder.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading model or vectorizer: {e}")
    st.stop()

text_input = st.text_area("text goes here", height=150)

if st.button("🔍 Predict"):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        try:
    
            X = vectorizer.transform([text_input])
            prediction_proba = model.predict_proba(X)[0][1]  # Probability of offensive
            result = "⚠️ Offensive" if prediction_proba > 0.7 else "✅ Non-Offensive"

            st.subheader("Prediction Result:")
            if result == "✅ Non-Offensive":
                st.success(result)
            else:
                st.error(result)

            st.caption(f"Confidence Score: {prediction_proba:.2f}")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
