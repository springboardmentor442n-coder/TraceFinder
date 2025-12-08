import streamlit as st
import requests
from PIL import Image


API_URL = "http://127.0.0.1:8000/api/predict"

st.title("📄 Scanner Detection App")

model_choice = st.selectbox("Choose model:", ["xgboost", "cnn"])
uploaded = st.file_uploader("Upload image", type=["tif", "tiff", "png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            data = {"model_choice": model_choice}

            res = requests.post(API_URL, files=files, data=data)
            
            if res.status_code == 200:
                out = res.json()

                # ------------------------------
                # CASE 1: CNN model result
                # ------------------------------
                if out["model"] == "cnn":
                    st.header("🧠 CNN Prediction")
                    st.success(f"Label: {out['label']}")
                    st.write(f"Confidence: {out['confidence'] * 100:.2f}%")
                    st.write("Probability Distribution:")
                    st.bar_chart(out["probs"])

                # ------------------------------
                # CASE 2: XGBoost model result
                # ------------------------------
                elif out["model"] == "xgboost":
                    st.header("🌲 XGBoost Predictions (150 DPI & 300 DPI)")

                    # 150 DPI
                    st.subheader("📌 150 DPI Prediction")
                    st.success(f"Label: {out['150dpi']['label']}")
                    st.write(f"Confidence: {out['150dpi']['confidence'] * 100:.2f}%")
                    st.write("Probability Distribution (150 DPI):")
                    st.bar_chart(out["150dpi"]["probs"])

                    # Divider
                    st.write("---")

                    # 300 DPI
                    st.subheader("📌 300 DPI Prediction")
                    st.success(f"Label: {out['300dpi']['label']}")
                    st.write(f"Confidence: {out['300dpi']['confidence'] * 100:.2f}%")
                    st.write("Probability Distribution (300 DPI):")
                    st.bar_chart(out["300dpi"]["probs"])

                else:
                    st.error("Unknown model response format!")

            else:
                st.error(f"Error: {res.text}")
