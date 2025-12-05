import streamlit as st
import requests
from PIL import Image
import io
import os

# ----------------- Config -----------------
API_URL = "http://127.0.0.1:8000/predict"  # FastAPI backend
st.set_page_config(page_title="AI Trace Finder", layout="wide")

# ----------------- UI -----------------
st.markdown("### Scan and detect forged or authentic documents instantly with AI.")
st.title("AI Trace Finder — Forensic Document Scanner")
st.write("Upload document image, choose model, and get prediction.")

# Sidebar
st.sidebar.header("Model Selection")
model_option = st.sidebar.radio("Choose model", ["Random Forest","CNN Model","Hybrid CNN Model"])

with st.sidebar.expander("Model Info", expanded=True):
    img_map = {
        "Random Forest":"assets/cyclincline.png",
        "CNN Model":"assets/cnn.png",
        "Hybrid CNN Model":"assets/hybrid.png"
    }
    desc_map = {
        "Random Forest": [
            "Lightweight forensic model tuned for texture analysis.",
            "Fast inference for texture-based detection.",
            "Ideal for simple document forgery detection.",
            "Handles scanned images with minimal preprocessing.",
            "Low computational resource requirements."
        ],
        "CNN Model": [
            "Classic Convolutional Neural Network for image feature extraction.",
            "Detects fine details and patterns in document images.",
            "Robust against noise and minor distortions.",
            "Suitable for various document types and layouts.",
            "Moderate computational requirements."
        ],
        "Hybrid CNN Model": [
            "Combines image features and metadata for higher accuracy.",
            "Best for complex or mixed-input documents.",
            "Handles both scanned and photographed documents effectively.",
            "Advanced feature extraction with deep learning.",
            "Higher computational requirements but most accurate."
        ]
    }
    st.subheader(model_option)
    for point in desc_map[model_option]:
        st.markdown(f"- {point}")
    if os.path.exists(img_map[model_option]):
        st.image(img_map[model_option], use_column_width=True)

# ----------------- Main Layout -----------------
uploaded_file = st.file_uploader("Upload document image", type=["jpg","jpeg","png","tif","tiff"])
st.markdown("---")
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded document", use_column_width=True)
st.markdown("---")
if st.button("Predict"):
    if not uploaded_file:
        st.warning("Please upload an image first.")
    else:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {"model_choice": model_option}
        try:
            with st.spinner("Contacting backend..."):
                resp = requests.post(API_URL, files=files, data=data, timeout=60)
            if resp.status_code == 200:
                out = resp.json()
                
                # ---- Random Forest ----
                if out["model"] == "Random Forest":
                    st.header("🌲 Random Forest Prediction")
                    st.success(f"Label: {out['label']}")
                    st.write(f"Confidence: {out['prob_real']*100:.2f}%")
                    st.bar_chart(out["probs"])

                # ---- CNN Model ----
                elif out["model"] == "CNN Model":
                    st.header("🧠 CNN Model Prediction")
                    st.success(f"Label: {out['label']}")
                    st.write(f"Confidence: {out['prob_real']*100:.2f}%")
                    st.bar_chart(out["probs"])

                # ---- Hybrid CNN Model ----
                elif out["model"] == "Hybrid CNN Model":
                    st.header("🤖 Hybrid CNN Model Prediction")
                    st.success(f"Label: {out['label']}")
                    st.write(f"Confidence: {out['prob_real']*100:.2f}%")
                    st.bar_chart(out["probs"])

                else:
                    st.error("Unknown model response format!")

            else:
                st.error(f"Backend error: {resp.status_code} - {resp.text}")

        except Exception as e:
            st.error(f"Failed to contact backend: {e}")
