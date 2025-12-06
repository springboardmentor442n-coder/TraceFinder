import streamlit as st
import requests
from PIL import Image
import io
import os
import pandas as pd

# ===============================
# CONFIG
# ===============================
BACKEND_URL = "http://127.0.0.1:8000/predict"
CONFUSION_DIR = "models/confusion_matrices"

st.set_page_config(
    page_title="TraceFinder | Scanner Identification",
    page_icon="🖨️",
    layout="centered",
)

# ===============================
# HEADER
# ===============================
st.markdown(
    """
    <div style="text-align:center; background:linear-gradient(to right, #00c6ff, #0072ff); padding:20px; border-radius:15px;">
        <h1 style="color:white;">🖨️ TraceFinder</h1>
        <h4 style="color:#e0e0e0;">Document Scanner Identification System</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Select Detection Model",
    ["cnn", "svm", "rf", "hybrid"]
)

st.sidebar.info(
    """
    **Models Available**
    - CNN → Image-based deep learning
    - SVM → Texture & wavelet features
    - RF → Statistical + edge features
    - Hybrid → CNN + ML fusion (Best)
    """
)

# ===============================
# FILE UPLOAD
# ===============================
st.subheader("📂 Upload Document Image")
uploaded_file = st.file_uploader(
    "Choose a scanned document image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption="Uploaded Image", use_container_width=True)

# ===============================
# PREDICTION
# ===============================
if uploaded_file and st.button("🚀 Identify Scanner", help="Click to identify the scanner"):
    with st.spinner("Analyzing document... 🔍 Please wait"):
        try:
            response = requests.post(
                BACKEND_URL,
                files={"file": uploaded_file.getvalue()},
                params={"model": model_choice},
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                st.success("✅ Prediction Completed")

                # -------------------------------
                # Colored metrics for predictions
                # -------------------------------
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                        <div style="padding:20px; border-radius:10px; background:linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%); text-align:center;">
                        <h4 style="color:white;">Predicted Scanner</h4>
                        <h2 style="color:white;">{data['prediction']}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:
                    st.markdown(
                        f"""
                        <div style="padding:20px; border-radius:10px; background:linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%); text-align:center;">
                        <h4 style="color:white;">Confidence</h4>
                        <h2 style="color:white;">{data['confidence']*100:.2f}%</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # -------------------------------
                # Probability Distribution Bar Chart (if exists)
                # -------------------------------
                if "probs" in data:
                    st.subheader("📊 Prediction Probability Distribution")
                    probs_df = pd.DataFrame(data["probs"], index=data.get("labels", []), columns=["Probability"])
                    st.bar_chart(probs_df)

                # -------------------------------
                # Confusion Matrix
                # -------------------------------
                cm_path = os.path.join(CONFUSION_DIR, f"{model_choice}_confusion.png")
                if os.path.exists(cm_path):
                    st.subheader("📈 Model Confusion Matrix")
                    st.image(cm_path, use_container_width=True)

            else:
                st.error(f"❌ Backend Error ({response.status_code})")

        except Exception as e:
            st.error("⚠️ Unable to connect to backend")
            st.code(str(e))

# ===============================
# FOOTER
# ===============================
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:gray; font-size:14px;">
    TraceFinder • Hybrid Forensic Scanner Identification System<br>
    Built with FastAPI & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
