import streamlit as st
import requests
from PIL import Image
import io
import os
import pandas as pd

# ===============================
# CONFIG
# ===============================
BACKEND_URL = "http://127.0.0.1:8501/predict"
CONFUSION_DIR = "models/confusion_matrices"

st.set_page_config(
    page_title="TraceFinder | Scanner Identification",
    page_icon="🖨️",
    layout="centered",
)

# ===============================
# BLACK THEME CSS
# ===============================
st.markdown("""
<style>

/* Animated Background */
body {
    background: linear-gradient(-45deg, #000000, #001f3f, #000000, #003366);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass container effect */
section.main > div {
    background: rgba(10, 10, 10, 0.85) !important;
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 25px;
    box-shadow: 0 0 25px rgba(0, 255, 255, 0.15);
}

/* Neon title */
.neon-title {
    text-align: center;
    font-size: 46px;
    font-weight: bold;
    color: #00f7ff;
    text-shadow: 0 0 12px #00f7ff, 0 0 25px #00c8ff;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00f7ff, #0062ff);
    color: white;
    font-size: 18px;
    padding: 12px 30px;
    border-radius: 12px;
    border: none;
    transition: 0.3s;
    box-shadow: 0 0 20px rgba(0, 247, 255, 0.5);
}
.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 35px rgba(0, 247, 255, 1);
}

</style>
""", unsafe_allow_html=True)


# ===============================
# HEADER
# ===============================
st.markdown("""
<div style="padding:20px;">
    <div class="neon-title">🖨️ TraceFinder</div>
    <div class="subtitle">Forensic Scanner Identification System</div>
</div>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Detection Model",
    ["CNN", "SVM", "RF", "Hybrid"]
)

st.sidebar.markdown("""
<div style="font-size:14px; color:#9beeff;">
<b>Models:</b><br>
CNN → Deep Image Scanner Model<br>
SVM → Texture-Based Analysis<br>
RF → Edge + Pattern Detection<br>
Hybrid → Fusion AI Model ⚡
</div>
""", unsafe_allow_html=True)

# ===============================
# UPLOAD
# ===============================
st.subheader("📂 Upload Document Image")

uploaded_file = st.file_uploader(
    "Upload scanned document image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption="📸 Preview Image", use_container_width=True)

# ===============================
# PREDICTION
# ===============================
if uploaded_file and st.button("🚀 Identify Scanner"):
    with st.spinner("🔍 Analyzing document..."):

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

                # Metrics
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h4 style="color:white;">Detected Scanner</h4>
                        <h2 style="color:white;">{data['prediction']}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h4 style="color:white;">Confidence</h4>
                        <h2 style="color:white;">{data['confidence']*100:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)

                # Probability chart
                if "probs" in data:
                    st.subheader("📊 Prediction Confidence Distribution")
                    probs_df = pd.DataFrame(
                        data["probs"],
                        index=data.get("labels", []),
                        columns=["Probability"]
                    )
                    st.bar_chart(probs_df)

                # Confusion matrix
                cm_path = os.path.join(CONFUSION_DIR, f"{model_choice}_confusion.png")
                if os.path.exists(cm_path):
                    st.subheader("📈 Confusion Matrix")
                    st.image(cm_path, use_container_width=True)

            else:
                st.error(f"❌ Backend Error: {response.status_code}")

        except Exception as e:
            st.error("⚠️ Could not connect to backend")
            st.code(str(e))

# ===============================
# FOOTER
# ===============================
st.markdown("""
<hr>
<p style="text-align:center; color:#7defff; font-size:14px;">
TraceFinder • AI Scanner Forensics Platform<br>
Powered by FastAPI + Streamlit
</p>
""", unsafe_allow_html=True)
