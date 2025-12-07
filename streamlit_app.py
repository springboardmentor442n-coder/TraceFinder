import streamlit as st
import requests
import tempfile
import os

# -------------------------------------
# GLOBAL SETTINGS
# -------------------------------------
BACKEND_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="TraceFinder – Scanner Identification",
    page_icon="🔍",
    layout="wide"
)

# -------------------------------------
# MODEL METADATA + DESCRIPTIONS
# -------------------------------------
MODEL_INFO = {
    "svm": {
        "title": "Support Vector Machine (SVM)",
        "accuracy": "41.14%",
        "images": "1982",
        "description": (
            "The SVM model is trained on scaled features. Its parameters (C, gamma, kernel) "
            "were optimized using GridSearchCV. Accuracy and confusion matrix were used for evaluation."
        )
    },
    "rf": {
        "title": "Random Forest (RF)",
        "accuracy": "47.05%",
        "images": "1982",
        "description": (
            "Random Forest is trained on unscaled features with GridSearchCV tuning the number of trees, "
            "depth and split criteria. Performance was verified using accuracy and confusion matrix."
        )
    },
    "cnn": {
        "title": "Convolutional Neural Network (CNN)",
        "accuracy": "93.10%",
        "images": "4568",
        "description": (
            "CNN uses grayscale, resized, normalized, and wavelet-denoised residual images. "
            "A 3-layer CNN was trained using Adam optimizer and evaluated using a test split."
        )
    },
    "hybrid": {
        "title": "Hybrid CNN Model",
        "accuracy": "93.78%",
        "images": "4568",
        "description": (
            "Hybrid CNN combines residual images with handcrafted features using a dual-branch network. "
            "Both branches are fused to improve classification accuracy."
        )
    }
}

# -------------------------------------
# SESSION STATE
# -------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(model_key):
    st.session_state.page = model_key

# -------------------------------------
# GLOBAL CSS FIXES
# -------------------------------------
st.markdown("""
<style>

.stApp { background-color: #F7F9FC; }

.title-large {
    font-size:42px; font-weight:800; color:#2D3748;
    text-align:center; margin-top:18px; margin-bottom:6px;
}
.subtitle {
    color:#4A5568; text-align:center; margin-bottom:30px;
}

/* Model Card */
.model-box {
    background: #ffffff; border: 1px solid #E2E8F0;
    border-radius: 14px; padding: 22px; height: 200px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    display:flex; flex-direction:column; justify-content:center;
    text-align:center;
}

.model-title {
    font-size:25px; font-weight:700; color:#2D3748; margin-bottom:6px;
}
.model-meta { font-size:19px; color:#4A5568; margin: 3px 0; }

.select-btn-spacing { height: 30px; }

/* Column Cards */
div[data-testid="column"] > div:first-child {
    background:#ffffff;
    border:1px solid #E2E8F0;
    border-radius:14px;
    padding:22px;
    min-height:360px;
    box-shadow:0 6px 16px rgba(0,0,0,0.06);
}

/* Confusion Matrix */
.cm-img { display:block; margin:auto; max-width:100%; height:auto; }

/* Preprocessing Pipeline Styling */
.preprocess-block, .preprocess-block * {
    font-size:19px !important;
    color:#2D3748 !important;
    line-height:1.55 !important;
}

.preprocess-block ul { margin-left:18px !important; }

</style>
""", unsafe_allow_html=True)

# -------------------------------------
# HOME PAGE
# -------------------------------------
def home_page():
    st.markdown("<div class='title-large'>🔍 TraceFinder – Scanner Identification</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>TraceFinder identifies the source scanner of an image using device-specific artifacts.</div>", unsafe_allow_html=True)

    st.markdown("### <span style='color:#D53F8C;'>🧰 Choose a Model</span>", unsafe_allow_html=True)
    st.write("")

    cols = st.columns(4, gap="large")

    for col, (key, info) in zip(cols, MODEL_INFO.items()):
        with col:
            st.markdown(
                f"""
                <div class="model-box">
                    <div class="model-title">{info['title']}</div>
                    <div class="model-meta">Accuracy: <b>{info['accuracy']}</b></div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div class='select-btn-spacing'></div>", unsafe_allow_html=True)

            center_btn = st.columns([1,1,1])[1]
            with center_btn:
                if st.button("Select", key=f"select_{key}", use_container_width=True):
                    go_to(key)

# -------------------------------------
# MODEL PAGE
# -------------------------------------
def model_page(model_key):

    info = MODEL_INFO[model_key]
    st.markdown(f"<div class='title-large'>{info['title']}</div>", unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1, 1], gap="large")

    # LEFT — Upload
    with left:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader("Upload TIFF / Image",
                                         type=["png", "jpg", "jpeg", "tif", "tiff"])
        if st.button("🔍 Detect Scanner", use_container_width=True):
            if not uploaded_file:
                st.error("⚠ Please upload an image first.")
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False)
                tmp.write(uploaded_file.read())
                tmp.close()

                with open(tmp.name, "rb") as f:
                    try:
                        files = {"file": (uploaded_file.name, f, uploaded_file.type)}
                        res = requests.post(BACKEND_URL, files=files,
                                            params={"model": model_key}, timeout=30)
                    except:
                        st.error("❌ Backend not reachable")
                        return

                os.remove(tmp.name)

                if res.status_code == 200:
                    out = res.json()
                    pred = out["prediction"]
                    conf = out.get("confidence")
                    if conf:
                        conf = round(float(conf)*100,2)
                        st.success(f"🎉 Scanner Detected: **{pred}** — {conf}%")
                    else:
                        st.success(f"🎉 Scanner Detected: **{pred}**")
                else:
                    st.error("❌ Backend error!")

    # RIGHT — Model Details
    with right:
        st.markdown("### 📊 Model Details")
        st.markdown(f"<p style='font-size:20px; font-weight:700;'>Model Accuracy: {info['accuracy']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:20px; font-weight:700;'>Training Images: {info['images']} images</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:20px; font-weight:700;'>Model Description:</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:19px; color:#4A5568; line-height:1.6;'>{info['description']}</p>", unsafe_allow_html=True)

    # Technical Summary
    st.markdown("### 🔍 Performance & Technical Summary")
    c1, c2 = st.columns([1,1], gap="large")

    # LEFT — Confusion Matrix
    with c1:
        st.markdown("#### 📉 Confusion Matrix")

        cm_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "confusion_matrices", f"{model_key}_cm.png")
        )

        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.info("Confusion matrix not available.")

    # RIGHT — Preprocessing Summary
    with c2:
        st.markdown("#### 🧠 Model Summary")

        PREPROCESSING = {
    "svm": """
<ul>
<li>Grayscale conversion</li>
<li>Resize → 256×256</li>
<li>Extract 14 handcrafted features:
<ul>
<li>Intensity stats (mean, std, skew, kurtosis)</li>
<li>Entropy</li>
<li>Sobel edge density</li>
<li>GLCM: contrast, homogeneity, energy, correlation</li>
<li>LBP entropy</li>
<li>FFT mean & FFT std</li>
</ul>
</li>
<li>Scaled using StandardScaler</li>
</ul>
""",

    "rf": """
<ul>
<li>Grayscale conversion</li>
<li>Resize → 256×256</li>
<li>Extract 14 handcrafted features</li>
<li>No feature scaling used</li>
</ul>
""",

    "cnn": """
<ul>
<li>Grayscale conversion</li>
<li>Resize → 256×256</li>
<li>Pixel normalization (0–1)</li>
<li>Haar wavelet denoising</li>
<li>Residual creation (original − denoised)</li>
<li>Reshape to (256×256×1)</li>
</ul>
""",

    "hybrid": """
<p><b>Residual branch</b></p>
<ul>
<li>Grayscale → Resize → Normalize</li>
<li>Haar wavelet denoise</li>
<li>Residual map extraction</li>
<li>Reshape to (256×256×1)</li>
</ul>

<p><b>Handcrafted features (27)</b></p>
<ul>
<li>11 fingerprint correlation features</li>
<li>6 FFT features</li>
<li>10-bin LBP histogram</li>
<li>StandardScaler normalization</li>
</ul>
"""
}


        # *** SINGLE BLOCK (NO SPLIT) — FIXES the </div> problem ***
        full_html = f"""
<div class="preprocess-block">
    <p style='font-size:20px; font-weight:700;'>Name: {info['title']}</p>
    <p style='font-size:20px; font-weight:700; margin-bottom:12px;'>Preprocessing Pipeline:</p>
    {PREPROCESSING[model_key]}
</div>
"""
        st.markdown(full_html, unsafe_allow_html=True)

    # Back Button
    st.write("")
    mid = st.columns([1,2,1])[1]
    with mid:
        if st.button("⬅ Back to Home", use_container_width=True):
            st.session_state.page = "home"

# Router
if st.session_state.page == "home":
    home_page()
else:
    model_page(st.session_state.page)
