# backend/models.py

import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO

# Import preprocessing functions

from backend.preprocessing_cnn import preprocess_image_cnn
from backend.preprocessing_hybridcnn import preprocess_image_hybridcnn

# -------- MODEL PATHS --------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_model.pkl")
HYBRID_MODEL_PATH = os.path.join(BASE_DIR, "models", "hybrid_cnn_model.pkl")
XGB_MODEL_PATH = os.path.join(BASE_DIR, "models", "ai_trace_finder_model.pkl")  # Random Forest renamed

# -------- LOAD MODELS --------

with open(CNN_MODEL_PATH, "rb") as f:
    cnn_model = pickle.load(f)

with open(HYBRID_MODEL_PATH, "rb") as f:
    hybrid_model = pickle.load(f)

with open(XGB_MODEL_PATH, "rb") as f:
    xgb_model = pickle.load(f)

# -------- HELPER FUNCTION --------

def predict_by_name(model_name, image_bytes):
    """
    model_name: 'cnn', 'hybrid', 'cyclincline'
    image_bytes: uploaded image file in bytes
    """
    # Convert bytes to PIL Image
    image = Image.open(BytesIO(image_bytes)).convert("RGB")


    if model_name.lower() == "cnn":
        img_array = preprocess_image_cnn(image)  # returns np.array
        img_array = np.expand_dims(img_array, axis=0)
        pred = cnn_model.predict(img_array)
        label = np.argmax(pred)
        prob_real = float(pred[0][1])
        return {"model": "CNN", "label": int(label), "prob_real": prob_real}

    elif model_name.lower() == "hybrid":
        img_array = preprocess_image_hybridcnn(image)  # returns np.array
        img_array = np.expand_dims(img_array, axis=0)
        pred = hybrid_model.predict(img_array)
        label = np.argmax(pred)
        prob_real = float(pred[0][1])
        return {"model": "Hybrid CNN", "label": int(label), "prob_real": prob_real}

    elif model_name.lower() == "cyclincline":
    # Assuming XGB model works on flattened features or tabular data
    # Here you may need to convert image to feature vector
        img_array = preprocess_image_cnn(image).flatten().reshape(1, -1)
        pred = xgb_model.predict(img_array)
        prob_real = float(xgb_model.predict_proba(img_array)[0][1])
        return {"model": "Cyclincline", "label": int(pred[0]), "prob_real": prob_real}

    else:
        raise ValueError(f"Unknown model: {model_name}")

