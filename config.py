# backend/config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_final_model.keras")
CNN_ENCODER_PATH = os.path.join(BASE_DIR, "models", "cnn_label_encoder.pkl")

XGB_MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model.pkl")
XGB_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder_scikit.pkl")

IMG_SIZE = (256, 256)