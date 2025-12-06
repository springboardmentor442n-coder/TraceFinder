import os

# Paths
DATASET_ROOT = r"E:\Project\DocumentFounder\Dataset"
OUTPUT_DIR = r"E:\Project\DocumentFounder\models"
FLATFIELD_RESIDUALS_PKL = os.path.join(OUTPUT_DIR, "flat_common_residual.pkl")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model Paths
XGB_MODEL_PATH = os.path.join(OUTPUT_DIR, "xgb_model.pkl")
LABEL_ENCODER_PATH = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
CNN_MODEL_PATH = os.path.join(OUTPUT_DIR, "cnn_residual_model.keras")
CNN_LABEL_ENCODER_PATH = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
HYBRID_MODEL_PATH = os.path.join(OUTPUT_DIR, "scanner_hybrid_final.keras")
HYBRID_LABEL_ENCODER_PATH = os.path.join(OUTPUT_DIR, "hybrid_label_encoder.pkl")
HYBRID_SCALER_PATH = os.path.join(OUTPUT_DIR, "hybrid_feat_scaler.pkl")
SCANNER_FINGERPRINTS_PATH = os.path.join(OUTPUT_DIR, "scanner_fingerprints.pkl")
FP_KEYS_PATH = os.path.join(OUTPUT_DIR, "fp_keys.npy")
COMBINED_RESIDUALS_PKL = os.path.join(OUTPUT_DIR, "combined_residuals.pkl")
COMBINED_FEATURES_PKL = os.path.join(OUTPUT_DIR, "combined_features.pkl")

# Hyperparameters
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 20  # You can increase this for better accuracy
