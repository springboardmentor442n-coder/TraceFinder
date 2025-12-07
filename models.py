import os
import joblib
import numpy as np
import tensorflow as tf

from .preprocessing import (
    preprocess_cnn_input,
    preprocess_features_14,
    preprocess_features_27     # <-- UPDATED
)

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE, "..", "models"))


class ModelRegistry:
    def __init__(self):
        self.cnn = None
        self.cnn_label = None

        self.svm = None
        self.rf = None
        self.svm_label = None

        self.hybrid = None
        self.feat_scaler = None
        self.hybrid_label = None


    def load_all(self):

        # ------------------- CNN --------------------
        self.cnn = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, "cnn_model.keras")
        )
        self.cnn_label = joblib.load(
            os.path.join(MODELS_DIR, "cnn_label_encoder.pkl")
        )

        # ------------------- SVM + RF --------------------
        self.svm = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
        self.rf = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))

        self.svm_label = joblib.load(
            os.path.join(MODELS_DIR, "label_encoder.pkl")
        )

        # ------------------- HYBRID 38-FEATURE MODEL --------------------
        self.hybrid = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, "scanner_hybrid_best.keras")
        )

        self.feat_scaler = joblib.load(
            os.path.join(MODELS_DIR, "hybrid_feat_scaler.pkl")
        )

        self.hybrid_label = joblib.load(
            os.path.join(MODELS_DIR, "hybrid_label_encoder.pkl")
        )

        print("✅ All models loaded successfully.")


registry = ModelRegistry()



# ---------------------------------------------------------------------
#                        PREDICT FUNCTIONS
# ---------------------------------------------------------------------

def predict_cnn(path):
    x = preprocess_cnn_input(path)
    probs = registry.cnn.predict(x)[0]

    idx = probs.argmax()
    label = registry.cnn_label.inverse_transform([idx])[0]
    conf = float(probs[idx])

    return label, conf



def predict_svm(path):
    feats = preprocess_features_14(path)
    pred = registry.svm.predict([feats])[0]

    label = registry.svm_label.inverse_transform([pred])[0]
    return label, 1.0     # SVM has no true probability output



def predict_rf(path):
    feats = preprocess_features_14(path)

    # if RF supports probas
    if hasattr(registry.rf, "predict_proba"):
        proba = registry.rf.predict_proba([feats])[0]
        idx = int(np.argmax(proba))
        conf = float(proba[idx])
    else:
        conf = 1.0

    pred = registry.rf.predict([feats])[0]
    label = registry.svm_label.inverse_transform([pred])[0]

    return label, conf


def normalize_scanner_name(name):
    return name.replace("dataset-trace-2_", "").replace("dataset-trace_", "")

def predict_hybrid(path):
    """
    Hybrid model expects:
    - residual image tensor → (1,256,256,1)
    - handcrafted features → (1,27)
    """

    img_tensor, feat_27 = preprocess_features_27(path)

    # scale handcrafted features
    feat_scaled = registry.feat_scaler.transform(feat_27)   # already shape (1,38)

    # run prediction
    probs = registry.hybrid.predict([img_tensor, feat_scaled])[0]

    idx = np.argmax(probs)
    raw_label = registry.hybrid_label.inverse_transform([idx])[0]
    clean_label = normalize_scanner_name(raw_label)
    confidence = float(probs[idx])

    return clean_label, confidence
