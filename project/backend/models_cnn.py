import pickle
from tensorflow import keras
import numpy as np

from .config import CNN_MODEL_PATH, CNN_ENCODER_PATH
from .preprocessing_cnn import preprocess_image_cnn

class CNNScannerModel:
    def __init__(self):
        self.model = keras.models.load_model(CNN_MODEL_PATH)
        with open(CNN_ENCODER_PATH, "rb") as f:
            self.le = pickle.load(f)

    def predict(self, img_path):
        x = preprocess_image_cnn(img_path)
        probs = self.model.predict(x)[0]
        idx = int(np.argmax(probs))
        label = self.le.inverse_transform([idx])[0]
        confidence = float(probs[idx])

        return {
            "model": "cnn",
            "label": label,
            "confidence": confidence,
            "probs": probs.tolist()
        }