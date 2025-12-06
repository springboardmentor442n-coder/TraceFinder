import pickle
import numpy as np
from backend.preprocessing_hybridcnn import preprocess_image_hybridcnn

class HybridCNNScannerModel:
    def __init__(self):
        try:
            with open("models/hybrid_cnn_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            print("✅ Hybrid CNN model loaded")
        except Exception as e:
            print("❌ Error loading Hybrid CNN model:", e)
            self.model = None

    def predict(self, file_path):
        if self.model is None:
            return {"error": "Model not loaded"}

        img = preprocess_image_hybridcnn(file_path)
        if img is None:
            return {"error": "Invalid Image"}

        img = np.expand_dims(img, axis=0)
        pred = self.model.predict(img)
        label = int(np.argmax(pred))

        return {
            "model": "Hybrid CNN",
            "prediction": label
        }
