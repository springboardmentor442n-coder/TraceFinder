import os
import pickle
import numpy as np
from tensorflow.keras import models
from backend.preprocessing_cnn import preprocess_image_cnn


class CNNScannerModel:
    """
    CNN Model Loader + Predictor for AI Trace Finder
    """

    def __init__(self):
        self.model = None
        self.label_classes = None
        self.load_model()

    def load_model(self):
        """Load CNN model & label encoder from /models folder"""
        try:
            model_path = os.path.join("models", "cnn_model.pkl")
            label_path = os.path.join("models", "label_encoder.pkl")

            # Load model json + weights
            with open(model_path, "rb") as f:
                data = pickle.load(f)

            self.model = models.model_from_json(data["model_json"])
            self.model.set_weights(data["model_weights"])
            self.model.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

            # Load label classes
            with open(label_path, "rb") as f:
                label_data = pickle.load(f)
                self.label_classes = label_data["classes"]

            print("✔ CNN Model Loaded Successfully")

        except Exception as e:
            print("❌ Error loading CNN model:", e)

    def predict(self, image_path):
        """
        Predict class of a single image.
        Returns JSON decoded label + confidence.
        """

        try:
            img = preprocess_image_cnn(image_path)
            img = np.expand_dims(img, axis=0)  # (1,128,128,1)

            pred = self.model.predict(img)
            class_id = int(np.argmax(pred))
            confidence = float(np.max(pred))

            return {
                "label_id": class_id,
                "label_name": self.label_classes[class_id],
                "confidence": confidence,
                "probabilities": pred[0].tolist(),
            }

        except Exception as e:
            return {"error": str(e)}
