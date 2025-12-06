import os
import cv2
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from skimage.filters import sobel
import pickle

class RandomForestScannerModel:
    """Random Forest model for AI Trace Finder using image-derived features"""

    def __init__(self, model_path="models/ai_trace_finder_model.pkl"):
        """Load RF model from pickle file"""
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            print("✅ RF model loaded successfully")
        except Exception as e:
            print("❌ Error loading RF model:", e)
            self.model = None

    def extract_features(self, image_path):
        """Extract image features matching CSV columns used for training"""
        if not os.path.exists(image_path):
            return None

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        height, width = img.shape
        pixel_density = np.count_nonzero(img)
        aspect_ratio = width / height
        file_size_kb = os.path.getsize(image_path) / 1024
        mean_intensity = np.mean(img)
        std_intensity = np.std(img)
        skewness = skew(img.flatten())
        kurt_val = kurtosis(img.flatten())

        hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 255), density=True)
        ent_val = entropy(hist + 1e-7)  # avoid log(0)
        edges = sobel(img)
        edge_density = np.sum(edges > 0) / (width * height)

        features = [
            pixel_density,
            width,
            height,
            aspect_ratio,
            file_size_kb,
            mean_intensity,
            std_intensity,
            skewness,
            kurt_val,
            ent_val,
            edge_density
        ]
        return np.array(features).reshape(1, -1)

    def predict(self, image_path):
        """Predict RF label from image"""
        if self.model is None:
            return {"error": "RF model not loaded"}

        features = self.extract_features(image_path)
        if features is None:
            return {"error": "Invalid image"}

        pred_prob = self.model.predict_proba(features)[0]
        label_idx = int(np.argmax(pred_prob))
        prob_real = float(pred_prob[label_idx])

        return {
            "model": "Random Forest",
            "label": "Real" if label_idx == 0 else "Fake",
            "prob_real": prob_real,
            "probs": pred_prob.tolist()
        }

# ----------- Test -------------
if __name__ == "__main__":
    rf = RandomForestScannerModel()
    res = rf.predict("test_image.tif")  # replace with your test image
    print(res)
