# backend/preprocessing_xgb.py

import os
import numpy as np
import cv2
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy

def load_gray(path, size=(512, 512)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = img.astype("float32") / 255.0
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def extract_features_xgb(path):
    img = load_gray(path)
    pixels = img.flatten()

    file_size_kb = os.path.getsize(path) / 1024
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt_val = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256, range=(0, 1))[0] + 1e-6)
    edge_density = np.mean(sobel(img) > 0.1)

    base_features = np.array([
        file_size_kb,
        mean_intensity,
        std_intensity,
        skewness,
        kurt_val,
        ent,
        edge_density
    ], dtype=np.float32)


    # Create two versions: one with density=150, one with density=300
    feat_150 = np.concatenate(([150.0], base_features)).astype(np.float32)
    feat_300 = np.concatenate(([300.0], base_features)).astype(np.float32)

    return feat_150, feat_300       

