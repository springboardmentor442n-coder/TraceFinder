# backend/preprocessing_cnn.py

import cv2
import numpy as np
import pywt

from .config import IMG_SIZE

def to_grey(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def resize_to(img, size=IMG_SIZE):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_img(img):
    return img.astype(np.float32) / 255.0

def denoise_wavelet(img):
    coeffs = pywt.dwt2(img, "haar")
    cA, (cH, cV, cD) = coeffs
    cH[:] = 0; cV[:] = 0; cD[:] = 0
    return pywt.idwt2((cA, (cH, cV, cD)), "haar")

def preprocess_image_cnn(fpath):
    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {fpath}")

    img = to_grey(img)
    img = resize_to(img)
    img = normalize_img(img)
    den = denoise_wavelet(img)
    res = (img - den).astype(np.float32)

    return res.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
