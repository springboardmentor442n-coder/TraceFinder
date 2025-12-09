# backend/inference/features.py
import os
import cv2
import numpy as np
import pywt
from scipy.stats import skew, kurtosis

IMG_TARGET = (256, 256)

def to_gray(img):
    if img is None:
        return None
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def resize_to(img, size=IMG_TARGET):
    # size is (w,h) or (256,256) -> cv2.resize uses (w,h) but we accept (h,w) tuple
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA) \
        if img.shape[:2] != (size[0], size[1]) else img

def normalize_img(img):
    im = img.astype(np.float32)
    maxv = im.max() if im.max() > 0 else 1.0
    return im / maxv

def denoise_wavelet(img):
    """Single-level Haar DWT denoise: zero out detail coeffs and inverse."""
    coeffs = pywt.dwt2(img, "haar")
    cA, (cH, cV, cD) = coeffs
    cH[:] = 0; cV[:] = 0; cD[:] = 0
    rec = pywt.idwt2((cA, (cH, cV, cD)), "haar")
    # ensure same shape as input
    if rec.shape != img.shape:
        rec = resize_to(rec, img.shape)
    return rec

def compute_basic_features(img_path):
    """Return dict of the *flat / sklearn* features used earlier:
       width,height,aspect_ratio,file_size_kb,mean_intensity,std_intensity,skewness,kurtosis,entropy,edge_density
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read image: {img_path}")

    gray = to_gray(img)
    h, w = gray.shape[:2]
    aspect_ratio = float(w) / float(h) if h != 0 else 0.0
    file_size_kb = float(os.path.getsize(img_path)) / 1024.0

    # normalize intensities to 0..1 for moments
    arr = gray.astype(np.float32)
    maxv = arr.max() if arr.max() > 0 else 1.0
    arrn = arr / maxv
    mean_intensity = float(arrn.mean())
    std_intensity = float(arrn.std())

    flat = arrn.flatten()
    skewness = float(skew(flat))
    kurt = float(kurtosis(flat))

    # entropy (histogram-based)
    hist, _ = np.histogram(flat, bins=256, range=(0.0, 1.0), density=True)
    hist = hist + 1e-12
    entropy = float(-np.sum(hist * np.log2(hist)))

    # edge density
    edges = cv2.Canny((arrn * 255).astype(np.uint8), 100, 200)
    edge_density = float(edges.astype(bool).sum()) / (w * h)

    return {
        "width": int(w),
        "height": int(h),
        "aspect_ratio": float(aspect_ratio),
        "file_size_kb": float(file_size_kb),
        "mean_intensity": float(mean_intensity),
        "std_intensity": float(std_intensity),
        "skewness": float(skewness),
        "kurtosis": float(kurt),
        "entropy": float(entropy),
        "edge_density": float(edge_density)
    }

def make_residual_image(image_path, target_size=IMG_TARGET):
    """Return residual image shaped (1, H, W, 1) float32 scaled to [-1,1] (same approach as training)."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    gray = to_gray(img)
    resized = cv2.resize(gray, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    # normalize
    maxv = resized.max() if resized.max() > 0 else 1.0
    norm = resized / maxv
    den = denoise_wavelet(norm)
    res = (norm - den).astype(np.float32)
    m = np.max(np.abs(res)) if np.max(np.abs(res)) > 0 else 1.0
    res = res / m
    res = np.expand_dims(res, axis=(0, -1))  # (1, H, W, 1)
    return res
