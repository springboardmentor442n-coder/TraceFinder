import os
import cv2
import numpy as np
import pandas as pd
from skimage import img_as_float
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy
from skimage.feature import local_binary_pattern
from scipy.fft import fft2, fftshift
import pywt
from scipy.signal import wiener as scipy_wiener
from tqdm import tqdm
import pickle

def load_gray(img_path, size=(512, 512)):
    """Load image as grayscale and resize."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = img.astype(np.float32) / 255.0
    img = img_as_float(img)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def extract_features(img, file_path, class_label, pixel_density):
    """Extract handcrafted features for XGBoost."""
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024  # in KB
    pixels = img.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256, range=(0, 1))[0] + 1e-6)
    edges = sobel(img)
    edge_density = np.mean(edges > 0.1)
    return {
        "file_name": os.path.basename(file_path),
        "class_label": class_label,
        "pixel_density": pixel_density,
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density
    }

def process_dataset_root_xgboost(root_folder):
    """Process dataset for XGBoost features."""
    rows = []
    if not os.path.exists(root_folder):
        print(f"Dataset root not found: {root_folder}")
        return rows

    for scanner_folder in sorted(os.listdir(root_folder)):
        scanner_path = os.path.join(root_folder, scanner_folder)
        if not os.path.isdir(scanner_path):
            continue

        # list possible DPI subfolders
        dpi_subfolders = [
            d for d in os.listdir(scanner_path)
            if os.path.isdir(os.path.join(scanner_path, d))
        ]

        # Case 1: Scanner has DPI folders
        if dpi_subfolders:
            for dpi in dpi_subfolders:
                dpi_path = os.path.join(scanner_path, dpi)
                tif_files = [f for f in os.listdir(dpi_path) if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))]

                for f in tqdm(tif_files, desc=f"{scanner_folder}/{dpi}"):
                    fpath = os.path.join(dpi_path, f)
                    img = load_gray(fpath)
                    if img is not None:
                        row = extract_features(img, fpath, scanner_folder, dpi)
                        rows.append(row)

        # Case 2: Scanner has images directly (no DPI folder)
        else:
            tif_files = [f for f in os.listdir(scanner_path) if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))]
            for f in tqdm(tif_files, desc=f"{scanner_folder}"):
                fpath = os.path.join(scanner_path, f)
                img = load_gray(fpath)
                if img is not None:
                    row = extract_features(img, fpath, scanner_folder, "none")
                    rows.append(row)

    return rows

# CNN Preprocessing
def to_grey(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def resize_to(img, size=(256, 256)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_img(img):
    return img.astype(np.float32) / 255.0

def denoise_wavelet(img):
    coeffs = pywt.dwt2(img, "haar")
    cA, (cH, cV, cD) = coeffs
    cH[:] = 0; cV[:] = 0; cD[:] = 0
    return pywt.idwt2((cA, (cH, cV, cD)), 'haar')

def preprocess_image(fpath, size=(256, 256)):
    """Preprocess image for CNN (denoising)."""
    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if len(img.shape) == 3:
        img = to_grey(img)
    img = resize_to(img, size)
    img = normalize_img(img)
    den = denoise_wavelet(img)
    return (img - den).astype(np.float32)

def load_dataset_cnn(root_path):
    """Load dataset for CNN."""
    X = []
    y = []
    if not os.path.exists(root_path):
        print(f"Dataset root not found: {root_path}")
        return X, y

    scanners = sorted(os.listdir(root_path))

    for scanner in scanners:
        scanner_path = os.path.join(root_path, scanner)
        if not os.path.isdir(scanner_path):
            continue

        dpi_folders = sorted([d for d in os.listdir(scanner_path) if os.path.isdir(os.path.join(scanner_path, d))])
        
        # Handle case with no DPI folders
        if not dpi_folders:
             files = [f for f in os.listdir(scanner_path) if f.lower().endswith(('.tif', ".tiff", ".png", ".jpg"))]
             for f in tqdm(files, desc=f"{scanner}"):
                fpath = os.path.join(scanner_path, f)
                res = preprocess_image(fpath)
                if res is not None:
                    X.append(res.reshape(256, 256, 1))
                    y.append(scanner)
        else:
            for dpi in dpi_folders:
                dpi_path = os.path.join(scanner_path, dpi)
                files = [f for f in os.listdir(dpi_path) if f.lower().endswith(('.tif', ".tiff", ".png", ".jpg"))]
                for f in tqdm(files, desc=f"{scanner}/{dpi}"):
                    fpath = os.path.join(dpi_path, f)
                    res = preprocess_image(fpath)
                    if res is not None:
                        X.append(res.reshape(256, 256, 1))
                        y.append(scanner)
    return X, y

# Hybrid Features
def corr2d(a, b):
    """Normalized correlation between residual and fingerprint"""
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / denom) if denom > 0 else 0.0

def fft_radial_energy(img, K=6):
    """FFT energy in K radial bins (global frequency pattern)"""
    f = fftshift(fft2(img))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    bins = np.linspace(0, r.max()+1e-6, K+1)
    features = [mag[(r >= bins[i]) & (r < bins[i+1])].mean() for i in range(K)]
    return list(map(float, features))

def lbp_hist_safe(img, P=8, R=1.0):
    """LBP histogram (texture pattern of noise)"""
    rng = float(np.ptp(img))
    if rng < 1e-12:
        return [0.0] * (P + 2)
    g = (img - img.min()) / (rng + 1e-8)
    g8 = (g * 255).astype(np.uint8)
    codes = local_binary_pattern(g8, P, R, method="uniform")
    hist, _ = np.histogram(codes, bins=np.arange(P+3), density=True)
    return hist.astype(np.float32).tolist()

def process_dataset_residuals(base_dir):
    """Process dataset to get residuals for Hybrid model."""
    residuals_dict = {}
    if not os.path.exists(base_dir):
        print(f"Dataset not found: {base_dir}")
        return residuals_dict

    scanners = sorted(os.listdir(base_dir))

    for scanner in tqdm(scanners, desc=f"Processing {os.path.basename(base_dir)}"):
        scanner_dir = os.path.join(base_dir, scanner)
        if not os.path.isdir(scanner_dir):
            continue

        all_residuals = []
        dpi_dirs = [os.path.join(scanner_dir, d) for d in os.listdir(scanner_dir) if os.path.isdir(os.path.join(scanner_dir, d))]

        if dpi_dirs:
            for dpi_dir in dpi_dirs:
                image_files = [os.path.join(dpi_dir, fname) for fname in os.listdir(dpi_dir) if fname.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
                for img_path in image_files:
                    res = preprocess_image(img_path)
                    if res is not None:
                        all_residuals.append(res)
        else:
            image_files = [os.path.join(scanner_dir, fname) for fname in os.listdir(scanner_dir) if fname.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
            for img_path in image_files:
                res = preprocess_image(img_path)
                if res is not None:
                    all_residuals.append(res)

        residuals_dict[scanner] = all_residuals

    return residuals_dict
