import cv2
import numpy as np
import pywt
from skimage import io, img_as_float
from skimage.filters import sobel
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft2, fftshift
import pickle
import os

# ======================================================
#                 PATHS FOR HYBRID FINGERPRINTS
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

FP_KEYS_PATH = os.path.join(MODELS_DIR, "fp_keys.npy")
FP_PATH = os.path.join(MODELS_DIR, "scanner_fingerprints.pkl")

# Load fingerprint data once
FP_KEYS = np.load(FP_KEYS_PATH, allow_pickle=True)
SCANNER_FPS = pickle.load(open(FP_PATH, "rb"))


# ======================================================
#        CNN PREPROCESSING  (RESIDUAL INPUT)
# ======================================================
def preprocess_cnn_input(path):
    """
    CNN residual preprocessing:
    - grayscale → resize → normalize → wavelet denoise → residual → reshape
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0

    # wavelet denoising
    cA, (cH, cV, cD) = pywt.dwt2(img, "haar")
    cH[:] = 0
    cV[:] = 0
    cD[:] = 0
    den = pywt.idwt2((cA, (cH, cV, cD)), "haar")

    residual = (img - den).astype(np.float32)
    return residual.reshape(1, 256, 256, 1)



# ======================================================
#     SVM / RF — 14 FEATURE EXTRACTION
# ======================================================
def preprocess_features_14(path):
    img = io.imread(path, as_gray=True)
    img = img_as_float(img)
    img = cv2.resize(img, (256, 256))

    h, w = img.shape
    aspect_ratio = w / h

    pixels = img.flatten()

    # basic stats
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt_val = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256, range=(0, 1))[0] + 1e-6)

    # edges
    edges = sobel(img)
    edge_density = np.mean(edges > 0.1)

    # GLCM
    img_uint8 = (img * 255).astype(np.uint8)
    glcm = graycomatrix(img_uint8, [1], [0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # LBP entropy
    lbp = local_binary_pattern(img_uint8, 8, 1, "uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    lbp_entropy = entropy(lbp_hist + 1e-6)

    # FFT features
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)

    fft_mean = np.mean(magnitude)
    fft_std = np.std(magnitude)

    return [
        aspect_ratio,
        mean_intensity, std_intensity,
        skewness, kurt_val,
        ent,
        edge_density,
        contrast, homogeneity, energy, correlation,
        lbp_entropy,
        fft_mean, fft_std,
    ]


# ======================================================
#       HYBRID MODEL – TRUE 27 FEATURE EXTRACTION
# ======================================================
def corr2d(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / denom) if denom != 0 else 0.0


def preprocess_features_27(path):
    
    # ---------------- Residual Image ----------------
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0

    cA, (cH, cV, cD) = pywt.dwt2(img, "haar")
    cH[:] = 0; cV[:] = 0; cD[:] = 0
    den = pywt.idwt2((cA, (cH, cV, cD)), "haar")

    residual = (img - den).astype(np.float32)
    residual_tensor = residual.reshape(256, 256, 1)

    # ---------------- 11 Correlation Features ----------------
    corr_feats = [
        corr2d(residual, SCANNER_FPS[k])
        for k in FP_KEYS
    ]

    # ---------------- FFT (6 bands) ----------------
    f = np.abs(fftshift(fft2(residual)))
    h, w = f.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)

    bins = np.linspace(0, r.max(), 7)
    fft_feats = [
        float(np.mean(f[(r >= bins[i]) & (r < bins[i+1])]))
        for i in range(6)
    ]

    # ---------------- LBP (10 bins) ----------------
    lbp = local_binary_pattern((residual * 255).astype(np.uint8), 8, 1, "uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(12), density=True)
    lbp_feats = lbp_hist[:10].tolist()

    # ---------------- Final 27 features ----------------
    handcrafted = corr_feats + fft_feats + lbp_feats
    handcrafted = np.array(handcrafted, dtype=np.float32)

    # ---------------- ADD BATCH DIM — FIX ----------------
    residual_tensor = residual_tensor.reshape(1, 256, 256, 1)
    handcrafted = handcrafted.reshape(1, 27)

    return residual_tensor, handcrafted
