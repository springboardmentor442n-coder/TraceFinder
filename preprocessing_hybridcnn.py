import cv2
import numpy as np
from scipy.ndimage import median_filter
import pywt

IMG_SIZE = (128, 128)

# -----------------------------
# Preprocessing Functions
# -----------------------------
def to_gray(img):
    """Convert image to grayscale if it's RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def resize_to(img, size=IMG_SIZE):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_img(img):
    return img.astype(np.float32) / 255.0

def denoise_wavelet_img(img):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    cH[:] = 0; cV[:] = 0; cD[:] = 0
    return pywt.idwt2((cA, (cH, cV, cD)), 'haar')

def preprocess_image_hybridcnn(fpath):
    """Preprocess a single image for Hybrid CNN."""
    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️ Image not found: {fpath}")
        return None
    img = to_gray(img)
    img = resize_to(img)
    img = normalize_img(img)
    den = denoise_wavelet_img(img)
    residual = img - den
    residual = residual.astype(np.float32)
    # Add channel dimension
    residual = np.expand_dims(residual, axis=-1)
    return residual

if __name__ == "__main__":
    # Test preprocessing
    img_path = r"C:\Users\Rishabh\Downloads\enrollment form .jpg"
    img = preprocess_image_hybridcnn(img_path)
    if img is not None:
        print("Image preprocessed successfully:", img.shape)
    else:
        print("Image not found.")
