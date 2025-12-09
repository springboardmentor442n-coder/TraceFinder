#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import cv2
import numpy as np
import pywt
import pickle
from tqdm import tqdm
from scipy.signal import wiener as scipy_wiener
from scipy.fft import fft2, fftshift
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage import img_as_ubyte
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------- CONFIG (edit these before running) ----------------
WORK = r"D:\Infosys_AI-Tracefinder\Notebooks"   # where outputs will be written
OFFICIAL_DIR = r"D:\Infosys_AI-Tracefinder\Data\Official"
WIKI_DIR     = r"D:\Infosys_AI-Tracefinder\Data\Wikipedia"
FLATFIELD_DIR= r"D:\Infosys_AI-Tracefinder\Data\Flatfield"   # optional
OUTPUT_DIR   = WORK
IMG_SIZE     = (256, 256)
DENOISE_METHOD = "wavelet"   # "wavelet" (mentor) or "wiener"
VALID_EXTS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("WORK:", WORK)
print("OFFICIAL_DIR exists:", os.path.exists(OFFICIAL_DIR))
print("WIKI_DIR exists:", os.path.exists(WIKI_DIR))
print("FLATFIELD_DIR exists:", os.path.exists(FLATFIELD_DIR))


# In[11]:


# ---------------- helpers ----------------
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print("Saved:", path)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_listdir(p):
    try:
        return os.listdir(p)
    except Exception:
        return []


# ---------------- denoise / preprocessing ----------------
def to_gray(img):
    if img is None:
        return None
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def resize_to(img, size=IMG_SIZE):
    # size expected (w,h) for cv2.resize when passed as tuple
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_img(img):
    # keep simple stable scaling: divide by max or by 255 if max==0
    arr = img.astype(np.float32)
    m = arr.max() if arr.max() > 0 else 255.0
    return arr / float(m)

def denoise_wavelet_safe(img, wavelet='db4', level=2):
    arr = np.asarray(img, dtype=np.float32)
    try:
        coeffs = pywt.wavedec2(arr, wavelet=wavelet, level=level)
    except Exception:
        return arr.copy()
    # robust MAD thresholding
    details = []
    for d in coeffs[1:]:
        for comp in d:
            details.append(np.ravel(np.nan_to_num(comp)))
    if len(details) == 0:
        return arr.copy()
    details = np.concatenate(details)
    mad = np.median(np.abs(details - np.median(details)))
    sigma = mad / 0.6745 if mad > 0 else 0.0
    uthresh = sigma * np.sqrt(2 * np.log(arr.size + 1e-12))
    new_coeffs = [coeffs[0]]
    for d in coeffs[1:]:
        new_level = tuple(pywt.threshold(np.nan_to_num(comp), value=uthresh, mode='soft') for comp in d)
        new_coeffs.append(new_level)
    den = pywt.waverec2(new_coeffs, wavelet)
    den = den[:arr.shape[0], :arr.shape[1]]
    return np.nan_to_num(den, nan=arr).astype(np.float32)

def denoise_mentor(img, method=DENOISE_METHOD):
    if method == "wiener":
        return scipy_wiener(img, mysize=(5,5)).astype(np.float32)
    return denoise_wavelet_safe(img, wavelet='db4', level=2)

def preprocess_image_path(path, method=DENOISE_METHOD, size=IMG_SIZE):
    """
    Returns residual image (H,W) float32 normalized by max abs -> range approx [-1,1]
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = to_gray(img)
    img = resize_to(img, size=size)
    img = normalize_img(img)
    den = denoise_mentor(img, method=method)
    residual = (img - den).astype(np.float32)
    # normalize residual by its max abs to keep values stable (-1..1)
    m = np.max(np.abs(residual)) if np.max(np.abs(residual)) > 0 else 1.0
    return (residual / m).astype(np.float32)



# In[12]:


# ---------------- helpers ----------------
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print("Saved:", path)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_listdir(p):
    try:
        return os.listdir(p)
    except Exception:
        return []

# ---------------- denoise / preprocessing ----------------
def to_gray(img):
    if img is None:
        return None
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def resize_to(img, size=IMG_SIZE):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_img(img):
    # keep simple stable scaling: divide by max or by 255 if max==0
    arr = img.astype(np.float32)
    m = arr.max() if arr.max() > 0 else 255.0
    return arr / float(m)

def denoise_wavelet_safe(img, wavelet='db4', level=2):
    arr = np.asarray(img, dtype=np.float32)
    try:
        coeffs = pywt.wavedec2(arr, wavelet=wavelet, level=level)
    except Exception:
        return arr.copy()
    # robust MAD thresholding
    details = []
    for d in coeffs[1:]:
        for comp in d:
            details.append(np.ravel(np.nan_to_num(comp)))
    if len(details) == 0:
        return arr.copy()
    details = np.concatenate(details)
    mad = np.median(np.abs(details - np.median(details)))
    sigma = mad / 0.6745 if mad > 0 else 0.0
    uthresh = sigma * np.sqrt(2 * np.log(arr.size + 1e-12))
    new_coeffs = [coeffs[0]]
    for d in coeffs[1:]:
        new_level = tuple(pywt.threshold(np.nan_to_num(comp), value=uthresh, mode='soft') for comp in d)
        new_coeffs.append(new_level)
    den = pywt.waverec2(new_coeffs, wavelet)
    den = den[:arr.shape[0], :arr.shape[1]]
    return np.nan_to_num(den, nan=arr).astype(np.float32)

def denoise_mentor(img, method=DENOISE_METHOD):
    if method == "wiener":
        return scipy_wiener(img, mysize=(5,5)).astype(np.float32)
    return denoise_wavelet_safe(img, wavelet='db4', level=2)

def preprocess_image_path(path, method=DENOISE_METHOD, size=IMG_SIZE):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = to_gray(img)
    img = resize_to(img, size=size)
    img = normalize_img(img)
    den = denoise_mentor(img, method=method)
    residual = (img - den).astype(np.float32)
    # normalize residual by its max abs to keep values stable (-1..1)
    m = np.max(np.abs(residual)) if np.max(np.abs(residual)) > 0 else 1.0
    return (residual / m).astype(np.float32)


# In[13]:


# ---------------- dataset scanning ----------------
def process_dataset_dir(base_dir, save_path=None):
    residuals = {}
    if not os.path.exists(base_dir):
        print(f"⚠ Dataset missing: {base_dir}")
        return residuals
    scanners = sorted([d for d in safe_listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    for scanner in tqdm(scanners, desc=f"Processing {os.path.basename(base_dir)}"):
        scanner_dir = os.path.join(base_dir, scanner)
        collected = []
        # detect dpi subfolders
        subdirs = [os.path.join(scanner_dir, s) for s in safe_listdir(scanner_dir) if os.path.isdir(os.path.join(scanner_dir, s))]
        if subdirs:
            for sdir in subdirs:
                files = [os.path.join(sdir, f) for f in safe_listdir(sdir) if f.lower().endswith(VALID_EXTS)]
                for fp in files:
                    r = preprocess_image_path(fp)
                    if r is not None:
                        collected.append(r)
        else:
            files = [os.path.join(scanner_dir, f) for f in safe_listdir(scanner_dir) if f.lower().endswith(VALID_EXTS)]
            for fp in files:
                r = preprocess_image_path(fp)
                if r is not None:
                    collected.append(r)
        residuals[scanner] = collected
    if save_path:
        save_pickle(residuals, save_path)
    return residuals


# In[14]:


 #---------------- run scanning for all 3 datasets ----------------#
OFFICIAL_OUT = os.path.join(OUTPUT_DIR, "official_residuals.pkl")
WIKI_OUT     = os.path.join(OUTPUT_DIR, "wikipedia_residuals.pkl")
FLATFIELD_OUT= os.path.join(OUTPUT_DIR, "flatfield_residuals.pkl")

official_residuals = process_dataset_dir(OFFICIAL_DIR, save_path=OFFICIAL_OUT)
wikipedia_residuals = process_dataset_dir(WIKI_DIR, save_path=WIKI_OUT)
if os.path.exists(FLATFIELD_DIR):
    flatfield_residuals = process_dataset_dir(FLATFIELD_DIR, save_path=FLATFIELD_OUT)
else:
    flatfield_residuals = {}
    print("Flatfield folder missing; continuing without flatfield.")

print("Official scanners:", list(official_residuals.keys())[:10])
print("Wikipedia scanners:", list(wikipedia_residuals.keys())[:10])
print("Flatfield scanners:", list(flatfield_residuals.keys())[:10])


# ---------------- compute fingerprints ----------------
FP_OUT = os.path.join(OUTPUT_DIR, "scanner_fingerprints.pkl")
FP_KEYS_OUT = os.path.join(OUTPUT_DIR, "fp_keys.npy")

scanner_fps = {}
if flatfield_residuals:
    for scanner, res_list in tqdm(flatfield_residuals.items(), desc="Computing fingerprints"):
        if not res_list:
            scanner_fps[scanner] = np.zeros(IMG_SIZE, dtype=np.float32)
            continue
        stack = np.stack(res_list, axis=0)
        scanner_fps[scanner] = np.mean(stack, axis=0).astype(np.float32)
else:
    # No flatfield available: create zero fingerprints for union of scanners found in official/wiki
    all_scanners = sorted(set(list(official_residuals.keys()) + list(wikipedia_residuals.keys())))
    print("No flatfield found: creating zero fingerprints for scanners:", len(all_scanners))
    for sc in all_scanners:
        scanner_fps[sc] = np.zeros(IMG_SIZE, dtype=np.float32)

save_pickle(scanner_fps, FP_OUT)
np.save(FP_KEYS_OUT, np.array(sorted(list(scanner_fps.keys()))))
print("Saved fingerprints:", FP_OUT)



# In[15]:


# ---------------- feature extraction helpers (improved) ----------------
def corr2d(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / denom) if denom > 0 else 0.0

def fft_radial_energy(img, K=6, log_scale=True):
    f = fftshift(fft2(img))
    mag = np.abs(f)
    if log_scale:
        mag = 20 * np.log1p(mag)
    h, w = mag.shape
    cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    bins = np.linspace(0, r.max()+1e-12, K+1)
    feats = []
    for i in range(K):
        mask = (r >= bins[i]) & (r < bins[i+1])
        sel = mag[mask]
        feats.append(float(np.mean(sel)) if sel.size else 0.0)
    return feats

def lbp_hist_safe(img, P=8, R=1.0, bins=None):
    if bins is None:
        bins = P + 2   # classic 'uniform' bins
    rng = float(img.max() - img.min())
    if rng < 1e-12:
        g8 = np.zeros_like(img, dtype=np.uint8)
    else:
        g8 = ((img - img.min()) / (rng + 1e-12) * 255).astype(np.uint8)
    codes = local_binary_pattern(g8, P, R, method="uniform")
    hist, _ = np.histogram(codes.ravel(), bins=np.arange(bins+1), density=True)
    return hist.astype(np.float32).tolist()

def glcm_features(img, distances=(1,), angles=(0,)):
    # img expected float-like 0..1 or similar; convert to uint8 for graycomatrix
    try:
        g8 = img_as_ubyte((img - img.min()) / ((img.max() - img.min()) + 1e-12))
    except Exception:
        g8 = img_as_ubyte(np.clip(img, 0.0, 1.0))
    try:
        glcm = graycomatrix(g8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        contrast = float(graycoprops(glcm, 'contrast')[0,0])
        homogeneity = float(graycoprops(glcm, 'homogeneity')[0,0])
        energy = float(graycoprops(glcm, 'energy')[0,0])
        correlation = float(graycoprops(glcm, 'correlation')[0,0])
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]
    return [contrast, homogeneity, energy, correlation]


# In[16]:


# ---------------- combine Official + Wikipedia residuals and extract features ----------------
combined_residuals = {}
# merge official then wiki (so order is predictable)
for sc, lst in official_residuals.items():
    combined_residuals.setdefault(sc, []).extend(lst)
for sc, lst in wikipedia_residuals.items():
    combined_residuals.setdefault(sc, []).extend(lst)

print("Combined scanners count:", len(combined_residuals))

fp_keys = sorted(list(scanner_fps.keys()))
features = []
labels = []
img_list = []

for scanner, res_list in tqdm(combined_residuals.items(), desc="Extracting features (official+wiki)"):
    for res in res_list:
        # ensure shapes
        if res is None:
            continue
        # correlation features across fingerprints (11 typically)
        v_corr = [corr2d(res, scanner_fps.get(k, np.zeros_like(res))) for k in fp_keys]
        # FFT radial energies (6)
        v_fft = fft_radial_energy(res, K=6, log_scale=True)
        # LBP (10 bins using P=8 uniform => P+2=10)
        v_lbp = lbp_hist_safe(res, P=8, R=1.0, bins=10)
        # GLCM four features
        v_glcm = glcm_features(res, distances=(1,), angles=(0,))
        # final feature vector
        feat_vec = v_corr + v_fft + v_lbp + v_glcm
        features.append(feat_vec)
        labels.append(scanner)
        img_list.append(np.expand_dims(res, -1))  # H,W,1

if len(features) == 0:
    raise RuntimeError("No features extracted — check dataset folders and image readability.")

FEATURES_OUT = os.path.join(OUTPUT_DIR, "combined_features.pkl")
save_pickle({"features": features, "labels": labels, "fp_keys": fp_keys}, FEATURES_OUT)
print("Feature vector length:", len(features[0]), "samples:", len(features))


# ---------------- prepare final arrays and encoders ----------------
features = np.array(features, dtype=np.float32)
labels = np.array(labels, dtype=object)
X_img = np.array(img_list, dtype=np.float32)
y_raw = np.array(labels, dtype=object)

print("X_img shape:", X_img.shape)
print("features shape:", features.shape)
print("num labels:", y_raw.shape)

le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)

scaler = StandardScaler()
X_feat_scaled = scaler.fit_transform(features)

# Save final artifacts (Notebooks/ used by training & backend inference)
np.save(os.path.join(OUTPUT_DIR, "X_img.npy"), X_img)
np.save(os.path.join(OUTPUT_DIR, "X_feat.npy"), X_feat_scaled)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y_encoded)

with open(os.path.join(OUTPUT_DIR, "hybrid_label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)
with open(os.path.join(OUTPUT_DIR, "hybrid_feat_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("Saved: X_img.npy, X_feat.npy, y.npy and encoder/scaler in", OUTPUT_DIR)


# ---------------- sanity checks & diagnostics ----------------
for p in ["official_residuals.pkl", "wikipedia_residuals.pkl", "flatfield_residuals.pkl",
          "scanner_fingerprints.pkl", "fp_keys.npy", "combined_features.pkl", "X_img.npy", "X_feat.npy", "y.npy"]:
    full = os.path.join(OUTPUT_DIR, p)
    print(p, "->", "FOUND" if os.path.exists(full) else "MISSING")

print("Classes:", list(le.classes_))
print("Feature vector length saved:", X_feat_scaled.shape[1])


# ---------------- quick diagnostic helper (optional) ----------------
def diagnostic_check_data_dirs(data_root=r"D:\Infosys_AI-Tracefinder\Data"):
    VALID_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    CHECK_DIRS = {
        "Official": os.path.join(data_root, "Official"),
        "Wikipedia": os.path.join(data_root, "Wikipedia"),
        "Flatfield": os.path.join(data_root, "Flatfield")
    }
    def find_scanner_files(base):
        scanners = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base,d))])
        info = {}
        for sc in scanners:
            scpath = os.path.join(base, sc)
            candidates = []
            subdirs = [d for d in os.listdir(scpath) if os.path.isdir(os.path.join(scpath, d))]
            if subdirs:
                for s in subdirs:
                    sdpath = os.path.join(scpath, s)
                    for f in os.listdir(sdpath):
                        if f.lower().endswith(VALID_EXTS):
                            candidates.append(os.path.join(sdpath, f))
            else:
                for f in os.listdir(scpath):
                    if f.lower().endswith(VALID_EXTS):
                        candidates.append(os.path.join(scpath, f))
            info[sc] = candidates
        return info

    for name, path in CHECK_DIRS.items():
        print("\n===", name, "at", path, "===")
        if not os.path.exists(path):
            print("MISSING:", path); continue
        info = find_scanner_files(path)
        for sc, files in sorted(info.items(), key=lambda x: -len(x[1])):
            print(f"{sc:20s} => {len(files):4d} files", end="")
            if len(files)>0:
                print("  sample:", files[0])
            else:
                print()
        if "HP" in info:
            files = info["HP"]
            print("\nHP present: count =", len(files))
            unreadable = []
            for f in files:
                img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                if img is None:
                    unreadable.append(f)
            if unreadable:
                print("Unreadable HP files (cv2.imread returned None):", len(unreadable))
            else:
                if files:
                    print("All HP files readable by OpenCV. First HP file preview:", files[0])

# Call diagnostic if you want:
# diagnostic_check_data_dirs()

# ---------------- END ----------------

