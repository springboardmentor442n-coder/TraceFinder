#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 0 — Diagnostics
import os, sys, pickle, numpy as np
import tensorflow as tf

print("Python:", sys.version.splitlines()[0])
print("TensorFlow:", tf.__version__)
print("Physical GPUs:", tf.config.list_physical_devices('GPU'))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))

# Work folder used for saves (change if needed)
WORK = r"D:\Infosys_AI-Tracefinder\Notebooks"
OUTPUT_DIR = r"D:\Infosys_AI-Tracefinder\Output"
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("WORK:", WORK)
print("OUTPUT_DIR:", OUTPUT_DIR)


# In[28]:


# Cell 1 — Imports & config
import os, math, pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import cv2
import pywt
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern
from scipy.fft import fft2, fftshift

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Paths (edit if needed)
DATA_ROOT    = r"D:\Infosys_AI-Tracefinder\Data"
OFFICIAL_DIR = os.path.join(DATA_ROOT, "Official")
FLATFIELD_DIR= os.path.join(DATA_ROOT, "Flatfield")
WIKI_DIR     = os.path.join(DATA_ROOT, "Wikipedia")

WORK         = r"D:\Infosys_AI-Tracefinder\Notebooks"  # where outputs / models are saved
OUTPUT_DIR   = r"D:\Infosys_AI-Tracefinder\Output"     # CSVs / pickles

os.makedirs(WORK, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (256,256)   # residual image size

# Training-config (CPU friendly)
BATCH = 16        # small batch for CPU; increase if you have GPU
EPOCHS = 32      # ceiling (EarlyStopping used)
LEARNING_RATE = 1e-3
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("Paths:")
print(" OFFICIAL:", OFFICIAL_DIR)
print(" FLATFIELD:", FLATFIELD_DIR)
print(" WIKI:", WIKI_DIR)
print(" WORK:", WORK)
print(" OUTPUT:", OUTPUT_DIR)
print("TensorFlow:", tf.__version__)
print("BATCH, EPOCHS, LR:", BATCH, EPOCHS, LEARNING_RATE)


# In[29]:


# ==== PATCH: Restore old stable training settings ====

BATCH = 32
EPOCHS = 64
LEARNING_RATE = 1e-3
RANDOM_STATE = 42

import numpy as np, tensorflow as tf, os

# restore old reproducibility behaviour
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# restore older callbacks (same as previous 91% run)
CKPT = os.path.join(WORK, "scanner_hybrid_best.keras")
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(CKPT, save_best_only=True, monitor="val_accuracy")
]

print("PATCH APPLIED → BATCH=32, EPOCHS=64, LR=1e-3, CKPT restored")


# In[3]:


# Cell 2 — Image helpers & residual extraction
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None and img.ndim == 3 else img

def resize_to(img, size=IMG_SIZE):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_img(img):
    im = img.astype(np.float32)
    maxv = im.max() if im.max() > 0 else 1.0
    return im / maxv

def denoise_wavelet(img):
    coeffs = pywt.dwt2(img, "haar")
    cA, (cH, cV, cD) = coeffs
    cH[:] = 0; cV[:] = 0; cD[:] = 0
    rec = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    # if shapes differ, resize to original shape
    if rec.shape != img.shape:
        rec = resize_to(rec, img.shape[::-1]) if img.ndim==2 else resize_to(rec, (img.shape[1], img.shape[0]))
    return rec

def compute_residual_from_tiff(path):
    """
    Returns residual array shape (H,W) float32 normalized (-1..1 approx)
    Raises RuntimeError if image not readable.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Unreadable image: {path}")
    g = to_gray(img)
    if g is None:
        raise RuntimeError(f"Cannot convert to gray: {path}")
    g = resize_to(g, IMG_SIZE)
    g = normalize_img(g)
    den = denoise_wavelet(g)
    res = (g - den).astype(np.float32)
    m = np.max(np.abs(res)) if np.max(np.abs(res)) > 0 else 1.0
    return (res / m).astype(np.float32)


# In[4]:


# Cell 3 — Dataset scanner (handles both structures)
def process_dataset(base_dir, dataset_name="dataset"):
    result = {}
    if not os.path.exists(base_dir):
        print(f"[WARN] dataset folder not found: {base_dir}")
        return result

    scanners = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,d))])
    print(f"\nScanning dataset '{dataset_name}' at: {base_dir}  (scanners: {len(scanners)})")
    for scanner in tqdm(scanners, desc=f"{dataset_name} scanners"):
        scanner_path = os.path.join(base_dir, scanner)
        if not os.path.isdir(scanner_path):
            continue
        # detect DPI subfolders
        subdirs = [d for d in os.listdir(scanner_path) if os.path.isdir(os.path.join(scanner_path, d))]
        residuals = []
        if subdirs:
            # structure: scanner/dpi/*.tif
            for dpi in subdirs:
                dpi_path = os.path.join(scanner_path, dpi)
                if not os.path.isdir(dpi_path):
                    continue
                files = [os.path.join(dpi_path, f) for f in os.listdir(dpi_path)
                         if f.lower().endswith((".tif", ".tiff")) and not f.startswith("._")]
                for fpath in files:
                    try:
                        res = compute_residual_from_tiff(fpath)
                        residuals.append(res)
                    except Exception as e:
                        # keep going, but you can log
                        # print("SKIP:", fpath, "->", e)
                        continue
        else:
            # structure: scanner/*.tif (flatfield)
            files = [os.path.join(scanner_path, f) for f in os.listdir(scanner_path)
                     if f.lower().endswith((".tif", ".tiff")) and not f.startswith("._")]
            for fpath in files:
                try:
                    res = compute_residual_from_tiff(fpath)
                    residuals.append(res)
                except Exception as e:
                    # print("SKIP:", fpath, "->", e)
                    continue
        result[scanner] = residuals
    return result

# Run scans
print("Processing Official ...")
res_official = process_dataset(OFFICIAL_DIR, "Official")

print("\nProcessing Wikipedia ...")
res_wiki = process_dataset(WIKI_DIR, "Wikipedia")

print("\nProcessing Flatfield ...")
res_flatfield = process_dataset(FLATFIELD_DIR, "Flatfield")


# In[5]:


# Cell 4 — Summaries & save flatfield residuals
def summarize_dict(d, name):
    total = sum(len(v) for v in d.values())
    print(f"\nSummary: {name}  (scanners: {len(d)}, total residuals: {total})")
    for k in sorted(d.keys()):
        print("  ", k, "->", len(d[k]))

summarize_dict(res_official, "Official")
summarize_dict(res_wiki, "Wikipedia")
summarize_dict(res_flatfield, "Flatfield")

# Save flatfield residuals for reuse
flatfield_pkl = os.path.join(WORK, "flatfield_residuals.pkl")   # saved to WORK
with open(flatfield_pkl, "wb") as f:
    pickle.dump(res_flatfield, f)
print("\nSaved flatfield residuals ->", flatfield_pkl)


# In[6]:


# Cell 5 — fingerprints (mean of flatfield residuals)
FP_OUT = os.path.join(WORK, "scanner_fingerprints.pkl")
FP_KEYS = os.path.join(WORK, "fp_keys.npy")

scanner_fps = {}
for scanner, res_list in res_flatfield.items():
    if not res_list:
        # no flatfield images -> zeros fingerprint
        print("Warning: no flatfield for", scanner, "- using zero fingerprint")
        scanner_fps[scanner] = np.zeros(IMG_SIZE, dtype=np.float32)
        continue
    stack = np.stack(res_list, axis=0)  # (N, H, W)
    scanner_fps[scanner] = np.mean(stack, axis=0).astype(np.float32)

with open(FP_OUT, "wb") as f:
    pickle.dump(scanner_fps, f)
np.save(FP_KEYS, np.array(sorted(scanner_fps.keys())))
print("Saved fingerprints:", FP_OUT)
print("Saved fp keys:", FP_KEYS)


# In[7]:


# Cell 6 — Merge residuals and extract features
# Merge dictionaries: official then wiki
combined_residuals = {}
all_scanners = sorted(set(list(res_official.keys()) + list(res_wiki.keys())))
for sc in all_scanners:
    list_off = res_official.get(sc, [])
    list_w   = res_wiki.get(sc, [])
    combined_residuals[sc] = list_off + list_w

print("Merged scanners count:", len(combined_residuals))

# feature helper functions
def corr2d(a, b):
    a = a.ravel().astype(np.float32); b = b.ravel().astype(np.float32)
    a -= a.mean(); b -= b.mean()
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float((a @ b) / denom) if denom > 0 else 0.0

def fft_radial_energy(img, K=6):
    f = fftshift(fft2(img)); mag = np.abs(f)
    h,w = mag.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    bins = np.linspace(0, r.max()+1e-6, K+1)
    feats = []
    for i in range(K):
        mask = (r >= bins[i]) & (r < bins[i+1])
        feats.append(float(np.mean(mag[mask])) if np.any(mask) else 0.0)
    return feats

def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    if rng < 1e-12:
        return [0.0] * (P+2)
    g = (img - img.min()) / (rng + 1e-8)
    g8 = (g*255).astype(np.uint8)
    codes = local_binary_pattern(g8, P, R, method="uniform")
    hist, _ = np.histogram(codes, bins=np.arange(P+3), density=True)
    return hist.astype(np.float32).tolist()

# prepare fingerprint keys in consistent order
fp_keys = sorted(scanner_fps.keys())

features = []
labels = []
X_img_list = []

for sc in tqdm(sorted(combined_residuals.keys()), desc="Extract features"):
    res_list = combined_residuals[sc]
    for res in res_list:
        v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]
        v_fft = fft_radial_energy(res, K=6)
        v_lbp = lbp_hist_safe(res, P=8, R=1.0)
        feat_vec = v_corr + v_fft + v_lbp
        features.append(feat_vec)
        labels.append(sc)
        X_img_list.append(res[..., None])  # (H,W,1)

X_feat = np.array(features, dtype=np.float32)
X_img  = np.array(X_img_list, dtype=np.float32)
y_raw  = np.array(labels, dtype=object)

print("Prepared shapes -> X_img:", X_img.shape, "X_feat:", X_feat.shape, "y:", y_raw.shape)

# Encode & scale
le = LabelEncoder()
y = le.fit_transform(y_raw)
scaler = StandardScaler()
X_feat_scaled = scaler.fit_transform(X_feat)

# Save datasets & preprocessors to WORK (Notebooks)
np.save(os.path.join(WORK, "X_img.npy"), X_img)
np.save(os.path.join(WORK, "X_feat.npy"), X_feat_scaled)
np.save(os.path.join(WORK, "y.npy"), y)

with open(os.path.join(WORK, "hybrid_label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)
with open(os.path.join(WORK, "hybrid_feat_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(WORK, "scanner_fingerprints.pkl"), "wb") as f:
    pickle.dump(scanner_fps, f)
np.save(os.path.join(WORK, "fp_keys.npy"), np.array(fp_keys))

print("Saved X_img, X_feat, y, label encoder, scaler, fingerprints to WORK")
print("Classes:", list(le.classes_))
print("Feature vector length:", X_feat.shape[1])


# In[8]:


# Cell 7 — Build CPU-friendly Hybrid CNN
IMG_SHAPE = X_img.shape[1:]   # e.g. (256,256,1)
FEAT_DIM  = X_feat.shape[1]
num_classes = len(le.classes_)

img_in  = keras.Input(shape=IMG_SHAPE, name="residual", dtype="float32")
feat_in = keras.Input(shape=(FEAT_DIM,), name="handcrafted", dtype="float32")

# small high-pass kernel layer (fixed)
hp_kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], dtype=np.float32).reshape((3,3,1,1))
hp_layer = layers.Conv2D(1, (3,3), padding="same", use_bias=False, trainable=False, name="hp_filter")
hp_layer.build((None,) + IMG_SHAPE)
hp_layer.set_weights([hp_kernel])
hp = hp_layer(img_in)

# Reduced CNN branch (16/32/64)
x = layers.Conv2D(16, (3,3), activation="relu", padding="same")(hp)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Dropout(0.15)(x)

x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Dropout(0.15)(x)

x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
x = layers.GlobalAveragePooling2D()(x)

# Handcrafted branch
f = layers.Dense(32, activation="relu")(feat_in)
f = layers.Dropout(0.2)(f)

# Fusion
z = layers.Concatenate()([x, f])
z = layers.Dense(128, activation="relu")(z)
z = layers.Dropout(0.25)(z)
out = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(z)

model = keras.Model(inputs=[img_in, feat_in], outputs=out)

opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()


# In[30]:


# Cell 8 — Train hybrid model (tf.data pipeline + callbacks + evaluation)
# Train/val split
X_img_tr, X_img_te, X_feat_tr, X_feat_te, y_tr, y_te = train_test_split(
    X_img, X_feat_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

num_classes = len(le.classes_)
y_tr_cat = keras.utils.to_categorical(y_tr, num_classes)
y_te_cat = keras.utils.to_categorical(y_te, num_classes)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices(((X_img_tr, X_feat_tr), y_tr_cat)) \
    .shuffle(buffer_size=len(y_tr)) \
    .batch(BATCH).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(((X_img_te, X_feat_te), y_te_cat)) \
    .batch(BATCH).prefetch(AUTOTUNE)

# callbacks & checkpoint files in WORK
CKPT = os.path.join(WORK, "scanner_hybrid_best.keras")
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-6),
    keras.callbacks.ModelCheckpoint(CKPT, save_best_only=True, monitor="val_accuracy")
]

print("Training on:", X_img_tr.shape[0], "samples; batch_size:", BATCH)

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

# Save final model + history (WORK)
MODEL_OUT = os.path.join(WORK, "scanner_hybrid_final.keras")
HIST_OUT  = os.path.join(WORK, "hybrid_training_history.pkl")
model.save(MODEL_OUT)
with open(HIST_OUT, "wb") as f:
    pickle.dump(history.history, f)

print("Saved model:", MODEL_OUT)
print("Saved history:", HIST_OUT)


# In[31]:


# Cell 9 — Evaluation & confusion matrices
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Get predictions on validation set
test_ds = tf.data.Dataset.from_tensor_slices(((X_img_te, X_feat_te), y_te_cat)).batch(BATCH)
probs = model.predict(test_ds, verbose=0)
pred_idx = np.argmax(probs, axis=1)
true_idx = y_te

acc = accuracy_score(true_idx, pred_idx)
print("Validation accuracy:", acc)
print("\nClassification report:")
print(classification_report(true_idx, pred_idx, target_names=list(le.classes_), zero_division=0))

# Raw CM
cm = confusion_matrix(true_idx, pred_idx)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title("Confusion Matrix (counts)")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.show()

# Normalized CM (per-row)
cm_norm = (cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True)+1e-12))
plt.figure(figsize=(12,10))
sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title("Confusion Matrix (normalized)")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.show()


# In[32]:


# Cell 10 — Save metadata for inference (optional)
meta = {
    "classes": list(map(str, le.classes_)),
    "feature_columns_count": X_feat.shape[1],
    "n_samples": X_img.shape[0]
}
meta_path = os.path.join(WORK, "hybrid_metadata.pkl")
with open(meta_path, "wb") as f:
    pickle.dump(meta, f)
print("Saved hybrid metadata:", meta_path)


# In[33]:


import os, cv2, pywt, pickle, numpy as np
from tensorflow import keras
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.fft import fft2, fftshift
from skimage import img_as_ubyte

WORK = r"D:\Infosys_AI-Tracefinder\Notebooks"
sample_img = r"D:\Infosys_AI-Tracefinder\Data\Flatfield\HP\150.tif"

# load artifacts
hyb = keras.models.load_model(os.path.join(WORK, "scanner_hybrid_best.keras"))
hyb_le = pickle.load(open(os.path.join(WORK, "hybrid_label_encoder.pkl"), "rb"))
hyb_scaler = pickle.load(open(os.path.join(WORK, "hybrid_feat_scaler.pkl"), "rb"))
scanner_fps = pickle.load(open(os.path.join(WORK, "scanner_fingerprints.pkl"), "rb"))
fp_keys = list(np.load(os.path.join(WORK, "fp_keys.npy")).tolist())

# helpers (compact)
def make_residual(path, size=(256,256)):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: raise ValueError("Unreadable image")
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
    g = cv2.resize(g, size).astype(np.float32); g /= (g.max() if g.max()>0 else 1.0)
    cA,(cH,cV,cD)=pywt.dwt2(g,"haar"); den=pywt.idwt2((cA,(np.zeros_like(cH),np.zeros_like(cV),np.zeros_like(cD))),"haar")
    den = cv2.resize(den, size); r = (g-den).astype(np.float32); m = np.max(np.abs(r)) or 1.0
    return r / m

def corr2d(a,b):
    a=a.ravel().astype(np.float32); b=b.ravel().astype(np.float32); a-=a.mean(); b-=b.mean()
    d=np.linalg.norm(a)*np.linalg.norm(b); return float((a@b)/d) if d>0 else 0.0

def fft_energy(img,K=6):
    f=fftshift(fft2(img)); mag=np.abs(f); h,w=mag.shape; cy,hx=h//2,w//2
    yy,xx=np.ogrid[:h,:w]; r=np.sqrt((yy-cy)**2+(xx-hx)**2); bins=np.linspace(0,r.max()+1e-12,K+1)
    return [float(np.mean(mag[(r>=bins[i])&(r<bins[i+1])])) if np.any((r>=bins[i])&(r<bins[i+1])) else 0.0 for i in range(K)]

def lbp_hist(img,P=8):
    g8=img_as_ubyte((img-img.min())/((img.max()-img.min())+1e-12)); codes=local_binary_pattern(g8,P,1.0,method="uniform")
    h,_=np.histogram(codes.ravel(), bins=np.arange(P+3), density=True); return h.astype(np.float32).tolist()

def glcm_feats(img):
    g8=img_as_ubyte((img-img.min())/((img.max()-img.min())+1e-12)); gl=graycomatrix(g8,[1],[0],256,symmetric=True,normed=True)
    return [float(graycoprops(gl,p)[0,0]) for p in ("contrast","homogeneity","energy","correlation")]

def make_feat_from_res(res):
    v=[corr2d(res, scanner_fps.get(k, np.zeros_like(res))) for k in fp_keys]
    v+=fft_energy(res); v+=lbp_hist(res); v+=glcm_feats(res)
    return np.array(v, dtype=np.float32).reshape(1,-1)

# handle path styles (just for clarity — hybrid features don't use resolution directly)
bn = os.path.basename(sample_img); parent = os.path.basename(os.path.dirname(sample_img))
is_flatfield = bn.split('.')[0].isdigit()  # filename numeric => flatfield
# resolution_level = int(bn.split('.')[0]) if is_flatfield else (int(parent) if parent.isdigit() else 0)

# compute residual + features, pad if scaler expects different dims
res = make_residual(sample_img)
res_in = res.reshape(1,256,256,1)
feat = make_feat_from_res(res)
expected = getattr(hyb_scaler, "n_features_in_", feat.shape[1])
if feat.shape[1] != expected:
    new = np.zeros((1, expected), dtype=np.float32); new[0,:min(feat.shape[1],expected)] = feat[0,:min(feat.shape[1],expected)]; feat = new
feat_s = hyb_scaler.transform(feat)

probs = hyb.predict([res_in, feat_s], verbose=0)[0]; idx = int(np.argmax(probs))
print("Hybrid:", hyb_le.inverse_transform([idx])[0], " confidence:", float(probs[idx]))

