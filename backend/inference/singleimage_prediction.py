#!/usr/bin/env python
# coding: utf-8

# In[7]:


# backend/inference/singleimage_prediction.py
import os
import cv2
import numpy as np
import pickle
import joblib
from collections import Counter
from scipy.stats import skew, kurtosis
from scipy.fft import fft2, fftshift
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage import img_as_ubyte
import traceback

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras

# ----------------- Utilities -----------------
def _to_py(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return _to_py(x.tolist())
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_py(v) for k, v in x.items()}
    return x

# ----------------- Paths & try-locate -----------------
MODELS_DIRS_TO_TRY = [
    r"D:\Infosys_AI-Tracefinder\Notebooks",
    r"D:\Infosys_AI-Tracefinder\Output",
    os.path.join(os.getcwd(), "models"),
    os.getcwd()
]

def find_file(filename):
    for d in MODELS_DIRS_TO_TRY:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None

# default names (match your notebook outputs)
CNN_LE_FNAME = "cnn_label_encoder.pkl"
CNN_MODEL_FNAME = "cnn_residual_best.keras"
HYB_MODEL_FNAME = "scanner_hybrid_best.keras"
SKL_MODEL_FNAME = "best_model.pkl"
SKL_SCALER_FNAME = "scaler.pkl"
SKL_LE_FNAME = "label_encoder.pkl"
HYB_SCALER_FNAME = "hybrid_feat_scaler.pkl"
HYB_LE_FNAME = "hybrid_label_encoder.pkl"
FPICKS_FNAME = "scanner_fingerprints.pkl"
FP_KEYS = "fp_keys.npy"

# ----------------- Basic feature (CSV) helpers -----------------
def guess_resolution_from_path(path):
    p = path.replace("\\","/").lower()
    if "/150" in p or "_150" in p or "/150/" in p:
        return 150
    if "/300" in p or "_300" in p or "/300/" in p:
        return 300
    return 0

def compute_basic_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    h, w = img_gray.shape[:2]
    aspect_ratio = float(w)/float(h) if h!=0 else 0.0
    file_size_kb = float(os.path.getsize(img_path))/1024.0

    imgf = img_gray.astype(np.float32)
    maxv = imgf.max() if imgf.max()>0 else 1.0
    imgn = imgf / maxv

    mean_intensity = float(imgn.mean())
    std_intensity = float(imgn.std())

    flat = imgn.flatten()
    skewness = float(skew(flat)) if flat.size>0 else 0.0
    kurt = float(kurtosis(flat)) if flat.size>0 else 0.0

    hist, _ = np.histogram(flat, bins=256, range=(0.0,1.0), density=True)
    hist = hist + 1e-12
    entropy = float(-np.sum(hist * np.log2(hist)))

    edges = cv2.Canny((imgn*255).astype(np.uint8), 100, 200)
    edge_density = float(edges.astype(bool).sum()) / (w*h) if (w*h)>0 else 0.0

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

def make_feature_vector(image_path):
    # CSV/basic features (11)
    res_level = guess_resolution_from_path(image_path)
    feats = compute_basic_features(image_path)
    ordered = [
        float(res_level),
        float(feats["width"]),
        float(feats["height"]),
        float(feats["aspect_ratio"]),
        float(feats["file_size_kb"]),
        float(feats["mean_intensity"]),
        float(feats["std_intensity"]),
        float(feats["skewness"]),
        float(feats["kurtosis"]),
        float(feats["entropy"]),
        float(feats["edge_density"])
    ]
    return np.array(ordered, dtype=np.float32).reshape(1, -1)

# ----------------- Residual (CNN) helper -----------------
def make_residual_image(image_path, target_size=(256,256)):
    import pywt
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    img_resized = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    maxv = img_resized.max() if img_resized.max()>0 else 1.0
    imgn = img_resized / maxv
    coeffs = pywt.dwt2(imgn, "haar")
    cA, (cH, cV, cD) = coeffs
    cH[:] = 0; cV[:] = 0; cD[:] = 0
    den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    res = imgn - den
    m = np.max(np.abs(res)) if np.max(np.abs(res))>0 else 1.0
    res = (res / m).astype(np.float32)
    return np.expand_dims(res, axis=(0,-1))   # shape (1,256,256,1)

# ----------------- Hybrid features builders (31) -----------------
def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float((a @ b)/denom) if denom>0 else 0.0

def fft_radial_energy(img, K=6, log_scale=True):
    f = fftshift(fft2(img))
    mag = np.abs(f)
    if log_scale:
        mag = 20 * np.log1p(mag)
    h,w = mag.shape
    cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    bins = np.linspace(0, r.max()+1e-12, K+1)
    feats = []
    for i in range(K):
        mask = (r >= bins[i]) & (r < bins[i+1])
        sel = mag[mask]
        feats.append(float(np.mean(sel)) if sel.size else 0.0)
    return feats

def lbp_hist_safe(img, P=8, R=1.0, bins=None):
    if bins is None:
        bins = P + 2
    rng = float(img.max() - img.min())
    if rng < 1e-12:
        g8 = np.zeros_like(img, dtype=np.uint8)
    else:
        g8 = ((img - img.min())/(rng + 1e-12) * 255).astype(np.uint8)
    codes = local_binary_pattern(g8, P, R, method="uniform")
    hist, _ = np.histogram(codes.ravel(), bins=np.arange(bins+1), density=True)
    return hist.astype(np.float32).tolist()

def glcm_features(img, distances=(1,), angles=(0,)):
    try:
        g8 = img_as_ubyte((img - img.min())/((img.max()-img.min())+1e-12))
    except Exception:
        g8 = img_as_ubyte(np.clip(img,0.0,1.0))
    try:
        glcm = graycomatrix(g8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        return [float(graycoprops(glcm,p)[0,0]) for p in ("contrast","homogeneity","energy","correlation")]
    except Exception:
        return [0.0,0.0,0.0,0.0]

def make_hybrid_feature_vector_from_residual(residual, scanner_fps, fp_keys):
    # residual expected 2D (H,W) float
    # returns (1, N) where N is typically 31 in your new pipeline
    v_corr = [corr2d(residual, scanner_fps.get(k, np.zeros_like(residual))) for k in fp_keys]   # 11
    v_fft = fft_radial_energy(residual, K=6, log_scale=True)                                   # 6
    v_lbp = lbp_hist_safe(residual, P=8, R=1.0, bins=10)                                       # 10
    v_glcm = glcm_features(residual, distances=(1,), angles=(0,))                              # 4
    feat_vec = v_corr + v_fft + v_lbp + v_glcm
    return np.array(feat_vec, dtype=np.float32).reshape(1, -1)

def make_hybrid_feature_vector(image_path, scanner_fps, fp_keys, target_size=(256,256)):
    # convenience: compute residual then features
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Unreadable image for hybrid features.")
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img.copy()
    g = cv2.resize(g, target_size).astype(np.float32)
    g /= (g.max() if g.max()>0 else 1.0)
    # simple wavelet denoise (safe)
    try:
        import pywt
        cA,(cH,cV,cD) = pywt.dwt2(g, "haar")
        cH[:] = cV[:] = cD[:] = 0
        den = pywt.idwt2((cA,(cH,cV,cD)),"haar")
        den = cv2.resize(den, target_size)
    except Exception:
        den = np.zeros_like(g)
    res = (g - den).astype(np.float32)
    m = np.max(np.abs(res)) if np.max(np.abs(res))>0 else 1.0
    res = (res / m).astype(np.float32)
    return make_hybrid_feature_vector_from_residual(res, scanner_fps, fp_keys)

# ----------------- Load artifacts -----------------
# find files
cnn_le_path = find_file(CNN_LE_FNAME)
cnn_model_path = find_file(CNN_MODEL_FNAME)
hyb_model_path = find_file(HYB_MODEL_FNAME)
skl_model_path = find_file(SKL_MODEL_FNAME)
skl_scaler_path = find_file(SKL_SCALER_FNAME)
skl_le_path = find_file(SKL_LE_FNAME)
hyb_scaler_path = find_file(HYB_SCALER_FNAME)
hyb_le_path = find_file(HYB_LE_FNAME)
fpicks_path = find_file(FPICKS_FNAME)
fp_keys_path = find_file(FP_KEYS)

# load encoders / models (best-effort)
cnn_le = None
if cnn_le_path:
    with open(cnn_le_path,"rb") as f: cnn_le = pickle.load(f)

cnn = None
if cnn_model_path:
    try:
        cnn = keras.models.load_model(cnn_model_path)
        try: cnn.predict(np.zeros((1,256,256,1),dtype=np.float32))
        except: pass
    except Exception as e:
        print("Failed to load CNN:", e); cnn = None

hybrid = None
if hyb_model_path:
    try:
        hybrid = keras.models.load_model(hyb_model_path)
        # warm-up: best effort
        try:
            inps=[]
            for inp in hybrid.inputs:
                s = inp.shape
                if len(s)==4: inps.append(np.zeros((1,256,256,1),dtype=np.float32))
                elif len(s)==2: inps.append(np.zeros((1, max(1, int(s[1]) if s[1] is not None else 27)), dtype=np.float32))
                else: inps.append(np.zeros((1,1),dtype=np.float32))
            hybrid.predict(inps)
        except Exception:
            pass
    except Exception as e:
        print("Failed to load Hybrid:", e); hybrid = None

sklearn_model = None
skl_scaler = None
skl_le = None
if skl_model_path:
    try: sklearn_model = joblib.load(skl_model_path)
    except Exception as e: print("Failed to load sklearn model:", e); sklearn_model = None
if skl_scaler_path:
    try: skl_scaler = joblib.load(skl_scaler_path)
    except Exception as e: print("Failed to load sklearn scaler:", e); skl_scaler = None
if skl_le_path:
    try: skl_le = joblib.load(skl_le_path)
    except Exception as e: print("Failed to load sklearn label encoder:", e); skl_le = None

hyb_scaler = None
hyb_le = None
if hyb_scaler_path:
    try: hyb_scaler = joblib.load(hyb_scaler_path)
    except Exception as e: print("Failed to load hybrid scaler:", e); hyb_scaler = None
if hyb_le_path:
    try: hyb_le = joblib.load(hyb_le_path)
    except Exception as e: print("Failed to load hybrid label encoder:", e); hyb_le = None

scanner_fps = {}
fp_keys = []
if fpicks_path:
    try:
        with open(fpicks_path,"rb") as f: scanner_fps = pickle.load(f)
    except Exception as e:
        print("Failed to load scanner_fingerprints:", e)
if fp_keys_path:
    try:
        fp_keys = list(np.load(fp_keys_path).tolist())
    except Exception as e:
        fp_keys = sorted(list(scanner_fps.keys()))

print("\nSUMMARY of loaded models/artifacts:")
print(" CNN:", "loaded" if cnn is not None else "missing")
print(" Hybrid:", "loaded" if hybrid is not None else "missing")
print(" Sklearn:", "loaded" if sklearn_model is not None else "missing")
print(" CNN label encoder:", "loaded" if cnn_le is not None else "missing")
print(" Sklearn label encoder:", "loaded" if skl_le is not None else "missing")
print(" Sklearn scaler:", "loaded" if skl_scaler is not None else "missing")
print(" Hybrid scaler:", "loaded" if hyb_scaler is not None else "missing")
print(" Hybrid label encoder:", "loaded" if hyb_le is not None else "missing")
print(" scanner_fps:", "loaded" if scanner_fps else "missing")
print(" fp_keys:", len(fp_keys), "entries\n")

# ----------------- Prediction main -----------------
def predict_single_image(image_path, model_choice="all", verbose=True):
    """
    model_choice: "cnn", "hybrid", "sklearn", or "all"
    returns dict with per-model label/confidence/probs and 'selected_model' field.
    """
    results = {"cnn": None, "hybrid": None, "sklearn": None, "ensemble_vote": None, "notes": [], "selected_model": model_choice}

    # 1) residual image for CNN/hybrid image input
    try:
        img_res = make_residual_image(image_path, target_size=(256,256))   # shape (1,256,256,1)
    except Exception as e:
        raise RuntimeError(f"Residual creation failed: {e}")

    # 2) CSV/basic features (11)
    try:
        feats_basic = make_feature_vector(image_path)   # (1,11)
    except Exception as e:
        raise RuntimeError(f"Basic feature extraction failed: {e}")

    # Run CNN?
    if model_choice in ("cnn","all") and cnn is not None:
        try:
            proba = cnn.predict(img_res, verbose=0)[0]
            idx = int(np.argmax(proba))
            label = cnn_le.inverse_transform([idx])[0] if cnn_le is not None else str(idx)
            results["cnn"] = {"label": label, "index": idx, "confidence": float(proba[idx]), "probs": _to_py(proba)}
            if verbose: print("CNN ->", label, float(proba[idx]))
        except Exception as e:
            results["cnn"] = {"error": str(e)}
            results["notes"].append("CNN prediction failed: " + str(e))

    # Run Hybrid?
    if model_choice in ("hybrid","all") and hybrid is not None:
        try:
            # build raw 31-feature vector from residual (uses scanner_fps / fp_keys)
            try:
                # prefer computing features from residual image to match pipeline
                residual_2d = np.squeeze(img_res[0,:,:,0])
                raw_feat = make_hybrid_feature_vector_from_residual(residual_2d, scanner_fps, fp_keys)  # shape (1,31 typically)
            except Exception:
                # fallback compute from image path
                raw_feat = make_hybrid_feature_vector(image_path, scanner_fps, fp_keys)

            # detect expected length (prefer hyb_scaler.n_features_in_, then hybrid model input)
            expected = None
            if hyb_scaler is not None:
                expected = getattr(hyb_scaler, "n_features_in_", None) or getattr(hyb_scaler, "n_features_in", None)
            if expected is None and hybrid is not None:
                try:
                    for inp in hybrid.inputs:
                        s = inp.shape
                        if len(s) == 2 and s[1] is not None:
                            expected = int(s[1]); break
                except Exception:
                    expected = None
            if expected is None:
                expected = raw_feat.shape[1]  # fallback

            # pad/truncate to expected
            if raw_feat.shape[1] != expected:
                adj = np.zeros((1, int(expected)), dtype=np.float32)
                take = min(raw_feat.shape[1], int(expected))
                adj[0, :take] = raw_feat[0, :take]
                feat_for_scaler = adj
                results["notes"].append(f"Hybrid features padded/truncated {raw_feat.shape[1]} -> {expected}")
            else:
                feat_for_scaler = raw_feat

            # scale if scaler present
            try:
                if hyb_scaler is not None:
                    # hyb_scaler may expect dataframe with feature_names_in_ -> handle both
                    if hasattr(hyb_scaler, "feature_names_in_"):
                        import pandas as pd
                        cols = list(getattr(hyb_scaler, "feature_names_in_"))
                        df = pd.DataFrame(np.zeros((1,len(cols))), columns=cols, dtype=np.float32)
                        for i in range(min(feat_for_scaler.shape[1], len(cols))):
                            df.iloc[0,i] = feat_for_scaler[0,i]
                        feat_scaled = hyb_scaler.transform(df)
                    else:
                        feat_scaled = hyb_scaler.transform(feat_for_scaler)
                else:
                    feat_scaled = feat_for_scaler
            except Exception as e:
                results["notes"].append("Hybrid scaler transform failed: " + str(e))
                feat_scaled = feat_for_scaler

            # prepare inputs in correct order: put image where hybrid expects image input
            inputs = []
            img_input_index = None
            for i, inp in enumerate(hybrid.inputs):
                s = inp.shape
                if len(s) == 4:
                    img_input_index = i
                    break
            if img_input_index is None:
                img_input_index = 0
            for i, inp in enumerate(hybrid.inputs):
                if i == img_input_index:
                    inputs.append(img_res)
                else:
                    inputs.append(np.asarray(feat_scaled, dtype=np.float32))

            proba_h = hybrid.predict(inputs, verbose=0)
            if isinstance(proba_h, list): proba_h = proba_h[0]
            proba_h = np.asarray(proba_h)[0]
            idx_h = int(np.argmax(proba_h))
            if hyb_le is not None:
                label_h = hyb_le.inverse_transform([idx_h])[0]
            else:
                label_h = (cnn_le.inverse_transform([idx_h])[0] if cnn_le is not None else str(idx_h))
            results["hybrid"] = {"label": label_h, "index": idx_h, "confidence": float(proba_h[idx_h]), "probs": _to_py(proba_h)}
            if verbose: print("Hybrid ->", label_h, float(proba_h[idx_h]))
        except Exception as e:
            tb = traceback.format_exc()
            results["hybrid"] = {"error": str(e), "trace": tb}
            results["notes"].append("Hybrid processing error: " + str(e))

    # Run Sklearn?
    if model_choice in ("sklearn","all") and sklearn_model is not None and skl_scaler is not None:
        try:
            feats_skl = feats_basic.copy()   # (1,11)
            expected_skl = int(getattr(skl_scaler, "n_features_in_", feats_skl.shape[1]))
            if feats_skl.shape[1] != expected_skl:
                padded = np.zeros((1, expected_skl), dtype=np.float32)
                padded[0, :feats_skl.shape[1]] = feats_skl[0]
                feats_skl = padded
                results["notes"].append(f"Sklearn features padded {feats_basic.shape[1]} -> {expected_skl}")
            # feature_names_in_ handling
            if hasattr(skl_scaler, "feature_names_in_"):
                import pandas as pd
                cols = list(getattr(skl_scaler, "feature_names_in_"))
                df = pd.DataFrame(np.zeros((1,len(cols))), columns=cols, dtype=np.float32)
                for i in range(min(feats_skl.shape[1], len(cols))):
                    df.iloc[0,i] = feats_skl[0,i]
                Xs = skl_scaler.transform(df)
            else:
                Xs = skl_scaler.transform(feats_skl)
            pred = sklearn_model.predict(Xs)
            prob = None
            try:
                prob = sklearn_model.predict_proba(Xs)[0]
                conf = float(np.max(prob))
            except Exception:
                conf = None
            label_s = skl_le.inverse_transform(pred)[0] if skl_le is not None else str(pred[0])
            results["sklearn"] = {"label": label_s, "index": int(pred[0]), "confidence": conf, "probs": _to_py(prob)}
            if verbose: print("Sklearn ->", label_s, conf)
        except Exception as e:
            results["sklearn"] = {"error": str(e)}
            results["notes"].append("Sklearn prediction failed: " + str(e))
    else:
        if model_choice in ("sklearn","all") and (sklearn_model is None or skl_scaler is None):
            results["notes"].append("sklearn model or scaler missing; skipped sklearn prediction.")

    # Ensemble vote (simple majority among available labels)
    votes = []
    for k in ("cnn","hybrid","sklearn"):
        v = results.get(k)
        if v and isinstance(v, dict) and ("label" in v):
            votes.append(v["label"])
    results["ensemble_vote"] = None if len(votes)==0 else Counter(votes).most_common(1)[0][0]

    if verbose:
        print("Ensemble vote ->", results["ensemble_vote"])
        if len(results["notes"])>0:
            print("Notes:")
            for n in results["notes"]:
                print(" -", n)

    return _to_py(results)

# ---------------- Example usage -----------------
if __name__ == "__main__":
    # quick local test (change path)
    path = r"D:\Infosys_AI-Tracefinder\Data\Wikipedia\Canon120-1\150\s1_1.tif"
    out = predict_single_image(path, model_choice="all", verbose=True)
    print("\nRETURNED:\n", out)

