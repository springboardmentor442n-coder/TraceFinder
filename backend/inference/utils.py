# backend/inference/utils.py
import os
import numpy as np
import joblib

def guess_resolution_from_path(path):
    p = path.replace("\\", "/").lower()
    if "/150" in p or "_150" in p or "/150/" in p:
        return 150
    if "/300" in p or "_300" in p or "/300/" in p:
        return 300
    return 0

def safe_scale_and_pad(feat_vector, scaler=None, meta=None):
    """
    feat_vector: 1D numpy array shape (F,) or shape (1,F)
    scaler: StandardScaler instance (may be None)
    meta: optional metadata dict with "n_features" or "feature_columns"
    Returns scaled 2D array shape (1, n_expected)
    """
    arr = np.asarray(feat_vector).reshape(1, -1).astype(np.float32)
    if scaler is None:
        # if no scaler, pad/truncate to meta size if meta provided
        if meta is not None and "n_features" in meta:
            n_expected = int(meta["n_features"])
            if arr.shape[1] != n_expected:
                out = np.zeros((1, n_expected), dtype=np.float32)
                out[0, :min(arr.shape[1], n_expected)] = arr[0, :min(arr.shape[1], n_expected)]
                return out
        return arr
    # scaler present: use scaler.n_features_in_ (sklearn>=1.0)
    n_expected = getattr(scaler, "n_features_in_", None)
    if n_expected is None and meta is not None:
        n_expected = int(meta.get("n_features", arr.shape[1]))
    if n_expected is None:
        # fallback: try transform (may raise)
        try:
            return scaler.transform(arr)
        except Exception:
            # return original arr
            return arr
    # pad/truncate to n_expected
    if arr.shape[1] != n_expected:
        padded = np.zeros((1, n_expected), dtype=np.float32)
        padded[0, :min(arr.shape[1], n_expected)] = arr[0, :min(arr.shape[1], n_expected)]
        arr = padded
    # now transform
    return scaler.transform(arr)
