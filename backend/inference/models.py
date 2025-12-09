# backend/inference/models.py
import os
import pickle
import joblib
import numpy as np
from scipy import stats

# tensorflow import may be heavy; import here
import tensorflow as tf
from tensorflow import keras

# local helper modules (must exist)
from .features import make_residual_image, compute_basic_features
from .utils import guess_resolution_from_path, safe_scale_and_pad

# Project layout helpers
BASE_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND_MODELS_DIR = os.path.join(BASE_PROJECT, "models")       # where you copied models earlier
NOTEBOOKS_DIR = os.path.join(BASE_PROJECT, "Notebooks")
OUTPUT_DIR = os.path.join(BASE_PROJECT, "Output")
CWD = os.getcwd()

# candidate dirs to search for model files (in priority order)
MODEL_SEARCH_PATHS = [BACKEND_MODELS_DIR, NOTEBOOKS_DIR, OUTPUT_DIR, CWD]

def find_file_in_paths(fname):
    for d in MODEL_SEARCH_PATHS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None

# expected filenames (adjust if your filenames differ)
CNN_FN = find_file_in_paths("cnn_residual_best.keras")
HYBRID_FN = find_file_in_paths("scanner_hybrid_best.keras")
SKLEARN_MODEL_FN = find_file_in_paths("best_model.pkl")
SKLEARN_SCALER_FN = find_file_in_paths("scaler.pkl")
SKLEARN_LE_FN = find_file_in_paths("label_encoder.pkl")
CNN_LE_FN = find_file_in_paths("cnn_label_encoder.pkl")
HYBRID_SCALER_FN = find_file_in_paths("hybrid_feat_scaler.pkl")
HYBRID_LE_FN = find_file_in_paths("hybrid_label_encoder.pkl")
# optional metadata
SKLEARN_META_FN = find_file_in_paths("sklearn_metadata.pkl")

# Globals for loaded artifacts
cnn_model = None
hybrid_model = None
sk_model = None
sk_scaler = None
sk_le = None
cnn_le = None
hybrid_scaler = None
hybrid_le = None
sk_meta = None

def _safe_load_keras(path):
    try:
        m = keras.models.load_model(path)
        return m
    except Exception as e:
        print(f"Failed to load keras model {path}: {e}")
        return None

def load_artifacts():
    """Load models/scalers/encoders into module globals (idempotent)."""
    global cnn_model, hybrid_model, sk_model, sk_scaler, sk_le, cnn_le, hybrid_scaler, hybrid_le, sk_meta

    # CNN label encoder
    if CNN_LE_FN and os.path.exists(CNN_LE_FN):
        try:
            with open(CNN_LE_FN, "rb") as f:
                cnn_le = pickle.load(f)
                print("Loaded CNN label encoder:", CNN_LE_FN)
        except Exception as e:
            print("Failed loading cnn label encoder:", e)
            cnn_le = None

    # load cnn
    if cnn_model is None and CNN_FN:
        print("Trying to load CNN model from:", CNN_FN)
        cnn_model = _safe_load_keras(CNN_FN)
        if cnn_model is not None:
            try:
                cnn_model.predict(np.zeros((1,256,256,1), dtype=np.float32))
            except Exception:
                pass
            print("Loaded CNN model.")

    # hybrid
    if hybrid_model is None and HYBRID_FN:
        print("Trying to load Hybrid model from:", HYBRID_FN)
        hybrid_model = _safe_load_keras(HYBRID_FN)
        if hybrid_model is not None:
            # attempt a warm-up with best-effort feature dim detection
            try:
                # find expected feature dim by inspecting inputs
                expected_feat = None
                for inp in hybrid_model.inputs:
                    s = getattr(inp, "shape", None)
                    if s is not None and len(s) == 2 and s[1] is not None and int(s[1]) not in (256, 256):
                        expected_feat = int(s[1])
                        break
                if expected_feat is None:
                    expected_feat = 27
                dummy_img = np.zeros((1,256,256,1), dtype=np.float32)
                dummy_feat = np.zeros((1, expected_feat), dtype=np.float32)
                # attempt both orders
                try:
                    hybrid_model.predict([dummy_img, dummy_feat])
                except Exception:
                    try:
                        hybrid_model.predict([dummy_feat, dummy_img])
                    except Exception:
                        pass
            except Exception:
                pass
            print("Loaded Hybrid model.")

    # sklearn artifacts
    if SKLEARN_MODEL_FN and os.path.exists(SKLEARN_MODEL_FN):
        try:
            sk_model = joblib.load(SKLEARN_MODEL_FN)
            print("Loaded sklearn model:", SKLEARN_MODEL_FN)
        except Exception as e:
            print("Failed to load sklearn model:", e)
            sk_model = None
    if SKLEARN_SCALER_FN and os.path.exists(SKLEARN_SCALER_FN):
        try:
            sk_scaler = joblib.load(SKLEARN_SCALER_FN)
            print("Loaded sklearn scaler:", SKLEARN_SCALER_FN)
        except Exception as e:
            print("Failed to load sklearn scaler:", e)
            sk_scaler = None
    if SKLEARN_LE_FN and os.path.exists(SKLEARN_LE_FN):
        try:
            sk_le = joblib.load(SKLEARN_LE_FN)
            print("Loaded sklearn label encoder:", SKLEARN_LE_FN)
        except Exception as e:
            print("Failed to load sklearn label encoder:", e)
            sk_le = None
    if SKLEARN_META_FN and os.path.exists(SKLEARN_META_FN):
        try:
            sk_meta = joblib.load(SKLEARN_META_FN)
            print("Loaded sklearn metadata.")
        except Exception:
            sk_meta = None

    # hybrid scaler + encoder
    if HYBRID_SCALER_FN and os.path.exists(HYBRID_SCALER_FN):
        try:
            hybrid_scaler = joblib.load(HYBRID_SCALER_FN)
            print("Loaded hybrid scaler:", HYBRID_SCALER_FN)
        except Exception as e:
            print("Failed to load hybrid scaler:", e)
            hybrid_scaler = None
    if HYBRID_LE_FN and os.path.exists(HYBRID_LE_FN):
        try:
            with open(HYBRID_LE_FN, "rb") as f:
                hybrid_le = pickle.load(f)
                print("Loaded hybrid label encoder:", HYBRID_LE_FN)
        except Exception as e:
            print("Failed to load hybrid label encoder:", e)
            hybrid_le = None

    # push to globals (in case some were locally assigned earlier)
    globals().update({
        "cnn_model": cnn_model,
        "hybrid_model": hybrid_model,
        "sk_model": sk_model,
        "sk_scaler": sk_scaler,
        "sk_le": sk_le,
        "cnn_le": cnn_le,
        "hybrid_scaler": hybrid_scaler,
        "hybrid_le": hybrid_le,
        "sk_meta": sk_meta
    })

# load once at import
load_artifacts()


def predict_single_image(image_path, model: str = None, verbose=False):
    """Return dict with predictions.

    If `model` is provided (one of 'cnn', 'hybrid', 'sklearn'), only that model's
    prediction will be computed and returned (keys named 'cnn'|'hybrid_cnn'|'sklearn').
    When `model` is None, all available models are run and an ensemble_vote is returned.
    """
    notes = []
    out = {"cnn": None, "hybrid_cnn": None, "sklearn": None, "ensemble_vote": None, "notes": notes}

    # normalize model param
    model_requested = None
    if isinstance(model, str):
        m = model.strip().lower()
        if m in ("cnn", "hybrid", "hybrid_cnn", "sklearn"):
            # normalize hybrid -> hybrid_cnn
            if m == "hybrid":
                model_requested = "hybrid_cnn"
            elif m == "sklearn":
                model_requested = "sklearn"
            else:
                model_requested = m
    # which models to run
    if model_requested:
        to_run = {model_requested}
    else:
        to_run = {"cnn", "hybrid_cnn", "sklearn"}

    # 1) residual image
    try:
        res = make_residual_image(image_path)  # should produce shape (1,H,W,1)
    except Exception as e:
        raise RuntimeError("Residual image creation failed: " + str(e))

    # 2) basic features vector (11-dim)
    try:
        basic = compute_basic_features(image_path)
        reslvl = guess_resolution_from_path(image_path)
        feat_vect = np.array([[
            float(reslvl),
            float(basic["width"]),
            float(basic["height"]),
            float(basic["aspect_ratio"]),
            float(basic["file_size_kb"]),
            float(basic["mean_intensity"]),
            float(basic["std_intensity"]),
            float(basic["skewness"]),
            float(basic["kurtosis"]),
            float(basic["entropy"]),
            float(basic["edge_density"])
        ]], dtype=np.float32)  # shape (1,11)
    except Exception as e:
        raise RuntimeError("Feature extraction failed: " + str(e))

    # 3) CNN
    if "cnn" in to_run:
        if cnn_model is not None:
            try:
                p = cnn_model.predict(res)
                p = np.asarray(p)[0]
                idx = int(np.argmax(p))
                label = cnn_le.inverse_transform([idx])[0] if cnn_le is not None else str(idx)
                out["cnn"] = {"label": label, "index": idx, "probs": p.tolist()}
                if verbose:
                    print("CNN ->", label, float(p[idx]))
            except Exception as e:
                out["cnn"] = {"error": str(e)}
                notes.append("CNN prediction failed: " + str(e))
        else:
            notes.append("CNN model not loaded; skipped.")

    # 4) Hybrid: pad/scale features to expected dim then call model
    if "hybrid_cnn" in to_run:
        if hybrid_model is not None:
            try:
                feats_for_hybrid = safe_scale_and_pad(feat_vect, scaler=hybrid_scaler, meta=None)
                # decide image input index
                img_idx = 0
                for i, inp in enumerate(hybrid_model.inputs):
                    if len(getattr(inp, "shape", [])) == 4:
                        img_idx = i
                        break
                inputs = []
                for i, inp in enumerate(hybrid_model.inputs):
                    if i == img_idx:
                        inputs.append(res)
                    else:
                        inputs.append(np.array(feats_for_hybrid, dtype=np.float32))
                ph = hybrid_model.predict(inputs)
                ph = np.asarray(ph)[0]
                idxh = int(np.argmax(ph))
                labh = hybrid_le.inverse_transform([idxh])[0] if hybrid_le is not None else str(idxh)
                out["hybrid_cnn"] = {"label": labh, "index": idxh, "probs": ph.tolist()}
                if verbose:
                    print("Hybrid ->", labh, float(ph[idxh]))
            except Exception as e:
                out["hybrid_cnn"] = {"error": str(e)}
                notes.append("Hybrid prediction failed: " + str(e))
        else:
            notes.append("Hybrid model not loaded; skipped.")

    # 5) sklearn
    if "sklearn" in to_run:
        if sk_model is not None and sk_scaler is not None:
            try:
                feats_sk = safe_scale_and_pad(feat_vect, scaler=sk_scaler, meta=sk_meta)
                pred = sk_model.predict(feats_sk)
                prob = None
                try:
                    prob = sk_model.predict_proba(feats_sk)[0].tolist()
                except Exception:
                    prob = None
                lab_s = sk_le.inverse_transform(pred)[0] if sk_le is not None else str(int(pred[0]))
                out["sklearn"] = {"label": lab_s, "index": int(pred[0]), "probs": prob}
                if verbose:
                    print("Sklearn ->", lab_s, "probs:", prob)
            except Exception as e:
                out["sklearn"] = {"error": str(e)}
                notes.append("Sklearn prediction failed: " + str(e))
        else:
            notes.append("sklearn model or scaler not available; skipped sklearn prediction.")

    # ensemble vote (only when multiple models were requested/run)
    if model_requested is None:
        votes = []
        for k in ("cnn", "hybrid_cnn", "sklearn"):
            v = out.get(k)
            if v and isinstance(v, dict) and ("label" in v):
                votes.append(v["label"])
        if votes:
            try:
                vote = stats.mode(votes)[0][0]
            except Exception:
                vote = votes[0]
            out["ensemble_vote"] = vote
        else:
            out["ensemble_vote"] = None
    else:
        out["ensemble_vote"] = None

    out["notes"] = notes
    return out


if __name__ == "__main__":
    # quick local test (runs only if this file executed directly)
    sample = os.path.join(NOTEBOOKS_DIR or ".", "test_sample.tif")
    if os.path.exists(sample):
        print(predict_single_image(sample, verbose=True))
    else:
        print("No test_sample.tif found. Call predict_single_image(image_path) from your script.")
