#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ===== Updated minimal denoising module =====
# - outputs: raw .npy residuals + two PNG previews per image
# - default image resize: 256x256
# - fallback denoising chain (db4 L2 -> db1 L1 -> gaussian highpass)
# - tiny-range reprocessing supported
# - no pickles, no global_abs

import os
import glob
import cv2
import numpy as np
import pywt
from tqdm import tqdm

# -------------------
# CONFIG - EDIT PATHS
# -------------------
SRC_ROOT = r"D:\Infosys_AI-Tracefinder\Data\Official"  # original dataset root
OUT_VIS  = r"D:\Infosys_AI-Tracefinder\Data\Residuals_vis"  # preview PNGs
OUT_RAW  = r"D:\Infosys_AI-Tracefinder\Data\Residuals_raw"  # raw .npy residuals
IMG_SIZE = (256, 256)
VALID_EXTS = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')
TINY_RANGE_THRESH = 1e-3  # keep same threshold logic (tweak if needed)

os.makedirs(OUT_VIS, exist_ok=True)
os.makedirs(OUT_RAW, exist_ok=True)

# -------------------
# UTILITIES
# -------------------
def safe_read_gray(path):
    """Read image in grayscale and return float32 array."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    return img

def resize_and_norm(img, size=IMG_SIZE):
    """Resize and keep float32 (not normalized to [0,1] because earlier code works in original range)."""
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)

def map_signed_to_uint8_zero128(raw):
    """
    Map signed float residual to uint8 such that zero maps near 128 (safe float math).
    If raw is constant, return mid-gray (128).
    """
    raw = np.asarray(raw, dtype=np.float32)
    mn = float(np.nanmin(raw))
    mx = float(np.nanmax(raw))
    if np.isclose(mx, mn):
        return np.full(raw.shape, 128, dtype=np.uint8)
    # linear mapping raw->[0,255], then shift so 0 -> 128
    scale = 255.0 / (mx - mn + 1e-12)
    mapped = (raw - mn) * scale
    zero_mapped = (0.0 - mn) * scale
    shift = 128.0 - zero_mapped
    mapped_shifted = mapped + shift
    mapped_clipped = np.clip(mapped_shifted, 0.0, 255.0)
    return mapped_clipped.astype(np.uint8)

def raw_to_vis_uint8(raw):
    """Map raw residual to visually informative uint8 (0..255) for quick checks."""
    raw = np.asarray(raw, dtype=np.float32)
    mn = float(np.nanmin(raw))
    mx = float(np.nanmax(raw))
    if np.isclose(mx, mn):
        return np.zeros_like(raw, dtype=np.uint8)
    vis = 255.0 * (raw - mn) / (mx - mn + 1e-12)
    return np.clip(vis, 0.0, 255.0).astype(np.uint8)

# -------------------
# DENOISING FUNCTIONS
# -------------------
def wavelet_threshold_denoise(img, wavelet='db4', level=2):
    """
    Wavelet denoising using soft-threshold on detail coeffs.
    Returns denoised image same shape as input (float32).
    """
    # ensure float32
    arr = np.asarray(img, dtype=np.float32)
    try:
        coeffs = pywt.wavedec2(arr, wavelet=wavelet, level=level)
    except Exception:
        # fallback: return original if wavelet fails
        return arr.copy()
    # collect detail coefficients to compute robust sigma
    details = []
    for d in coeffs[1:]:
        for comp in d:
            details.append(np.ravel(np.nan_to_num(comp)))
    if len(details) == 0:
        return arr.copy()
    details = np.concatenate(details)
    mad = np.median(np.abs(details - np.median(details)))
    sigma = mad / 0.6745 if mad > 0 else 0.0
    uth = sigma * np.sqrt(2 * np.log(arr.size + 1e-12))
    # soft threshold each detail array
    new_coeffs = [coeffs[0]]
    for d in coeffs[1:]:
        new_level = tuple(pywt.threshold(np.nan_to_num(comp), value=uth, mode='soft') for comp in d)
        new_coeffs.append(new_level)
    den = pywt.waverec2(new_coeffs, wavelet)
    den = den[:arr.shape[0], :arr.shape[1]]
    return np.nan_to_num(den, nan=arr).astype(np.float32)

def gaussian_highpass(img, ksize=11):
    """High-pass via original - gaussian blur."""
    arr = np.asarray(img, dtype=np.float32)
    if ksize % 2 == 0:
        ksize += 1
    blur = cv2.GaussianBlur(arr, (ksize, ksize), 0)
    return (arr - blur).astype(np.float32)

# -------------------
# RESIDUAL COMPUTE / FALLBACK CHAIN
# -------------------
def compute_residual_with_fallback(img_gray):
    """
    Try a sequence of denoising methods; return (raw_residual, used_method, rng).
    Methods attempted:
        1) db4 level=2
        2) db1 level=1
        3) gaussian highpass
    """
    img = img_gray.astype(np.float32)

    # 1) db4 L2
    den = wavelet_threshold_denoise(img, wavelet='db4', level=2)
    raw = img - den
    rng = float(np.nanmax(raw) - np.nanmin(raw))
    if rng >= TINY_RANGE_THRESH:
        return raw.astype(np.float32), "db4_L2", rng

    # 2) db1 L1
    den = wavelet_threshold_denoise(img, wavelet='db1', level=1)
    raw = img - den
    rng = float(np.nanmax(raw) - np.nanmin(raw))
    if rng >= TINY_RANGE_THRESH:
        return raw.astype(np.float32), "db1_L1", rng

    # 3) gaussian highpass fallback
    raw = gaussian_highpass(img, ksize=11)
    rng = float(np.nanmax(raw) - np.nanmin(raw))
    return raw.astype(np.float32), "gaussian_hp", rng

# -------------------
# MAIN PROCESSING LOOPS
# -------------------
def process_and_save_all(src_root=SRC_ROOT, out_vis=OUT_VIS, out_raw=OUT_RAW):
    """
    Walks dataset structure:
      src_root/
        scanner_name/
          dpi_folder/   (optional)
             image.tif
          image.tif   (also supported)
    Saves for each image:
      <base>_res_raw.npy  (raw float32 residual)
      <base>_res_vis.png  (visual scaled)
      <base>_res_signed.png (signed uint8 mapping 0->128)
    Also saves per-(scanner,dpi) stack and fingerprint files:
      scanner/dpi/_stack.npy, _fingerprint.npy
    """
    for scanner in sorted(os.listdir(src_root)):
        scanner_path = os.path.join(src_root, scanner)
        if not os.path.isdir(scanner_path):
            continue

        # find all image files recursively inside scanner folder
        pattern = os.path.join(scanner_path, "**", "*")
        files = [f for f in glob.glob(pattern, recursive=True) if f.lower().endswith(VALID_EXTS)]
        if not files:
            continue

        # group by subfolder (e.g., dpi) so outputs keep similar structure
        grouped = {}
        for f in files:
            rel = os.path.relpath(f, scanner_path)
            parts = rel.split(os.sep)
            # if images are inside a dpi folder, use that subfolder name; else root scanner folder
            key = parts[0] if len(parts) > 1 else "_root"
            grouped.setdefault(key, []).append(f)

        for group_name, file_list in grouped.items():
            out_vis_dir = os.path.join(out_vis, scanner, group_name)
            out_raw_dir = os.path.join(out_raw, scanner, group_name)
            os.makedirs(out_vis_dir, exist_ok=True)
            os.makedirs(out_raw_dir, exist_ok=True)

            residuals_stack = []
            for path in tqdm(sorted(file_list), desc=f"{scanner}/{group_name}", leave=False):
                base = os.path.splitext(os.path.basename(path))[0]
                img = safe_read_gray(path)
                if img is None:
                    continue
                img_r = resize_and_norm(img, size=IMG_SIZE)
                raw, method, rng = compute_residual_with_fallback(img_r)

                # Save raw residual (.npy)
                raw_path = os.path.join(out_raw_dir, base + "_res_raw.npy")
                np.save(raw_path, raw.astype(np.float32))

                # Save visual preview and signed preview
                vis = raw_to_vis_uint8(raw)
                vis_path = os.path.join(out_vis_dir, base + "_res_vis.png")
                cv2.imwrite(vis_path, vis)

                signed = map_signed_to_uint8_zero128(raw)
                signed_path = os.path.join(out_vis_dir, base + "_res_signed.png")
                cv2.imwrite(signed_path, signed)

                residuals_stack.append(raw.astype(np.float32))
            # save per-group stacked residuals and mean fingerprint
            if residuals_stack:
                stack_arr = np.stack(residuals_stack, axis=0)
                stack_path = os.path.join(out_raw_dir, f"{scanner}_{group_name}_residuals_stack.npy")
                np.save(stack_path, stack_arr)
                mean_fp = np.mean(stack_arr, axis=0).astype(np.float32)
                fp_path = os.path.join(out_raw_dir, f"{scanner}_{group_name}_fingerprint.npy")
                np.save(fp_path, mean_fp)

    print("✅ Denoising & residual export complete.")
    print("Visual previews in:", OUT_VIS)
    print("Raw residuals in:", OUT_RAW)

# -------------------
# Reprocess tiny-range files (optional helper)
# -------------------
def reprocess_tiny_range(dst_raw=OUT_RAW, src_root=SRC_ROOT):
    """
    Recompute residuals for any previously saved raw files whose range < TINY_RANGE_THRESH.
    Uses the same fallback chain and overwrites the .npy + previews.
    """
    updated = 0
    for root, _, files in os.walk(dst_raw):
        for f in files:
            if not f.endswith("_res_raw.npy"):
                continue
            p = os.path.join(root, f)
            arr = np.load(p)
            rng = float(np.nanmax(arr) - np.nanmin(arr))
            if rng >= TINY_RANGE_THRESH:
                continue
            # find original image by matching filename in src_root scanner structure
            base = f.replace("_res_raw.npy", "")
            # rough search (could be improved)
            # try to find a matching image file in corresponding SRC_ROOT path
            rel_root = os.path.relpath(root, dst_raw)
            guessed_src = os.path.join(src_root, rel_root)
            found = None
            for ext in VALID_EXTS:
                candidate = os.path.join(guessed_src, base + ext)
                if os.path.exists(candidate):
                    found = candidate
                    break
            if not found:
                continue
            img = safe_read_gray(found)
            if img is None:
                continue
            img_r = resize_and_norm(img, size=IMG_SIZE)
            raw, method, rng2 = compute_residual_with_fallback(img_r)
            np.save(p, raw.astype(np.float32))
            vis = raw_to_vis_uint8(raw)
            signed = map_signed_to_uint8_zero128(raw)
            vis_dir = root.replace(OUT_RAW, OUT_VIS) if OUT_RAW in root else root
            os.makedirs(vis_dir, exist_ok=True)
            cv2.imwrite(os.path.join(vis_dir, base + "_res_vis.png"), vis)
            cv2.imwrite(os.path.join(vis_dir, base + "_res_signed.png"), signed)
            updated += 1
    print(f"✅ Reprocessed {updated} tiny-range files.")

# -------------------
# Run (example)
# -------------------
if __name__ == "__main__":
    process_and_save_all()
    # optionally call reprocess_tiny_range() if you want to re-run tiny-range files:
    # reprocess_tiny_range()

