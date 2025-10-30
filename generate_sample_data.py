"""
generate_sample_data.py
Create synthetic scanned-document samples for multiple scanner models.

Features / context:
- Produces sample images that mimic scanned documents from several scanner models.
- Adds optional transformations: rotation, gaussian blur, synthetic noise.
- Embeds simple PNG text metadata (scanner_model, stamp).
- Writes a detailed labels CSV: scanner_model, file_name, file_path (absolute), format,
  width, height, color_mode, rotation_deg, noise_level, blur_applied, stamp, seed.
- Intended for prototyping preprocessing and model training pipelines when real scans
  from different devices are not available.

Usage (Windows):
  python "c:\Users\91628\OneDrive\Desktop\infosys\scanned-docs-preprocessing\scripts\generate_sample_data.py" --per-model 3 --out-root "..\data" --seed 42 --noise 0.02 --max-rotation 2 --blur-prob 0.3

Dependencies:
  - Pillow
  - numpy (optional; if not present, noise will be skipped)

Output:
  - data/raw/*  : generated images
  - data/annotations/labels.csv : metadata for each generated image
"""
import os
import csv
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter, PngImagePlugin

# optional numpy for noise
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except Exception:
    _NUMPY_AVAILABLE = False

# Project paths (relative to script)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
DEFAULT_ANNOT_DIR = PROJECT_ROOT / 'data' / 'annotations'

# Default scanner models
SCANNER_MODELS = [
    "CanonScan_Lide",
    "HP_ScanJet",
    "Epson_Perfection",
    "Brother_ADS",
    "Fujitsu_ScanSnap"
]

PRESETS = [
    ('PNG', (1240, 1754), 'RGB'),    # ~150 DPI A4
    ('PNG', (2480, 3508), 'RGB'),    # ~300 DPI A4
    ('PNG', (1654, 2339), 'L'),      # grayscale ~200 DPI A4
    ('PNG', (800, 1100), 'RGB'),     # smaller scan
]

def ensure_dirs(raw_dir: Path, annot_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    annot_dir.mkdir(parents=True, exist_ok=True)

def _add_noise_pil(img: Image.Image, noise_level: float) -> Image.Image:
    """
    Add multiplicative Gaussian noise (approx) to the image using numpy.
    noise_level: standard deviation relative to 1.0 (e.g., 0.02)
    """
    if noise_level <= 0 or not _NUMPY_AVAILABLE:
        return img
    arr = np.asarray(img).astype(np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=noise_level, size=arr.shape).astype(np.float32)
    arr_noisy = arr + noise
    arr_noisy = np.clip(arr_noisy, 0.0, 1.0)
    if arr_noisy.ndim == 2:
        arr_noisy = (arr_noisy * 255).astype(np.uint8)
    else:
        arr_noisy = (arr_noisy * 255).astype(np.uint8)
    return Image.fromarray(arr_noisy, mode=img.mode)

def make_sample_image(path: Path, text: str, size: tuple, mode: str,
                      scanner_model: str, stamp: str,
                      rotation_deg: float = 0.0, noise_level: float = 0.0,
                      apply_blur: bool = False, bg_color=None):
    if bg_color is None:
        bg_color = (255, 255, 255) if mode == 'RGB' else 255
    img = Image.new(mode, size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # font selection (fallback to default)
    try:
        font = ImageFont.truetype("arial.ttf", size=36)
    except Exception:
        font = ImageFont.load_default()

    # center text
    try:
        w, h = draw.textsize(text, font=font)
    except Exception:
        w, h = (len(text) * 8, 24)
    x = max(10, (size[0] - w) // 2)
    y = max(10, (size[1] - h) // 2)
    text_color = (0, 0, 0) if mode == 'RGB' else 0
    draw.text((x, y), text, fill=text_color, font=font)

    # stamp in corner
    draw.text((10, size[1]-40), stamp, fill=text_color, font=font)

    # small decorations to mimic scans (lines / boxes)
    if mode == 'RGB':
        draw.line([(50, 150), (size[0]-50, 150)], fill=(200,200,200), width=2)
    else:
        draw.line([(50, 150), (size[0]-50, 150)], fill=200, width=2)

    # optional rotation
    if rotation_deg:
        img = img.rotate(rotation_deg, expand=False, fillcolor=bg_color if mode == 'RGB' else 255)

    # optional blur
    if apply_blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=1.0 + random.random()*1.5))

    # optional noise (requires numpy)
    if noise_level > 0 and _NUMPY_AVAILABLE:
        img = _add_noise_pil(img, noise_level)

    # embed PNG text metadata for scan model and stamp (PNG only)
    save_kwargs = {}
    if path.suffix.lower() == '.png':
        png_meta = PngImagePlugin.PngInfo()
        png_meta.add_text("scanner_model", scanner_model)
        png_meta.add_text("stamp", stamp)
        save_kwargs['pnginfo'] = png_meta

    img.save(path, **save_kwargs)

def generate_samples(per_model: int = 3, raw_dir: Path = DEFAULT_RAW_DIR,
                     annot_dir: Path = DEFAULT_ANNOT_DIR, models=None,
                     seed: int = None, noise: float = 0.0, max_rotation: float = 0.0,
                     blur_prob: float = 0.0):
    if seed is not None:
        random.seed(seed)
        if _NUMPY_AVAILABLE:
            import numpy as _np
            _np.random.seed(seed)

    if models is None:
        models = SCANNER_MODELS

    ensure_dirs(raw_dir, annot_dir)
    labels_path = annot_dir / 'labels.csv'
    rows = []

    for model in models:
        for i in range(per_model):
            fmt, size, mode = random.choice(PRESETS)
            file_name = f"{model}_sample_{i+1}.{fmt.lower()}"
            out_path = raw_dir / file_name
            label_text = f"{model} - sample {i+1}"
            # random stamp and transformations
            stamp = f"{random.randint(1000,9999)}"
            rotation_deg = random.uniform(-max_rotation, max_rotation) if max_rotation else 0.0
            noise_level = float(noise) if noise and _NUMPY_AVAILABLE else 0.0
            blur_applied = random.random() < blur_prob

            make_sample_image(out_path, label_text, size, mode,
                              scanner_model=model, stamp=stamp,
                              rotation_deg=rotation_deg, noise_level=noise_level,
                              apply_blur=blur_applied)

            rows.append({
                'scanner_model': model,
                'file_name': file_name,
                'file_path': str(out_path.resolve()),
                'format': fmt,
                'width': size[0],
                'height': size[1],
                'color_mode': mode,
                'rotation_deg': round(rotation_deg, 3),
                'noise_level': round(noise_level, 5),
                'blur_applied': bool(blur_applied),
                'stamp': stamp
            })

    # write labels CSV
    fieldnames = ['scanner_model','file_name','file_path','format','width','height','color_mode','rotation_deg','noise_level','blur_applied','stamp']
    with open(labels_path, 'w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Generated {len(rows)} images -> {raw_dir}")
    print(f"Labels written -> {labels_path}")
    if not _NUMPY_AVAILABLE and noise:
        print("Note: numpy not installed, noise was skipped. Install numpy for noise support.")

def parse_args_and_run():
    parser = argparse.ArgumentParser(description="Generate sample scanned-document images with metadata.")
    parser.add_argument('--per-model', type=int, default=3, help='Samples per scanner model')
    parser.add_argument('--out-root', type=str, default=str(PROJECT_ROOT / 'data'),
                        help='Root output folder (contains raw/ and annotations/)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise level (std dev, e.g., 0.02) - requires numpy')
    parser.add_argument('--max-rotation', type=float, default=0.0, help='Max rotation degrees (uniform in [-max,max])')
    parser.add_argument('--blur-prob', type=float, default=0.0, help='Probability to apply mild Gaussian blur to each image')
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    raw_dir = out_root / 'raw'
    annot_dir = out_root / 'annotations'
    generate_samples(per_model=args.per_model, raw_dir=raw_dir, annot_dir=annot_dir,
                     seed=args.seed, noise=args.noise, max_rotation=args.max_rotation, blur_prob=args.blur_prob)

if __name__ == "__main__":
    parse_args_and_run()