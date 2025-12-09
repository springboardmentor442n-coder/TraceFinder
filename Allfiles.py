#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import cv2
import glob
import numpy as np
import pandas as pd
from skimage.filters import sobel
from skimage import img_as_float
from scipy.stats import skew, kurtosis, entropy
from concurrent.futures import ThreadPoolExecutor


# In[9]:


def extract_features(img_path, class_label, resolution_level):
    """Extract features from a single image."""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path}")
            return None

        # Basic attributes
        h, w = img.shape
        aspect_ratio = round(w / h, 6)
        file_size_kb = round(os.path.getsize(img_path) / 1024, 6)

        # Intensity-based features
        pixels = img.flatten()
        mean_intensity = round(np.mean(pixels) / 255, 6)
        std_intensity = round(np.std(pixels) / 255, 6)
        skewness = round(skew(pixels), 6)
        kurt = round(kurtosis(pixels), 6)
        ent = round(entropy(np.histogram(pixels, bins=256)[0] + 1), 6)

        # Edge detection
        edges = sobel(img_as_float(img))
        edge_density = round(np.mean(edges > 0.1), 6)

        return {
            "file_name": os.path.basename(img_path),
            "class_label": class_label,
            "resolution_level": resolution_level,
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

    except Exception as e:
        print(f"⚠️ Error processing {img_path}: {e}")
        return None


# In[10]:


import os

base_folder = r"D:\Infosys_AI-Tracefinder\Data\Official"
expected_folders = [
    "Canon120-1", "Canon120-2", "Canon220",
    "Canon9000-1", "Canon9000-2",
    "EpsonV39-1", "EpsonV39-2",
    "EpsonV370-1", "EpsonV370-2", "EpsonV550",
    "HP"
]

print("🔍 Checking dataset folders...\n")

for folder in expected_folders:
    folder_path = os.path.join(base_folder, folder)
    if os.path.exists(folder_path):
        print(f"✅ Found: {folder}")
    else:
        print(f"❌ Missing: {folder}")


# In[11]:


def process_all_images(base_folder, output_csv, max_workers=8):
    """Scan all class folders and extract image features."""

    # Auto-detect class folders dynamically
    class_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    print("📂 Auto-detected class folders:", class_folders)

    subfolders = ["150", "300"]
    valid_exts = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')

    all_images = []

    # Collect all images
    for class_folder in class_folders:
        class_path = os.path.join(base_folder, class_folder)
        if not os.path.exists(class_path):
            print(f"⚠️ Skipping missing folder: {class_folder}")
            continue

        for subfolder in subfolders:
            sub_path = os.path.join(class_path, subfolder)
            if not os.path.exists(sub_path):
                print(f"⚠️ Subfolder not found: {sub_path}")
                continue

            for ext in valid_exts:
                images = glob.glob(os.path.join(sub_path, f"*{ext}"))
                for img_path in images:
                    all_images.append((img_path, class_folder, subfolder))

    print(f"\n🖼️ Total images found: {len(all_images)}\n")

    data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda x: extract_features(*x), all_images)
        for result in results:
            if result:
                data.append(result)

    # Convert to DataFrame once after processing
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Feature extraction complete. Saved to: {output_csv}")
    print(f"📊 Total images processed: {len(df)}")

    if not df.empty:
        print("\n📋 Sample preview of data:\n")
        print(df.head(10))
    else:
        print("⚠️ No valid images were processed.")


# In[12]:


# === RUN ===
base_folder = r"D:\Infosys_AI-Tracefinder\Data\Official"
output_csv = r"D:\Infosys_AI-Tracefinder\Output\Output_for_Allfiles.csv"

process_all_images(base_folder, output_csv, max_workers=8)


# In[1]:


import os

base_folder = r"D:\Infosys_AI-Tracefinder\Data\Official"

for class_folder in os.listdir(base_folder):
    class_path = os.path.join(base_folder, class_folder)
    if not os.path.isdir(class_path):
        continue

    print(f"\n📁 {class_folder}:")
    for subfolder in os.listdir(class_path):
        sub_path = os.path.join(class_path, subfolder)
        if os.path.isdir(sub_path):
            files = len([f for f in os.listdir(sub_path) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))])
            print(f"   🔹 {subfolder}/ → {files} images")

