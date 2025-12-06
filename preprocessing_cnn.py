import os
import cv2
import numpy as np
from scipy.ndimage import median_filter

IMG_SIZE = (128, 128)

def preprocess_image_cnn(image_path, size=IMG_SIZE):
    """
    Preprocess a single image for CNN.
    Returns a normalized RGB image of given size.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = median_filter(img, size=3)
    img = img.astype("float32") / 255.0
    return img

def load_cnn_dataset(folder_path, label, size=IMG_SIZE):
    """
    Load all images recursively from folder, preprocess them,
    and assign the given label.
    Returns numpy arrays (images, labels).
    """
    images = []
    labels = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                img = preprocess_image_cnn(path, size)
                if img is not None:
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)

#