import os
import cv2
import numpy as np

def preprocess_image(image_path, output_size=(256, 256), denoise=False):
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image
    image_resized = cv2.resize(image, output_size)

    # Convert to grayscale if the image has color channels
    if len(image_resized.shape) == 3:
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image_resized

    # Denoise the image if required
    if denoise:
        image_denoised = cv2.fastNlMeansDenoising(image_gray, None, 30, 7, 21)
    else:
        image_denoised = image_gray

    # Normalize the image
    image_normalized = cv2.normalize(image_denoised, None, 0, 255, cv2.NORM_MINMAX)

    return image_normalized

def save_processed_image(image, output_path):
    cv2.imwrite(output_path, image)

def process_images(input_dir, output_dir, output_size=(256, 256), denoise=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            processed_image = preprocess_image(image_path, output_size, denoise)
            output_path = os.path.join(output_dir, filename)
            save_processed_image(processed_image, output_path)

if __name__ == "__main__":
    input_directory = '../data/raw'  # Adjust the path as necessary
    output_directory = '../data/processed'  # Adjust the path as necessary
    process_images(input_directory, output_directory, output_size=(256, 256), denoise=True)