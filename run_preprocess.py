import os
import cv2
import pandas as pd
from src.preprocess import preprocess_image
from src.dataset import create_dataset

def main():
    # Define paths
    raw_data_dir = 'data/raw'
    processed_data_dir = 'data/processed'
    labels_file = 'data/annotations/labels.csv'

    # Ensure directories exist
    os.makedirs(processed_data_dir, exist_ok=True)

    # Load labels
    labels = pd.read_csv(labels_file)

    # Process each image
    for index, row in labels.iterrows():
        file_name = row['file_name']
        scanner_model = row['scanner_model']
        image_path = os.path.join(raw_data_dir, file_name)

        # Preprocess the image
        processed_image = preprocess_image(image_path)

        # Save the processed image
        processed_image_path = os.path.join(processed_data_dir, f'processed_{file_name}')
        cv2.imwrite(processed_image_path, processed_image)

    # Create dataset for model training
    create_dataset(processed_data_dir)

if __name__ == '__main__':
    main()