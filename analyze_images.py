import os
import cv2
import pandas as pd

def analyze_images(raw_data_dir):
    image_properties = []

    # Iterate through all files in the raw data directory
    for filename in os.listdir(raw_data_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            file_path = os.path.join(raw_data_dir, filename)
            # Read the image
            image = cv2.imread(file_path)

            # Get image properties
            height, width, channels = image.shape
            resolution = f"{width}x{height}"
            format = filename.split('.')[-1]
            color_channel = channels

            # Append properties to the list
            image_properties.append({
                'file_name': filename,
                'resolution': resolution,
                'format': format,
                'color_channel': color_channel
            })

    # Create a DataFrame and save to CSV
    properties_df = pd.DataFrame(image_properties)
    properties_df.to_csv(os.path.join('data', 'annotations', 'labels.csv'), index=False)

if __name__ == "__main__":
    raw_data_directory = os.path.join('data', 'raw')
    analyze_images(raw_data_directory)