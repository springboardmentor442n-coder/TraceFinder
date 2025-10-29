import os
import shutil
import pandas as pd

def collect_samples(scanner_models, raw_data_dir):
    """
    Collect scanned document samples from different scanner devices and save them in the raw data directory.
    
    Parameters:
    - scanner_models: List of tuples containing (scanner_model, file_paths)
    - raw_data_dir: Directory to save the collected raw samples
    """
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)

    collected_samples = []

    for model, file_paths in scanner_models:
        for file_path in file_paths:
            # Copy the scanned document to the raw data directory
            file_name = os.path.basename(file_path)
            destination = os.path.join(raw_data_dir, file_name)
            shutil.copy(file_path, destination)
            collected_samples.append({'scanner_model': model, 'file_name': file_name})

    # Save the collected samples to a CSV file
    labels_df = pd.DataFrame(collected_samples)
    labels_csv_path = os.path.join(os.path.dirname(raw_data_dir), 'annotations', 'labels.csv')
    labels_df.to_csv(labels_csv_path, index=False)

if __name__ == "__main__":
    # Example usage
    scanner_models = [
        ('Scanner A', ['/path/to/scanned/doc1.pdf', '/path/to/scanned/doc2.pdf']),
        ('Scanner B', ['/path/to/scanned/doc3.pdf']),
        ('Scanner C', ['/path/to/scanned/doc4.pdf', '/path/to/scanned/doc5.pdf']),
    ]
    raw_data_directory = 'data/raw'
    collect_samples(scanner_models, raw_data_directory)