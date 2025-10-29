import os
import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, raw_data_dir, labels_file, processed_data_dir):
        self.raw_data_dir = raw_data_dir
        self.labels_file = labels_file
        self.processed_data_dir = processed_data_dir
        self.data = self.load_labels()

    def load_labels(self):
        return pd.read_csv(self.labels_file)

    def organize_dataset(self):
        organized_data = []
        for index, row in self.data.iterrows():
            scanner_model = row['scanner_model']
            file_name = row['file_name']
            file_path = os.path.join(self.raw_data_dir, file_name)
            if os.path.exists(file_path):
                organized_data.append({
                    'scanner_model': scanner_model,
                    'file_name': file_name,
                    'file_path': file_path
                })
        return organized_data

    def split_dataset(self, test_size=0.2):
        organized_data = self.organize_dataset()
        train_data, test_data = train_test_split(organized_data, test_size=test_size)
        return train_data, test_data

    def save_processed_data(self, processed_data):
        for item in processed_data:
            # Here you would implement the logic to save the processed images
            pass

    def get_dataset(self):
        return self.organize_dataset()