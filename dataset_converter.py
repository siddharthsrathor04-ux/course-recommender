# Updated dataset_converter.py

import os
import json

class DatasetConverter:
    def __init__(self, dataset_directory):
        self.dataset_directory = dataset_directory
        self.genres = []

    def find_dataset_files(self):
        
        # Improved extraction of dataset files from directory
        dataset_files = []
        for dirpath, _, filenames in os.walk(self.dataset_directory):
            for filename in filenames:
                if filename.endswith('.json'):
                    dataset_files.append(os.path.join(dirpath, filename))
        return dataset_files

    def extract_genres(self):
        
        # Enhanced genre handling
        for dataset_file in self.find_dataset_files():
            with open(dataset_file, 'r') as f:
                data = json.load(f)
                for item in data:
                    if 'genres' in item:
                        self.genres.extend(item['genres'])

    def get_unique_genres(self):
        return set(self.genres)

# Usage:
# converter = DatasetConverter('/path/to/dataset')
# converter.extract_genres()
# unique_genres = converter.get_unique_genres()