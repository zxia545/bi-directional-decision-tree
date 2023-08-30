from typing import List, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
class SimpleTrainingDataset(Dataset):
    def __init__(self, sensor_data_file_path: str):
        """
        Initialize the dataset.
        
        Parameters:
        - sensor_data_file_path: Path to the .data file containing sensor readings.
        """
        self.features, self.labels = self._load_data(sensor_data_file_path)

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch a single item (feature and label) from the dataset."""
        return self.features[idx], self.labels[idx]
    
    def _load_data(self, data_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the data from the .data file."""
        
        # Define a label mapping
        label_map = {'Move-Forward': 0, 'Slight-Right-Turn': 1, 'Sharp-Right-Turn': 2, 'Other': 3}
        
        # Read the .data file into a DataFrame
        df = pd.read_csv(data_file_path, header=None)
        d = df.values
        
        # Initialize lists for features and labels
        features_list = []
        labels_list = []
        
        # Iterate over each row in the DataFrame
        for i in range(len(d)):
            features_list.append(d[i, :-1])
            label_str = d[i, -1]
            
            # Map the label string to its corresponding integer
            label_int = label_map.get(label_str, 3)  # Default to 3 ('Other') if label is not in the mapping
            labels_list.append(label_int)

        # Convert lists to NumPy arrays
        features_np = np.array(features_list, dtype=np.float32)
        labels_np = np.array(labels_list, dtype=np.int64)

        return features_np, labels_np



def load_simple_classifer_data(dataset_path, num_workers=16, batch_size=16, shuffle=True, **kwargs):
    dataset = SimpleTrainingDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)