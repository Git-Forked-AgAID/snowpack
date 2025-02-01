import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

    # Data will be in CSV form with the following columns:
        # Date, Name, Lat, Long, elevation, southness, SWE, <------ From SWE_Values.csv & Station_info.csv
        # Windspeed, Tmin, Tmax, SRAD, SPH, Rmin, Rmax, Precipitation <------ From meteorological data and corresponding .csv files

class SWEDataset(Dataset):
    def __init__(self, csv_file, sequence_length=3650, target_column="SWE"):
        """
        Custom PyTorch Dataset for SWE prediction.

        Args:
        - csv_file (str): Path to the CSV file.
        - sequence_length (int): Number of past timesteps to include. (default 10 years)
        - target_column (str): The column to predict.
        """

        # Read data, set sequence length and target col
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.target_column = target_column 
            
        # Normalize numerical features (excluding non-numeric ones)
        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.data[self.numeric_cols] = (self.data[self.numeric_cols] - self.data[self.numeric_cols].mean()) / (self.data[self.numeric_cols].std() + 1e-8)

    # Get the number of samples in the dataset
    def __len__(self):
        return len(self.data) - self.sequence_length

    # retreive a data sample
    def __getitem__(self, idx):
        past_sequence = self.data.iloc[idx : idx + self.sequence_length].drop(columns=["Date", "Name"]).values.astype(np.float32)
        target = self.data.iloc[idx + self.sequence_length][self.target_column]
        
        return torch.tensor(past_sequence), torch.tensor(target, dtype=torch.float32)