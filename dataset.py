import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Data will be in CSV form with the following columns:
    # Date, Name, Lat, Long, elevation, southness, SWE, <------ From SWE_Values.csv & Station_info.csv
    # Windspeed, Tmin, Tmax, SRAD, SPH, Rmin, Rmax, Precipitation <------ From meteorological data and corresponding .csv files

class SWEDataset(Dataset):
    def __init__(self, csv_file, sequence_length=365, target_column="SWE"):
        """
        Custom PyTorch Dataset for SWE prediction.

        Args:
        - csv_file (str): Path to the CSV file.
        - sequence_length (int): Number of past timesteps to include. (default 10 years)
        - target_column (str): The column to predict.
        """

        # Read data, set sequence length and make labelled columns for data
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        #print(self.data.head())  # Check data
        self.data.columns = ['lat', 'lon', 'date', 'station', 'elevation', 'southness', 'SWE', 'windspeed', 'tmin', 'tmax', 'SRAD', 'SPH', 'rmin', 'rmax', 'precipitation', 'distance_from_met']
        
        # Convert 'Date' to datetime & drop NAS
        self.data.columns = self.data.columns.str.strip()
        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
        self.data = self.data.dropna()  # Drop rows with NaT/NaN

    # Get the number of samples in the dataset
    def __len__(self):
        return len(self.data) - self.sequence_length # Subtract sequence length to avoid index out of bounds

    # retreive a data sample
    def __getitem__(self, idx):
        # Generate a sequence of days days
        sequence = self.data.iloc[idx:idx + self.sequence_length]
        
        # Get the features and target for the sequence
        features = sequence[['lat', 'lon', 'elevation', 'southness', 'SWE', 'windspeed', 'tmin', 'tmax', 'SRAD', 'SPH', 'rmin', 'rmax', 'precipitation']].values
        target = sequence['SWE'].iloc[-1]  # Target is SWE at the last timestep in the sequence

        # Convert to tensors for PyTorch use
        features = np.array(features, dtype=np.float32)  # Convert to np array so compatible with torch.tensor
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)