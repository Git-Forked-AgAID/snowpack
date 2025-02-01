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

        # Read data, set sequence length and target col
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        print(self.data.head())  # Check data
        self.data.columns = ['Lat', 'Long', 'Date', 'station', 'Elevation', 'Southness', 'SWE', 'Windspeed', 'Tmin', 'Tmax', 'SRAD', 'SPH', 'Rmin', 'Rmax', 'Precipitation', 'Distance_from_met']
        
        # Convert 'Date' to datetime & drop NAS
        self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
        #self.data = self.data.dropna()  # Drop rows with NaT/NaN
  

    # Get the number of samples in the dataset
    def __len__(self):
        print("Data Length")
        print(len(self.data))
        return len(self.data) - self.sequence_length

    # retreive a data sample
    def __getitem__(self, idx):
        # Retrieve the row for the given index
        row = self.data.iloc[idx]
        
        # Example of how you can prepare your features and target
        features = row[['Lat', 'Lon', 'Elevation', 'Southness', 'SWE', 'Windspeed', 'Tmin', 'Tmax', 'SRAD', 'SPH', 'Rmin', 'Rmax', 'Precipitation']].values
        target = row['SWE']  # Assuming we are predicting SWE

        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)