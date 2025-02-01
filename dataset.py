import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Data will be in CSV form with the following columns:
    # Date, Name, Lat, Long, elevation, southness, SWE, <------ From SWE_Values.csv & Station_info.csv
    # Windspeed, Tmin, Tmax, SRAD, SPH, Rmin, Rmax, Precipitation <------ From meteorological data and corresponding .csv files

class SWEDataset(Dataset):
    def __init__(self, csv_file):
        # Read data, set sequence length and make labelled columns for data
        self.data = pd.read_csv(csv_file)

        # drop cols we don't need
        for col in ['id', 'county', 'state', 'station', 'geometry']:
            self.data = self.data.drop(col, axis=1)

        print(self.data)
        print(self.data.columns)

        self.data['date'] = pd.to_datetime(self.data['date']).astype(int)

    # Get the number of samples in the dataset
    def __len__(self):
        return len(self.data) # Subtract sequence length to avoid index out of bounds

    # retreive a data sample
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        features = row[['lat', 'lon', 'date', 'elevation', 'southness', 'windspeed', 'tmin', 'tmax', 'srad', 'sph', 'rmin', 'rmax', 'precip', 'dist_from_met']].values
        target = row['swe']

        # Convert to tensors for PyTorch use
        features = np.array(features, dtype=np.float32)  # Convert to np array so compatible with torch.tensor
        # print(features.shape)
        # input()

        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
