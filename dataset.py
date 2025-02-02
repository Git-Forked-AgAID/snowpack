import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

VALS = ['lat', 'lon', 'date', 'elevation', 'southness', 'windspeed', 'tmin', 'tmax', 'srad', 'sph', 'rmin', 'rmax', 'precip', 'dist_from_met']
# VALS = ['lat', 'lon']
INPUT_SIZE = len(VALS)
HIDDEN_LAYERS = 8
EPOCHS = 10
BATCH_SIZE = 256
SEQ_SIZE = 100
NEURONS = 156

# Data will be in CSV form with the following columns:
    # Date, Name, Lat, Long, elevation, southness, SWE, <------ From SWE_Values.csv & Station_info.csv
    # Windspeed, Tmin, Tmax, SRAD, SPH, Rmin, Rmax, Precipitation <------ From meteorological data and corresponding .csv files

class SWEDataset(Dataset):
    def __init__(self, csv_file):
        # Read data, set sequence length and make labelled columns for data
        self.data = pd.read_csv(csv_file)

        # drop cols we don't need
        if "test.csv" not in csv_file:
            for col in ['id', 'county', 'state', 'station']:
                self.data = self.data.drop(col, axis=1)

        # print(self.data)
        # print(self.data.columns)

        self.data['date'] = pd.to_datetime(self.data['date']).astype(int)


        # Normalize features
        self.feature_scaler = MinMaxScaler()
        self.data[VALS] = self.feature_scaler.fit_transform(self.data[VALS])

        N = 0

        #print('norm_train_feat', list(self.data[VALS].iloc[N]))

        # Normalize target
        self.target_scaler = MinMaxScaler()
        self.data['swe'] = self.target_scaler.fit_transform(self.data[['swe']])
        #print('norm_train_swe',self.data['swe'].iloc[N])
        # self.inptest = torch.tensor(self.data[VALS].iloc[N])

        self.data.to_csv("normalize_clean_main.csv", index=False)

    # Get the number of samples in the dataset
    def __len__(self):
        return len(self.data) - SEQ_SIZE# Subtract sequence length to avoid index out of bounds

    # retreive a data sample
    def __getitem__(self, idx):
        rows = self.data.iloc[idx: idx+SEQ_SIZE]

        features = rows[VALS].values
        target = rows['swe']

        # Convert to tensors for PyTorch use
        features = np.array(features, dtype=np.float32)  # Convert to np array so compatible with torch.tensor
        target = np.array(target, dtype=np.float32)
        # print(features.shape)
        # input()
        # print(target)

        rvalues = torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
        # print(rvalues[0].shape, rvalues[1].shape)
        return rvalues
