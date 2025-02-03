# rewrite of geeksforgeeks tutorial
# UNUSED

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

VALS = ['lat', 'lon', 'date', 'elevation', 'southness', 'windspeed', 'tmin', 'tmax', 'srad', 'sph', 'rmin', 'rmax', 'precip', 'dist_from_met']
csv_file = "clean_main_day1.csv"
BATCHP_SIZE = 256
# Generate synthetic sine wave data
# t = np.linspace(0, 100, 1000)
# data = np.sin(t)

# Read data, set sequence length and make labelled columns for data
df = pd.read_csv(csv_file)

# drop cols we don't need
if "test.csv" not in csv_file:
    for col in ['id', 'county', 'state', 'station']:
        df = df.drop(col, axis=1)

# print(self.data)
# print(self.data.columns)

df['date'] = pd.to_datetime(df['date']).astype(int)


# Normalize features
feature_scaler = MinMaxScaler()
df[VALS] = feature_scaler.fit_transform(df[VALS])

N = 0

print('norm_train_feat', list(df[VALS].iloc[N]))

# Normalize target
target_scaler = MinMaxScaler()
df['swe'] = target_scaler.fit_transform(df[['swe']])
print('norm_train_swe',df['swe'].iloc[N])


# Function to create sequences
# def create_sequences(data, seq_length):
#     xs = []
#     ys = []
#     for i in range(len(data)-seq_length):
#         x = data[i:(i+seq_length)]
#         y = data[i+seq_length]
#         xs.append(x)
#         ys.append(y)
#     return np.array(xs), np.array(ys)

#seq_length = 10
# X, y = create_sequences(df, seq_length)

# Convert data to PyTorch tensors
trainX = torch.tensor(df[VALS].values, dtype=torch.float32)
trainY = torch.tensor(df['swe'].values, dtype=torch.float32).view(-1, 1)
#trainY = torch.tensor(df['swe'].values, dtype=torch.float32).unsqueeze(1)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)
        # If hidden and cell states are not provided, initialize them as zeros
        if h0 is None or c0 is None:
            # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            # c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)


        # Forward pass through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Selecting the last output
        return out, hn, cn

# Initialize model, loss, and optimizer
model = LSTMModel(input_dim=len(VALS), hidden_dim=100, layer_dim=14, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
h0, c0 = None, None  # Initialize hidden and cell states
# dataset = torch.utils.data.TensorDataset(trainX, trainY)
# trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCHP_SIZE, shuffle=True)

# Setup Graph
ax = plt.gca()
ax.set_ylim([.01, .03])
counter = 0;

for epoch in range(num_epochs):
    model.train()
    # print("X Shape")
    # print(trainX.shape)
    # print("Y Shape")
    # print(trainY.shape)

    #for batchX, batchY in trainloader:
    counter += 1
    optimizer.zero_grad()

    # Forward pass
    #trainX = trainX
    trainX = trainX.view(-1, 1, len(VALS))
    #trainX = trainX.view(-1, seq_length, len(VALS))
    outputs, h0, c0 = model(trainX, h0, c0)

    print("Outputs Shape")
    print(outputs.shape)

    # Compute loss
    print("Computing loss")
    loss = criterion(outputs, trainY)
    print("Back propogating")
    loss.backward()
    print("Optimizing")
    optimizer.step()

    # Detach hidden and cell states to prevent backpropagation through the entire sequence
    h0 = h0.detach()
    c0 = c0.detach()

    if (epoch+1) % 10 == 0:
        print(f'Epooch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        input()

                # graph loss in real time
    ax.set_xlim([0, counter * (epoch +1)])
    plt.scatter((counter * (epoch + 1)), loss.item())
    plt.pause(0.05)

    # Predicted outputs
model.eval()
predicted, _, _ = model(trainX, h0, c0)

# Adjusting the original data and prediction for plotting
# original = df[seq_length:]  # Original data from the end of the first sequence
# time_steps = np.arange(seq_length, len(df))  # Corresponding time steps

# # Plotting
# plt.figure(figsize=(12, 6))
# plt.plot(time_steps, original, label='Original Data')
# plt.plot(time_steps, predicted.detach().numpy(), label='Predicted Data', linestyle='--')
# plt.title('LSTM Model Predictions vs. Original Data')
# plt.xlabel('Time Step')
# plt.ylabel('Value')
# plt.legend()
# plt.show()
