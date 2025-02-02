import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import dataset
from dataset import INPUT_SIZE, EPOCHS, HIDDEN_LAYERS, BATCH_SIZE
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

SWEdataset = dataset.SWEDataset("./clean_main.csv")
dataloader = DataLoader(SWEdataset, batch_size=BATCH_SIZE, shuffle=True)

with open("log2.txt", "w") as f:
    f.write(f"loaded\n")

class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)


        out, (hn, cn) = self.lstm(x)  # lstm_out will have shape (batch_size, seq_len, hidden_size)
        # print(out)
        # print(out)
        out = self.fc(out[-1, :])
        # out = self.fc(out[-1, :])

        return out, hn, cn

def train_model(model, train_loader, num_epochs):

    # Training Configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU is available
    model.to(device) # Move model to assigned device
    criterion = nn.MSELoss() # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Optimizer

    h0, c0 = None, None

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        counter= 0
        # Loop through each batch in the dataloader

        # outputs = []
        for inputs, targets in train_loader:
            # print(len(inputs), len(targets))
            # print("inputs")
            # print (inputs)
            # print("targets")
            # print (targets)
            batch_X, batch_y = inputs.to(device), targets.to(device)
            outputs = []
            for row in batch_X:
                # print(row)
                row = row.view(1, INPUT_SIZE)
                output1, h0, c0 = model(row, h0, c0)  # Forward pass (Generate predictions)
                outputs.append(output1)
            # print("@@@@@@@@@", outputs)

            # optimizer.zero_grad()  # Clear gradients from previous batch

            # print("\n\n\n")
            # print(batch_X, batch_X.shape)
            # print(outputs, outputs.shape)
            # outputs = outputs.view(*batch_y.shape)
            outputs = torch.tensor(outputs, requires_grad=True).view(-1, 1)
            outputs = outputs.to(device)
            # print(outputs.shape)
            loss = criterion(outputs, batch_y)  # Compute the loss
            loss.backward()  # Backward pass (Calculate gradients based on loss)
            optimizer.step()  # Update model weights
            h0 = h0.detach()
            c0 = c0.detach()

            with open("log2.txt", "a") as f:
                f.write(f"{counter}, loss:{loss.item()}\n")
            print({counter}, "Loss", loss.item())
            counter += 1

        print (f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        with open("log2.txt", "a") as f:
            f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}\n")

    return h0, c0
        # input("Press Enter to continue...")  # Debugging

# Save the model
def save_model(model):
    torch.save(model.state_dict(), 'SWE_Predictor.pth')

def load_model(model, fname):
    model.load_state_dict(torch.load(fname))

def predict(model, dataloader, h0, c0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU is available
    model.to(device) # Move model to assigned device
    model.eval()

    all_predictions = []  # To store all predictions

    with torch.no_grad():
        for inputs, _ in dataloader:  # We don't need targets for prediction
            inputs = inputs.to(device)
            predictions = model(inputs, h0, c0)
            all_predictions.append(predictions.cpu().numpy())  # Store predictions

    return all_predictions

def predict_single_day(model, weather_tensor, h0=None, c0=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to the correct device
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        weather_tensor = weather_tensor.to(device)  # Move input to device
        prediction = model(weather_tensor, h0, c0)  # Run prediction
        return prediction  # Convert tensor to Python float

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU is available
    input_size = INPUT_SIZE  # Number of numerical features
    hidden_size = HIDDEN_LAYERS # number of layers connecting input to output
    num_layers = 2   # LSTM layers (input/output layers)
    output_size = 1  # Predicting SWE
    model = WeatherLSTM(input_size, hidden_size, num_layers, output_size)
    load_model(model, "./kamiak512_20epochs.pth")
    # model.to(device)

    fmin = SWEdataset.feature_scaler.data_min_
    fmax = SWEdataset.feature_scaler.data_max_
    tmin = SWEdataset.target_scaler.data_min_
    tmax = SWEdataset.target_scaler.data_max_
    # print(fmin, fmax, tmin, tmax)

    inp = torch.tensor([33.65352,-109.30877,int(pd.to_datetime("1991-01-01").timestamp()*(10**9)),9027,0.888151729,2.16,-9.3,7.92,117.325,0.0016,19.78,69.42,0.0,0.0276550772915165], dtype=torch.float64)
    # inp = SWEdataset.inptest

    inp = (inp-fmin)/(fmax-fmin)
    # inp = torch.tensor(inp, dtype=torch.float32)

    # TODO::: WORRY ABOUT THE dtype F64??

    print(inp.view(-1).tolist())
    inp = inp.view(1, -1)
    pred = predict_single_day(model, inp.type(torch.float32))[0]

    pred = pred.cpu()

    print(pred.tolist()[0])


    print(tmax, tmin)
    scaled_pred = (pred * (tmax-tmin)) + (tmin)
    print(scaled_pred.tolist()[0])

    # print("Model created")

    ## while not exit_program:
    #    # print("Train model? (y/n)")
    #h0, c0 = None, None
    ## if input().lower() == 'y':
    #print("Splitting data...")
    ##train_data, test_data = train_test_split(SWEdataset, test_size=0.2, shuffle=True)
    #print("Training model...")
    #h0, c0 = train_model(model, dataloader, EPOCHS)  # Pass dataloader to train_model
    #save_model(model)
