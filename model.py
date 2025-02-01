import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Define Data & Load Dataset


# TODO: Figure out data normalization
# Normalize data
from sklearn.preprocessing import MinMaxScaler

# TODO: Define the DataLoader
# Define the DataLoader

### Define the Neural Network Architecture Model ###########################################################
# Long Short Term Memory (LSTM) Model (These are commonly used for Time-series forcasting)
# Type of RNN that is capable of learning long-term dependencies
class WeatherLSTM(nn.Module):
    # Constructor
    # takes input size (should be 15), Number of hidden layers (Non input/output layer), number of LSTM layers (input/output layers), and output size
    # Data will be in CSV form with the following columns:
        # Date, Name, Lat, Long, elevation, southness, SWE, <------ From SWE_Values.csv & Station_info.csv
        # Windspeed, Tmin, Tmax, SRAD, SPH, Rmin, Rmax, Precipitation <------ From meteorological data and corresponding .csv files
    

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # Input layer
                                                                                # Returns a tuple (output, hidden_state)
        self.fc = nn.Linear(hidden_size, output_size) # Output layer (Fully connected layer)

    def forward(self, x): 
        out, _ = self.lstm(x) # Takes first element
        out = self.fc(out[:, -1, :])  # Selects last timesteps output
        return out 


# Notes:
# Loss function is a method of evaluating how well a model's predictions match the actual data (Lower = more accurate)
# Optimizer is a method of adjusting the model's internal parameters to reduce the loss function

# Training process defenition ###########################################################
def train_model(model, train_loader, num_epochs):

    # Training Configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU is available
    model.to(device) # Move model to assigned device
    criterion = nn.MSELoss() # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Optimizer

    for epoch in range(num_epochs):
        model.train() # Set model to training mode

        ###############################
        # TODO: Link up the DataLoader
        # TODO: Figure out batch size (How many samples to use in each iteration)
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        ###############################

            optimizer.zero_grad() # Clear gradients (this clears previous gradiants in previous batch ensuring consistent training)
            outputs = model(batch_X) # Forward pass (Generate predicitions)
            loss = criterion(outputs, batch_y) 
            loss.backward() # Backward pass (Calculate gradients based on loss)
            optimizer.step() # Update weights

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model
def save_model(model):
    torch.save(model.state_dict(), 'SWE_Predictor.pth')

# Main 
if __name__ == "__main__":
    # Model Parameters
    input_size = 15  # Number of numerical features
    hidden_size = 64 # number of layers connecting input to output
    num_layers = 2 # LSTM layers (input/output layers)

    # TODO: Figure out what our output will look like
    output_size = 1  # Predicting precipitation or another target variable
    model = WeatherLSTM(input_size, hidden_size, num_layers, output_size)
    train_model(model, 10)
    save_model(model)