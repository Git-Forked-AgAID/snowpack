import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import dataset

# Define Data & Load Dataset ###########################################################
csv_file = "processed_data.csv" # TODO: Update path
SWEdataset = dataset.SWEDataset(csv_file)
dataloader = DataLoader(SWEdataset, batch_size=32, shuffle=True)

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

        # Loop through each batch in the dataloader
        for inputs, targets in dataloader:
            batch_X, batch_y = inputs.to(device), targets.to(device)

            optimizer.zero_grad() # Clear gradients (this clears previous gradiants in previous batch ensuring consistent training)
            outputs = model(batch_X) # Forward pass (Generate predicitions)
            loss = criterion(outputs, batch_y) 
            loss.backward() # Backward pass (Calculate gradients based on loss)
            optimizer.step() # Update weights

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model
def save_model(model):
    torch.save(model.state_dict(), 'SWE_Predictor.pth')

def load_model(model):
    model.load_state_dict(torch.load('SWE_Predictor.pth'))

def predict(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU is available
    model.to(device) # Move model to assigned device
    model.eval()

    with torch.no_grad():
        for inputs, _ in dataloader: # returns tuple, use _ to ignore second element
            inputs = inputs.to(device)
            predictions = model(inputs)
            print(predictions[:5])
            break

# Main 
if __name__ == "__main__":

    to_train = False
    # Model Parameters
    input_size = 15  # Number of numerical features
    hidden_size = 64 # number of layers connecting input to output
    num_layers = 2 # LSTM layers (input/output layers)
    output_size = 1  # Predicting SWE
    model = WeatherLSTM(input_size, hidden_size, num_layers, output_size)

    while exit == False:
        print("Train model? (y/n)")
        if input() == 'y':
            train_model(model, SWEdataset, 10)
            save_model(model)

        print("Predict? (y/n)")
        if input() == 'y':
            if load_model:
                load_model(model)
                predict(model, dataloader)
            else:
                print("Model not found")
        print("Exit? (y/n)")
        if input() == 'y':
            exit = True
            break
    print("Goodbye!")


