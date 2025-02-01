import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import dataset

# Define Data & Load Dataset ###########################################################
csv_file = "test.csv" # TODO: Update path
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
        lstm_out, _ = self.lstm(x)  # lstm_out will have shape (batch_size, seq_len, hidden_size)
        
        # Select the output of the last timestep
        last_out = lstm_out[:, -1, :]  # Selects the last timestep output (batch_size, hidden_size)

        # Pass the output through the fully connected layer
        return self.fc(last_out)  # Final output (batch_size, output_size)


# Notes:
# Loss function is a method of evaluating how well a model's predictions match the actual data (Lower = more accurate)
# Optimizer is a method of adjusting the model's internal parameters to reduce the loss function

# Training process defenition ###########################################################
def train_model(model, train_loader, num_epochs):

    # Training Configurations
    idx = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU is available
    model.to(device) # Move model to assigned device
    criterion = nn.MSELoss() # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Optimizer

    # Debugging
    # print("Training Model...")
    # print(f"Device: {device}")

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        # Loop through each batch in the dataloader
        for inputs, targets in train_loader:
            idx += 1
            batch_X, batch_y = inputs.to(device), targets.to(device)
            
            # Debugging
            # print("Batch X Shape")
            # print(batch_X.shape) 
            # print("Batch Y Shape")
            # print(batch_y.shape)
            #                              /
            # Old unsequential re-shaping V
            #batch_X = batch_X.unsqueeze(1)  # Adds an extra dimension for seq_len (Is set to one, need to update to something much larger)
            #batch_y = batch_y.view(-1, 1)  # Reshape the target tensor to (32, 1) forcing it to fit dimensionality

            batch_X = batch_X.view(-1, 365, 13)  # Ensure correct shape for LSTM
            batch_y = batch_y.view(-1, 1)  # Reshape target tensor to (batch_size, 1)


            optimizer.zero_grad()  # Clear gradients from previous batch
            outputs = model(batch_X)  # Forward pass (Generate predictions)
            loss = criterion(outputs, batch_y)  # Compute the loss
            loss.backward()  # Backward pass (Calculate gradients based on loss)
            optimizer.step()  # Update model weights

        print (f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        wait = input("Press Enter to continue...")  # Debugging

# Save the model
def save_model(model):
    torch.save(model.state_dict(), 'SWE_Predictor.pth')

def load_model(model):
    model.load_state_dict(torch.load('SWE_Predictor.pth'))

def predict(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU is available
    model.to(device) # Move model to assigned device
    model.eval()

    all_predictions = []  # To store all predictions

    with torch.no_grad():
        for inputs, _ in dataloader:  # We don't need targets for prediction
            inputs = inputs.to(device)
            predictions = model(inputs)
            all_predictions.append(predictions.cpu().numpy())  # Store predictions

    return all_predictions

def predict_single_day(model, weather_tensor):
    """
    Predicts SWE for a single day's weather conditions.

    Args:
    - model (torch.nn.Module): Trained PyTorch model.
    - weather_tensor (torch.Tensor): Tensor of shape (1, sequence_length, input_size).

    Returns:
    - Predicted SWE value as a float.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to the correct device
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        weather_tensor = weather_tensor.to(device)  # Move input to device
        prediction = model(weather_tensor)  # Run prediction
        return prediction.item()  # Convert tensor to Python float
    
# Main 
if __name__ == "__main__":


    exit_program = False
    to_train = False
    # Model Parameters
    input_size = 13  # Number of numerical features
    hidden_size = 64 # number of layers connecting input to output
    num_layers = 2 # LSTM layers (input/output layers)
    output_size = 1  # Predicting SWE
    model = WeatherLSTM(input_size, hidden_size, num_layers, output_size)

    while not exit_program:
        print("Train model? (y/n)")
        if input().lower() == 'y':
            train_model(model, dataloader, 10)  # Pass dataloader to train_model
            save_model(model)

        print("Predict? (y/n)")
        if input().lower() == 'y':
            print(model)
            #try:
            load_model(model)  # Attempt to load the model
            first_predictions = predict(model, dataloader)
            ####### Testing ################################################################
            weather_data = np.array([[40.7128, -74.0060, 10, 1.5, 0, 3.2, 5, 10, 250, 1.3, 2.1, 1.9, 0.2]], dtype=np.float32)

            # Convert to tensor and reshape to match (batch_size=1, sequence_length=1, input_size=13)
            weather_tensor = torch.tensor(weather_data, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1, 13)

            # Predict using the fixed function
            second_prediction = predict_single_day(model, weather_tensor)

            print("Predicted SWE:", second_prediction)
                #################################################################################
            #except Exception as e:
                #print(f"Error: Model not found or not loaded correctly.")

        print("Exit? (y/n)")
        if input().lower() == 'y':
            exit_program = True
    print("Goodbye!")


