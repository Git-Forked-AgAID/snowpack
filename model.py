import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import dataset

BATCH_SIZE = 160

SWEdataset = dataset.SWEDataset("./clean_main_day1.csv")
dataloader = DataLoader(SWEdataset, batch_size=BATCH_SIZE, shuffle=True)

class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
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
        out = self.fc(out[-1, :])
        # out = self.fc(out[-1, :])

        return out, hn, cn

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
    h0, c0 = None, None

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        # Loop through each batch in the dataloader
        for inputs, targets in train_loader:
            idx += 1
            batch_X, batch_y = inputs.to(device), targets.to(device)

            # Debugging
            print("Batch X Shape")
            print(batch_X.shape)
            # print("Batch Y Shape")
            # print(batch_y.shape)
            #                              /
            # Old unsequential re-shaping V
            # batch_X = batch_X.unsqueeze(1)  # Adds an extra dimension for seq_len (Is set to one, need to update to something much larger)
            # batch_y = batch_y.view(-1, 1)  # Reshape the target tensor to (32, 1) forcing it to fit dimensionality

            # print(batch_X)
            batch_X = batch_X.view(-1, BATCH_SIZE, 14)  # Ensure correct shape for LSTM
            batch_y = batch_y.view(-1, 1)  # Reshape target tensor to (batch_size, 1)


            optimizer.zero_grad()  # Clear gradients from previous batch
            outputs, h0, c0 = model(batch_X, h0, c0)  # Forward pass (Generate predictions)
            loss = criterion(outputs, batch_y)  # Compute the loss
            # input("A")
            loss.backward()  # Backward pass (Calculate gradients based on loss)
            # input("B")
            optimizer.step()  # Update model weights


        print (f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    return h0, c0
        # input("Press Enter to continue...")  # Debugging

# Save the model
def save_model(model):
    torch.save(model.state_dict(), 'SWE_Predictor.pth')

def load_model(model):
    model.load_state_dict(torch.load('SWE_Predictor.pth'))

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

def predict_single_day(model, weather_tensor, h0, c0):
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
        prediction = model(weather_tensor, h0, c0)  # Run prediction
        return prediction.item()  # Convert tensor to Python float

if __name__ == "__main__":
    exit_program = False
    to_train = False

    input_size = 14  # Number of numerical features
    hidden_size = 2 # number of layers connecting input to output
    num_layers = 2   # LSTM layers (input/output layers)
    output_size = 1  # Predicting SWE
    model = WeatherLSTM(input_size, hidden_size, num_layers, output_size)

    while not exit_program:
        print("Train model? (y/n)")
        h0, c0 = None, None
        if input().lower() == 'y':
            h0, c0 = train_model(model, dataloader, 75)  # Pass dataloader to train_model
            save_model(model)

        #print("Predict? (y/n)")
        #if input().lower() == 'y':
        #    print(model)
        #    #try:
        #    load_model(model)  # Attempt to load the model
        #    first_predictions = predict(model, dataloader)
        #    ####### Testing ################################################################
        #    # lat,lon,date,station,elevation,southness,swe,windspeed,tmin,tmax,srad,sph,rmin,rmax,precip,dist_from_met
        #    # 48.17478,-109.64728,1991-01-01,Rocky Boy,4700,-0.744754638,58.42,2.68,-23.27,-5.81,33.1,0.0005,43.12,87.36,7.0,0.0205869327487129,917,MT,Hill,POINT (-109.64728 48.17478)
        #    weather_data = np.array([[48.17478,-109.64728,1991-01-01,Rocky Boy,4700,-0.744754638,58.42,2.68,-23.27,-5.81,33.1,0.0005,43.12,87.36,7.0,0.0205869327487129]], dtype=np.float32)

        #    # Convert to tensor and reshape to match (batch_size=1, sequence_length=1, input_size=13)
        #    weather_tensor = torch.tensor(weather_data, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1, 13)

        #    # Predict using the fixed function
        #    second_prediction = predict_single_day(model, weather_tensor)

        #    print("Predicted SWE:", second_prediction)
        #        #################################################################################
        #    #except Exception as e:
        #        #print(f"Error: Model not found or not loaded correctly.")

        #print("Exit? (y/n)")
        #if input().lower() == 'y':
        #    exit_program = True
    print("Goodbye!")
