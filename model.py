import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from dataset import INPUT_SIZE, EPOCHS, HIDDEN_LAYERS, BATCH_SIZE, SEQ_SIZE, NEURONS, VALS
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

<<<<<<< HEAD
mms = MinMaxScaler()
alldata = pd.read_csv("./stations/0.csv")
alldata['date'] = pd.to_datetime(alldata['date']).astype(int)
alldata[VALS + ['swe']] = mms.fit_transform(alldata[VALS+['swe']])
# alldata[['swe']] = mms.fit_transform(alldata['swe'])

train_ind = int(len(alldata)*0.8)
train = alldata[train_ind:]
train_X = torch.tensor(np.array(train[VALS])).type(torch.float32)
train_Y = torch.tensor(np.array(train['swe'])).type(torch.float32)

test = alldata[:train_ind]
test_X = torch.tensor(np.array(test[VALS])).type(torch.float32)
test_Y = torch.tensor(np.array(test['swe'])).type(torch.float32)

SWEdataset = dataset.SWEDataset(train, test=False)
=======
SWEdataset = dataset.SWEDataset("clean_main.csv")
>>>>>>> 86f497c139fef46efae212b16905b3b8b12f0e78
dataloader = DataLoader(SWEdataset, batch_size=BATCH_SIZE, shuffle=False)

with open("log2.txt", "w") as f:
    f.write(f"loaded\n")

class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, output_size, bias=True),
        )

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.lstm(x)  # lstm_out will have shape (batch_size, seq_len, hidden_size)
        # print(out)
        out = self.fc(out)
        # out = self.fc(out[-1, :])

        return out, hn, cn

def train_model(model, train_loader, num_epochs):

    # Training Configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU is available
    model.to(device) # Move model to assigned device
    criterion = nn.MSELoss() # Loss function
    optimizer = optim.SGD(model.parameters(), lr=.00001) # Optimizer

    # Setup Graph
    ax = plt.gca()
<<<<<<< HEAD
    fig, ax = plt.subplots()

    # ax.set_ylim([0, .1])
=======
    ax.set_ylim([0, .005])
>>>>>>> 86f497c139fef46efae212b16905b3b8b12f0e78

    h0, c0 = None, None

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        counter= 0
        # Loop through each batch in the dataloader

        # outputs = []
        for inputs, targets in train_loader:
            batch_X, batch_y = inputs.to(device), targets.to(device)
            batch_y = batch_y.view(len(targets), SEQ_SIZE, 1)
            outputs, h0, c0 = model(batch_X, h0, c0)  # Forward pass (Generate predictions)
            outputs = outputs.to(device)
            loss = criterion(outputs, batch_y)  # Compute the loss
            loss.backward()  # Backward pass (Calculate gradients based on loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradients to avoid exploding gradients
            optimizer.step()  # Update model weights
            h0 = h0.detach()
            c0 = c0.detach()

            with open("log2.txt", "a") as f:
                f.write(f"{counter}, loss:{loss.item()}\n")
<<<<<<< HEAD
            counter += 1

=======
            print("\t", {counter}, "Loss", loss.item())
            counter += 1

            # graph loss in real time
            ax.set_xlim([0, counter * (epoch +1)])
        #ax.set_xlim([0, num_epochs])
            plt.scatter((counter * (epoch + 1)), loss.item())
        #plt.scatter(epoch, loss.item())
            plt.pause(0.05)

>>>>>>> 86f497c139fef46efae212b16905b3b8b12f0e78

        model.eval()
        with torch.no_grad():
            y_pred,*_ = model(train_X)
            y_pred = y_pred.view(-1)
            # print(y_pred)
            train_rmse = np.sqrt(criterion(y_pred, train_Y))
            y_pred,*_ = model(test_X)
            y_pred = y_pred.view(-1)
            test_rmse = np.sqrt(criterion(y_pred, test_Y))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

        ax.set_xlim([0, epoch+1])
        plt.scatter(epoch, test_rmse, c="blue")
        plt.scatter(epoch, train_rmse, c="red")
        plt.pause(0.05)


#         print (f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        with open("log2.txt", "a") as f:
            f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}\n")

    return h0, c0
        # input("Press Enter to continue...")  # Debugging

# Save the model
def save_model(model):
    torch.save(model.state_dict(), f'SWE_Predictor_{HIDDEN_LAYERS}_{EPOCHS}epochs.pth')

def load_model(model, fname):
    model.load_state_dict(torch.load(fname))

def predict_single_day(model, weather_tensor, h0=None, c0=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to the correct device
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        weather_tensor = weather_tensor.to(device)  # Move input to device
        prediction = model(weather_tensor, h0, c0)  # Run prediction
        return prediction

if __name__ == "__main__":
    exit_program = False
    to_train = True

    input_size = INPUT_SIZE  # Number of numerical features
    hidden_size = NEURONS #
    num_layers = HIDDEN_LAYERS   # LSTM layers (input/output layers)
    output_size = 1  # Predicting SWE
    model = WeatherLSTM(input_size, hidden_size, num_layers, output_size)

    if to_train:
        h0, c0 = None, None
        # if input().lower() == 'y':
        print("Splitting data...")
        #train_data, test_data = train_test_split(SWEdataset, test_size=0.2, shuffle=True)
        print("Training model...")
        h0, c0 = train_model(model, dataloader, EPOCHS)  # Pass dataloader to train_model
        print("Saving model...")
        save_model(model)
    else:
        load_model(model, "./SWE_Predictor_8_10epochs.pth")
        fmin = SWEdataset.feature_scaler.data_min_
        fmax = SWEdataset.feature_scaler.data_max_
        tmin = SWEdataset.target_scaler.data_min_
        tmax = SWEdataset.target_scaler.data_max_
        print(fmin, fmax, tmin, tmax)
        print(type(fmin))

        #inp = torch.tensor([33.65352,-109.30877,int(pd.to_datetime("1991-01-01").timestamp()*(10**9)),9027,0.888151729,2.16,-9.3,7.92,117.325,0.0016,19.78,69.42,0.0,0.0276550772915165], dtype=torch.float64)
        # inp = SWEdataset.inptest
        # inp = np.array([45.19085,-119.25392,int(pd.to_datetime("1991-01-01").timestamp()*(10**9)),5770,-0.124565305,4.01,-7.94,-0.86,71.25,0.0025,64.75,92.91,0.0,0.0448925261040153])
        # inp = np.array()
        inp = SWEdataset.features[0:100]
        inp = torch.tensor(np.array(inp))


#         # print((inp-fmin))
#         # print((fmax-fmin))
#         # input()
#         inp = (inp-fmin)/(fmax-fmin)
#         inp[np.isnan(inp)] = 0
#         inp = torch.tensor(inp)
#         print(inp)
#         # inp = torch.tensor(inp, dtype=torch.float32)

#         # TODO::: WORRY ABOUT THE dtype F64??

        # print(inp.view(-1).tolist())
        print(inp.shape)
        # inp = inp.view(1, -1)
        pred = predict_single_day(model, inp.type(torch.float32))[0]

        pred = pred.cpu()

        print(pred.tolist()[0])


        print(tmax, tmin)
        scaled_pred = (pred * (tmax-tmin)) + (tmin)
        print(scaled_pred.tolist()[0])
