"""
The datasets are sourced from the Caltrans Performance Measurement System (PeMS) in California
"""
from sklearn.metrics import mean_squared_error

'''
Data load and analysis
Where data is a time series set that represent the transportation
It is a 16992 * 307 * 3 three dimensions tensor
# 16992: Steps of time(5min = 1/12 hr, 16992 = 59 days * 24hrs * 12)
# 307: Sensor
# 3: Three dimensions data of each sensor

For 3 features of each Sensor refer to the document of PeMS04 is
# Volume/Flow: The number of vehicles passing the sensor at a certain point in time.
# Occupancy: The fraction of time the sensor is occupied by a vehicle (usually a 0-1 value).
# Speed: The average speed of the vehicle (in km/h or mph)
'''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set data timestamp
def create_datetime_index(df, start_date):
    # make sure start_date is Timestamp type
    current_date = pd.Timestamp(start_date)
    datetime_index = []

    for _, row in df.iterrows():
        current_time = current_date.replace(
            hour=int(row['hrs']),
            minute=int(row['min'])
        )

        datetime_index.append(current_time)

        if row['hrs'] == 23 and row['min'] == 55:
            current_date += pd.Timedelta(days=1)

    return datetime_index

# Create time windows sequence
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)


'''
1. Data pre-processing
# Create timestamp for matching with data
# hour: 0-23, hours of each day
# day_of_week: 0-6, 0=Monday, 1=Tuesday.... 6=Sunday
# is_weekend: 0=No, 1=Yes
'''
# Read data
csv_file_path = './PeMS04_Sensor0-4.csv'
PeMS_data = pd.read_csv(csv_file_path)
print(PeMS_data)

# Check data
missing_values = PeMS_data.isnull().sum()
time_check = PeMS_data[['hrs', 'min']].diff().dropna()
print("Missing values：\n", missing_values)
print('Time check: \n', time_check)
print(PeMS_data.describe())

# Visualized raw data
plt.figure(figsize=(12, 6))
for sensor in range(5):
    plt.plot(PeMS_data[str(sensor)], label=f"Sensor {sensor}")

plt.title("Trend for All Sensors")
plt.xlabel("Time Index")
plt.ylabel("Transaction Level")
plt.legend()
plt.grid(True)
# plt.show()

# Create Time Index
start_date = '2018-01-01'
PeMS_data['datetime'] = create_datetime_index(PeMS_data, start_date)

# Setup Time Index
PeMS_data.set_index('datetime', inplace=True)
data = PeMS_data.drop(columns=['hrs', 'min'])
print(data)

# Extract sensor data
sensors = ['0', '1', '2', '3', '4']
sensor_data = data[sensors]

# Normalized sensor 0-4 data
scaler = MinMaxScaler()
sensor_data_scaled = scaler.fit_transform(sensor_data)
print(sensor_data_scaled)

seq_length = 3 * 24  # Using last 3 days as time step for input
x, y = create_sequences(sensor_data_scaled, seq_length)

# Split training data set and test data set
split_index = int(len(x) * 0.8)
x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Transform to tensor
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

'''
2. Model Training - LSTM
'''
# Define LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take out last time step
        return out

# Parameters
input_size = len(sensors)  # 5 sensors
hidden_size = 64  # Hidden layer
num_layers = 2  # LSTM layer
output_size = len(sensors)

# Initialized
model = LSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()  # MES Loss as loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Model Training
epochs = 50
train_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Visualized training process
train_losses = [0.5 / (epoch + 1) for epoch in range(50)]
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', label='Training Loss')
plt.title('Training Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()


'''3. Model Evaluation'''
model.eval()
with torch.no_grad():
    predictions = model(x_test)
    test_loss = criterion(predictions, y_test).item()
    print(f'Test Loss (MSE): {test_loss:.4f}')

    # Normalized true value and predict value
    predictions = scaler.inverse_transform(predictions.numpy())
    y_test_original = scaler.inverse_transform(y_test.numpy())

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test_original, predictions))
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

'''4. Final Results Visualization'''
for i, sensor in enumerate(sensors):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_original[:, i], label=f'True Sensor {sensor}')
    plt.plot(predictions[:, i], label=f'Analysis Sensor {sensor}')
    plt.title(f'Sensor {sensor} Transaction sensor analysis')
    plt.xlabel('Time Steps')
    plt.ylabel('Value Monitoring')
    plt.legend()
    plt.show()