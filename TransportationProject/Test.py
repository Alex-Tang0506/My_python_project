import pandas as pd
import numpy as np

csv_file_path = './PEMS04.csv'
npz_file_path = './PEMS04.npz'
data_npz = np.load(npz_file_path)
csv_data = pd.read_csv(csv_file_path)
# print(csv_data)
# print("Keys in the .npz file:", data_npz.keys())
traffic_data = data_npz['data']
print(traffic_data.shape)

# Extract data from .npz
occupancy_data = traffic_data[:,:,1]
flow_data = traffic_data[:,:,2]
speed_data = traffic_data[:,:,0]
print('Occupancy Data\n',occupancy_data)
print('Speed Data:\n',speed_data)


timestamps = pd.date_range(start="2018-01-01", periods=16992, freq="5min")
time_features = pd.DataFrame({
    "hrs": timestamps.hour,
    "min": timestamps.minute,
    "day_of_week": timestamps.dayofweek,
    "is_weekend": (timestamps.dayofweek >= 5).astype(int)
})

combined_data = pd.concat([time_features, pd.DataFrame(occupancy_data[:,:5])], axis=1)
print(combined_data)
# combined_data.to_csv('PeMS04_Sensor0-4.csv', index=False)
