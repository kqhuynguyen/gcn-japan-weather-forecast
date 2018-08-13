import numpy as np
import pandas as pd
import os
from datetime import datetime

# parameters for the time series
periods = 276048 # number of periods is the same in each of the sensor's data
locations = 1252 # number of sensor locations
start_date = datetime(2012, 1, 1, 0, 10) # the starting date for the data collection
freq = "10T" # the frequency of the collection. In this case it is per 10 minutes
# aids = pd.read_csv("datasets/japan/amd_master.tsv", delimiter="\t", usecols=["aid"]).loc[:,"aid"] # the ids of the sensors
datapath = "datasets/japan"
forecast_feature = "max_tp" # the feature we want to forecast
train_val_split = 0.6 # point at which we split the dataset into train and val
val_test_split = 0.8 # point at which we split the dataset into val and test
sampling_rate = "6H" # sampling rate, for resampling, thus reducing the size of the data
train_fname = os.path.join("datasets/japan", "amd_{}_{}_train.csv".format(sampling_rate, forecast_feature)) # the train output file path
test_fname = os.path.join("datasets/japan", "amd_{}_{}_test.csv".format(sampling_rate, forecast_feature)) # the test output file path
val_fname = os.path.join("datasets/japan", "amd_{}_{}_val.csv".format(sampling_rate, forecast_feature)) # the validation output file path
seasonal_features = True # whether to add seasonal features to the data

# load the data
all_data_list = []
for i in range(5):
    current_amd_path = os.path.join("datasets/japan", "amd{}".format(i + 1))
    files_in_current_folder = os.listdir(current_amd_path) 
    for file_name in files_in_current_folder:
        print("Currently reading: {}".format(file_name))
        aid = file_name.split("_")[1].split(".")[0] # get the current aid
        all_data_list.append(pd.read_csv(os.path.join(current_amd_path, file_name), delimiter="\t", usecols=[forecast_feature]).rename(columns={forecast_feature: aid})) # append the dataframe into list
print("Concatenating the dataframes...")
output_dataframe = pd.concat(all_data_list, axis=1) # concat the dataframes in the list into a new dataframe
output_dataframe.set_index(pd.date_range(start=start_date, periods=periods, freq=freq), inplace=True) # set a DatetimeIndex for the dataframe
print("Dealing with N/A's...")
output_dataframe.dropna(axis=1, thresh=250000, inplace=True) # only keep sensors with at least 250000 non-NA values
output_dataframe.fillna(method="bfill", axis=1, inplace=True) # fill the NA values with back fill
output_dataframe.fillna(method="ffill", axis=1, inplace=True) # fill the NA values with forward fill because some NA values have no precedents
print("Resampling the data...")
output_dataframe = output_dataframe.resample(sampling_rate).mean() # resample the data

if (seasonal_features):
    output_dataframe.loc[output_dataframe.month.isin(range(3,6)),"spring"] = 1
    output_dataframe.loc[output_dataframe.month.isin(range(6,9)),"summer"] = 1
    output_dataframe.loc[output_dataframe.month.isin(range(9,11)),"autumn"] = 1
    output_dataframe.loc[output_dataframe.month.isin([12,1,2]),"winter"] = 1

print("Splitting the data...")
total_length = len(output_dataframe)
train, val, test = np.split(output_dataframe, [int(.6 * total_length), int(.8 * total_length)]) # split the data into train, validation and test sets
# dealing with NA values
print("Saving the data...")
train.to_csv(train_fname, index_label="datetime")
test.to_csv(test_fname, index_label="datetime")
val.to_csv(val_fname, index_label="datetime")
print("Successfully preprocessed.")