from os import listdir, makedirs
from os.path import isfile, join, dirname
from datetime import datetime
import pandas as pd

kanto_data_path = "./data/tokyo"
resampled_data_path = "./resampled"
file_names = [f for f in listdir(kanto_data_path) if isfile(join(kanto_data_path,f))] # list all files

def parser(x):
    if x[-4:] == '2400':
        x = x[:-4] + '0000'
    return datetime.strptime(x, "%Y%m%d%H%M")
                             
def load_sensor_data(name):
    df = pd.read_csv(join(kanto_data_path, name), delimiter='\t', parse_dates=[0], date_parser=parser, index_col=0)
    return df

def write_sensor_data(sensor_dataframe, destination):
    makedirs(dirname(destination), exist_ok=True)
    sensor_dataframe.to_csv(destination)

def load_resampled_sensor_data(name):
    df = pd.read_csv(join(resampled_data_path, name))
    df.set_index(pd.DatetimeIndex(df['datetime']), inplace=True)
    df = df.drop("datetime", axis=1)
    return df