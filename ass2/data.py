# Used since I save some files and stuff
import os

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import functions
from sklearn.preprocessing import MinMaxScaler
from typing import *

# Defined by choosing approximately the 700:th item in the dataset, didn't want to do this one dynamically.
split_date = "2004-02-17 07:12:39"

def get_data(path:str="./ass2/dataset2.csv", export_df:bool=False):
    if os.path.exists("./ass2/df/traindf.tfds") and os.path.exists("./ass2/df/valdf.tfds"):
        train = tf.data.Dataset.load("./ass2/df/traindf.tfds")
        val = tf.data.Dataset.load("./ass2/df/valdf.tfds")
    else:
        if os.path.exists(path) and path.split(".")[-1] == "csv":
            # Get data as PD dataframe
            print(f'Found data file at "{path}"!')
            df, data_cols = _get_df(path=path)
            print(f'DF shape: {df.shape}')
            print(df.head(5))

            # There are no NaN values in the data, but just to be sure, we replace any NaN values with 0 
            # since we still want the data points, and it'll result in an anomaly later on.
            df.fillna(0)

            # Plot sensor data
            _plot_data(df)

            # After plotting the data we can see that it is not optimal as it more or less stay the same then 
            # suddenly rise quicly, menaing that all anomalies will be at the end. W can however say that any 
            # unnecessary removal of outliers wouldn't make it better, so we skip that step.
            
            # Normalize data per column. We do not wish to normalize over the entire dataset because it could 
            # affect any anomlies.
            scalers = [_normalize_data(df=df, col=col) for col in data_cols]

            # Split the dataset into train and validation set
            train, val = _train_val_split(ds=df)
            
            # Sequencify the data
            trainDS = _sequencify_df(df=train, sl=5, data_columns=data_cols)
            print(f'Norm train data shape: {trainDS.shape}')
            print(trainDS[0])

            valDS = _sequencify_df(df=val, sl=5, data_columns=data_cols)
            print(f'Norm val data shape: {valDS.shape}')
            print(valDS[0])
            
            if export_df:
                train.save("./ass2/df/traindf.tfds")
                val.save("./ass2/df/valdf.tfds")

    return ((trainDS, trainDS), (valDS, valDS), scalers, df)

# Basic data plotting function
def _plot_data(ds):
    fig, ax = plt.subplots(figsize=(14,6),dpi=80)
    ax.plot(ds['Bearing 1'], label="Bearing 1", color='Blue', animated=True, linewidth=1)
    ax.plot(ds['Bearing 2'], label="Bearing 2", color='Red', animated=True, linewidth=1)
    ax.plot(ds['Bearing 3'], label="Bearing 3", color='Green', animated=True, linewidth=1)
    ax.plot(ds['Bearing 4'], label="Bearing 4", color='Black', animated=True, linewidth=1)
    plt.legend(loc="lower left")
    ax.set_title("Sensor data")
    plt.savefig("./ass2/img/SensorData.jpg")

# Function for turning a CSV file into a pandas dataframe
def _get_df(path:str="./ass2/dataset2.csv"):
    df = pd.read_csv(path)
    return (df, ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4'])

# Normalizes the dataframe at df[[col]] using MinMaxScaling.
def _normalize_data(df, col):
    scaler = MinMaxScaler()
    df[[col]] = scaler.fit_transform(df[[col]])
    return scaler

# Sequnecifies the data into a numpy array of of sl long sequences. 
def _sequencify_df(df, sl:int=5, data_columns:list=["Bearing 1", "Bearing 2", "Bearing 3", "Bearing 4"]):
    xs = []
    df_length = len(df.index)
    for i in range(df_length-sl+1):
        data_point = [df[data_col].iloc[i:i+sl].to_list() for data_col in data_columns]
        data_point = [d for d in zip(*data_point)]
        xs.append(data_point)
    return np.array(xs)

# Splits the dataframe into training and validation datasets at split_date from the data.
def _train_val_split(ds):
    train_ds = ds[ds["timestamp"] < split_date]
    val_ds = ds[ds["timestamp"] >= split_date].reset_index()
    return train_ds, val_ds

if __name__ == "__main__":
    train, val = get_data(path="./ass2/dataset2.csv")
    #print(train, val)