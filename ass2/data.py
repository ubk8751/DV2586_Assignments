import os
import joblib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import *

def get_data(path:str="./ass2/dataset2.csv", export_df:bool=False):
    if os.path.exists("./ass2/df/traindf.tfds") and os.path.exists("./ass2/df/valdf.tfds"):
        train = tf.data.Dataset.load("./ass2/df/traindf.tfds")
        val = tf.data.Dataset.load("./ass2/df/valdf.tfds")
    else:
        if os.path.exists(path) and path.split(".")[-1] == "csv":
            # Get data as PD dataframe
            print(f'Found data file at "{path}"!')
            df, data_cols = _get_df(path=path)

            # Plot sensor data
            _plot_data(df)
            
            # Normalize per column
            for col in data_cols:
                _normalize_col(df, col)
            
            # Sequencify the data
            Sdfx = _sequencify_df(df=df, sl=5, data_columns=data_cols)
            
            # The goal is to recreat the data, so our y = x
            Sdfy = Sdfx

            ds = tf.data.Dataset.from_tensor_slices((Sdfx, Sdfy))

            train, val = _train_val_split(ds=ds, vs_split=0.3, batch_size=64, buffer_size=128)

            if export_df:
                train.save("./ass2/df/traindf.tfds")
                val.save("./ass2/df/valdf.tfds")

    return train, val

def _plot_data(ds):
    fig, ax = plt.subplots(figsize=(14,6),dpi=80)
    ax.plot(ds['Bearing 1'], label="Bearing 1", color='Blue', animated=True, linewidth=1)
    ax.plot(ds['Bearing 2'], label="Bearing 2", color='Red', animated=True, linewidth=1)
    ax.plot(ds['Bearing 3'], label="Bearing 3", color='Green', animated=True, linewidth=1)
    ax.plot(ds['Bearing 4'], label="Bearing 4", color='Black', animated=True, linewidth=1)
    plt.legend(loc="lower left")
    ax.set_title("Sensor data")
    plt.savefig("./ass2/img/SensorData.jpg")

def _get_df(path:str="./ass2/dataset2.csv"):
    df = pd.read_csv(path, header=0, index_col="timestamp", delimiter=",")
    return (df, df.columns)

def _normalize_col(df, col):
    df[col] = df[col] / df[col].max()

def _add(row):
    if row[3] + row[4] + row[5] > 1:
        return 1
    return 0

def _sequencify_df(df, sl:int=5, data_columns:list=["Bearing 1", "Bearing 2", "Bearing 3", "Bearing 4"]):
    xs = []
    df_length = len(df.index)
    for i in range(df_length-sl):
        data_point = [df[data_col].iloc[i:i+sl].to_list() for data_col in data_columns]
        data_point = [d for d in zip(*data_point)]
        xs.append(data_point)
    return xs

def _train_val_split(ds, vs_split:float=0.3, batch_size:int=64, buffer_size:int=128):
    count = tf.data.experimental.cardinality(ds)
    val_count = tf.floor(tf.cast(count, tf.float32) * vs_split).numpy()
    val_ds = ds.take(val_count)
    ds = ds.skip(val_count)
    
    ds = ( 
        ds
        .shuffle(buffer_size=buffer_size)
        .batch(batch_size=batch_size)
    )
    
    val_ds = (
        val_ds
        .batch(1000)
    )
    
    return ds, val_ds



if __name__ == "__main__":
    train, val = get_data(path="./ass2/dataset2.csv")
    print(train, val)