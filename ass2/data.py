import os
import joblib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


"""
Private function for  turning a pandas dataframe

Input:
    data_path (str)     : path to csv data file

Output:
    df (pd.DataFrame)   : A pandas dataframe        
"""
def _get_df(data_path:str = "./ass2/ass2data.csv", index:str="timestamp"):
    if index=="Numbers":
        df = pd.read_csv(
            data_path, parse_dates=True
        )
    else:
        df = pd.read_csv(
            data_path, parse_dates=True, index_col=index
        )
    return df

def _add(row):
    if row[3] + row[4] + row[5] > 1:
        return 1
    return 0

"""
Function for retrieving a pandas data frame and clean it up

Input:
    df (pd.DataFrame)   : A pandas dataframe
    filter (bool)       : A boolean for deciding wether to filter out the top 
                          and bottom [quantile] percent of data in an attempt 
                          to remove some outliers.
    quantile (float)    : only necessary if filter == True. The qunatile to 
                          be cut off on top and bottom.

Output:
    df (pd.DataFrame)   : A clenaed pandas dataframe        
"""
def _clean_data(df:pd.DataFrame=None, filter:bool=True, quantile:float=0.01):
    init_row_count = len(df. index)
    # Drop any NaN values (if necessary)
    df.dropna()

    # Replace "normal" status with 0 and "anomaly" status with 1
    df.replace("normal", 0, inplace=True)
    df.replace("anomalous", 1, inplace=True)
    
    # Create a "status column"
    df["status"] = df.apply(_add, axis=1)
    df = df.drop(["temperature_status","pressure_status","humidity_status"], axis=1)
    
    # Clean out any values in the top and bottom 1 percentiles
    if filter:
        for col in ["temperature", "pressure", "humidity"]:
            q_low = df[col].quantile(quantile)
            q_hi  = df[col].quantile(1-quantile)

            df= df[(df[col] < q_hi) & (df[col] > q_low)]
    else:
        print("Did not remove lower and upper quantiles of data.")
        print()
    removed_row_count = init_row_count - len(df.index)
    return df, removed_row_count

"""
Private function for displaying the data in a pandas dataframe as a boxplot and 
a distance plot for visualization.

Input:
    df (pd.DataFrame)   : A pandas dataframe

Output:
    -         
"""
def _display(df):
    # Some path management
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'img/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        
    # Create plots to view tha data
    sns.boxplot(x = df["temperature"])
    plt.savefig(results_dir + "temp_boxplot.jpg")
    sns.displot(df["temperature"], bins=15, kde=False)
    plt.savefig(results_dir + "temp_displot.jpg")
        
    sns.boxplot(x = df["pressure"])
    plt.savefig(results_dir + "pressure_boxplot.jpg")
    sns.displot(df["pressure"], bins=15, kde=False)
    plt.savefig(results_dir + "pressure_displot.jpg")

    sns.boxplot(x = df["humidity"])
    plt.savefig(results_dir + "humidity_boxplot.jpg")
    sns.displot(df["humidity"], bins=15, kde=False)
    plt.savefig(results_dir + "humidity_displot.jpg")

"""
Function for plotting one column's data in a pandas dataframe over time and save 
as an image file

Input:
    df (pd.DataFrame)   : A pandas dataframe
    col_name (str)      : The name of the column to plot.

Output:
    -         
"""
def plot_column(df, col_name:str="temperature"):
    # Some path management
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'img/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    
    plt.plot(df["timestamp"],df[col_name])
    plt.xlabel("Time")
    plt.ylabel(col_name)
    plt.legend()
    plt.savefig(results_dir + col_name + "_plot.png")

"""
Function for splitting a pandas dataframe into a test and a train daframe.

Input:
    df (pd.DataFrame)   : A pandas dataframe
    test_size (float)   : The percentage of the dataframe that should be test data. 

Output:
    train (pd.DataFrame): A pands dataframe
    test (pd.DataFrame) : A pandas dataframe   
"""
def get_train_test(df, test_size:float=0.3):
    train, test = train_test_split(df, test_size=test_size)
    train = pd.DataFrame(train, columns=df.columns)
    test = pd.DataFrame(test, columns=df.columns)
    print(f'Train shape: {train.shape}; Test shape: {test.shape}')
    print()
    return train, test

"""
Function for normalizing data in the training and testing datasets

Input:
    train (pd.DataFrame) : A pandas dataframe
    test (pd.DataFrame)  : A pandas dataframe

Output:
    s_train(pd.DataFrame): A normalized pands dataframe
    s_test(pd.DataFrame) : A normalized pandas dataframe   
"""
def normalize_df(train, test):
    scaler = MinMaxScaler()
    s_train = scaler.fit_transform(train)
    s_test = scaler.transform(test)
    joblib.dump(scaler, "scaler")
    sr_train = s_train.reshape(s_train.shape[0], 1, s_train.shape[1])
    sr_test = s_test.reshape(s_test.shape[0], 1, s_test.shape[1])
    print(f'Normalized train shape: {sr_train.shape}; Normalized test shape: {sr_test.shape}')
    print()
    return sr_train, sr_test

"""
Function for retrieving a pandas data frame and clean it up

Input:
    data_path (str)     : path to csv data file
    plot_df   (bool)    : True: create plots of data; False: do not create plots of data

Output:
    df (pd.DataFrame)   : A clenaed pandas dataframe        
"""
def get_df_and_cleanup(data_path:str = "./ass2/ass2data.csv", plot_df:bool=False, filter:bool=True, quantile:float=0.01):
    if os.path.exists(data_path) and data_path.split(".")[-1] == "csv":
        print(f'Found data file at "{data_path}"!')
        df = _get_df(data_path=data_path, index = "timestamp")
        print(f'Shape of dataframe before cleaning: {df.shape}')
        print(f'Number of NaN values in data: {df.isna().sum().sum()}')
        if plot_df:
            tdf = _get_df(data_path=data_path, index="Numbers")
            _display(tdf)
        df, rm_rows = _clean_data(df, filter=filter, quantile=quantile)
        print(f'Shape of dataframe after cleaning: {df.shape}. Removed rows: {rm_rows}')
        print()
        return df

def get_x_y(ds):
    x = []
    y = []
    for r in ds:
        y.append(r[0][-1])
        x.append(r[0][:-1])
    print(x)
    x = pd.DataFrame(x,columns=["temperature", "pressure", "humidity"])
    y = pd.DataFrame(y,columns=["status"])
    # y = np.array([x[-1] for x[0] in ds]).reshape(1,ds.shape[2])
    # x = np.array([x[:-1] for x[0] in ds]).reshape(ds.shape[1],1,ds.shape[2])
    return x, y

if __name__ == "__main__":
    df = get_df_and_cleanup("./ass2/ass2data.csv", True)
    print(df.head(2))